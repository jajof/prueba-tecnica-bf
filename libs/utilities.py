from copy import deepcopy
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder, TargetEncoder, MinMaxScaler
from sklearn.model_selection import KFold, cross_val_score, TimeSeriesSplit
import optuna
import joblib
import numpy as np
import pandas as pd
import operator
from importlib import reload
import os
from datetime import datetime
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, median_absolute_error

plt.style.use('seaborn-v0_8-dark')
palette = sns.color_palette()

class OptunaCallback(object):
    """
    Callback for Optuna for logging and early stopping

    Args:
        early_stopping_rounds (int): Number of rounds to wait for improvement.
        direction (str): Direction to optimize ("minimize" or "maximize").

    Attributes:
        early_stopping_rounds (int): Number of rounds to wait for improvement.
        _iter (int): Internal counter for tracking the number of rounds without improvement.
        logger: logger object
        _operator (function): Operator for comparing study.best_value with the current score.
        _score (float): Current best score.

    """

    def __init__(self, early_stopping_rounds: int, direction: str = "maximize") -> None:
        """
        Initialize EarlyStoppingCallback.

        Args:
            early_stopping_rounds (int): Number of rounds to wait for improvement.
            direction (str): Direction to optimize ("minimize" or "maximize").

        """
        self.early_stopping_rounds = early_stopping_rounds
        self._iter = 0

        if direction == "minimize":
            self._operator = operator.lt
            self._score = np.inf
        elif direction == "maximize":
            self._operator = operator.gt
            self._score = -np.inf
        else:
            raise ValueError(f"Invalid direction: {direction}")

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """
        Perform early stopping.

        Args:
            study (optuna.Study): Optuna study.
            trial (optuna.Trial): Optuna trial.

        """
        if self._operator(study.best_value, self._score):
            self._iter = 0
            self._score = study.best_value
        else:
            self._iter += 1


        if self._iter >= self.early_stopping_rounds:
            study.stop()

class Model():
    """
    Custom model for training and prediction using Optuna for hyperparameter tuning.

    Args:
        base_estimator: Base estimator for the model.
        vars (dict): Dictionary containing continuous and categorical variables.
        partition_by (str): Column name for data partitioning.
        study_parameters (dict): Dictionary containing hyperparameter names, types, min, and max values.
        early_stopping_rounds (int): Number of early stopping iterations.

    Attributes:
        cont_vars (list): List of continuous variables.
        cat_var_low_card (list): List of categorical variables with low cardinality.
        cat_var_high_card (list): List of categorical variables with high cardinality.
        partition_by (str): Column name for data partitioning.
        base_estimator: Base estimator for the model.
        initialized_model: Initialized model pipeline.
        models_dict (dict): Dictionary to store models for each partition.
        studies_dict (dict): Dictionary to store Optuna studies for each partition.
        study_parameters (dict): Dictionary containing hyperparameter names, types, min, and max values.
        iteraciones_early_stop (int): Number of early stopping iterations.

    """

    def __init__(self, base_estimator, vars, study_parameters, partition_by=None, early_stopping_rounds=5):
        """
        Initialize Modelo.

        Args:
            estimador_base: Base estimator for the model.
            variables (dict): Dictionary containing continuous and categorical variables.
            particionado_por (str): Column name for data partitioning.
            parametros_estudio (dict): Dictionary containing hyperparameter names, types, min, and max values.
            iteraciones_early_stop (int): Number of early stopping iterations.

        """
        self.cont_vars = vars.get('continuous', [])
        self.cat_var_low_card = vars.get('categorical_low_cardinality', [])
        self.cat_var_high_card = vars.get('categorical_high_cardinality', [])
        self.partition_by = partition_by
        self.base_estimator = deepcopy(base_estimator)
        self.initialized_model = None
        self.models_dict = {}
        self.studies_dict = {}
        self.study_parameters = study_parameters
        self.iteraciones_early_stop = early_stopping_rounds
        self.estimator_id = f'{type(base_estimator).__name__}_{partition_by}'

        self.initialize_model()


    def initialize_model(self):
        """
        Initialize the model pipeline.

        """
        transformers = []
        if len(self.cat_var_high_card) > 0:
            transformers.append(('TargetEncoder', TargetEncoder(target_type='continuous', cv=5),
                                 self.cat_var_high_card))
        if len(self.cat_var_low_card) > 0:
            transformers.append(('OneHotEncoder', OneHotEncoder(drop='first', sparse_output=False),
                                 self.cat_var_low_card))
        encoder = ColumnTransformer(transformers, remainder='passthrough', verbose_feature_names_out=False)
        pipeline = Pipeline([('Encoder', encoder),
                             ('Scaler', MinMaxScaler()),
                             ('Estimator', self.base_estimator)])
        self.current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.initialized_model = deepcopy(pipeline)

    def get_data(self, X, y=None, partition_value=None):
        """
        Extract data for a specific partition.

        Args:
            X: Input features.
            y: Target variable.
            partition_value: Value for data partitioning.

        Returns:
            tuple: Tuple containing X and y data.

        """
        _X = X.copy()
        columns_list = self.cont_vars + \
                       self.cat_var_high_card + \
                       self.cat_var_low_card

        if self.partition_by:
            _X = _X.loc[_X[self.partition_by] == partition_value, columns_list].copy()
        else:
            _X = _X[columns_list].copy()
        if y is not None:
            return _X, y[_X.index]
        else:
            return _X

    def build_optuna_dictionary(self, trial):
        """
        Build a dictionary for hyperparameters from Optuna trial.

        Args:
            trial (optuna.Trial): Optuna trial.

        Returns:
            dict: Dictionary containing hyperparameter values.

        """
        aux_dict = {}  # Dictionary for hyperparameters
        for par_name, value in self.study_parameters.items():
            data_type, min, max = value[0], value[1], value[2]
            if data_type == 'int':
                parametro = trial.suggest_int(par_name, min, max)
            elif data_type == 'float':
                parametro = trial.suggest_float(par_name, min, max)
            elif data_type == 'str':
                parametro = trial.suggest_categorical(par_name, [min, max])
            aux_dict[par_name] = parametro
        return aux_dict

    def study(self, X, y, iter=1):
        """
        Optimize hyperparameters using Optuna.

        Args:
            X: Input features.
            y: Target variable.
            iter (int): Number of optimization iterations.

        """
        try:
            partition_values = X[self.partition_by].unique()
        except:
            partition_values = ['Total']
        for partition_value_i in partition_values:
            def objective(trial):
                aux_dict = self.build_optuna_dictionary(trial)
                trial_estimator = deepcopy(self.initialized_model)
                trial_estimator.set_params(**aux_dict)
                cv = TimeSeriesSplit(n_splits=3)
                scores = cross_val_score(trial_estimator, _X, _y, cv=cv, n_jobs=-1, scoring='neg_mean_absolute_error')
                return scores[~np.isnan(scores)].mean()                
            _X, _y = self.get_data(X, y, partition_value_i)
            if self.studies_dict.get(partition_value_i) is None:
                pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=2)
                self.studies_dict[partition_value_i] = optuna.create_study(direction="maximize", 
                                                                           pruner=pruner,
                                                                           storage="sqlite:///db.sqlite3",
                                                                           study_name = f'{self.current_time}_{self.estimator_id}_{partition_value_i}',
                                                                           load_if_exists=True)
            early_stopping = OptunaCallback(10, direction='maximize')
            self.studies_dict[partition_value_i].optimize(objective, n_trials=iter, callbacks=[early_stopping])
            params = self.studies_dict[partition_value_i].best_params
            self.models_dict[partition_value_i] = deepcopy(self.initialized_model)
            self.models_dict[partition_value_i].set_params(**params)
    def train(self, X, y):
        """
        Train the model.

        Args:
            X: Input features.
            y: Target variable.

        """
        try:
            partition_values = X[self.partition_by].unique()
        except:
            partition_values = ['Total']
        for partition_value_i in partition_values:
            _X, _y = self.get_data(X, y, partition_value_i)
            if self.models_dict.get(partition_value_i) is None:
                self.models_dict[partition_value_i] = deepcopy(self.initialized_model)
            self.models_dict[partition_value_i].fit(_X, _y)

    def predict(self, X):
        """
        Make predictions using the trained model.

        Args:
            X: Input features.

        Returns:
            pd.Series: Predicted values.

        """
        _X = X.copy()
        try:
            partition_values = X[self.partition_by].unique()
        except:
            partition_values = ['Total']
        for partition_value_i in partition_values:
            X_i = self.get_data(X, partition_value=partition_value_i)
            model_i = deepcopy(self.models_dict[partition_value_i])
            y_hat_i = pd.DataFrame(model_i.predict(X_i), columns=['y_hat'], index=X_i.index)
            _X.loc[X_i.index, ['y_hat']] = y_hat_i
        return _X['y_hat']

    def save(self, name):
        """
        Save the trained model to a file.

        Args:
            name (str): Name of the file.

        """
        joblib.dump(self, f'models/{name}_{self.partition_by}.joblib')


def load_or_create_model(base_estimator, vars, partition_by, study_parameters, early_stopping_rounds):
    """
    Load an existing model or create a new one if it doesn't exist.

    Args:
        base_estimator: Base estimator for the model.
        vars (dict): Dictionary containing continuous and categorical variables.
        partition_by (str): Column name for data partitioning.
        study_parameters (dict): Dictionary containing hyperparameter names, types, min, and max values.
        early_stopping_rounds (int): Number of early stopping iterations.

    Returns:
        tuple: Model instance and estimator name.

    """
    estimator_name = type(base_estimator).__name__
    try:
        model_1 = joblib.load(f'models/{estimator_name}_{partition_by}.joblib')
    except:
        model_1 = Model(base_estimator=base_estimator,
                        vars=vars,
                        study_parameters=study_parameters,
                        early_stopping_rounds=early_stopping_rounds,
                        partition_by=partition_by)

    return model_1, estimator_name

def plot_hist(dataframe, var, min_value, max_value):
    """
    Generate a histogram of the specified variable in the given DataFrame.

    Parameters:
    - dataframe (pd.DataFrame): The pandas DataFrame containing the data.
    - var (str): The name of the variable/column for which the histogram is to be plotted.
    - min_value (float): The minimum value for the x-axis range of the histogram.
    - max_value (float): The maximum value for the x-axis range of the histogram.

    Returns:
    - None: This function produces a matplotlib histogram plot with mean and median lines,
      as well as textual information about the mean and median.
    """
    tmp = dataframe[var].copy()
    ax = sns.histplot(tmp, color='grey', alpha=.2)
    plt.title(var)
    plt.axvline(np.mean(tmp), color='dodgerblue', label ="Media")
    plt.axvline(np.median(tmp), color='tomato', label = "Mediana")
    plt.gca().set(title= var)
    at = AnchoredText(
            f"Media: {np.mean(tmp):.2f}\nMediana: {np.median(tmp):.2f}",
            prop=dict(size="large"),
            frameon=True,
            loc="upper left",
        )
    at.patch.set_boxstyle("square, pad=0.0")
    ax.add_artist(at)
    ax.set_xlim(min_value, max_value)
    plt.legend(loc='upper right', frameon=True,)



def plot_evolutivo(df_sku, i, prod_id, path_name):
    """
    Generate an evolutionary plot for a specific product, comparing actual sales, current model predictions,
    and new model predictions over time.

    Parameters:
    - df_sku (pd.DataFrame): DataFrame containing the product data with columns 'mes', 'ventas', 'modelo_actual', 'y_hat'.
    - i (int): Index or identifier for the specific plot.
    - prod_id (str or int): Identifier for the product being plotted.
    - path_name (str): Name of the directory where the plots will be saved.

    Returns:
    - None: This function generates and saves a matplotlib line plot with three lines representing actual sales,
      current model predictions, and new model predictions over time. Additionally, it includes Mean Absolute Error (MAE)
      information for both the original and new models.
    """
    plt.figure(figsize=(16, 4))
    ax = sns.lineplot(data=df_sku, x='mes', y='ventas', label='Ventas', color=palette[0])
    sns.lineplot(data=df_sku, x='mes', y='modelo_actual', label='Modelo Actual', color=palette[1])
    sns.lineplot(data=df_sku, x='mes', y='y_hat', label='Modelo nuevo', color=palette[3])

    error_absoluto_medio_original = mean_absolute_error(df_sku['ventas'], df_sku['modelo_actual'])
    error_absoluto_medio_nuevo = mean_absolute_error(df_sku['ventas'], df_sku['y_hat'])


    at = AnchoredText(
                f"MAE original: {error_absoluto_medio_original:.2f}\nMAE nuevo: {error_absoluto_medio_nuevo:.2f}",
                prop=dict(size="large"),
                frameon=True,
                loc="upper left",
            )
    at.patch.set_boxstyle("square, pad=0.0")
    ax.add_artist(at)
    ax.set(title=f'Gráfico producto {prod_id}', ylabel='Unidades')
    plt.legend(loc='lower right', frameon=True,)

    plt.tight_layout()
    plt.savefig(f'plots/evolutivos_{path_name}/{i:03d}_{prod_id}.png')



def reportar_error(y_true, y_hat):
    """
    Report Mean Absolute Error (MAE) and Median Absolute Error (MedAE) between true and predicted values.

    Parameters:
    - y_true (array-like): Ground truth (actual) values.
    - y_hat (array-like): Predicted values.

    Returns:
    - None: This function prints the calculated Mean Absolute Error (MAE) and Median Absolute Error (MedAE)
      between the true and predicted values."""

    error_absoluto_medio = mean_absolute_error(y_true, y_hat)
    error_absoluto_mediano = median_absolute_error(y_true, y_hat)
    print(f'Error absoluto medio: {error_absoluto_medio:.2f}')
    print(f'Error absoluto mediano: {error_absoluto_mediano:.2f}')




