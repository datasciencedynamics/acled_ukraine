################################################################################
######################### Import Requisite Libraries ###########################
################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import mlflow
from mlflow.tracking import MlflowClient
import pickle
import os
import networkx as nx
import shap
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    precision_score,
    recall_score,
    brier_score_loss,
    precision_score,
    average_precision_score,
)

from tqdm import tqdm

from core.constants import (
    mlflow_artifacts_data,
    mlflow_models_data,
    databricks_username,
)


################################################################################
############################## Preprocessing ###################################
#                                                                              #
# ########################## Cleaning DataFrames ###############################
################################################################################


def clean_dataframe(df, cols_with_thousand_separators=None):
    """
    Cleans a pandas DataFrame by replacing specific values with NaN, optionally
    removing thousand separators, and converting columns to numeric types.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be cleaned.

    cols_with_thousand_separators : list of str, optional
        A list of column names that contain thousand separators and need to be
        processed. If None, this step is skipped.

    Returns
    -------
    pandas.DataFrame
        The cleaned DataFrame.
    """

    # Step 1: Replace None and blank values with NaN
    replacements = {
        None: np.nan,
        "": np.nan,
        "-{2,}": np.nan,
        "\.{2,}": np.nan,
    }

    for col in tqdm(df.columns, desc="Replacing values in columns"):
        for to_replace, value in replacements.items():
            if to_replace is None:
                df[col] = df[col].map(lambda x: value if x is to_replace else x)
            else:
                df[col] = df[col].replace(to_replace, value, regex=True)

    # Step 2: Remove thousand separators and convert to numeric if specified
    if cols_with_thousand_separators:
        desc_text = "Processing columns with thousand separators"
        for col in tqdm(cols_with_thousand_separators, desc=desc_text):
            if col in df.columns:
                if df[col].dtype == "object":
                    df[col] = df[col].str.replace(",", "", regex=False)
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # Step 3: Convert all other columns to numeric if possible
    for col in tqdm(df.columns, desc="Converting columns to numeric"):
        if (
            not cols_with_thousand_separators
            or col not in cols_with_thousand_separators
        ):
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                # If conversion fails, keep the column as is
                pass

    return df


def clean_feature_selection_params(pipeline_steps, tuned_parameters):
    """
    Remove feature selection parameters from tuned_parameters if RFE is not in
    0pipeline_steps.

    Args:
        pipeline_steps (list): List of tuples containing pipeline steps
        (name, estimator).
        tuned_parameters (list): List of dictionaries with parameters to tune.
    """
    # Check if any step in pipeline_steps is an RFE instance
    has_rfe = any(isinstance(step[1], RFE) for step in pipeline_steps)

    # If no RFE is found, remove feature selection-related parameters
    if not has_rfe:
        for key in list(tuned_parameters[0].keys()):
            if "feature_selection" in key:
                del tuned_parameters[0][key]


def get_cat_feature_indices(preprocessor, num_cols, cat_cols):
    """
    Return categorical feature indices AFTER ColumnTransformer.
    Assumes transformer order: num → cat
    """

    if len(cat_cols) == 0:
        return []

    return list(range(len(num_cols), len(num_cols) + len(cat_cols)))


def to_str_func(X):
    # Convert categorical values to strings.
    # CatBoost can consume string categories directly, and this ensures
    # consistent dtype handling after imputation.
    return X.astype(str)


def adjust_preprocessing_pipeline(
    model_type,
    pipeline_steps,
    numerical_cols,
    categorical_cols,
    sampler=None,
):
    no_scale_models = ["xgb", "cat"]
    has_rfe = any(isinstance(step[1], RFE) for step in pipeline_steps)
    use_smote = isinstance(sampler, SMOTE)

    if model_type in no_scale_models:
        if has_rfe or use_smote:
            numerical_transformer = Pipeline(
                steps=[("imputer", SimpleImputer(strategy="mean"))],
            )
            categorical_transformer = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value="missing"),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            )
        else:
            numerical_transformer = Pipeline(
                steps=[("passthrough", "passthrough")],
            )
            if model_type == "xgb":
                categorical_transformer = Pipeline(
                    steps=[
                        (
                            "to_cat",
                            FunctionTransformer(
                                lambda X: X.astype("category"),
                                feature_names_out="one-to-one",
                            ),
                        ),
                    ]
                )
            elif model_type == "cat":
                categorical_transformer = Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant", fill_value="__MISSING__"
                            ),
                        ),
                        (
                            "to_str",
                            FunctionTransformer(
                                to_str_func,
                                feature_names_out="one-to-one",
                            ),
                        ),
                    ]
                )

        adjusted_preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_cols),
                ("cat", categorical_transformer, categorical_cols),
            ],
            remainder="passthrough",
        )

        if model_type in {"xgb"} and not has_rfe and not use_smote:
            adjusted_preprocessor.set_output(transform="pandas")

        return [
            (name, adjusted_preprocessor if name == "Preprocessor" else step)
            for name, step in pipeline_steps
        ]

    return pipeline_steps


################################################################################
# Compare 2 dataframes
################################################################################


# Function to compare two DataFrames
def compare_dataframes(df1, df2):
    if df1.shape != df2.shape:
        print("DataFrames have different shapes:", df1.shape, df2.shape)
        return

    if list(df1.columns) != list(df2.columns):
        print("DataFrames have different columns")
        print("df1 columns:", df1.columns)
        print("df2 columns:", df2.columns)
        return

    if df1.dtypes.equals(df2.dtypes) == False:
        print("DataFrames have different data types")
        print("Differences:\n", df1.dtypes.compare(df2.dtypes))
        return

    if not df1.equals(df2):
        print("DataFrames have different content")
        diff = (df1 != df2).stack()
        print(diff[diff].index.tolist())  # Show locations of differences
        return

    print("No differences found between the DataFrames!")


################################################################################


def extract_relevant_days_hcc_ccs_columns(df):
    """
    Extracts column names from the dataframe based on specific substring conditions.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe.

    Returns:
    --------
    list
        A list of unique column names that match the filtering criteria.
    """
    # Create a list 'days_to' containing all column names in 'df' that
    # include the substring "Daysto"
    days_to = [col for col in df.columns if "Daysto" in col]

    # Create a list 'hcc' containing all column names in 'df' that include
    # the substring "HCC" but do not end with "_HCC"
    hcc = [col for col in df.columns if "HCC" in col and not col.endswith("_HCC")]

    # Create a list 'ccs' containing all column names in 'df' that include
    # the substring "CCS" but do not end with "_CCS"
    ccs = [col for col in df.columns if "CCS" in col and not col.endswith("_CCS")]

    return np.unique(days_to + hcc + ccs).tolist()


################################################################################
################## Read in Train, Valid, Test Data W/ Outcomes #################
################################################################################


def load_variant_data(data_variants, data_path, outcomes_map, return_sets=None):
    """
    Load datasets for multiple data variants and organize them into dictionaries.

    Parameters:
    -----------
    data_variants : dict
        Dictionary mapping keys to data variant names.
    data_path : str
        Path to the directory containing the parquet files.
    outcomes_map : dict
        Dictionary mapping outcome names to their respective variant keys.
    return_sets : list, optional
        List of dataset names to return (e.g., ["X_train", "X_test", "y_test",
        "outcomes"]). If None, returns all datasets.

    Returns:
    --------
    tuple
        Selected datasets based on `return_sets` parameter.
    """

    # Initialize empty dictionaries
    X_train_data, X_valid_data, X_test_data = {}, {}, {}
    y_train_data, y_valid_data, y_test_data = {}, {}, {}

    # Load data for each variant
    for key, variant in data_variants.items():
        X_train_data[key] = pd.read_parquet(
            os.path.join(data_path, f"X_train_{variant}.parquet"),
        )
        X_valid_data[key] = pd.read_parquet(
            os.path.join(data_path, f"X_valid_{variant}.parquet"),
        )
        X_test_data[key] = pd.read_parquet(
            os.path.join(data_path, f"X_test_{variant}.parquet"),
        )
        y_train_data[key] = pd.read_parquet(
            os.path.join(data_path, f"y_train_{variant}.parquet"),
        )
        y_valid_data[key] = pd.read_parquet(
            os.path.join(data_path, f"y_valid_{variant}.parquet"),
        )
        y_test_data[key] = pd.read_parquet(
            os.path.join(data_path, f"y_test_{variant}.parquet"),
        )

    # Organize train, valid, and test datasets dynamically based on outcomes_map
    datasets = {
        "X_train": {
            outcome: X_train_data.get(variant)
            for outcome, variant in outcomes_map.items()
        },
        "X_valid": {
            outcome: X_valid_data.get(variant)
            for outcome, variant in outcomes_map.items()
        },
        "X_test": {
            outcome: X_test_data.get(variant)
            for outcome, variant in outcomes_map.items()
        },
        "y_train": {
            outcome: y_train_data.get(variant)
            for outcome, variant in outcomes_map.items()
        },
        "y_valid": {
            outcome: y_valid_data.get(variant)
            for outcome, variant in outcomes_map.items()
        },
        "y_test": {
            outcome: y_test_data.get(variant)
            for outcome, variant in outcomes_map.items()
        },
        "outcomes": {
            outcome: {
                "X_train": X_train_data.get(variant),
                "X_valid": X_valid_data.get(variant),
                "X_test": X_test_data.get(variant),
                "y_train": y_train_data.get(variant),
                "y_valid": y_valid_data.get(variant),
                "y_test": y_test_data.get(variant),
            }
            for outcome, variant in outcomes_map.items()
        },
    }

    ## If return_sets is specified, return only those datasets
    if return_sets:
        return tuple(datasets[ds] for ds in return_sets)

    ## Otherwise, return all datasets
    return tuple(datasets.values())


################################################################################
############################# Handle Missing Values ############################
################################################################################


def handle_missing_values(df, columns, fillna_value=None):
    """
    Handles missing values in specified columns of a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to modify.
    columns : list
        List of column names to process.
    fillna_value : optional
        The value to fill NaN values with. If None, no filling is performed.

    Returns:
    --------
    pandas.DataFrame
        The modified DataFrame with optional missing value handling.
    """
    if fillna_value is not None:
        df[columns] = df[columns].fillna(fillna_value)

    return df


################################################################################
####################### Safe Conversion of Numeric Features ####################
################################################################################


def safe_to_numeric(series):
    """
    Safely converts a pandas Series to a numeric type, handling errors explicitly.

    This function attempts to convert a pandas Series to a numeric type using
    `pd.to_numeric`. If the conversion fails due to a `ValueError` or `TypeError`,
    it will return the original Series unmodified.

    Parameters:
    -----------
    series : pandas.Series
        The input Series to be converted to a numeric type.

    Returns:
    --------
    pandas.Series
        The converted Series if the conversion is successful; otherwise, the
        original Series is returned.
    """
    try:
        return pd.to_numeric(series)
    except (ValueError, TypeError):
        return series  # If conversion fails, return the original series


################################################################################
############################## Top N Features ##################################
################################################################################


def top_n(series, n=10):
    """
    Returns the top N most frequent unique values from a Pandas Series.

    Parameters:
    -----------
    series : pandas.Series
        The input Series from which to extract the top N most frequent values.
    n : int, optional (default=10)
        The number of top values to return.

    Returns:
    --------
    set
        A set containing the N most frequently occurring unique values in the Series.
    """

    out = set(series.value_counts().head(n).index)

    return out


################################################################################


# Customize the background color for missing values using applymap
def highlight_null(val):
    """
    Highlights null (NaN) values in a DataFrame with a red background color.

    This function checks if a given value is null (NaN) and returns a CSS style
    to apply a red background color if the value is null. If the value is not
    null, it returns an empty string, meaning no styling will be applied.

    Parameters:
    -----------
    val : any
        The value to check. Typically an element of a pandas DataFrame.

    Returns:
    --------
    str
        A string representing the CSS style to apply. If the value is null,
        'background-color: red' is returned; otherwise, an empty string is
        returned.

    Examples:
    ---------
    >>> df.style.applymap(highlight_null)

    This applies the `highlight_null` function element-wise to the DataFrame
    `df`, highlighting any null values with a red background.
    """
    color = "background-color: red" if pd.isnull(val) else ""
    return color


################################################################################
####################### MLFlow Models and Artifacts ############################
################################################################################


###################### MLFlow Helper Functions #################################
def set_or_create_experiment(experiment_name, verbose=True, databricks=False):
    """
    Set up or create an MLflow experiment.

    Args:
        experiment_name (str): Name of the experiment.
        verbose (bool): Whether to print progress messages.
        databricks (bool): If True, prepend the Databricks username path.

    Returns:
        str: Experiment ID.
    """
    # Build full experiment path conditionally
    if databricks:
        full_experiment_name = databricks_username + experiment_name
    else:
        full_experiment_name = experiment_name

    if verbose:
        print(f"Using experiment path: {full_experiment_name}")

    existing_experiment = mlflow.get_experiment_by_name(full_experiment_name)

    if existing_experiment is None:
        if verbose:
            print(f"Experiment '{experiment_name}' does not exist. Creating a new one.")
        experiment_id = mlflow.create_experiment(full_experiment_name)
    else:
        experiment_id = existing_experiment.experiment_id
        if verbose:
            print(f"Using Existing Experiment_ID: {experiment_id}")

    mlflow.set_experiment(full_experiment_name)
    return experiment_id


def start_new_run(run_name):
    """
    Start a new MLflow run with the given name.

    Args:
        run_name: Name of the run.

    Returns:
        Run ID of the newly started run.
    """
    run = mlflow.start_run(run_name=run_name)
    run_id = run.info.run_id
    mlflow.end_run()
    print(f"Starting New Run_ID: {run_id} for {run_name}")
    return run_id


def get_run_id_by_name(
    experiment_name,
    run_name,
    verbose=True,
    databricks=False,
):
    """
    Query MLflow to find the run_id for the given run_name in the experiment.
    If no run exists, create a new one.

    Args:
        experiment_name: Name of the MLflow experiment.
        run_name: Name of the run to search for or create.

    Returns:
        Run ID of the most recent run matching the run_name, or a new run ID
        if none exists.
    """
    client = MlflowClient()

    # Get the experiment
    if databricks:
        experiment = mlflow.get_experiment_by_name(
            databricks_username + experiment_name
        )
    else:
        experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment {experiment_name} not found.")

    # Search for existing runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],  # Get the most recent run
    )

    if runs:
        run_id = runs[0].info.run_id  # Use the latest run_id for this run_name
        if verbose:
            print(
                f"Found Run_ID: {run_id} for run_name '{run_name}' in experiment '{experiment_name}'"
            )
    else:
        # No runs found, create a new one
        if verbose:
            print(
                f"No runs found with run_name '{run_name}' in experiment '{experiment_name}'. Creating a new run."
            )
        run_id = start_new_run(run_name)

    return run_id


################## Dump artificats (e.g. to preprocessing) #####################


def mlflow_dumpArtifact(
    experiment_name,
    run_name,
    obj_name,
    obj,
    get_existing_id=True,
    artifact_run_id=None,
    databricks=False,
    artifacts_data_path=mlflow_artifacts_data,
    artifact_format="pkl",
):
    """
    Log an object as an MLflow artifact with a persistent run ID.

    Args:
        experiment_name: Name of the MLflow experiment.
        run_name: Name of the run within the experiment.
        obj_name: Name of the artifact (without extension).
        obj: Object to serialize and log.
        get_existing_id: If True, try to reuse an existing run ID (default: True).
        artifact_run_id: Specific run ID to use (optional).
        artifacts_data_path: Path to MLflow artifacts directory
        (default: mlflow_artifacts_data from constants).
        artifact_format: Format to save the artifact ('pkl', 'csv', or 'svg').

    Returns:
        None
    """

    # Initialize or reuse the artifacts_run_id as a function attribute
    if not hasattr(mlflow_dumpArtifact, "artifacts_run_id"):
        mlflow_dumpArtifact.artifacts_run_id = None
    else:
        mlflow_dumpArtifact.artifacts_run_id = artifact_run_id
    abs_mlflow_data = os.path.abspath(artifacts_data_path)

    if databricks:
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
    else:
        mlflow.set_tracking_uri(f"file://{abs_mlflow_data}")

    # Set or create experiment
    experiment_id = set_or_create_experiment(
        experiment_name,
        databricks=databricks,
    )
    print(f"Experiment_ID for artifact {obj_name}: {experiment_id}")

    if get_existing_id:
        mlflow_dumpArtifact.artifacts_run_id = get_run_id_by_name(
            experiment_name, run_name, databricks=databricks
        )

    # Get or create a single run_id for all artifacts
    if mlflow_dumpArtifact.artifacts_run_id:
        run_id = mlflow_dumpArtifact.artifacts_run_id
        print(f"Reusing Existing Artifacts Run_ID: {run_id} for {run_name}")
    else:
        run_id = start_new_run(run_name)
        # Store the run_id for future calls
        mlflow_dumpArtifact.artifacts_run_id = run_id

    with mlflow.start_run(run_id=run_id, nested=True):
        if artifact_format == "csv":
            temp_file = f"{obj_name}.csv"

            if hasattr(obj, "to_csv"):
                # pandas DataFrame
                obj.to_csv(temp_file, index=False)
            elif isinstance(obj, str):
                # already a CSV string
                with open(temp_file, "w") as f:
                    f.write(obj)
            else:
                raise TypeError(
                    "artifact_format='csv' requires a DataFrame or CSV string"
                )

        elif artifact_format == "svg":
            # matplotlib Figure
            temp_file = f"{obj_name}.svg"

            if hasattr(obj, "savefig"):
                # matplotlib Figure object
                obj.savefig(temp_file, format="svg", bbox_inches="tight", dpi=300)
            else:
                raise TypeError(
                    "artifact_format='svg' requires a matplotlib Figure object"
                )

        else:
            # default: pickle
            temp_file = f"{obj_name}.pkl"
            with open(temp_file, "wb") as f:
                pickle.dump(obj, f)

        mlflow.log_artifact(temp_file)
        os.remove(temp_file)

    print(
        f"Artifact {obj_name}.{artifact_format} logged successfully in MLflow "
        f"under Run_ID: {run_id}."
    )
    return None


################# Load artificats (e.g. from preprocessing) ####################


def mlflow_loadArtifact(
    experiment_name,
    run_name,  # Use run_name to query the single artifacts run_id
    obj_name,
    verbose=True,
    databricks=False,
    artifacts_data_path=mlflow_artifacts_data,
):
    """
    Load an object from MLflow artifacts by experiment and run name.

    Args:
        experiment_name: Name of the MLflow experiment.
        run_name: Name of the run within the experiment.or
        obj_name: Name of the artifact (without .pkl extension).

    Returns:
        Deserialized object from the artifact.

    Raises:
        ValueError: If experiment or run is not found.
    """
    abs_mlflow_data = os.path.abspath(artifacts_data_path)

    if databricks:
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
    else:
        mlflow.set_tracking_uri(f"file://{abs_mlflow_data}")

    set_or_create_experiment(experiment_name, verbose=verbose, databricks=databricks)

    # Get the run_id using the helper function
    run_id = get_run_id_by_name(
        experiment_name,
        run_name,
        verbose=verbose,
        databricks=databricks,
    )

    # Download the artifact from the run's artifact directory
    client = MlflowClient()

    local_path = client.download_artifacts(run_id, f"{obj_name}.pkl")
    with open(local_path, "rb") as f:
        obj = pickle.load(f)
    return obj


################### Return model metrics to be used in MlFlow ##################


def return_model_metrics(inputs: dict, model, estimator_name) -> pd.Series:
    """
    Compute and return model performance metrics for multiple input types.

    Parameters:
    ----------
    inputs : dict
        A dictionary where keys are dataset names (e.g., "train", "test") and
        values are tuples containing feature matrices (X) and target arrays (y).
    model : object
        A model instance with a `return_metrics` method that computes evaluation
        metrics.
    estimator_name : str
        The name of the estimator to label the output.

    Returns:
    -------
    pd.Series
        A Series containing the computed metrics, indexed by input type and
        metric name.
    """

    all_metrics = []
    for input_type, (X, y) in inputs.items():
        print(input_type)
        return_metrics_dict = model.return_metrics(
            X,
            y,
            optimal_threshold=True,
            print_threshold=True,
            model_metrics=True,
        )

        metrics = pd.Series(return_metrics_dict).to_frame(estimator_name)
        metrics = round(metrics, 3)
        metrics.index = [input_type + " " + ind for ind in metrics.index]
        all_metrics.append(metrics)
    return pd.concat(all_metrics)


####################### Enter the model plots into MlFlow ######################


def mlflow_log_parameters_model(
    model_type: str = None,
    n_iter: int = None,
    kfold: bool = None,
    outcome: str = None,
    run_name: str = None,
    experiment_name: str = None,
    model_name: str = None,
    model=None,
    databricks=False,
    hyperparam_dict=None,
):
    """
    Log model-specific parameters, hyperparameters from a dictionary, and the
    trained model under a single MLflow run in mlruns/modeling.

    Args:
        model_type: Type of the model (e.g., 'lr', 'rf', 'xgb', 'cat').
        n_iter: Number of iterations for hyperparameter tuning.
        kfold: Whether k-fold cross-validation was used.
        outcome: Target variable name.
        run_name: Name of the MLflow run.
        experiment_name: Name of the MLflow experiment.
        model_name: Name for the logged model artifact.
        model: The trained model object.
        hyperparam_dict: Dictionary of hyperparameters to loop through and log
        (default None).
    """

    abs_mlflow_data = os.path.abspath(mlflow_models_data)

    if databricks:
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
    else:
        mlflow.set_tracking_uri(f"file://{abs_mlflow_data}")

    # Set or create the experiment_id for the model and parameters
    experiment_id = set_or_create_experiment(experiment_name, databricks=databricks)
    run_id = get_run_id_by_name(experiment_name, run_name, databricks=databricks)

    print(f"Experiment_ID for model {model_type} and parameters: {experiment_id}")

    with mlflow.start_run(experiment_id=experiment_id, run_id=run_id) as run:
        print(f"experiment_id={experiment_id}, run_id={run_id}")

        # Log parameters under the active run
        if model_type is not None:
            mlflow.log_param("model_type", model_type)
        if n_iter is not None:
            mlflow.log_param("n_iter", n_iter)
        if kfold is not None:
            mlflow.log_param("kfold", kfold)
        if outcome is not None:
            mlflow.log_param("outcome", outcome)

        # Logging best model hyperparameters
        if hyperparam_dict is not None:
            mlflow.log_params(hyperparam_dict)

        # Logging model
        mlflow.sklearn.log_model(
            model,
            model_name,
        )

        print("Parameters and model logged successfully in MLflow.")

    return None


########################## Load the model object ###############################


def mlflow_load_model(
    experiment_name,
    run_name,
    model_name,
    databricks=False,
    mlruns_location: str = None,
):
    """
    Load a scikit-learn model from MLflow by experiment and run name.

    Args:
        experiment_name: Name of the MLflow experiment.
        run_name: Name of the run within the experiment.
        model_name: Name of the model artifact.

    Returns:
        Scikit-learn model instance.

    Raises:
        ValueError: If experiment or run is not found.
    """
    if mlruns_location is None:
        abs_mlflow_data = os.path.abspath(mlflow_models_data)
    else:
        abs_mlflow_data = os.path.abspath(mlruns_location)

    if databricks:
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
    else:
        mlflow.set_tracking_uri(f"file://{abs_mlflow_data}")

    # Query MLflow to find the latest run_id for the given run_name in the
    # experiment (for models)
    client = MlflowClient()
    if databricks:
        experiment = mlflow.get_experiment_by_name(
            databricks_username + experiment_name
        )
    else:
        experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment {experiment_name} not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],  # Get the most recent run
    )

    if not runs:
        raise ValueError(
            f"No runs found with run_name '{run_name}' in experiment '{experiment_name}'."
        )

    run_id = runs[
        0
    ].info.run_id  # Use the latest run_id for this run_name (for the specific model)

    # Load the scikit-learn model
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/{model_name}")
    return model


########################### MLFlow Model Evaluation ############################


def log_mlflow_metrics(
    experiment_name,
    run_name,
    metrics=None,
    databricks=False,
    images={},
):
    """
    Logs experiment metrics and visualizations to MLflow.

    This function sets up an MLflow experiment, retrieves the appropriate run,
    and logs key metrics and visual artifacts for model evaluation.

    Parameters:
    -----------
    - experiment_name (str):
        The name of the MLflow experiment.
    - run_name (str):
        The name of the specific run within the experiment.
    - metrics (pd.Series, optional):
        A Pandas Series containing performance metrics (e.g., precision,
        recall, F1-score). Each metric is logged individually.
    - images (dict, optional):
        A dictionary where keys are filenames and values are Matplotlib figure
        objects. These visualizations are logged to MLflow as artifacts.
    Returns:
    --------
    None
    """

    # Set the tracking URI to the specified mlflow_data location
    abs_mlflow_data = os.path.abspath(mlflow_models_data)  # Use models path

    if databricks:
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
    else:
        mlflow.set_tracking_uri(f"file://{abs_mlflow_data}")

    # Set or create the experiment_id for the model and parameters
    experiment_id = set_or_create_experiment(experiment_name, databricks=databricks)
    run_id = get_run_id_by_name(experiment_name, run_name, databricks=databricks)

    # Iterate over the models and log their metrics and parameters
    with mlflow.start_run(experiment_id=experiment_id, run_id=run_id):

        # Extract the row for the current model
        if metrics is not None:
            result = metrics
            if not result.empty:

                # Log the parameters and metrics
                for col in result.index:
                    value = result[col]
                    mlflow.log_metric(col, float(value))

        for name, image in images.items():
            mlflow.log_figure(image, name)


def find_best_model(
    experiment_name: str,
    metric_name: str,
    mode: str = "max",
    databricks=False,
    mlruns_location: str = None,
) -> str:
    """
    Finds the best model from a given MLflow experiment based on a specified
    metric.

    :param experiment_name: The name of the MLflow experiment to search in.
    :param metric_name: The metric used to determine the best model.
    :param mode: Specify "max" to select model based on maximum metric value
                 or "min" for minimum. Default is "max".
    :return: The run ID of the best model.
    :raises ValueError: If the experiment does not exist.
    """
    # Get experiment by name
    if mlruns_location is None:
        abs_mlflow_data = os.path.abspath(mlflow_models_data)
    else:
        abs_mlflow_data = os.path.abspath(mlruns_location)

    if databricks:
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
    else:
        mlflow.set_tracking_uri(f"file://{abs_mlflow_data}")

    if databricks:
        experiment = mlflow.get_experiment_by_name(
            databricks_username + experiment_name
        )
    else:
        experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' does not exist.")

    experiment_id = experiment.experiment_id

    # Get all runs for the experiment
    order_clause = (
        f"metrics.`{metric_name}` DESC"
        if mode == "max"
        else f"metrics.`{metric_name}` ASC"
    )

    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        order_by=[order_clause],
    )
    if runs.empty:
        raise ValueError(f"No runs found for experiment '{experiment_name}'")
    # Return the run ID with the best performance metric
    best_run = runs.iloc[0]  # Get the best run
    best_run_id = runs.iloc[0]["run_id"]
    best_metric_value = runs.iloc[0][f"metrics.{metric_name}"]
    print(f"Best Run ID: {best_run_id}, Best {metric_name}: {best_metric_value}")

    # Extract model_type from run_name or parameters
    run_name = best_run["tags.mlflow.runName"]

    # Extract estimator name
    estimator_name = run_name.split("_")[0]
    return run_name, estimator_name


def return_best_model(outcome, metric, mlruns_location=None, databricks=False):

    outcome = "ISDEATHDATElead1yr"
    experiment_name = outcome + "_model"
    if databricks:
        run_name, estimator_name = find_best_model(
            experiment_name,
            metric,
            databricks=databricks,
        )
    else:
        run_name, estimator_name = find_best_model(
            experiment_name,
            metric,
            mlruns_location=mlruns_location,
            databricks=databricks,
        )

    model_name = f"{estimator_name}_{outcome}"

    if databricks:
        best_model = mlflow_load_model(
            experiment_name=experiment_name,
            run_name=run_name,
            model_name=model_name,
            databricks=databricks,
        )
    else:
        best_model = mlflow_load_model(
            experiment_name,
            run_name,
            model_name,
            mlruns_location=mlruns_location,
            databricks=databricks,
        )

    return best_model


################################################################################
############################### Actor Embeddings ###############################
################################################################################


def normalize_actor(actor: str) -> str:
    """
    Normalize ACLED actor strings into low-cardinality actor families
    for embedding and modeling. Collapses geographic and unit-level
    variation while preserving actor alignment and role.
    """
    if actor is None or not isinstance(actor, str):
        return "None_Actor"

    a = actor.lower()

    # civilians / protesters
    if "civilians" in a:
        return "Civilians"
    if "protesters" in a:
        return "Protesters"

    # unidentified
    if "unidentified" in a:
        return "Unidentified"

    # communal militias
    if "communal militia" in a:
        return "Communal_Militia_Ukraine"

    # Ukraine military
    if "military forces of ukraine" in a:
        if "azov" in a:
            return "MF_Ukraine_Azov"
        if "air force" in a:
            return "MF_Ukraine_AirForce"
        if "navy" in a:
            return "MF_Ukraine_Navy"
        if "marines" in a:
            return "MF_Ukraine_Marines"
        if "intelligence" in a or "gur" in a:
            return "MF_Ukraine_Intelligence"
        return "MF_Ukraine"

    # Russia military
    if "military forces of russia" in a:
        if "kadyrov" in a or "chechen" in a:
            return "MF_Russia_Kadyrov"
        if "air force" in a:
            return "MF_Russia_AirForce"
        if "navy" in a:
            return "MF_Russia_Navy"
        if "gru" in a:
            return "MF_Russia_GRU"
        return "MF_Russia"

    # police
    if "police forces of ukraine" in a:
        return "Police_Ukraine"
    if "police forces of russia" in a:
        return "Police_Russia"

    # proxies and special groups
    if "wagner" in a:
        return "Wagner"
    if "donetsk people's republic" in a or "luhansk people's republic" in a:
        return "Russia_Proxy"
    if "lsr" in a or "freedom of russia legion" in a:
        return "Anti_Russia_Proxy"
    if "rdk" in a:
        return "Anti_Russia_Proxy"

    # fallback
    return actor


################################################################################
# Actor Interactions Embeddings
################################################################################


def parse_assoc_actors(x):
    if not isinstance(x, str) or x.strip() == "":
        return []
    return [a.strip() for a in x.split(";")]


def build_actor_interaction_graph(
    df,
    actor1_col="actor1_root",
    actor2_col="actor2_root",
    assoc1_col="assoc_actor_1",
    assoc2_col="assoc_actor_2",
    none_actor_label="None_Actor",
    main_weight=1.0,
    assoc_weight=0.5,
    none_actor_weight=0.3,
):
    """
    Build a weighted, undirected actor interaction graph.
    Primary actors define main edges; associated actors contribute
    lower-weight contextual edges used for representation learning.
    """

    G = nx.Graph()

    for _, row in tqdm(
        df.iterrows(), total=len(df), desc="Building actor interaction graph"
    ):
        a1 = row[actor1_col]
        a2 = row[actor2_col]

        # main interaction
        if a1 != a2:
            w = main_weight
            if a2 == none_actor_label:
                w = none_actor_weight

            if G.has_edge(a1, a2):
                G[a1][a2]["weight"] += w
            else:
                G.add_edge(a1, a2, weight=w)

        # assoc actors for actor1
        for assoc in parse_assoc_actors(row.get(assoc1_col)):
            assoc_root = normalize_actor(assoc)
            if assoc_root != a1:
                G.add_edge(
                    a1,
                    assoc_root,
                    weight=G.get_edge_data(a1, assoc_root, {}).get("weight", 0)
                    + assoc_weight,
                )

        # assoc actors for actor2
        for assoc in parse_assoc_actors(row.get(assoc2_col)):
            assoc_root = normalize_actor(assoc)
            if assoc_root != a2:
                G.add_edge(
                    a2,
                    assoc_root,
                    weight=G.get_edge_data(a2, assoc_root, {}).get("weight", 0)
                    + assoc_weight,
                )

    return G


################################################################################
## Actor Embedding Feature Engineering
################################################################################


def add_pairwise_embedding_features(
    df: pd.DataFrame,
    emb_prefix: str = "emb_",
    a2_prefix: str = "a2_emb_",
    add_diff: bool = False,
    add_dot: bool = False,
) -> pd.DataFrame:
    """
    Add optional pairwise actor embedding interaction features.

    Options:
    - add_diff: emb_diff_k = emb_k - a2_emb_k
    - add_dot:  emb_dot = dot(emb, a2_emb)
    """

    emb_cols = sorted(
        c
        for c in df.columns
        if c.startswith(emb_prefix) and not c.startswith(a2_prefix)
    )
    a2_cols = sorted(c for c in df.columns if c.startswith(a2_prefix))

    if len(emb_cols) == 0 or len(a2_cols) == 0:
        raise ValueError("Embedding columns not found in DataFrame.")

    if len(emb_cols) != len(a2_cols):
        raise ValueError(
            f"Actor1 and Actor2 embedding dimensions do not match "
            f"({len(emb_cols)} vs {len(a2_cols)})."
        )

    # Optional per-dimension differences
    if add_diff:
        for c1, c2 in zip(emb_cols, a2_cols):
            dim = c1.replace(emb_prefix, "")
            col_name = f"emb_diff_{dim}"
            if col_name not in df.columns:
                df[col_name] = df[c1] - df[c2]

    # Optional dot product
    if add_dot:
        df["emb_dot"] = np.sum(df[emb_cols].values * df[a2_cols].values, axis=1)

    return df


################################################################################
#################### Actual vs. Predicted Regression Plots #####################
################################################################################


def plot_actual_vs_predicted(
    y_true: pd.Series,
    y_pred: pd.Series,
    title: str = "Actual vs Predicted Fatalities",
    log_scale: bool = False,
    show_log_metrics: bool = False,
):
    """
    Scatter plot of observed vs predicted fatalities.

    Parameters:
    -----------
    y_true : pd.Series
        True values in log(1 + fatalities) scale
    y_pred : pd.Series
        Predicted values in log(1 + fatalities) scale
    title : str
        Plot title
    log_scale : bool
        If True, use log scale for axes. Default False for clarity.
    show_log_metrics : bool
        If True, show R² on log scale (training objective).
        If False, show R² on actual scale (interpretable). Default False.

    Returns:
    --------
    fig : matplotlib.figure.Figure
    """

    # Inverse transform to original scale
    y_true_actual = np.expm1(y_true)
    y_pred_actual = np.expm1(y_pred)

    # Handle any negative predictions (shouldn't happen but be safe)
    y_pred_actual = np.maximum(y_pred_actual, 0)

    # Calculate SEPARATE data ranges for X and Y
    x_max = y_true_actual.max()
    y_max = y_pred_actual.max()

    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot
    ax.scatter(
        y_true_actual,
        y_pred_actual,
        alpha=0.4,
        s=20,
        edgecolors="none",
    )

    # Perfect prediction line - draw to the LARGER of the two maxes
    line_max = max(x_max, y_max)
    ax.plot(
        [0, line_max],
        [0, line_max],
        "r--",
        linewidth=2,
        label="Perfect prediction",
        alpha=0.7,
    )

    # Set axis limits based on INDIVIDUAL ranges (not square!)
    if log_scale:
        # For log scale, start from a small positive number
        x_min = max(0.1, y_true_actual[y_true_actual > 0].min() * 0.5)
        y_min = max(0.1, y_pred_actual[y_pred_actual > 0].min() * 0.5)

        # Set limits with 10% padding
        ax.set_xlim(x_min, x_max * 1.1)
        ax.set_ylim(y_min, y_max * 1.1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Observed Fatalities (log scale)", fontsize=12)
        ax.set_ylabel("Predicted Fatalities (log scale)", fontsize=12)
    else:
        # Linear scale - INDEPENDENT limits for X and Y
        x_limit = x_max * 1.05
        y_limit = y_max * 1.05

        ax.set_xlim(-x_limit * 0.02, x_limit)
        ax.set_ylim(-y_limit * 0.02, y_limit)

        ax.set_xlabel("Observed Fatalities", fontsize=12)
        ax.set_ylabel("Predicted Fatalities", fontsize=12)

    # Title, legend, grid
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Calculate metrics on appropriate scale
    if show_log_metrics:
        # Metrics on LOG scale (matches training objective)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        scale_label = "(log scale)"
    else:
        # Metrics on ACTUAL scale (interpretable for stakeholders)
        r2 = r2_score(y_true_actual, y_pred_actual)
        mae = mean_absolute_error(y_true_actual, y_pred_actual)
        rmse = np.sqrt(mean_squared_error(y_true_actual, y_pred_actual))
        scale_label = ""

    # Add text box with metrics
    metrics_text = (
        f"R² = {r2:.3f} {scale_label}\n" f"MAE = {mae:.2f}\n" f"RMSE = {rmse:.2f}"
    )

    ax.text(
        0.95,
        0.05,
        metrics_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.tight_layout()
    return fig


################################################################################
############# Cumulative Fatalities Captured (impact curve) Plots ##############
################################################################################


def plot_cumulative_fatalities_captured(
    y_true_log: pd.Series,
    y_pred_log: pd.Series,
    model_name: str = "Model",
    title: str = "Cumulative Fatalities Captured",
    return_table: bool = False,
):
    """
    Plot cumulative gains curve showing fraction of total fatalities captured
    as events are ranked by predicted severity.

    Optionally returns the underlying cumulative table.
    """

    df = pd.DataFrame(
        {
            "fatal_true": np.expm1(y_true_log),
            "fatal_pred": np.expm1(y_pred_log),
        }
    )

    df = df.sort_values("fatal_pred", ascending=False)

    cum_fatalities = df["fatal_true"].cumsum()
    total_fatalities = cum_fatalities.iloc[-1]

    cum_frac = cum_fatalities / total_fatalities
    event_frac = np.arange(1, len(df) + 1) / len(df)

    # area under cumulative curve
    auc_capture = np.trapz(cum_frac.values, event_frac)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Shaded gain area
    ax.fill_between(
        event_frac,
        event_frac,  # Random baseline
        cum_frac,  # Model curve
        alpha=0.15,
        color="blue",
        label="Efficiency gain over random",
    )

    # Model curve
    ax.plot(
        event_frac,
        cum_frac,
        label=f"{model_name} (AUC = {auc_capture:.3f})",
        linewidth=2.5,
        color="#1f77b4",
    )

    # Random baseline
    ax.plot(
        event_frac,
        event_frac,
        "--",
        color="gray",
        label="Random",
        linewidth=2,
        alpha=0.7,
    )

    # Reference lines and annotations at key points
    key_thresholds = [0.10, 0.20, 0.50]

    for pct in key_thresholds:
        # Vertical line
        ax.axvline(x=pct, color="gray", linestyle=":", alpha=0.4, linewidth=1)

        # Find capture rate at this threshold
        idx = int(pct * len(df)) - 1
        if idx >= 0 and idx < len(cum_frac):
            capture_rate = cum_frac.iloc[idx]

            # Horizontal line
            ax.axhline(
                y=capture_rate, color="gray", linestyle=":", alpha=0.4, linewidth=1
            )

            # Annotation with actual counts
            actual_count = cum_fatalities.iloc[idx]
            ax.annotate(
                f"{int(pct*100)}% events\n→ {capture_rate*100:.0f}% casualties\n({int(actual_count):,} of {int(total_fatalities):,})",
                xy=(pct, capture_rate),
                xytext=(pct + 0.08, capture_rate - 0.12),
                fontsize=9,
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="wheat",
                    alpha=0.7,
                    edgecolor="gray",
                ),
                arrowprops=dict(
                    arrowstyle="->",
                    connectionstyle="arc3,rad=0.2",
                    lw=1.5,
                    color="gray",
                ),
            )

    # Labels and formatting
    ax.set_xlabel("Fraction of Events", fontsize=12, fontweight="bold")
    ax.set_ylabel(
        "Fraction of Total Fatalities Captured", fontsize=12, fontweight="bold"
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Secondary Y-axis with actual fatality counts
    ax2 = ax.twinx()
    ax2.set_ylabel(
        f"Cumulative Fatalities (Total = {int(total_fatalities):,})",
        fontsize=11,
        color="darkred",
    )
    ax2.set_ylim(0, total_fatalities)
    ax2.tick_params(axis="y", labelcolor="darkred")

    fig.tight_layout()

    if return_table:
        # RETURN FULL TABLE (every event)
        table = pd.DataFrame(
            {
                "event_fraction": event_frac,
                "cumulative_fatalities": cum_fatalities.values,
                "cumulative_fraction": cum_frac.values,
                "lift_over_random": cum_frac.values / event_frac,  # Added lift column
            }
        )
        return fig, table

    return fig


def print_capture_summary(
    capture_table: pd.DataFrame,
    split_name: str,
    ks=(0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 0.90, 0.95, 0.99),
):
    print("\n" + "=" * 70)
    print(f"CUMULATIVE FATALITY CAPTURE SUMMARY ({split_name})")
    print("=" * 70)

    for k in ks:
        idx = int(np.ceil(k * len(capture_table))) - 1
        idx = max(idx, 0)

        frac_events = capture_table.iloc[idx]["event_fraction"]
        frac_fatal = capture_table.iloc[idx]["cumulative_fraction"]

        print(f"Top {int(k*100):>2}% events " f"→ {frac_fatal*100:6.2f}% of fatalities")

    print("\nFirst 5 rows of capture table:")
    print(capture_table.head().round(4))
    print("=" * 70 + "\n")


################################################################################
############################# Temporal Splits ##################################
################################################################################


def create_temporal_splits(df, train_end, valid_end):
    """Create temporal train/valid/test splits"""

    # Ensure datetime
    df["event_date"] = pd.to_datetime(df["event_date"])

    # Sort by date
    df = df.sort_values("event_date").reset_index(drop=False)

    # Create splits
    train_mask = df["event_date"] <= pd.to_datetime(train_end)
    valid_mask = (df["event_date"] > pd.to_datetime(train_end)) & (
        df["event_date"] <= pd.to_datetime(valid_end)
    )
    test_mask = df["event_date"] > pd.to_datetime(valid_end)

    # Log split info
    print(f"\n{'='*60}")
    print("TEMPORAL SPLIT SUMMARY")
    print(f"{'='*60}")
    print(f"Train: up to {train_end}")
    print(f"  Events: {train_mask.sum():,} ({train_mask.sum()/len(df)*100:.1f}%)")
    print(f"  Avg fatalities/event: {df[train_mask]['fatalities'].mean():.2f}")
    print(f"\nValid: {train_end} to {valid_end}")
    print(f"  Events: {valid_mask.sum():,} ({valid_mask.sum()/len(df)*100:.1f}%)")
    print(f"  Avg fatalities/event: {df[valid_mask]['fatalities'].mean():.2f}")
    print(f"\nTest: after {valid_end}")
    print(f"  Events: {test_mask.sum():,} ({test_mask.sum()/len(df)*100:.1f}%)")
    print(f"  Avg fatalities/event: {df[test_mask]['fatalities'].mean():.2f}")
    print(f"{'='*60}\n")

    return df[train_mask], df[valid_mask], df[test_mask]


def normalize_split(df):
    """
    Create missing indicators for actor columns.

    Args:
        df: DataFrame with actor1 and actor2 columns

    Returns:
        DataFrame with actor1_missing and actor2_missing binary indicators
    """
    # Create binary missing indicators
    if "actor1" in df.columns:
        df["actor1_missing"] = df["actor1"].isna().astype(int)

    if "actor2" in df.columns:
        df["actor2_missing"] = df["actor2"].isna().astype(int)

    return df


def apply_embeddings(df, emb_df):
    df = df.merge(emb_df, left_on="actor1_root", right_index=True, how="left")
    df = df.merge(
        emb_df.add_prefix("a2_"),
        left_on="actor2_root",
        right_index=True,
        how="left",
    )
    return df


################################################################################
#### SHAP Value Plots for Model Explainability (for tree-based models) #########
################################################################################


def create_shap_plots(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path = Path("./models/eval"),
    max_display: int = 20,
    sample_size: int = 100,
):
    """
    Create comprehensive SHAP plots for model interpretation.

    Args:
        model: Trained model object (model_tuner.Model wrapper)
        X_train: Training features (for background samples)
        X_test: Test features (for SHAP values)
        y_test: Test target (for analysis)
        output_dir: Directory to save plots
        max_display: Number of features to show in summary plots
        sample_size: Number of test samples to explain (SHAP is slow!)

    Returns:
        expanded_importance_df: DataFrame with category-level SHAP importance
        feature_importance_df: DataFrame with collapsed feature importance
        figures: Dictionary of matplotlib figures
    """

    print("\n" + "=" * 80)
    print("GENERATING SHAP EXPLANATIONS")
    print("=" * 80)

    import scipy.sparse as sp
    from sklearn.pipeline import Pipeline

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dictionary to store figures
    figures = {}

    # --------------------------------------------------------------
    # STEP 1: Extract the actual estimator from wrapper
    # --------------------------------------------------------------
    print("\n[1/9] Extracting estimator from wrapper...")

    fitted_pipeline = model.estimator
    estimator = fitted_pipeline.steps[-1][1]
    estimator_type = type(estimator).__name__
    print(f"Extracted model type: {estimator_type}")

    # --------------------------------------------------------------
    # STEP 2: Transform training data through preprocessing pipeline
    # --------------------------------------------------------------
    print("\n[2/9] Transforming data through preprocessing pipeline...")

    preprocessing_steps = fitted_pipeline.steps[:-1]

    if preprocessing_steps:
        preprocessing_pipeline = Pipeline(preprocessing_steps)
        X_train_transformed = preprocessing_pipeline.transform(X_train)
        X_test_transformed = preprocessing_pipeline.transform(X_test)
    else:
        X_train_transformed = X_train.values
        X_test_transformed = X_test.values

    print(f"Training data shape after preprocessing: {X_train_transformed.shape}")
    print(f"Test data shape after preprocessing: {X_test_transformed.shape}")

    # --------------------------------------------------------------
    # STEP 3: Create SHAP Explainer based on model type
    # --------------------------------------------------------------
    print("\n[3/9] Creating SHAP Explainer...")

    tree_based = any(
        name in estimator_type.lower()
        for name in ["xgb", "catboost", "gradientboosting", "randomforest", "tree"]
    )

    if tree_based:
        explainer = shap.TreeExplainer(estimator)
        print("Using TreeExplainer")
    else:
        background = X_train_transformed
        if sp.issparse(background):
            background = background.toarray()
        explainer = shap.LinearExplainer(estimator, background)
        print("Using LinearExplainer")

    # --------------------------------------------------------------
    # STEP 4: Calculate SHAP Values on Test Set
    # --------------------------------------------------------------
    print(f"\n[4/9] Calculating SHAP values for {sample_size} test samples...")

    if X_test_transformed.shape[0] > sample_size:
        sample_indices = np.random.choice(
            X_test_transformed.shape[0], size=sample_size, replace=False
        )
        if hasattr(X_test_transformed, "iloc"):
            X_sample = X_test_transformed.iloc[sample_indices]
        else:
            X_sample = X_test_transformed[sample_indices]
        y_sample = y_test.iloc[sample_indices]
    else:
        X_sample = X_test_transformed
        y_sample = y_test

    # Calculate SHAP values
    if tree_based:
        shap_values = explainer.shap_values(X_sample, check_additivity=False)
    else:
        shap_values = explainer.shap_values(X_sample)

    print(f"SHAP values shape: {shap_values.shape}")

    # Get feature names after transformation
    try:
        feature_names = fitted_pipeline[:-1].get_feature_names_out()
    except:
        feature_names = [f"feature_{i}" for i in range(X_sample.shape[1])]

    print(f"Features: {len(feature_names)}")

    # Convert sparse matrix to dense DataFrame for SHAP plots
    if sp.issparse(X_sample):
        X_sample = X_sample.toarray()
        print("Converted sparse matrix to dense array")

    X_sample = pd.DataFrame(X_sample, columns=feature_names)
    print(f"Converted to DataFrame with feature names for SHAP visualization")

    # Build display copy with proper numeric types for SHAP coloring
    from sklearn.preprocessing import LabelEncoder

    X_sample_display = X_sample.copy()
    for col in X_sample_display.columns:
        if col.startswith("num__"):
            X_sample_display[col] = pd.to_numeric(
                X_sample_display[col], errors="coerce"
            )
        elif col.startswith("cat__"):
            le = LabelEncoder()
            X_sample_display[col] = le.fit_transform(X_sample_display[col].astype(str))

    # --------------------------------------------------------------
    # STEP 5: Summary Plot (Bar) - Feature Importance
    # --------------------------------------------------------------
    print(f"\n[5/9] Creating feature importance plot...")

    fig_importance = plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.title("SHAP Feature Importance", fontsize=14, fontweight="bold")
    plt.tight_layout()

    plt.savefig(output_dir / "shap_importance.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "shap_importance.svg", bbox_inches="tight")

    figures["shap_importance"] = fig_importance
    print(f"Saved: {output_dir}/shap_importance.png")

    # --------------------------------------------------------------
    # STEP 6: Summary Plot (Beeswarm) - Feature Effects
    # --------------------------------------------------------------
    print(f"\n[6/9] Creating beeswarm plot...")

    fig_beeswarm = plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample_display.values,  # numpy array, not DataFrame
        feature_names=list(X_sample_display.columns),
        max_display=max_display,
        show=False,
    )
    plt.title("SHAP Summary Plot (Feature Effects)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    plt.savefig(output_dir / "shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "shap_beeswarm.svg", bbox_inches="tight")

    figures["shap_beeswarm"] = fig_beeswarm
    print(f"Saved: {output_dir}/shap_beeswarm.png")

    # --------------------------------------------------------------
    # STEP 7: Expanded Category-Level Beeswarm
    # --------------------------------------------------------------
    # For models using native categorical support (e.g. XGBoost with
    # enable_categorical=True), SHAP returns one value per categorical
    # column. This step explodes that single value into per-category-
    # level rows: for each sample, the SHAP value is assigned to the
    # active category level and zeroed out for all others. This allows
    # the beeswarm and bar plots to show which specific category levels
    # (e.g. admin1=Donetsk, sub_event_type=Shelling) drive predictions.
    # --------------------------------------------------------------
    print(f"\n[7/9] Creating expanded category-level beeswarm plot...")

    cat_features = [c for c in feature_names if c.startswith("cat__")]
    num_features = [c for c in feature_names if c.startswith("num__")]

    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    X_sample_df = pd.DataFrame(
        X_sample.values if hasattr(X_sample, "values") else X_sample,
        columns=feature_names,
    )

    # Accumulate all columns into dicts first, then build DataFrames
    # in one pass — avoids repeated frame.insert which triggers the
    # "highly fragmented DataFrame" PerformanceWarning.
    shap_cols = {}
    feat_cols = {}

    for col in cat_features:
        categories = X_sample_df[col].astype(str).values
        shap_vals = shap_df[col].values
        for cat in np.unique(categories):
            mask = categories == cat
            col_name = f"{col} = {cat}"
            shap_cols[col_name] = np.where(mask, shap_vals, 0)
            feat_cols[col_name] = mask.astype(float)

    for col in num_features:
        shap_cols[col] = shap_df[col].values
        feat_cols[col] = X_sample_df[col].values

    exploded_shap = pd.DataFrame(shap_cols)
    exploded_features = pd.DataFrame(feat_cols)

    # Save per-sample expanded beeswarm data as pkl for the Dash app
    import pickle

    beeswarm_payload = {
        "shap_values": exploded_shap.values,
        "X": exploded_features,
    }
    with open(output_dir / "shap_beeswarm_expanded.pkl", "wb") as f:
        pickle.dump(beeswarm_payload, f)

    mean_abs = exploded_shap.abs().mean().sort_values(ascending=False)
    top_cols = mean_abs.head(max_display).index.tolist()

    shap_explanation = shap.Explanation(
        values=exploded_shap[top_cols].values,
        data=exploded_features[top_cols].values,
        feature_names=top_cols,
    )

    fig_beeswarm_expanded = plt.figure(figsize=(12, 10))
    shap.plots.beeswarm(shap_explanation, max_display=max_display, show=False)
    plt.title(
        "SHAP Summary Plot — Category-Level Drill-Down",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    plt.savefig(output_dir / "shap_beeswarm_expanded.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "shap_beeswarm_expanded.svg", bbox_inches="tight")

    figures["shap_beeswarm_expanded"] = fig_beeswarm_expanded
    print(f"Saved: {output_dir}/shap_beeswarm_expanded.png")

    # --------------------------------------------------------------
    # STEP 8: Expanded Category-Level Importance (Bar)
    # --------------------------------------------------------------
    print(f"\n[8/9] Creating expanded category-level importance plot...")

    fig_importance_expanded = plt.figure(figsize=(12, 10))
    shap.plots.bar(shap_explanation, max_display=max_display, show=False)
    plt.title(
        "SHAP Feature Importance — Category-Level Drill-Down",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    plt.savefig(
        output_dir / "shap_importance_expanded.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_dir / "shap_importance_expanded.svg", bbox_inches="tight")

    figures["shap_importance_expanded"] = fig_importance_expanded
    print(f"Saved: {output_dir}/shap_importance_expanded.png")

    # --------------------------------------------------------------
    # STEP 9: Feature Importance DataFrames
    # --------------------------------------------------------------
    print(f"\n[9/9] Creating feature importance table...")

    feature_importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": np.abs(shap_values).mean(axis=0),
            "mean_shap_value": shap_values.mean(axis=0),
            "abs_mean_shap": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("importance", ascending=False)

    feature_importance_df.to_csv(
        output_dir / "shap_feature_importance.csv", index=False
    )

    # Expanded importance table (category-level)
    expanded_importance_df = pd.DataFrame(
        {
            "feature": mean_abs.index,
            "abs_mean_shap": mean_abs.values,
        }
    ).sort_values("abs_mean_shap", ascending=False)

    expanded_importance_df.to_csv(
        output_dir / "shap_feature_importance_expanded.csv", index=False
    )

    print("\nTop 10 Most Important Features (Collapsed):")
    print(feature_importance_df.head(10).to_string(index=False))

    print("\nTop 10 Most Important Features (Expanded):")
    print(expanded_importance_df.head(10).to_string(index=False))

    print("\n" + "=" * 80)
    print("SHAP ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nGenerated files in {output_dir}:")
    print("  - shap_importance.png/svg - Feature importance (bar chart)")
    print("  - shap_beeswarm.png/svg - Feature effects (beeswarm)")
    print("  - shap_beeswarm_expanded.png/svg - Category-level beeswarm")
    print("  - shap_importance_expanded.png/svg - Category-level importance")
    print("  - shap_feature_importance.csv - Feature importance table")
    print("  - shap_feature_importance_expanded.csv - Expanded importance table")

    return expanded_importance_df, feature_importance_df, figures


############################## Regression Metrics ##############################


def adjusted_r2(r2: float, n: int, p: int) -> float:
    """Compute adjusted R² given R², sample size n, and number of features p."""
    if n <= p + 1:
        return float("nan")
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


############################## Haversine Distance ##############################


def haversine_km(lat, lon, ref_lat, ref_lon):
    R = 6371
    dlat = np.radians(lat - ref_lat)
    dlon = np.radians(lon - ref_lon)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(ref_lat)) * np.cos(np.radians(lat)) * np.sin(dlon / 2) ** 2
    )
    return R * 2 * np.arcsin(np.sqrt(a))


################################ End of functions.py ###########################
