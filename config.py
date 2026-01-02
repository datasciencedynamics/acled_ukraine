import numpy as np

from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.linear_model import ElasticNet
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from constants import (
    exp_artifact_name,
    preproc_run_name,
)
from functions import mlflow_loadArtifact

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATA_DIR_INFER = DATA_DIR / "processed/inference"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
RESULTS_DIR = PROJ_ROOT / MODELS_DIR / "results"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

features_path = PROCESSED_DATA_DIR / "X.parquet"

################################################################################
############################ Global Constants ##################################
################################################################################

rstate = 222  # random state for reproducibility
threshold_target_metric = "precision"  # target metric for threshold optimization
target_precision = 0.5  # target precision for threshold optimization

sampler_definitions = {
    "None": None,
    "SMOTE": SMOTE(random_state=rstate),
    "RandomUnderSampler": RandomUnderSampler(random_state=rstate),
}

rfe_estimator = LogisticRegression(
    max_iter=100,
    n_jobs=-2,
)

# Remove 10% of features per iteration
rfe = RFE(
    estimator=rfe_estimator,
    step=0.1,
)


################################################################################
# This section here is for categorical variables

categorical_cols = ["race", "sex"]

# Load feature column names from Mlflow
try:
    X_columns_list = mlflow_loadArtifact(
        experiment_name=exp_artifact_name,
        run_name=preproc_run_name,  # Use the same run_name as training
        obj_name="X_columns_list",
        verbose=False,
    )
    if X_columns_list is None:
        raise ValueError("X_columns_list is None - failed to load from artifacts")
except Exception as e:
    raise Exception(f"Failed to load X_columns_list: {str(e)}")

# Subset the numerical columns only; categorical columns are already defined above
numerical_cols = [col for col in X_columns_list if col not in categorical_cols]


################################################################################
############################### Transformers ###################################
################################################################################

numerical_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("imputer", SimpleImputer(strategy="mean")),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Create the ColumnTransformer with passthrough
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ],
    # remainder="passthrough",
    # prevents prepending transformer names (e.g., 'remainder_') to output
    # feature names
    # verbose_feature_names_out=False,
)

################################################################################
################################ Pipelines #####################################
################################################################################

pipeline_scale_imp_rfe = [
    ("Preprocessor", preprocessor),
    ("RFE", rfe),
]

pipeline_scale_imp = [
    ("Preprocessor", preprocessor),
]

pipelines = {
    "orig": {
        "pipeline": pipeline_scale_imp,
        "sampler": None,
        "feature_selection": False,  # No feature selection for orig
    },
    "smote": {
        "pipeline": pipeline_scale_imp,
        "sampler": SMOTE(random_state=rstate),
        "feature_selection": False,  # No feature selection for smote
    },
    "under": {
        "pipeline": pipeline_scale_imp,
        "sampler": RandomUnderSampler(random_state=rstate),
        "feature_selection": False,  # No feature selection for under
    },
    "orig_rfe": {
        "pipeline": pipeline_scale_imp_rfe,
        "sampler": None,
        "feature_selection": True,  # Feature selection (RFE) for orig_rfe
    },
    "smote_rfe": {
        "pipeline": pipeline_scale_imp_rfe,
        "sampler": SMOTE(random_state=rstate),
        "feature_selection": True,  # Feature selection (RFE) for smote_rfe
    },
    "under_rfe": {
        "pipeline": pipeline_scale_imp_rfe,
        "sampler": RandomUnderSampler(random_state=rstate),
        "feature_selection": True,  # Feature selection (RFE) for under_rfe
    },
}


################################################################################
############################# Path Variables ###################################
################################################################################

# model_output = "model_output"  # model output path
# mlflow_data = "mlflow_data"  # path to store mlflow artificats (i.e., results)

################################################################################
########################## Logistic Regression #################################
################################################################################

# Define the hyperparameters for Logistic Regression
lr_name = "lr"

# lr_penalties = ["elasticnet"]
# lr_penalties = ["l1"]
lr_penalties = ["l2"]
lr_Cs = np.logspace(-4, 0, 10)
l1_ratio = np.linspace(0, 1, 10)

# Structure the parameters similarly to the RF template
tuned_parameters_lr = [
    {
        "lr__penalty": lr_penalties,
        "lr__C": lr_Cs,
        "feature_selection_RFE__n_features_to_select": [10, 0.1, 0.5, 0.7, 1.0],
        # "lr__l1_ratio": l1_ratio,
    }
]

lr = LogisticRegression(
    class_weight="balanced",
    random_state=rstate,
    n_jobs=-2,
    # solver="saga",
    # solver="liblinear",
    solver="lbfgs",
)

lr_definition = {
    "clc": lr,
    "estimator_name": lr_name,
    "tuned_parameters": tuned_parameters_lr,
    "randomized_grid": True,
    "n_iter": 1,
    "early": False,
}


################################################################################
########################## Random Forest Classifier ############################
################################################################################

# Define the hyperparameters for Random Forest
rf_name = "rf"

rf_n_estimators = [100, 200, 300]
rf_max_depths = [None, 5, 10]
rf_criterions = ["gini", "entropy"]
rf_parameters = [
    {
        "rf__n_estimators": rf_n_estimators,
        "rf__max_depth": rf_max_depths,
        "rf__criterion": rf_criterions,
        "feature_selection_RFE__n_features_to_select": [10, 0.1, 0.5, 0.7, 1.0],
    }
]

rf = RandomForestClassifier(
    class_weight="balanced",
    random_state=rstate,
    n_jobs=-2,
)

rf_definition = {
    "clc": rf,
    "estimator_name": rf_name,
    "tuned_parameters": rf_parameters,
    "randomized_grid": True,
    "n_iter": 1,
    "early": False,
}

################################################################################
############################## XGBoost Classifier ##############################
################################################################################

# Estimator name prefix for use in GridSearchCV or similar tools
xgb_name = "xgb"

xgb = XGBClassifier(
    objective="binary:logistic",
    random_state=rstate,
    tree_method="hist",
    device="cuda",
    n_jobs=16,
)

# Define the hyperparameters for XGBoost
xgb_learning_rates = [0.01]  # Learning rate or eta
xgb_n_estimators = [10000]  # Number of trees. Equivalent to n_estimators in GB
xgb_max_depths = [3, 5, 7]  # Maximum depth of the trees
xgb_subsamples = [0.8, 1.0]  # Subsample ratio of the training instances
xgb_colsample_bytree = [0.8, 1.0]
xgb_alpha = [0, 0.1, 1, 10]  # L1 regularization (alpha)
xgb_lambda = [0, 0.1, 10, 100]  # L2 regularization (lambda)
xgb_eval_metric = ["logloss"]  # check out "aucpr"
xgb_early_stopping_rounds = [3]
xgb_verbose = [0]
# Subsample ratio of columns when constructing each tree

# Combining the hyperparameters in a dictionary
xgb_parameters = [
    {
        "xgb__learning_rate": xgb_learning_rates,
        "xgb__n_estimators": xgb_n_estimators,
        "xgb__max_depth": xgb_max_depths,
        "xgb__subsample": xgb_subsamples,
        "xgb__alpha": xgb_alpha,  # L1 regularization (alpha)
        "xgb__lambda": xgb_lambda,  # L2 regularization (lambda)
        "xgb__colsample_bytree": xgb_colsample_bytree,
        "xgb__eval_metric": xgb_eval_metric,
        "xgb__early_stopping_rounds": xgb_early_stopping_rounds,
        "xgb__verbose": xgb_verbose,
        "feature_selection_RFE__n_features_to_select": [10, 0.1, 0.5, 0.7, 1.0],
    }
]

xgb_definition = {
    "clc": xgb,
    "estimator_name": xgb_name,
    "tuned_parameters": xgb_parameters,
    "randomized_grid": True,
    "n_iter": 1,
    "early": True,
}

################################################################################
############################ CatBoost Classifier ###############################
################################################################################

cat_name = "cat"

cat = CatBoostClassifier(
    task_type="CPU",
    random_state=rstate,
    eval_metric="Logloss",
)

# Define the hyperparameters for CatBoost
cat_depths = [4, 6, 8, 10]  # Depth of the trees
cat_learning_rates = [0.01]  # Learning rate
cat_l2_leaf_regs = [3, 10, 100]  # L2 regularization
cat_bagging_temperatures = [0, 0.5, 1]  # Bagging temperature
cat_n_estimators = [10000]  # Number of trees
cat_early_stopping_rounds = [3]  # Early stopping rounds
cat_random_strengths = [1, 10]  # Random strength for feature score randomness
cat_verbose = [0]  # Verbosity level
cat_n_features_to_select = [10, 0.1, 0.5, 0.7, 1.0]  # Features to select for RFE

# Combining the hyperparameters in a dictionary
cat_parameters = [
    {
        "cat__depth": cat_depths,
        "cat__learning_rate": cat_learning_rates,
        "cat__l2_leaf_reg": cat_l2_leaf_regs,
        "cat__bagging_temperature": cat_bagging_temperatures,
        "cat__n_estimators": cat_n_estimators,
        "cat__early_stopping_rounds": cat_early_stopping_rounds,
        "cat__random_strength": cat_random_strengths,
        "cat__verbose": cat_verbose,
        "feature_selection_RFE__n_features_to_select": cat_n_features_to_select,
    }
]

cat_definition = {
    "clc": cat,
    "estimator_name": cat_name,
    "tuned_parameters": cat_parameters,
    "randomized_grid": True,
    "n_iter": 1,
    "early": True,
}


model_definitions = {
    lr_name: lr_definition,
    rf_name: rf_definition,
    xgb_name: xgb_definition,
    cat_name: cat_definition,
}