from pathlib import Path

import typer
from loguru import logger
import pandas as pd
from model_tuner import Model

################################################################################
# Step 1. Import Configurations and Constants
################################################################################

from core.constants import target_outcome

from core.config import (
    PROCESSED_DATA_DIR,
    model_definitions,
    rstate,
    pipelines,
    numerical_cols,
    categorical_cols,
)
from core.functions import (
    clean_feature_selection_params,
    mlflow_log_parameters_model,
    adjust_preprocessing_pipeline,
    mlflow_load_model,
)

app = typer.Typer()

################################################################################
# Step 2. Define CLI Arguments with Default Values
################################################################################


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ---
    model_type: str = "xgb",
    pipeline_type: str = "orig",
    outcome: str = target_outcome,
    features_path: Path = PROCESSED_DATA_DIR / "X.parquet",
    labels_path: Path = PROCESSED_DATA_DIR / f"y_{target_outcome}.parquet",
    scoring: str = "r2",
    pretrained: int = 0,

    # --------------------------------------------------------------------------
):

    ################################################################################
    # Step 3. Load Feature and Label Datasets
    ################################################################################

    X = pd.read_parquet(features_path)
    y = pd.read_parquet(labels_path)[outcome]  # series

    ################################################################################
    # Step 4. Retrieve Model and Pipeline Configurations
    ################################################################################

    clc = model_definitions[model_type]["clc"]
    estimator_name = model_definitions[model_type]["estimator_name"]
    pipeline_steps = pipelines[pipeline_type]["pipeline"]
    sampler = pipelines[pipeline_type]["sampler"]
    feature_selection = pipelines[pipeline_type]["feature_selection"]

    # Set the parameters
    tuned_parameters = model_definitions[model_type]["tuned_parameters"]
    randomized_grid = model_definitions[model_type]["randomized_grid"]
    n_iter = model_definitions[model_type]["n_iter"]
    early_stop = model_definitions[model_type]["early"]

    print("Sampler", sampler)

    ################################################################################
    # Step 5. Clean up pipeline
    # Step 5a. Clean up tuned_parameters by removing feature selection keys if
    # RFE isn't in the pipeline
    ################################################################################
    clean_feature_selection_params(pipeline_steps, tuned_parameters)

    # Step 5b. Adjust preproc. pipe. to skip imputer and scaler for 'rf', 'xgb', 'cat'
    # for binary classification models; also applicable to regression models for 'xgb' and 'cat'

    # Clean numerical_cols and categorical_cols

    num_cols = [c for c in numerical_cols if c in X.columns]
    cat_cols = [c for c in categorical_cols if c in X.columns]


    pipeline_steps = adjust_preprocessing_pipeline(
        model_type,
        pipeline_steps,
        num_cols,
        cat_cols,
        sampler=sampler,
    )

    # Always rebuild the preprocessor with updated columns
    for i, (name, step) in enumerate(pipeline_steps):
        if "Preprocessor" in name:
            step.transformers = [
                ("num", step.transformers[0][1], num_cols),
                ("cat", step.transformers[1][1], cat_cols),
            ]
            break

    ################################################################################
    # Step 6. Printing Outcome
    ################################################################################

    print()
    print(f"Outcome:")
    print("-" * 60)
    print()
    print("=" * 60)
    print(f"{outcome}")
    print("=" * 60)

    ################################################################################
    # Step 7. Define and Initialize the Model Pipeline
    ################################################################################

    logger.info(f"Training {estimator_name} for {outcome} ...")

    if pretrained:

        print("Loading Pretrained Model...")
        model = mlflow_load_model(
            experiment_name=f"{outcome}_model",
            run_name=f"{estimator_name}_{pipeline_type}_training",
            model_name=f"{estimator_name}_{outcome}",
        )

    else:
        model = Model(
            pipeline_steps=pipeline_steps,
            name=estimator_name,
            model_type="regression",
            estimator_name=estimator_name,
            calibrate=True,
            estimator=clc,
            kfold=False,
            grid=tuned_parameters,
            n_jobs=5,
            randomized_grid=randomized_grid,
            n_iter=n_iter,
            scoring=[scoring],
            random_state=rstate,
            stratify_cols=["admin1"],
            boost_early=early_stop,
            imbalance_sampler=sampler,
            feature_selection=feature_selection,
        )

        ################################################################################
        # Step 8. Perform Hyperparameter Tuning
        ################################################################################

        model.grid_search_param_tuning(X, y)

        ################################################################################
        # Step 9. Extract Training, Validation, and Test Splits
        ################################################################################

    X_train, y_train = model.get_train_data(X, y)
    X_valid, y_valid = model.get_valid_data(X, y)
    X_test, y_test = model.get_test_data(X, y)

    ################################################################################
    # Step 10. Train the Model
    ################################################################################

    # Boosting algorithms like XGBoost and CatBoost benefit from validation data
    # during training to optimize early stopping and prevent overfitting.
    # Hence, we explicitly provide the validation dataset in the `fit` method
    # for these models. For other models, validation data is not required at this
    # stage.

    if not pretrained:
        if model_type in {"xgb", "cat"}:
            model.fit(
                X_train,
                y_train,
                validation_data=(X_valid, y_valid),
                score=scoring,
            )
        else:
            model.fit(
                X_train,
                y_train,
                score=scoring,
            )

    ################################################################################
    # Step 12. Report Metrics (Train / Validation / Test)
    # see the results printed to the terminal for reference
    ################################################################################

    print()
    print("=" * 80)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 80)

    # ---- TRAIN METRICS ----
    print("\n[TRAIN METRICS]")
    model.return_metrics(
        X=X_train,
        y=y_train,
        model_metrics=True,
    )

    # ---- VALID METRICS ----
    print("\n[VALIDATION METRICS]")
    model.return_metrics(
        X=X_valid,
        y=y_valid,
        model_metrics=True,
    )

    # ---- TEST METRICS (optional) ----
    if X_test is not None and y_test is not None:
        print("\n[TEST METRICS]")
        model.return_metrics(
            X=X_test,
            y=y_test,
            model_metrics=True,
        )
        print()

    if pretrained:
        mlflow_log_parameters_model(
            experiment_name=f"{outcome}_model",
            run_name=f"{estimator_name}_{pipeline_type}_training",
            model_name=f"{estimator_name}_{outcome}",
            model=model,
        )

    else:
        mlflow_log_parameters_model(
            model_type=model_type,
            n_iter=n_iter,
            kfold=False,
            outcome=outcome,
            experiment_name=f"{outcome}_model",
            run_name=f"{estimator_name}_{pipeline_type}_training",
            model_name=f"{estimator_name}_{outcome}",
            model=model,
            hyperparam_dict=model.best_params_per_score[scoring],
        )

    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
