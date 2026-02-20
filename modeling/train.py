from pathlib import Path

import typer
from loguru import logger
import pandas as pd
from model_tuner import Model

################################################################################
# Step 1. Import Configurations and Constants
################################################################################

from core.constants import target_outcome, target_log_outcome

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
    outcome: str = target_log_outcome,
    data_path: Path = PROCESSED_DATA_DIR,
    scoring: str = "r2",
    pretrained: int = 0,
    # --------------------------------------------------------------------------
):

    ################################################################################
    # Step 3. Load Feature and Label Datasets
    ################################################################################

    print("\n" + "=" * 80)
    print("Loading temporal splits for training...")
    print("=" * 80)

    # Load X splits
    X_train = pd.read_parquet(data_path / "X_train.parquet")
    X_valid = pd.read_parquet(data_path / "X_valid.parquet")
    X_test = pd.read_parquet(data_path / "X_test.parquet")

    # Load y splits based on outcome
    y_train = pd.read_parquet(data_path / f"y_train_{outcome}.parquet").iloc[:, 0]
    y_valid = pd.read_parquet(data_path / f"y_valid_{outcome}.parquet").iloc[:, 0]
    y_test = pd.read_parquet(data_path / f"y_test_{outcome}.parquet").iloc[:, 0]

    print(f"\nOutcome: {outcome}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_valid shape: {X_valid.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_valid shape: {y_valid.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Combine for full dataset (needed for model_tuner)
    X = pd.concat([X_train, X_valid, X_test], axis=0)
    y = pd.concat([y_train, y_valid, y_test], axis=0)

    print(f"\nCombined X shape: {X.shape}")
    print(f"Combined y shape: {y.shape}")

    # Create custom splits dictionary
    custom_splits = {
        "X_train": X_train,
        "y_train": y_train,
        "X_valid": X_valid,
        "y_valid": y_valid,
        "X_test": X_test,
        "y_test": y_test,
    }

    print(f"\n{'-'*80}\nCustom Data Splits Summary:\n{'-'*80}")
    print(f"X_train = {X_train.shape[0]} rows")
    print(f"X_valid = {X_valid.shape[0]} rows")
    print(f"X_test = {X_test.shape[0]} rows\n{'-'*80}")
    print(f"Total = {X.shape[0]} rows\n{'-'*80}")
    print(f"X_train is {X_train.shape[0]/X.shape[0]*100:.2f}% of total data")
    print(f"X_valid is {X_valid.shape[0]/X.shape[0]*100:.2f}% of total data")
    print(f"X_test is {X_test.shape[0]/X.shape[0]*100:.2f}% of total data")
    print("=" * 80 + "\n")

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
    print("=" * 60)
    print(f"Outcome: {outcome}")
    print("=" * 60)
    print()

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
            estimator=clc,
            kfold=False,
            grid=tuned_parameters,
            n_jobs=5,
            randomized_grid=randomized_grid,
            n_iter=n_iter,
            scoring=[scoring],
            random_state=rstate,
            boost_early=early_stop,
            imbalance_sampler=sampler,
            feature_selection=feature_selection,
        )

        ################################################################################
        # Step 8. Perform Hyperparameter Tuning
        ################################################################################

        model.grid_search_param_tuning(X, y, custom_splits=custom_splits)

    ################################################################################
    # Step 9. Train the Model
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
    # Step 10. Report Metrics (Train / Validation / Test)
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
