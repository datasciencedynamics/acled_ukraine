################################################################################
# STEP 1: Import required libraries and modules
################################################################################
from pathlib import Path
import typer
import pandas as pd
import numpy as np
import json

from core.functions import mlflow_loadArtifact, mlflow_load_model
from modeling.predict import find_best_model
from core.constants import (
    target_log_outcome,
    shap_artifact_name,
    shap_run_name,
    shap_artifacts_data,
)

from core.config import (
    PROCESSED_DATA_DIR,
    PROCESSED_DATA_DIR_INFER,
    categorical_cols,
)

from tqdm import tqdm

tqdm.pandas()

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR_INFER / "X.parquet",
    outcome: str = target_log_outcome,
    metric_name: str = "test_r2",
    mode: str = "max",
    top_n: int = 5,
    shap_val_flag: int = 1,  # flag for whether or not to print vals next to feats.
    explanations_path: Path = "",
    training_data_path: Path = PROCESSED_DATA_DIR,
):
    ############################################################################
    # STEP 2: Set up experiment parameters
    ############################################################################

    experiment_name = f"{outcome}_model"

    ############################################################################
    # STEP 3: Find and load the best model
    ############################################################################

    run_name, estimator_name = find_best_model(
        experiment_name,
        metric_name,
        mode,
    )
    model_name = f"{estimator_name}_{outcome}"

    # Load best model and assign it to variable called model
    model = mlflow_load_model(experiment_name, run_name, model_name)

    ############################################################################
    # STEP 4: Load processed data (features)
    ############################################################################

    X = pd.read_parquet(features_path)

    # Load training splits for categorical alignment
    X_train = pd.read_parquet(training_data_path / "X_train.parquet")
    X_valid = pd.read_parquet(training_data_path / "X_valid.parquet")

    # Align categorical codes across all splits
    for col in categorical_cols:
        combined_cats = pd.Categorical(
            pd.concat([X_train[col], X_valid[col], X[col]])
        ).categories
        X[col] = pd.Categorical(X[col], categories=combined_cats)

    ############################################################################
    # STEP 5: Prepare the pipeline and transform data
    ############################################################################

    # Retrieve pipeline steps using built-in model_tuner getter
    X_transformed = model.get_preprocessing_and_feature_selection_pipeline().transform(
        X
    )

    ############################################################################
    # STEP 6: Load SHAP explainer
    ############################################################################

    # Load the SHAP explainer from artifact saved in explainer.py
    explainer = mlflow_loadArtifact(
        experiment_name=shap_artifact_name,
        run_name=shap_run_name,
        obj_name="explainer",
        artifacts_data_path=shap_artifacts_data,
    )

    ############################################################################
    # STEP 7: Compute SHAP values w/ progress bar
    ############################################################################
    print("Computing SHAP values...")
    with tqdm(total=X.shape[0], desc="SHAP Explaining") as pbar:
        shap_values = explainer(X_transformed, check_additivity=False)
        pbar.update(X.shape[0])

    ############################################################################
    # STEP 8: Process SHAP results — explode categories to per-level
    ############################################################################
    # Native categorical SHAP returns one value per categorical column.
    # Explode each categorical column into per-level columns by assigning
    # the SHAP value to the active category and zero to all others. This
    # ensures the top-N features report specific category levels (e.g.,
    # "cat__admin1 = Donetsk") rather than collapsed parent columns.
    ############################################################################

    shap_feature_names = model.estimator[:-1].get_feature_names_out()

    print("Shape of shap_values.values:", shap_values.values.shape)

    shap_df_raw = pd.DataFrame(
        shap_values.values,
        columns=shap_feature_names,
        index=X.index,
    )

    X_transformed_df = pd.DataFrame(
        X_transformed.values if hasattr(X_transformed, "values") else X_transformed,
        columns=shap_feature_names,
        index=X.index,
    )

    cat_features = [c for c in shap_feature_names if c.startswith("cat__")]
    num_features = [c for c in shap_feature_names if c.startswith("num__")]

    exploded_shap = pd.DataFrame(index=X.index)

    for col in cat_features:
        categories = X_transformed_df[col].astype(str).values
        shap_vals = shap_df_raw[col].values
        for cat in np.unique(categories):
            mask = categories == cat
            col_name = f"{col} = {cat}"
            exploded_shap[col_name] = np.where(mask, shap_vals, 0)

    for col in num_features:
        exploded_shap[col] = shap_df_raw[col].values

    shap_results = exploded_shap

    ############################################################################
    # STEP 9: Extract top n SHAP features
    ############################################################################
    print(f"Extracting Top {top_n} SHAP features per event...")

    top_shap_pairs = shap_results.progress_apply(
        lambda row: row.abs().round(2).nlargest(top_n).to_dict(),
        axis=1,
    )

    ############################################################################
    # STEP 10: Create SHAP DataFrame
    ############################################################################

    shap_df = pd.DataFrame(index=X.index)

    if shap_val_flag:
        shap_df[f"Top {top_n} Features"] = top_shap_pairs
    else:
        shap_df[f"Top {top_n} Features"] = top_shap_pairs.progress_apply(
            lambda d: ", ".join(d.keys())
        )

    ############################################################################
    # STEP 11: Add predictions to dataframe
    ############################################################################
    # Generate log-scale predictions and back-transform to raw fatality
    # counts using expm1 (inverse of log1p).
    ############################################################################

    y_pred_log = model.predict(X)

    shap_df["predicted_fatalities"] = (
        np.clip(np.expm1(y_pred_log), a_min=0, a_max=None).round().astype(int)
    )

    y_log_path = Path(features_path).parent / f"y_{outcome}.parquet"
    if y_log_path.exists():
        y_log = pd.read_parquet(y_log_path).iloc[:, 0]
        shap_df["actual_fatalities"] = (
            np.clip(np.expm1(y_log), a_min=0, a_max=None).round().astype(int)
        )
        shap_df["residual"] = (
            shap_df["actual_fatalities"] - shap_df["predicted_fatalities"]
        )

        # Binarized confusion matrix: did fatalities occur? (> 0)
        actual_bin = (shap_df["actual_fatalities"] > 0).astype(int)
        pred_bin = (shap_df["predicted_fatalities"] > 0).astype(int)

        shap_df["TP"] = ((actual_bin == 1) & (pred_bin == 1)).astype(int)
        shap_df["FN"] = ((actual_bin == 1) & (pred_bin == 0)).astype(int)
        shap_df["FP"] = ((actual_bin == 0) & (pred_bin == 1)).astype(int)
        shap_df["TN"] = ((actual_bin == 0) & (pred_bin == 0)).astype(int)

    # Append original feature columns for context
    context_cols = [
        "latitude",
        "longitude",
        "geo_precision",
        "days_since_invasion",
        "percentage_missing",
        "sub_event_type",
        "interaction",
        "admin1",
        "source_scale",
    ]

    for col in context_cols:
        if col in X.columns:
            shap_df[col] = X[col].values

    ############################################################################
    # STEP 12: Save results to CSV
    ############################################################################
    shap_df.to_csv(explanations_path, index=True)
    print(f"Results saved to '{explanations_path}'")


if __name__ == "__main__":
    app()
