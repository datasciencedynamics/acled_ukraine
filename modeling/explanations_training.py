################################################################################
# STEP 1: Import required libraries and modules
################################################################################
from pathlib import Path
import typer
import pandas as pd
import numpy as np

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
    categorical_cols,
)

from tqdm import tqdm

tqdm.pandas()

app = typer.Typer()


@app.command()
def main(
    outcome: str = target_log_outcome,
    features_path: Path = PROCESSED_DATA_DIR / "X_test.parquet",
    metric_name: str = "test_r2",
    mode: str = "max",
    explanations_path: Path = "",
    shap_val_flag: int = 1,  # flag for whether or not to print vals next to feats.
    top_n: int = 5,  # top n feats.
    residual_threshold: float = 5.0,  # absolute fatality tolerance δ
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
    # STEP 4: Load Processed Data (Features & Labels)
    ############################################################################

    X_test = pd.read_parquet(features_path)

    # Load train and valid splits for categorical alignment
    X_train = pd.read_parquet(features_path.parent / "X_train.parquet")
    X_valid = pd.read_parquet(features_path.parent / "X_valid.parquet")

    # Align categorical codes across all splits
    for col in categorical_cols:
        combined_cats = pd.Categorical(
            pd.concat([X_train[col], X_valid[col], X_test[col]])
        ).categories
        X_test[col] = pd.Categorical(X_test[col], categories=combined_cats)

    # Load actual target if available
    y_log_path = features_path.parent / f"y_test_{outcome}.parquet"
    y_log = None
    if y_log_path.exists():
        y_log = pd.read_parquet(y_log_path).iloc[:, 0]

    ############################################################################
    # STEP 5: Prepare the pipeline and transform data
    ############################################################################

    # Retrieve pipeline steps using built-in model_tuner getter
    X_transformed = model.get_preprocessing_and_feature_selection_pipeline().transform(
        X_test
    )

    ############################################################################
    # STEP 6: Load SHAP explainer
    ############################################################################

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
    with tqdm(total=X_test.shape[0], desc="SHAP Explaining") as pbar:
        shap_values = explainer(X_transformed, check_additivity=False)
        pbar.update(X_test.shape[0])

    ############################################################################
    # STEP 8: Process SHAP results — explode categories to per-level
    ############################################################################

    shap_feature_names = model.estimator[:-1].get_feature_names_out()

    print("Shape of shap_values.values:", shap_values.values.shape)

    shap_df_raw = pd.DataFrame(
        shap_values.values,
        columns=shap_feature_names,
        index=X_test.index,
    )

    X_transformed_df = pd.DataFrame(
        X_transformed.values if hasattr(X_transformed, "values") else X_transformed,
        columns=shap_feature_names,
        index=X_test.index,
    )

    cat_features = [c for c in shap_feature_names if c.startswith("cat__")]
    num_features = [c for c in shap_feature_names if c.startswith("num__")]

    exploded_shap = pd.DataFrame(index=X_test.index)

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

    shap_df = pd.DataFrame(index=X_test.index)

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

    y_pred_log = model.predict(X_test)

    shap_df["predicted_fatalities"] = (
        np.clip(np.expm1(y_pred_log), a_min=0, a_max=None).round().astype(int)
    )

    if y_log is not None:
        shap_df["actual_fatalities"] = (
            np.clip(np.expm1(y_log), a_min=0, a_max=None).round().astype(int)
        )
        shap_df["residual"] = (
            shap_df["actual_fatalities"] - shap_df["predicted_fatalities"]
        )

        # ------------------------------------------------------------------
        # Residual-based confusion matrix
        # ------------------------------------------------------------------
        # Predictions within a fixed absolute tolerance δ of actual are
        # considered "correct"; those outside are failure cases.
        #
        # δ = residual_threshold   [CLI param, default 5 fatalities]
        #
        # Quadrant definitions:
        #   TP       — real event (actual > 0), |residual| within δ
        #   FN       — real event (actual > 0), model underpredicted beyond δ
        #               (residual > δ, i.e. actual >> predicted)
        #   FP       — no event (actual == 0), model overpredicted beyond δ
        #               (|residual| > δ, i.e. predicted >> actual)
        #   TN       — no event (actual == 0), |residual| within δ
        #   over_pred — real event (actual > 0), model overpredicted beyond δ
        #               (residual < -δ, i.e. predicted >> actual)
        # ------------------------------------------------------------------

        # Inspect residual distribution to inform δ selection
        print("\nResidual distribution:")
        print(shap_df["residual"].describe())
        print("\n|Residual| percentiles:")
        print(
            dict(
                zip(
                    [25, 50, 75, 90, 95],
                    np.percentile(shap_df["residual"].abs(), [25, 50, 75, 90, 95]),
                )
            ),
            "\n",
        )

        delta = residual_threshold
        print(f"Residual tolerance δ = {delta:.1f} fatalities (fixed threshold)\n")

        shap_df["residual_delta"] = delta

        abs_residuals = shap_df["residual"].abs()
        within_tolerance = abs_residuals <= delta
        actual_nonzero = shap_df["actual_fatalities"] > 0

        # Core quadrants
        shap_df["TP"] = (actual_nonzero & within_tolerance).astype(int)
        shap_df["FN"] = (
            actual_nonzero & ~within_tolerance & (shap_df["residual"] > 0)
        ).astype(int)
        shap_df["FP"] = (
            ~actual_nonzero & ~within_tolerance & (shap_df["residual"] < 0)
        ).astype(int)
        shap_df["TN"] = (~actual_nonzero & within_tolerance).astype(int)

        # Events where actual > 0 but model over-predicted beyond δ
        shap_df["over_pred"] = (
            actual_nonzero & ~within_tolerance & (shap_df["residual"] < 0)
        ).astype(int)

        # Sanity check: every row should belong to exactly one category
        category_sum = shap_df[["TP", "FN", "FP", "TN", "over_pred"]].sum(axis=1)
        assert (category_sum == 1).all(), (
            "Confusion matrix category assignment is not mutually exclusive / exhaustive. "
            f"Rows with != 1 category:\n{shap_df[category_sum != 1]}"
        )

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
            if col in X_test.columns:
                shap_df[col] = X_test[col].values

    ############################################################################
    # STEP 12: Save results to CSV file
    ############################################################################

    shap_df.to_csv(explanations_path, index=True)
    print(f"Results saved to '{explanations_path}'")


if __name__ == "__main__":
    app()
