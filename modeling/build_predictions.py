from pathlib import Path
import typer
from loguru import logger
import pandas as pd

from core.functions import mlflow_load_model
from core.constants import target_log_outcome
from core.config import (
    PROCESSED_DATA_DIR,
    model_definitions,
    categorical_cols,
)

app = typer.Typer()


@app.command()
def main(
    outcome: str = target_log_outcome,
    pipeline_type: str = "orig",
    data_path: Path = PROCESSED_DATA_DIR,
    output_dir: Path = Path("./data/processed"),
):
    """
    Generate predictions from all models and save to a single predictions.csv.

    Columns: y_true_log, lr_log, lasso_log, xgb_log, catboost_log, split
    """

    experiment_name = f"{outcome}_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load splits
    # ------------------------------------------------------------------
    print("\nLoading temporal splits...")

    X_train = pd.read_parquet(data_path / "X_train.parquet")
    X_valid = pd.read_parquet(data_path / "X_valid.parquet")
    X_test = pd.read_parquet(data_path / "X_test.parquet")

    y_train = pd.read_parquet(data_path / f"y_train_{outcome}.parquet").iloc[:, 0]
    y_valid = pd.read_parquet(data_path / f"y_valid_{outcome}.parquet").iloc[:, 0]
    y_test = pd.read_parquet(data_path / f"y_test_{outcome}.parquet").iloc[:, 0]

    print(
        f"  Train: {X_train.shape[0]:,} | Valid: {X_valid.shape[0]:,} | Test: {X_test.shape[0]:,}"
    )

    splits = {
        "train": (X_train, y_train),
        "valid": (X_valid, y_valid),
        "test": (X_test, y_test),
    }

    # Model shorthand → estimator_name mapping
    # Keys must match model_definitions in core/config.py
    model_map = {
        "lr": model_definitions["lr"]["estimator_name"],
        "lasso": model_definitions["lasso"]["estimator_name"],
        "ridge": model_definitions["ridge"]["estimator_name"],
        "elastic_net": model_definitions["elastic_net"]["estimator_name"],
        "xgb": model_definitions["xgb"]["estimator_name"],
        "catboost": model_definitions["cat"]["estimator_name"],
    }

    # ------------------------------------------------------------------
    # Collect predictions per split
    # ------------------------------------------------------------------
    frames = []

    for split_name, (X_s, y_s) in splits.items():
        print(f"\nGenerating predictions for split: {split_name.upper()}")

        # Start with ground truth
        df_split = pd.DataFrame({"y_true_log": y_s.values}, index=y_s.index)

        for short, estimator_name in model_map.items():
            run_name = f"{estimator_name}_{pipeline_type}_training"
            print(f"  Loading {estimator_name}...")

            model = mlflow_load_model(
                experiment_name=experiment_name,
                run_name=run_name,
                model_name=f"{estimator_name}_{outcome}",
            )

            # XGBoost native categorical — align category codes across splits
            if short == "xgb" and pipeline_type == "orig":
                X_input = X_s.copy()
                for col in categorical_cols:
                    combined_cats = pd.Categorical(
                        pd.concat([X_train[col], X_valid[col], X_test[col]])
                    ).categories
                    X_input[col] = pd.Categorical(
                        X_input[col], categories=combined_cats
                    )
            else:
                X_input = X_s

            df_split[f"{short}_log"] = model.predict(X_input)

        df_split["split"] = split_name
        frames.append(df_split)

    # ------------------------------------------------------------------
    # Concatenate and save
    # ------------------------------------------------------------------
    predictions = pd.concat(frames)
    out_path = output_dir / "predictions.csv"
    predictions.to_csv(out_path, index=False)

    print(f"\n{'='*80}")
    print(f"Saved predictions to: {out_path}")
    print(f"Shape: {predictions.shape}")
    print(f"Columns: {list(predictions.columns)}")
    print(f"Split counts:\n{predictions['split'].value_counts().to_string()}")
    print("=" * 80)

    logger.success("Predictions saved.")


# ==================================================================
if __name__ == "__main__":
    app()
