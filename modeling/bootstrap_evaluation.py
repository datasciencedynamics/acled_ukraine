"""
evaluate_regression.py

CLI for bootstrapped regression model evaluation across multiple models.

Usage:
    python evaluate_regression.py
    python evaluate_regression.py --n-samples 1000 --num-resamples 1000
    python evaluate_regression.py --no-bootstrap
    python evaluate_regression.py --output results/bootstrap_results.csv
"""

import sys
import typer
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import box
from sklearn.metrics import r2_score
from model_tuner import loadObjects, evaluate_bootstrap_metrics
from model_metrics import summarize_model_performance

sys.path.append("../")

################################### Paths ######################################

BASE = Path("/home/lshpaner/Python_Projects/acled_ukraine")

MODEL_PATHS = {
    "Linear Regressor": BASE
    / "mlruns/models/618546375450881810/c00a6432dd054887a2858f29262c8662/artifacts/lr_log_fatalities/model.pkl",
    ## lasso_orig_rfe_training
    "Lasso RFE": BASE
    / "mlruns/models/618546375450881810/45a94ba689074cc6bfd4f723f5f7f38d/artifacts/lasso_log_fatalities/model.pkl",
    # ridge_orig_rfe_training
    "Ridge RFE": BASE
    / "mlruns/models/618546375450881810/c79d08249184436699ccb2869fc3eb35/artifacts/ridge_log_fatalities/model.pkl",
    "XGBoost Regressor": BASE
    / "mlruns/models/618546375450881810/cfa2d7dcc8604ce09eca44264b1ab2eb/artifacts/xgb_log_fatalities/model.pkl",
    "CatBoost Regressor": BASE
    / "mlruns/models/618546375450881810/ea4f31901bf449d3ba21fe1d3b85063a/artifacts/cat_log_fatalities/model.pkl",
}

DATA_DIR = BASE / "data/processed"
RESULTS_DIR = BASE / "results"

CATEGORICAL_COLS = [
    "admin1",
    "sub_event_type",
    "interaction",
    "source_scale",
    "geo_precision",
    "time_precision",
]

## Internal sklearn metric names used for computation
BOOTSTRAP_METRICS = [
    "r2",
    "adjusted_r2",
    "neg_root_mean_squared_error",
    "neg_mean_squared_error",
    "neg_mean_absolute_error",
    "explained_variance",
]

## Human-readable display names mapped from sklearn internal names
METRIC_DISPLAY_NAMES = {
    "r2": "R²",
    "adjusted_r2": "Adjusted R²",
    "neg_root_mean_squared_error": "RMSE",
    "neg_mean_squared_error": "MSE",
    "neg_mean_absolute_error": "MAE",
    "explained_variance": "Explained Variance",
}

################################### Helpers ####################################

console = Console()
app = typer.Typer(
    name="evaluate_regression",
    help="Bootstrapped regression evaluation across multiple models.",
    add_completion=False,
)


def load_data() -> tuple:
    """Load partitioned datasets from disk."""
    console.print("[bold cyan]Loading data...[/bold cyan]")
    X_train = pd.read_parquet(DATA_DIR / "X_train.parquet")
    X_valid = pd.read_parquet(DATA_DIR / "X_valid.parquet")
    X_test = pd.read_parquet(DATA_DIR / "X_test.parquet")
    y_valid = pd.read_parquet(DATA_DIR / "y_valid_log_fatalities.parquet")
    y_test = pd.read_parquet(DATA_DIR / "y_test_log_fatalities.parquet")
    return X_train, X_valid, X_test, y_valid, y_test


def load_models() -> dict:
    """Load all model objects from disk."""
    console.print("[bold cyan]Loading models...[/bold cyan]")
    return {name: loadObjects(str(path)) for name, path in MODEL_PATHS.items()}


def encode_categoricals(X_train, *splits) -> list:
    """
    Apply consistent categorical encoding across all splits.
    Returns encoded copies; originals are not mutated.
    """
    encoded = [split.copy() for split in splits]
    for col in CATEGORICAL_COLS:
        combined_cats = pd.Categorical(
            pd.concat([X_train[col]] + [s[col] for s in splits])
        ).categories
        for split in encoded:
            split[col] = pd.Categorical(split[col], categories=combined_cats)
    return encoded


def build_model_inputs(models, X_raw, X_encoded) -> dict:
    """
    Map each model to its correct X variant.
    Linear models use raw X; tree-based models use encoded X.
    """
    return {
        "Linear Regressor": (models["Linear Regressor"], X_raw),
        "Lasso RFE": (models["Lasso RFE"], X_raw),
        "Ridge RFE": (models["Ridge RFE"], X_raw),
        "XGBoost Regressor": (models["XGBoost Regressor"], X_encoded),
        "CatBoost Regressor": (models["CatBoost Regressor"], X_encoded),
    }


def rename_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace sklearn internal metric names with human-readable labels and
    flip the sign on negated error metrics so they read as positive values.
    """
    neg_metrics = {
        "neg_root_mean_squared_error",
        "neg_mean_squared_error",
        "neg_mean_absolute_error",
    }
    numeric_cols = ["Mean", "95% CI Lower", "95% CI Upper"]

    df = df.copy()

    # Flip sign on negated metrics so errors are positive
    mask = df["Metric"].isin(neg_metrics)
    df.loc[mask, numeric_cols] = df.loc[mask, numeric_cols] * -1

    # Swap CI bounds so Lower < Upper after negation
    df.loc[mask, ["95% CI Lower", "95% CI Upper"]] = df.loc[
        mask, ["95% CI Upper", "95% CI Lower"]
    ].values

    # Apply display names
    df["Metric"] = df["Metric"].map(METRIC_DISPLAY_NAMES).fillna(df["Metric"])

    return df


def print_r2_table(title: str, preds: dict, y) -> None:
    """Render a rich table of point-estimate R² scores."""
    table = Table(title=title, box=box.SIMPLE_HEAVY, show_lines=False)
    table.add_column("Model", style="bold")
    table.add_column("R²", justify="right")
    for name, pred in preds.items():
        table.add_row(name, f"{r2_score(y, pred):.4f}")
    console.print(table)


def run_point_estimates(models, X_train, X_valid, X_test, y_valid, y_test) -> None:
    """Print quick R² point estimates for validation and test sets."""
    X_valid_enc, X_test_enc = encode_categoricals(X_train, X_valid, X_test)

    valid_preds = {
        "Linear Regressor": models["Linear Regressor"].predict(X_valid),
        "Lasso RFE": models["Lasso RFE"].predict(X_valid),
        "Ridge RFE": models["Ridge RFE"].predict(X_valid),
        "XGBoost Regressor": models["XGBoost Regressor"].predict(X_valid_enc),
        "CatBoost Regressor": models["CatBoost Regressor"].predict(X_valid_enc),
    }
    test_preds = {
        "Linear Regressor": models["Linear Regressor"].predict(X_test),
        "Lasso RFE": models["Lasso RFE"].predict(X_test),
        "Ridge RFE": models["Ridge RFE"].predict(X_test),
        "XGBoost Regressor": models["XGBoost Regressor"].predict(X_test_enc),
        "CatBoost Regressor": models["CatBoost Regressor"].predict(X_test_enc),
    }

    print_r2_table("Validation Set — R²", valid_preds, y_valid)
    print_r2_table("Test Set — R²", test_preds, y_test)

    for split_label, X_raw, y, preds in [
        ("Validation", X_valid, y_valid, valid_preds),
        ("Test", X_test, y_test, test_preds),
    ]:
        console.print(f"\n[bold]{split_label} Set Performance Metrics:[/bold]")
        summarize_model_performance(
            model=list(models.values()),
            model_title=list(models.keys()),
            X=X_raw,
            y=y,
            y_pred=list(preds.values()),
            model_type="regression",
            return_df=True,
            decimal_places=3,
            include_adjusted_r2=True,
        )


def run_bootstrap(
    models,
    X_train,
    X_test,
    y_test,
    n_samples: int,
    num_resamples: int,
    random_state: int,
) -> pd.DataFrame:
    """Run bootstrapped CI evaluation for all models on the test set."""
    (X_test_enc,) = encode_categoricals(X_train, X_test)
    inputs = build_model_inputs(models, X_test, X_test_enc)

    all_results = []
    for name, (model, X) in inputs.items():
        console.print(f"[bold cyan]Bootstrap:[/bold cyan] {name}")
        result = evaluate_bootstrap_metrics(
            model=model,
            X=X,
            y=y_test,
            y_pred_prob=None,
            n_samples=n_samples,
            num_resamples=num_resamples,
            metrics=BOOTSTRAP_METRICS,
            random_state=random_state,
            threshold=0.5,
            average="macro",
            thresholds=None,
            model_type="regression",
            stratify=None,
            balance=False,
            class_proportions=None,
        )
        result.insert(0, "Model", name)
        all_results.append(result)

    combined = pd.concat(all_results, ignore_index=True)

    # Clean up metric names and fix negated signs before returning
    return rename_metrics(combined)


#################################### Main CLI ##################################


@app.command()
def main(
    n_samples: int = typer.Option(5000, help="Bootstrap sample size per resample."),
    num_resamples: int = typer.Option(5000, help="Number of bootstrap resamples."),
    random_state: int = typer.Option(42, help="Random seed for reproducibility."),
    no_bootstrap: bool = typer.Option(
        False, help="Skip bootstrap; show point estimates only."
    ),
    output: str = typer.Option(
        str(RESULTS_DIR / "bootstrap_results.csv"),
        help="Path to save combined bootstrap results CSV.",
    ),
):
    """
    Evaluate regression models on the ACLED Ukraine fatalities dataset.

    Runs point-estimate R² summaries for validation and test sets, then
    optionally runs bootstrapped confidence interval evaluation on the test set.
    Results are always saved to CSV.
    """
    models = load_models()
    X_train, X_valid, X_test, y_valid, y_test = load_data()

    console.rule("[bold]Point Estimates[/bold]")
    run_point_estimates(models, X_train, X_valid, X_test, y_valid, y_test)

    if no_bootstrap:
        console.print("\n[yellow]Skipping bootstrap (--no-bootstrap).[/yellow]")
        raise typer.Exit()

    console.rule("[bold]Bootstrap Evaluation — Test Set[/bold]")
    all_results = run_bootstrap(
        models, X_train, X_test, y_test, n_samples, num_resamples, random_state
    )

    console.print("\n[bold]Combined Bootstrap Results:[/bold]")
    console.print(all_results.round(3).to_string(index=False))

    ## Always save to CSV
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_results.round(3).to_csv(out_path, index=False)
    console.print(f"\n[green]✓ Results saved to {out_path}[/green]")


if __name__ == "__main__":
    app()
