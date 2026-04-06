import sys
import typer
import pandas as pd
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import box
from sklearn.metrics import r2_score
from sklearn.utils import resample
from model_tuner import loadObjects, evaluate_bootstrap_metrics
from model_metrics import summarize_model_performance

sys.path.append("../")

################################### Paths ######################################

BASE = Path("/home/lshpaner/Python_Projects/acled_ukraine")

MODEL_PATHS = {
    "Linear Regressor": BASE
    / "mlruns/models/306762030449779565/162f7fd63e104b25a59bb610c4439308/artifacts/lr_log_fatalities/model.pkl",
    ## lasso_orig_rfe_training
    "Lasso RFE": BASE
    / "mlruns/models/306762030449779565/3d48f6d0797b4477bdcff57e51428624/artifacts/lasso_log_fatalities/model.pkl",
    # ridge_orig_rfe_training
    "Ridge RFE": BASE
    / "mlruns/models/306762030449779565/37d760e4bca144708b8decb30afb8fa7/artifacts/ridge_log_fatalities/model.pkl",
    # elasticnet_orig_rfe_training
    "ElasticNet RFE": BASE
    / "mlruns/models/306762030449779565/335f28ea0fc54feb8016b00993ade2c5/artifacts/elastic_net_log_fatalities/model.pkl",
    "XGBoost Regressor": BASE
    / "mlruns/models/306762030449779565/04c10026de6d4dda9e318683f439be25/artifacts/xgb_log_fatalities/model.pkl",
    # cat_orig_rfe_training
    "CatBoost Regressor": BASE
    / "mlruns/models/306762030449779565/a11e1a6f5f98417cb1c1cc05a4d5b195/artifacts/cat_log_fatalities/model.pkl",
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
    "neg_mean_absolute_error",
]

## Human-readable display names mapped from sklearn internal names
METRIC_DISPLAY_NAMES = {
    "r2": "R²",
    "adjusted_r2": "Adjusted R²",
    "neg_root_mean_squared_error": "RMSE",
    "neg_mean_absolute_error": "MAE",
    "capture_auc": "Capture AUC",
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
        "ElasticNet RFE": (models["ElasticNet RFE"], X_raw),
        "XGBoost Regressor": (models["XGBoost Regressor"], X_encoded),
        "CatBoost Regressor": (models["CatBoost Regressor"], X_encoded),
    }


def rename_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Rename first
    df["Metric"] = df["Metric"].map(METRIC_DISPLAY_NAMES).fillna(df["Metric"])

    # Flip based on display names — no dependency on sklearn internal names
    error_metrics = {"RMSE", "MSE", "MAE"}
    mask = df["Metric"].isin(error_metrics)

    for col in ["Mean", "95% CI Lower", "95% CI Upper"]:
        df[col] = np.where(mask, df[col].abs(), df[col])

    # Re-sort CI bounds so Lower < Upper
    ci_min = df[["95% CI Lower", "95% CI Upper"]].min(axis=1)
    ci_max = df[["95% CI Lower", "95% CI Upper"]].max(axis=1)
    df["95% CI Lower"] = np.where(mask, ci_min, df["95% CI Lower"])
    df["95% CI Upper"] = np.where(mask, ci_max, df["95% CI Upper"])

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
        "ElasticNet RFE": models["ElasticNet RFE"].predict(X_valid),
        "XGBoost Regressor": models["XGBoost Regressor"].predict(X_valid_enc),
        "CatBoost Regressor": models["CatBoost Regressor"].predict(X_valid_enc),
    }
    test_preds = {
        "Linear Regressor": models["Linear Regressor"].predict(X_test),
        "Lasso RFE": models["Lasso RFE"].predict(X_test),
        "Ridge RFE": models["Ridge RFE"].predict(X_test),
        "ElasticNet RFE": models["ElasticNet RFE"].predict(X_test),
        "XGBoost Regressor": models["XGBoost Regressor"].predict(X_test_enc),
        "CatBoost Regressor": models["CatBoost Regressor"].predict(X_test_enc),
    }

    print_r2_table("Validation Set: R²", valid_preds, y_valid)
    print_r2_table("Test Set: R²", test_preds, y_test)

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


def bootstrap_capture_auc(
    y_true,
    y_pred,
    n_samples=500,
    num_resamples=1000,
    random_state=222,
):
    """
    Bootstrap the capture AUC metric independently, returning a single-row
    DataFrame matching evaluate_bootstrap_metrics output format.
    """
    from random import seed as set_seed, randint

    set_seed(random_state)

    y_true = pd.Series(np.ravel(y_true)).reset_index(drop=True)
    y_pred = pd.Series(np.ravel(y_pred)).reset_index(drop=True)

    auc_scores = []
    for _ in range(num_resamples):
        idx = resample(
            np.arange(len(y_true)),
            replace=True,
            n_samples=n_samples,
            random_state=randint(0, 1_000_000),
        )
        yt = np.expm1(y_true.iloc[idx].values)
        yp = y_pred.iloc[idx].values

        order = np.argsort(-yp)
        cumsum = np.cumsum(yt[order])
        total = cumsum[-1]

        if total == 0:
            auc_scores.append(0.0)
        else:
            y_norm = cumsum / total
            x_norm = np.linspace(1 / len(y_norm), 1.0, len(y_norm))
            auc_scores.append(float(np.trapz(y_norm, x_norm)))

    mean_score = np.mean(auc_scores)
    ci_lower = np.percentile(auc_scores, 2.5)
    ci_upper = np.percentile(auc_scores, 97.5)

    return pd.DataFrame(
        {
            "Metric": ["capture_auc"],
            "Mean": [mean_score],
            "95% CI Lower": [ci_lower],
            "95% CI Upper": [ci_upper],
        }
    )


def run_bootstrap(
    models,
    X_train,
    X,
    y,
    split_name: str,
    n_samples: int,
    num_resamples: int,
    random_state: int,
) -> pd.DataFrame:
    """Run bootstrapped CI evaluation for all models on a given split."""
    (X_enc,) = encode_categoricals(X_train, X)
    inputs = build_model_inputs(models, X, X_enc)

    all_results = []
    for name, (model, X_input) in inputs.items():
        console.print(f"[bold cyan]Bootstrap ({split_name}):[/bold cyan] {name}")
        result = evaluate_bootstrap_metrics(
            model=model,
            X=X_input,
            y=y,
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
            ci_method="percentile",
        )

        # Bootstrap capture AUC (computed outside model_tuner)
        y_pred = pd.Series(np.ravel(model.predict(X_input)))
        capture_row = bootstrap_capture_auc(
            y_true=y,
            y_pred=y_pred,
            n_samples=n_samples,
            num_resamples=num_resamples,
            random_state=random_state,
        )
        result = pd.concat([result, capture_row], ignore_index=True)

        result.insert(0, "Model", name)
        result.insert(0, "Split", split_name)
        all_results.append(result)

    combined = pd.concat(all_results, ignore_index=True)
    return rename_metrics(combined)


#################################### Main CLI ##################################


@app.command()
def main(
    n_samples: int = typer.Option(5000, help="Bootstrap sample size per resample."),
    num_resamples: int = typer.Option(5000, help="Number of bootstrap resamples."),
    random_state: int = typer.Option(42, help="Random seed for reproducibility."),
    split: str = typer.Option("test", help="Split to bootstrap: 'valid' or 'test'."),
    no_bootstrap: bool = typer.Option(
        False, help="Skip bootstrap; show point estimates only."
    ),
    output: str = typer.Option(
        str(RESULTS_DIR / "bootstrap_results.csv"),
        help="Path to save combined bootstrap results CSV.",
    ),
):
    models = load_models()
    X_train, X_valid, X_test, y_valid, y_test = load_data()

    console.rule("[bold]Point Estimates[/bold]")
    run_point_estimates(models, X_train, X_valid, X_test, y_valid, y_test)

    if no_bootstrap:
        console.print("\n[yellow]Skipping bootstrap (--no-bootstrap).[/yellow]")
        raise typer.Exit()

    X_split = X_valid if split == "valid" else X_test
    y_split = y_valid if split == "valid" else y_test

    console.rule(f"[bold]Bootstrap Evaluation — {split.capitalize()} Set[/bold]")
    all_results = run_bootstrap(
        models,
        X_train,
        X_split,
        y_split,
        split.capitalize(),
        n_samples,
        num_resamples,
        random_state,
    )

    console.print("\n[bold]Combined Bootstrap Results:[/bold]")
    console.print(all_results.round(3).to_string(index=False))

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_results.round(3).to_csv(out_path, index=False)
    console.print(f"\n[green] Results saved to {out_path}[/green]")


if __name__ == "__main__":
    app()
