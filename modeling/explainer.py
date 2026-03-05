################################################################################
# STEP 1: Import Required Libraries and Modules
################################################################################

# Import standard libraries, third-party tools, and custom modules for data
# processing, modeling, and SHAP analysis
import typer
import shap

from core.functions import (
    mlflow_load_model,
    mlflow_dumpArtifact,
)

from modeling.predict import find_best_model
from core.constants import (
    target_log_outcome,
    shap_artifact_name,
    shap_artifacts_data,
    shap_run_name,
)

app = typer.Typer()


@app.command()
def main(
    outcome: str = target_log_outcome,
    metric_name: str = "test_r2",
    mode: str = "max",
):

    ############################################################################
    # STEP 2: Define Experiment and Model Parameters
    ############################################################################
    # Set up the experiment name based on the outcome variable and retrieve
    # estimator details
    ############################################################################

    experiment_name = f"{outcome}_model"

    ############################################################################
    # STEP 3: Identify and Load the Best Model from MLflow
    ############################################################################
    # Find the best model run based on the specified metric and load it from
    # MLflow
    ############################################################################

    run_name, estimator_name = find_best_model(
        experiment_name,
        metric_name,
        mode,
    )
    model_name = f"{estimator_name}_{outcome}"
    model = mlflow_load_model(experiment_name, run_name, model_name)

    ############################################################################
    # STEP 4: Extract the Complete Pipeline from the Trained Model
    ############################################################################
    # Retrieve the full preprocessing and modeling pipeline from the model
    # object
    ############################################################################

    pipeline = model.estimator

    ############################################################################
    # STEP 5: Isolate the Final Estimator from the Pipeline
    ############################################################################
    # Extract the last step (regressor) from the pipeline for SHAP analysis
    ############################################################################

    final_model = pipeline.steps[-1][1]

    ############################################################################
    # STEP 6: Create SHAP Explainer for Model Interpretability
    ############################################################################
    # Initialize SHAP TreeExplainer using the final regressor. TreeExplainer
    # is used because our best-performing models are tree-based (XGBoost,
    # CatBoost). This is significantly faster than KernelExplainer and
    # produces exact SHAP values for tree models. No background data is
    # required — TreeExplainer derives expectations directly from the
    # tree structure.
    ############################################################################

    explainer = shap.TreeExplainer(final_model)

    ############################################################################
    # STEP 7: Persist SHAP Explainer for Future Use
    ############################################################################
    # Log the SHAP explainer as an MLflow artifact for tracking and
    # reproducibility
    ############################################################################

    mlflow_dumpArtifact(
        experiment_name=shap_artifact_name,
        run_name=shap_run_name,
        obj_name="explainer",
        obj=explainer,
        artifacts_data_path=shap_artifacts_data,
    )


if __name__ == "__main__":
    app()
