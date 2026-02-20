################################################################################
######################### Step 1: Import Requisite Libraries ###################
################################################################################

import os
import pandas as pd
import numpy as np
import typer

from core.config import numerical_cols, categorical_cols
from core.functions import mlflow_dumpArtifact, mlflow_loadArtifact

from core.constants import (
    var_index,
    event_date,
    exp_artifact_name,
    preproc_run_name,
    target_outcome,
    target_log_outcome,
)

################################################################################
################ Model Preprocessing and Feature Engineering ###################
################################################################################

################################################################################
################ Step 2: Define Typer Application ##############################
################################################################################

app = typer.Typer()

################################################################################
################ Step 3: Define Main Function ##################################
################################################################################


@app.command()
def main(
    input_data_file: str = "./data/processed/df_sans_zero_missing.parquet",
    stage: str = "training",
    data_path: str = "./data/processed",
):
    """
    Processes split dataframes and generates X, y for each split.

    Args:
        input_data_file: Path to input data (used for inference stage)
        stage: "training" or "inference"
        data_path: Path to data directory
    """

    ############################################################################
    ################ Step 4: Training Stage ####################################
    ############################################################################

    if stage == "training":

        # Load the pre-split dataframes
        print("\n" + "=" * 80)
        print("Loading temporal splits...")
        print("=" * 80)

        train_df = pd.read_parquet(os.path.join(data_path, "train.parquet"))
        valid_df = pd.read_parquet(os.path.join(data_path, "valid.parquet"))
        test_df = pd.read_parquet(os.path.join(data_path, "test.parquet"))

        # Ensure index is set for all splits
        for name, df in [
            ("train_df", train_df),
            ("valid_df", valid_df),
            ("test_df", test_df),
        ]:
            try:
                df.set_index(var_index, inplace=True)
            except KeyError:
                print(f"Index '{var_index}' already set or doesn't exist in {name}")

        # Drop event_date and index column - not needed for training
        cols_to_drop = [event_date]

        # Check if 'index' column exists (shouldn't be a feature)
        if "index" in train_df.columns:
            cols_to_drop.append("index")

        for df in [train_df, valid_df, test_df]:
            for col in cols_to_drop:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)

        print(f"\nTrain shape: {train_df.shape}")
        print(f"Valid shape: {valid_df.shape}")
        print(f"Test shape: {test_df.shape}")

        # Print percentage breakdown
        total_events = len(train_df) + len(valid_df) + len(test_df)
        print("\n" + "-" * 80)
        print("SPLIT DISTRIBUTION")
        print("-" * 80)
        train_pct = len(train_df) / total_events * 100
        valid_pct = len(valid_df) / total_events * 100
        test_pct = len(test_df) / total_events * 100
        print(f"Train: {len(train_df):,} events ({train_pct:.1f}%)")
        print(f"Valid: {len(valid_df):,} events ({valid_pct:.1f}%)")
        print(f"Test:  {len(test_df):,} events ({test_pct:.1f}%)\n")
        print(f"Total: {total_events:,} events")
        print("-" * 80)

        ########################################################################
        # Step 5: Process TRAINING split #######################################
        ########################################################################
        print("\n" + "=" * 80)
        print("Processing TRAIN split...")
        print("=" * 80)

        X_train = train_df.drop(columns=[target_outcome]).copy()
        y_train_regular = train_df[target_outcome].copy()
        y_train_log = np.log1p(train_df[target_outcome]).copy()

        # Retain numeric + object columns
        cols_to_keep = X_train.select_dtypes(include=np.number).columns.tolist()
        cols_to_keep.extend(X_train.select_dtypes(include="object").columns.tolist())

        X_train = X_train[cols_to_keep]
        X_columns_list = X_train.columns.to_list()

        print(f"\nX_train shape: {X_train.shape}")
        print(f"Number of features: {len(X_columns_list)}")
        print(f"\nFirst 5 feature names: {X_columns_list[:5]}")
        print(f"\nFirst 5 rows of X_train:")
        print(X_train.head())
        print(f"\nFirst 5 rows of y_train (regular):")
        print(y_train_regular.head())
        print(f"\nFirst 5 rows of y_train (log):")
        print(y_train_log.head())

        # Save X_columns_list to MLflow for inference
        mlflow_dumpArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="X_columns_list",
            obj=X_columns_list,
        )

        ########################################################################
        # Step 6: Process VALIDATION split (using same columns as train)
        ########################################################################
        print("\n" + "=" * 80)
        print("Processing VALID split...")
        print("=" * 80)

        X_valid = valid_df.drop(columns=[target_outcome]).copy()
        y_valid_regular = valid_df[target_outcome].copy()
        y_valid_log = np.log1p(valid_df[target_outcome]).copy()
        X_valid = X_valid[X_columns_list]  # Use same columns as train

        print(f"X_valid shape: {X_valid.shape}")

        ########################################################################
        # Step 7: Process TEST split (using same columns as train)
        ########################################################################
        print("\n" + "=" * 80)
        print("Processing TEST split...")
        print("=" * 80)

        X_test = test_df.drop(columns=[target_outcome]).copy()
        y_test_regular = test_df[target_outcome].copy()
        y_test_log = np.log1p(test_df[target_outcome]).copy()
        X_test = X_test[X_columns_list]  # Use same columns as train

        print(f"X_test shape: {X_test.shape}")

        ########################################################################
        # Step 8: Save all X and y files
        ########################################################################
        print("\n" + "=" * 80)
        print("Saving feature and target files...")
        print("=" * 80)

        # Save X files
        X_train.to_parquet(os.path.join(data_path, "X_train.parquet"))
        X_valid.to_parquet(os.path.join(data_path, "X_valid.parquet"))
        X_test.to_parquet(os.path.join(data_path, "X_test.parquet"))
        print("Saved X files (train, valid, test)")

        # Save y files - REGULAR versions
        pd.DataFrame(y_train_regular).to_parquet(
            os.path.join(data_path, f"y_train_{target_outcome}.parquet")
        )
        pd.DataFrame(y_valid_regular).to_parquet(
            os.path.join(data_path, f"y_valid_{target_outcome}.parquet")
        )
        pd.DataFrame(y_test_regular).to_parquet(
            os.path.join(data_path, f"y_test_{target_outcome}.parquet")
        )
        print(f"Saved y files for '{target_outcome}' (train, valid, test)")

        # Save y files - LOG versions
        y_train_log_df = pd.DataFrame(y_train_log)
        y_train_log_df.columns = [target_log_outcome]
        y_train_log_df.to_parquet(
            os.path.join(data_path, f"y_train_{target_log_outcome}.parquet")
        )

        y_valid_log_df = pd.DataFrame(y_valid_log)
        y_valid_log_df.columns = [target_log_outcome]
        y_valid_log_df.to_parquet(
            os.path.join(data_path, f"y_valid_{target_log_outcome}.parquet")
        )

        y_test_log_df = pd.DataFrame(y_test_log)
        y_test_log_df.columns = [target_log_outcome]
        y_test_log_df.to_parquet(
            os.path.join(data_path, f"y_test_{target_log_outcome}.parquet")
        )
        print(f"Saved y files for '{target_log_outcome}' (train, valid, test)")

        print("\n" + "=" * 80)
        print("Feature generation complete for all splits!")
        print("=" * 80)
        print("\nGenerated files:")
        print("  X_train.parquet, X_valid.parquet, X_test.parquet")
        print(
            f"  y_train_{target_outcome}.parquet, y_valid_{target_outcome}.parquet, y_test_{target_outcome}.parquet"
        )
        print(
            f"  y_train_{target_log_outcome}.parquet, y_valid_{target_log_outcome}.parquet, y_test_{target_log_outcome}.parquet"
        )
        print("\n")

    ############################################################################
    ################ Step 9: Inference Stage ###################################
    ############################################################################

    elif stage == "inference":

        print("\n" + "=" * 80)
        print("Inference mode: Loading X_columns_list from MLflow...")
        print("=" * 80)

        # Load X_columns_list from MLflow
        X_columns_list = mlflow_loadArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="X_columns_list",
        )

        print(f"Loaded {len(X_columns_list)} feature columns")

        # Load inference data from input_data_file
        df = pd.read_parquet(input_data_file)

        # Filter to X_columns_list only
        X = df[X_columns_list].copy()
        X.to_parquet(os.path.join(data_path, "X.parquet"))

        print(f"X shape for inference: {X.shape}")
        print("\nInference features generated!")


################################################################################

if __name__ == "__main__":
    app()
