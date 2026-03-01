################################################################################
######################### Import Requisite Libraries ###########################
import os
import typer
import pandas as pd
import numpy as np

# from node2vec import Node2Vec  # COMMENTED OUT: No longer using embeddings

# import pickling scripts
from model_tuner.pickleObjects import dumpObjects

################################################################################

from core.constants import (
    var_index,
    preproc_run_name,
    exp_artifact_name,
    percent_miss,
    drop_vars,
)

# import all user-defined functions and constants
from core.functions import (
    mlflow_dumpArtifact,
    mlflow_loadArtifact,
    haversine_km,
    safe_to_numeric,
    # build_actor_interaction_graph,  # COMMENTED OUT: No longer using embeddings
    # add_pairwise_embedding_features,  # COMMENTED OUT: No longer using embeddings
    # apply_embeddings,  # COMMENTED OUT: No longer using embeddings
)

app = typer.Typer()

print("\n" + "#" * 80)
print(f"Running script: {os.path.basename(__file__)}")
print("#" * 80 + "\n")


@app.command()
def main(
    input_data_file: str = "./data/raw/acled_ukraine_data_2026_01_02.parquet",
    output_data_file: str = "./data/processed/df_sans_zero_missing.parquet",
    stage: str = "training",
    data_path: str = "./data/processed",
):
    """
    Main script execution replacing sys.argv with typer.

    Args:
        input_data_file (str): Path to the input parquet file.
        output_data_file (str): Path to save the processed parquet file.
        stage (str): Processing stage (e.g., 'training' or 'inference').
    """

    ############################################################################
    # Step 1. Read the input data file
    ############################################################################
    # IMPORTANT UPDATE:
    # We do NOT build embeddings from the full raw dataframe anymore.
    # We load the temporal splits created by temporal_splits.py, then:
    #   1) Build the actor graph ONLY from train_df (prevents temporal leakage)
    #   2) Train Node2Vec embeddings on that training graph
    #   3) Apply the learned embeddings to train, valid, and test splits
    #
    # UPDATE 2026-02-16: EMBEDDINGS COMMENTED OUT FOR INTERPRETABILITY
    ############################################################################

    train_path = os.path.join(data_path, "train_df.parquet")
    valid_path = os.path.join(data_path, "valid_df.parquet")
    test_path = os.path.join(data_path, "test_df.parquet")

    if (
        os.path.exists(train_path)
        and os.path.exists(valid_path)
        and os.path.exists(test_path)
    ):
        print("Loading temporal splits...")
        train_df = pd.read_parquet(train_path)
        valid_df = pd.read_parquet(valid_path)
        test_df = pd.read_parquet(test_path)
    else:
        raise FileNotFoundError(
            "Temporal splits not found. Run temporal_splits.py first "
            "to create train_df.parquet, valid_df.parquet, test_df.parquet."
        )

    # Set index if needed (on each split)
    for _name, _df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        try:
            _df.set_index(var_index, inplace=True)
        except Exception:
            print(
                f"({_name}) Index already set or '{var_index}' doesn't exist in dataframe"
            )

    print(f"Train Data Shape: {train_df.shape}")
    print(f"Valid Data Shape: {valid_df.shape}")
    print(f"Test Data Shape:  {test_df.shape}")

    print(
        f"There are {train_df.index.unique().shape[0]} unique indices in the TRAIN dataframe."
    )

    ############################################################################
    # Step 2. String Columns Handling
    ############################################################################
    # String columns are identified and should be removed before modeling
    # because machine learning models typically require numerical inputs.
    # Keeping string columns in the dataset may lead to errors or
    # unintended behavior unless explicitly encoded.
    #
    # To ensure consistency between training and inference,
    # we save the list of string columns and track it using MLflow.
    ############################################################################

    if stage == "training":

        # Identify string columns on TRAIN only
        df_object = train_df.select_dtypes("object")
        print()
        print(
            "The following columns have strings and may need to be removed from "
            "modeling and/or otherwise transformed with `categorical_transformer` "
            f"\nas handled accordingly in the `config.py` file. This list is stored "
            f"as an artifact in MLflow for future reference if necessary for "
            f"retrieval at a later time. \n \n"
            f"There are {df_object.shape[1]} string columns:\n \n"
            f"{df_object.columns.to_list()}. \n "
        )

        # Extract column names to a list
        string_cols_list = df_object.columns.to_list()

        ############################################################################
        # Step 3. Save and Log String Column List
        ############################################################################
        # Save the list of string columns for consistency across training and
        # inference and log them in MLflow for reproducibility.
        # This list of string columns is dumped (stored) only to inform of what
        # the string columns are; no further action is taken; we do not need to
        # load this list into production, since it is only there for us to
        # see what the columns are.
        ############################################################################

        # Dump the string_cols_list into a pickle file for future reference
        dumpObjects(
            string_cols_list,
            os.path.join(data_path, "string_cols_list.pkl"),
        )

        # Log the string column list as an artifact in MLflow
        mlflow_dumpArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="string_cols_list",
            obj=string_cols_list,
        )

        ############################################################################
        # Step 4. Store Unique Actor Lists
        ############################################################################
        # Extract unique actor1 and actor2 lists from the dataset and store them
        # as pickle files for future reference. Additionally, log these lists as
        # artifacts in MLflow to ensure reproducibility and easy access during
        # model training and inference.
        #
        # IMPORTANT UPDATE:
        # Store lists from TRAIN only to avoid accidentally recording future-only
        # actors as part of the "known" set.
        ############################################################################

        actor1_list = train_df["actor1"].unique().tolist()
        actor2_list = train_df["actor2"].unique().tolist()

        dumpObjects(actor1_list, os.path.join(data_path, "actor1_list.pkl"))
        dumpObjects(actor2_list, os.path.join(data_path, "actor2_list.pkl"))

        mlflow_dumpArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="actor1_list",
            obj=actor1_list,
        )

        mlflow_dumpArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="actor2_list",
            obj=actor2_list,
        )

    if stage == "inference":

        ########################################################################
        # Load Previously Saved Object
        ########################################################################
        # During training, we identified and stored actor lists.
        # Now, we reload this to ensure that inference follows the same
        # preprocessing pipeline as training, maintaining consistency.
        ########################################################################

        actor1_list = mlflow_loadArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="actor1_list",
        )
        actor2_list = mlflow_loadArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="actor2_list",
        )

    ############################################################################
    ###################### Re-engineering Selected Features ####################
    ############################################################################

    ############################################################################
    # Step 5. Re-encode `civilian_targeting` column to binarized format
    ############################################################################
    # The `civilian_targeting` column is re-encoded to a binary format where
    # 1 indicates the presence of civilian targeting and 0 indicates its absence.
    # This transformation simplifies the variable for modeling purposes,
    # making it easier to interpret and utilize in predictive analyses.
    #
    # IMPORTANT UPDATE:
    # Apply transform to EACH split. Use errors-safe approach if column missing.
    ############################################################################

    to_binary = lambda x: 1 if x == "Civilian targeting" else 0
    for _df in (train_df, valid_df, test_df):
        if "civilian_targeting" in _df.columns:
            _df["civilian_targeting"] = _df["civilian_targeting"].apply(to_binary)

    ############################################################################
    # Step 6. Backfill missing admin1 using location
    ############################################################################
    # If admin1 is missing or blank, use location as a proxy regional label.
    # This preserves spatial signal and avoids introducing artificial categories.

    for _df in (train_df, valid_df, test_df):
        if "admin1" in _df.columns and "location" in _df.columns:

            _df["admin1"] = _df["admin1"].replace("", pd.NA).fillna(_df["location"])

            # ensure string dtype for encoder safety
            _df["admin1"] = _df["admin1"].astype(str)

    ############################################################################
    # Step 6b. Days Since Invasion Feature
    ############################################################################
    # Compute the number of days between each event and the start of the
    # full-scale Russian invasion (2022-02-24). This gives the model a
    # monotonic temporal signal without requiring the raw date column.
    ############################################################################

    INVASION_DATE = pd.Timestamp("2022-02-24")

    for _df in (train_df, valid_df, test_df):
        if "event_date" in _df.columns:
            _df["days_since_invasion"] = (
                pd.to_datetime(_df["event_date"]) - INVASION_DATE
            ).dt.days

    ############################################################################
    # Step 7. Drop Intermediate and Redundant Columns
    ############################################################################
    # Drop intermediate columns used for normalization and embedding
    # generation to streamline the dataset and retain only relevant features
    # for modeling. This step helps in reducing redundancy and improving
    # dataset clarity.
    #
    # Also drop event_type because sub_event_type is more granular and informative
    # and `event_date` because it is not needed for regression modeling
    # The full list of columns to be dropped are contained within `drop_vars`
    # inside constants.py
    #
    # IMPORTANT UPDATE:
    # Apply drops to EACH split.
    ############################################################################

    for _df in (train_df, valid_df, test_df):
        _df.drop(columns=drop_vars, errors="ignore", inplace=True)

    ########################################################################
    # Step 8. Ensure Numeric Data and Feature Engineering
    ########################################################################
    # Convert any possible numeric values that may have been incorrectly
    # classified as non-numeric. This avoids accidental labeling errors.
    #
    # IMPORTANT UPDATE:
    # Apply safe_to_numeric to EACH split.
    ########################################################################

    train_df = train_df.apply(lambda x: safe_to_numeric(x))
    valid_df = valid_df.apply(lambda x: safe_to_numeric(x))
    test_df = test_df.apply(lambda x: safe_to_numeric(x))

    ################################################################################
    # Step 9. Zero Variance Columns
    ################################################################################
    # Select only numeric columns s/t .var() can be applied since you can only
    # call this function on numeric columns; otherwise, if you include a mix
    # (object and numeric), it will throw the following FutureWarning:
    # Dropping of nuisance columns in DataFrame reductions
    # (with 'numeric_only=None') is deprecated; in a future version this will
    # raise TypeError.  Select only valid columns before calling the reduction.
    #
    # IMPORTANT UPDATE:
    # Fit zero-variance columns on TRAIN only, then drop the same columns
    # from VALID and TEST to keep feature space consistent without leakage.
    ################################################################################

    if stage == "training":
        numeric_cols = train_df.select_dtypes(include=["number"]).columns
        var_indf = train_df[numeric_cols].var()
        zero_var = var_indf[var_indf == 0]
        zero_varlist_list = list(zero_var.index)

        print("-" * 80)
        print(f"\n\nZero-variance columns: {zero_varlist_list}\n\n")
        print("-" * 80)

        dumpObjects(
            zero_varlist_list,
            os.path.join(data_path, "zero_varlist_list.pkl"),
        )

        mlflow_dumpArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="zero_varlist_list",
            obj=zero_varlist_list,
        )

    if stage == "inference":
        zero_varlist_list = mlflow_loadArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="zero_varlist_list",
        )

    # Drop zero-variance columns from each split (errors ignored)
    train_df = train_df.drop(columns=zero_varlist_list, errors="ignore")
    valid_df = valid_df.drop(columns=zero_varlist_list, errors="ignore")
    test_df = test_df.drop(columns=zero_varlist_list, errors="ignore")

    print(f"Train shape after dropping zero var cols: {train_df.shape}")
    print(f"Valid shape after dropping zero var cols: {valid_df.shape}")
    print(f"Test shape after dropping zero var cols:  {test_df.shape}")

    ############################################################################
    # Step 10. Calculate Row-wise Missingness Percentage
    ############################################################################
    # This step computes the proportion of missing values for each row in the
    # DataFrame. It helps identify rows with a high level of incompleteness, which
    # may be useful for filtering, imputation strategies, or downstream analysis.
    #
    # A new column is added to each split where each value represents
    # the percentage of columns that are missing for that row.
    ############################################################################

    for _df in (train_df, valid_df, test_df):
        _df[percent_miss] = _df.isna().mean(axis=1)

    ############################################################################
    # Step 16 Ensure parquet-safe dtypes
    ############################################################################
    # PyArrow requires consistent column types.
    # Some ACLED admin/location fields may contain mixed types.
    # Convert object columns to string to avoid ArrowTypeError.

    for _df in (train_df, valid_df, test_df):
        obj_cols = _df.select_dtypes(include="object").columns
        _df[obj_cols] = _df[obj_cols].astype(str)

    ############################################################################
    # Step 11. Save Processed Data
    ############################################################################
    ############################################################################

    train_out = os.path.join(data_path, "train.parquet")
    valid_out = os.path.join(data_path, "valid.parquet")
    test_out = os.path.join(data_path, "test.parquet")

    train_df.to_parquet(train_out)
    valid_df.to_parquet(valid_out)
    test_df.to_parquet(test_out)

    print("\nSaved enriched splits (NO EMBEDDINGS - commented out):")
    print(f"  TRAIN: {train_out}")
    print(f"  VALID: {valid_out}")
    print(f"  TEST:  {test_out}")


if __name__ == "__main__":
    app()
