################################################################################
######################### Step 1: Import Requisite Libraries ###################
################################################################################

import os
import pandas as pd
import typer

from core.functions import create_temporal_splits

from core.constants import (
    var_index,
    TRAIN_END_DATE,
    VALID_END_DATE,
    DATA_FILTER_DATE,
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
    data_path: str = "./data/processed",
):
    """Create temporal train/valid/test splits on FULL dataframe"""

    ############################################################################
    ################ Step 4: Load Input Data ###################################
    ############################################################################
    # df already loaded from .parquet
    # Example:
    # df = pd.read_parquet("path_to_file.parquet")

    # Read the input data file
    df = pd.read_parquet(input_data_file)

    try:
        df.set_index(var_index, inplace=True)
    except:
        print("Index already set or 'var_index' doesn't exist in dataframe")
    print("-" * 80)
    print(f"# of DataFrame Columns: {df.shape[1]}")

    ############################################################################

    # Filter out incomplete Jan 2025 data
    df["event_date"] = pd.to_datetime(df["event_date"])
    df = df[df["event_date"] < DATA_FILTER_DATE]

    print(f"Filtered data before {DATA_FILTER_DATE}")
    print(f"Total events after filter: {len(df):,}")
    ############################################################################

    ################ Step 5: Create Temporal Splits ############################
    ############################################################################

    # Create temporal splits on FULL dataframe
    train_df, valid_df, test_df = create_temporal_splits(
        df, train_end=TRAIN_END_DATE, valid_end=VALID_END_DATE
    )

    # Print summaries
    for name, split_df in [("Train", train_df), ("Valid", valid_df), ("Test", test_df)]:
        print(
            f"\n{name}: {len(split_df):,} events | "
            f"{split_df['event_date'].min()} to {split_df['event_date'].max()}"
        )

    # Save FULL dataframes (still contain all columns including target)
    train_df.to_parquet(os.path.join(data_path, "train_df.parquet"))
    valid_df.to_parquet(os.path.join(data_path, "valid_df.parquet"))
    test_df.to_parquet(os.path.join(data_path, "test_df.parquet"))

    print("\nTemporal splits created successfully!")


if __name__ == "__main__":
    app()
