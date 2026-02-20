################################################################################
######################### Step 1: Import Requisite Libraries ###################
################################################################################

import os
import pandas as pd
import typer

from core.functions import create_temporal_splits, normalize_split

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
    input_data_file: str = "./data/raw/acled_ukraine_data_2026_01_02.parquet",
    data_path: str = "./data/processed",
):
    """Create temporal train/valid/test splits on FULL dataframe"""

    ############################################################################
    ################ Step 4: Load Input Data ###################################
    ############################################################################

    # Read the input data file
    df = pd.read_parquet(input_data_file)

    try:
        if var_index in df.columns:
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

    ############################################################################
    ################ Step 6: LEAKAGE CHECKS ####################################
    ############################################################################

    print("\n" + "=" * 80)
    print("TEMPORAL LEAKAGE VERIFICATION")
    print("=" * 80)

    # Check 1: Verify chronological ordering
    train_max = train_df["event_date"].max()
    valid_min = valid_df["event_date"].min()
    valid_max = valid_df["event_date"].max()
    test_min = test_df["event_date"].min()

    print("\n[1] Temporal Ordering Check:")
    print(f"  Train ends:   {train_max}")
    print(f"  Valid starts: {valid_min}")
    print(f"  Valid ends:   {valid_max}")
    print(f"  Test starts:  {test_min}")

    if train_max < valid_min < valid_max < test_min:
        print("  PASS: Chronological ordering is correct")
    else:
        print("  FAIL: Dates overlap or are out of order!")

    # Check 2: No duplicate event_id_cnty across splits
    train_ids = set(train_df.index)
    valid_ids = set(valid_df.index)
    test_ids = set(test_df.index)

    print("\n[2] Event ID Overlap Check:")
    print(f"  Train unique IDs: {len(train_ids):,}")
    print(f"  Valid unique IDs: {len(valid_ids):,}")
    print(f"  Test unique IDs:  {len(test_ids):,}")

    train_valid_overlap = train_ids & valid_ids
    train_test_overlap = train_ids & test_ids
    valid_test_overlap = valid_ids & test_ids

    if len(train_valid_overlap) == 0:
        print("   PASS: No event ID overlap between Train and Valid")
    else:
        print(f"   FAIL: {len(train_valid_overlap)} IDs overlap Train/Valid!")
        print(f"    Example overlaps: {list(train_valid_overlap)[:5]}")

    if len(train_test_overlap) == 0:
        print("   PASS: No event ID overlap between Train and Test")
    else:
        print(f"   FAIL: {len(train_test_overlap)} IDs overlap Train/Test!")
        print(f"    Example overlaps: {list(train_test_overlap)[:5]}")

    if len(valid_test_overlap) == 0:
        print("   PASS: No event ID overlap between Valid and Test")
    else:
        print(f"   FAIL: {len(valid_test_overlap)} IDs overlap Valid/Test!")
        print(f"    Example overlaps: {list(valid_test_overlap)[:5]}")

    # Check 3: All original IDs accounted for
    total_split_ids = train_ids | valid_ids | test_ids
    original_ids = set(df.index)

    print("\n[3] Completeness Check:")
    print(f"  Original data: {len(original_ids):,} unique IDs")
    print(f"  After splits:  {len(total_split_ids):,} unique IDs")

    if len(total_split_ids) == len(original_ids):
        print("   PASS: All events accounted for in splits")
    else:
        missing = original_ids - total_split_ids
        print(f"   WARNING: {len(missing)} events missing from splits!")
        print(f"    Example missing: {list(missing)[:5]}")

    # Check 4: No duplicate IDs within each split
    print("\n[4] Internal Duplicate Check:")

    train_dupes = len(train_df) - len(train_ids)
    valid_dupes = len(valid_df) - len(valid_ids)
    test_dupes = len(test_df) - len(test_ids)

    if train_dupes == 0:
        print("   PASS: No duplicate IDs within Train")
    else:
        print(f"   FAIL: {train_dupes} duplicate IDs in Train!")

    if valid_dupes == 0:
        print("   PASS: No duplicate IDs within Valid")
    else:
        print(f"   FAIL: {valid_dupes} duplicate IDs in Valid!")

    if test_dupes == 0:
        print("   PASS: No duplicate IDs within Test")
    else:
        print(f"   FAIL: {test_dupes} duplicate IDs in Test!")

    # Summary
    print("\n" + "=" * 80)
    all_checks_pass = (
        train_max < valid_min < valid_max < test_min
        and len(train_valid_overlap) == 0
        and len(train_test_overlap) == 0
        and len(valid_test_overlap) == 0
        and len(total_split_ids) == len(original_ids)
        and train_dupes == 0
        and valid_dupes == 0
        and test_dupes == 0
    )

    if all_checks_pass:
        print(" ALL CHECKS PASSED - No temporal leakage detected!")
    else:
        print(" SOME CHECKS FAILED - Review issues above!")
    print("=" * 80 + "\n")

    ############################################################################
    ################ Step 7: Save Splits #######################################
    ############################################################################

    # Save FULL dataframes (still contain all columns including target)
    # Reset index to save event_id_cnty as a column
    train_df.reset_index(drop=False).to_parquet(
        os.path.join(data_path, "train_df.parquet"), index=False
    )
    valid_df.reset_index(drop=False).to_parquet(
        os.path.join(data_path, "valid_df.parquet"), index=False
    )
    test_df.reset_index(drop=False).to_parquet(
        os.path.join(data_path, "test_df.parquet"), index=False
    )

    print("Temporal splits created and saved successfully!")


if __name__ == "__main__":
    app()
