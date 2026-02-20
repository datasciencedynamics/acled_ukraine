################################################################################
######################### Step 1: Import Requisite Libraries ###################
################################################################################

import re
import pandas as pd
import typer

################################################################################
################ Remove Data Leakage from Notes ################################
################################################################################

################################################################################
################ Step 2: Define Typer Application ##############################
################################################################################

app = typer.Typer()

################################################################################
################ Step 3: Define Leakage Removal Function ######################
################################################################################


def remove_notes_leakage(text):
    """
    Remove leakage patterns from notes text.

    Removes:
    - Explicit fatality/injury counts
    - Casualty status statements
    - Outcome descriptions that imply deaths

    Keeps:
    - Event descriptions (what happened)
    - Location, actors, weapons
    - Temporal information
    """
    if pd.isna(text) or text == "":
        return text

    text = str(text)

    # CRITICAL: Remove explicit casualty mentions
    leakage_patterns = [
        # Explicit counts - deaths
        (
            r"\d+\s+(people|persons|soldiers|civilians|fighters|militants|"
            r"troops|combatants|men|women|children)\s+(were|was)\s+"
            r"(killed|dead|died|slain|deceased)",
            "",
        ),
        (r"\d+\s+(killed|dead|died|deaths|fatalities|casualties)", ""),
        (r"(killed|deaths?|fatalities|casualties):\s*\d+", ""),
        # No casualty statements
        (
            r"[Tt]here\s+(were|was)\s+no\s+"
            r"(injuries|fatalities|casualties|deaths|wounded)",
            "",
        ),
        (
            r"[Nn]o\s+(injuries|fatalities|casualties|deaths)\s+or\s+"
            r"(injuries|fatalities|casualties|deaths)",
            "",
        ),
        (
            r"[Nn]o\s+(one|person|soldier|civilian)\s+(was|were)\s+"
            r"(killed|injured|wounded|hurt)",
            "",
        ),
        (r"[Ww]ithout\s+(injuries|fatalities|casualties|deaths)", ""),
        # Injury/wounded mentions
        (r"\d+\s+(were|was)\s+(injured|wounded|hurt|hospitalized)", ""),
        (
            r"(one|two|three|four|five|six|seven|eight|nine|ten)\s+"
            r"(soldier|civilian|person)s?\s+(was|were)\s+"
            r"(injured|wounded|killed|hurt)",
            "",
        ),
        # Casualties unknown/unreported
        (
            r"[Cc]asualties\s+"
            r"(unknown|unreported|unclear|not\s+(yet\s+)?confirmed)",
            "",
        ),
        (
            r"(number\s+of\s+)?(deaths?|fatalities|casualties)\s+"
            r"(unknown|unreported|unclear)",
            "",
        ),
        # Outcome descriptions that strongly imply deaths
        (
            r"successfully\s+(repulsed|repelled|defeated|destroyed)",
            "engaged with",
        ),
        (r"(unsuccessful|failed)\s+(offensive|attack|assault)", "offensive"),
        # Medical aftermath
        (
            r"(taken|rushed|transported|evacuated)\s+to\s+"
            r"(hospital|medical\s+facility)",
            "",
        ),
        (
            r"(bodies|corpses)\s+(were|was)\s+" r"(found|discovered|recovered)",
            "",
        ),
        # Death verbs
        (r"(perished|succumbed|lost\s+(their\s+)?lives?)", ""),
        # During the day casualty statements
        (
            r"[Dd]uring\s+the\s+day,?\s+(one|\d+)\s+"
            r"(soldier|civilian)\s+was\s+(killed|wounded|injured)[^.]*\.?",
            "",
        ),
    ]

    for pattern, replacement in leakage_patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Clean up extra spaces
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


################################################################################
################ Step 4: Define Main Function ##################################
################################################################################


@app.command()
def main(
    input_file: str = "./data/processed/text_base.parquet",
    output_file: str = "./data/processed/text_base_no_leakage.parquet",
):
    """
    Remove leakage from notes column.

    Args:
        input_file: Path to input parquet file with notes column
        output_file: Path to save cleaned parquet file
    """

    print("\n" + "=" * 80)
    print("Loading data...")
    print("=" * 80)

    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df):,} rows")
    print(f"Columns: {df.columns.tolist()}")

    # Check if notes column exists
    if "notes_with_tags" in df.columns:
        notes_col = "notes_with_tags"
    elif "notes_clean_ml" in df.columns:
        notes_col = "notes_clean_ml"
    elif "notes" in df.columns:
        notes_col = "notes"
    else:
        print("ERROR: No notes column found!")
        print(f"Available columns: {df.columns.tolist()}")
        return

    print(f"\nUsing column: '{notes_col}'")

    print("\n" + "=" * 80)
    print("Removing leakage from notes...")
    print("=" * 80)

    # Apply leakage removal
    df[f"{notes_col}_clean"] = df[notes_col].apply(remove_notes_leakage)

    # Show examples
    print("\nExample transformations:")
    print("-" * 80)
    for i in range(min(5, len(df))):
        if (
            pd.notna(df[notes_col].iloc[i])
            and df[notes_col].iloc[i] != df[f"{notes_col}_clean"].iloc[i]
        ):
            print(f"\nRow {i}:")
            print(f"  Before: {df[notes_col].iloc[i][:150]}...")
            print(f"  After:  {df[f'{notes_col}_clean'].iloc[i][:150]}...")
            break

    # Statistics
    total_chars_before = df[notes_col].str.len().sum()
    total_chars_after = df[f"{notes_col}_clean"].str.len().sum()
    reduction_pct = (
        (total_chars_before - total_chars_after) / total_chars_before
    ) * 100

    print("\n" + "=" * 80)
    print("Statistics:")
    print("=" * 80)
    print(f"Total characters before: {total_chars_before:,}")
    print(f"Total characters after:  {total_chars_after:,}")
    print(f"Reduction: {reduction_pct:.1f}%")

    # Count how many rows changed
    changed = (df[notes_col] != df[f"{notes_col}_clean"]).sum()
    print(f"\nRows modified: {changed:,} ({changed/len(df)*100:.1f}%)")

    print("\n" + "=" * 80)
    print("Saving cleaned data...")
    print("=" * 80)

    df.to_parquet(output_file, index=False)
    print(f"Saved to: {output_file}")

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


################################################################################

if __name__ == "__main__":
    app()
