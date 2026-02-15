################################################################################
######################### Import Requisite Libraries ###########################
from pathlib import Path
import pandas as pd
import typer

################################################################################

app = typer.Typer()


def clean_notes_text_series(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.lower()

    s = s.str.replace(r"https?://\S+|www\.\S+", " ", regex=True)
    s = s.str.replace(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", regex=True)

    s = (
        s.str.replace("'", "'", regex=False)
         .str.replace(""", '"', regex=False)
         .str.replace(""", '"', regex=False)
    )

    s = s.str.replace(r"[^a-z0-9\s\.\,\!\?\-']", " ", regex=True)
    s = s.str.replace(r"([!\?.,])\1+", r"\1", regex=True)

    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def strip_leading_comma(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
         .astype(str)
         .str.replace(r"^\s*,\s*", "", regex=True)
    )


def remove_leading_date_phrase(notes: pd.Series) -> pd.Series:
    # Removes ONLY a leading "On 2 January 2026," or "Around 2 January 2026,"
    lead_date_pattern = r"^\s*(On|Around)\s+\d{1,2}\s+[A-Za-z]+\s+\d{4}\s*,?\s*"
    return (
        notes.fillna("").astype(str)
            .str.replace(lead_date_pattern, "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
    )


@app.command()
def main(
    input_data_file: str = "./data/raw/acled_ukraine_data_2026_01_02.parquet",
    output_data_file: str = "./data/processed/catboost_text.parquet",
) -> None:
    """
    Clean text data for CatBoost modeling.

    Args:
        input_data_file (str): Path to input parquet file with 'notes' column.
        output_data_file (str): Path to output parquet file.
    """
    
    input_path = Path(input_data_file)
    output_path = Path(output_data_file)

    ############################################################################
    # Step 1. Read Input Data
    ############################################################################
    
    print("Starting text preprocessing pipeline...")
    print(f"Reading data from: {input_path}")
    
    # Read with index - event_id_cnty is the index
    df = pd.read_parquet(input_path, columns=["notes"])
    print(f"Loaded {len(df):,} rows")
    
    ############################################################################
    # Step 2. Verify Index Uniqueness
    ############################################################################
    
    # Index should already be unique since it's the index, but verify
    if not df.index.is_unique:
        raise ValueError("Index event_id_cnty is not unique in the raw file.")
    print("Index uniqueness verified")

    ############################################################################
    # Step 3. Clean Text Data
    ############################################################################
    
    print()
    print("Cleaning text:")
    print("  - Removing leading date phrases...")
    notes_clean = remove_leading_date_phrase(df["notes"])
    
    print("  - Applying text cleaning (lowercasing, removing URLs, emails, special chars)...")
    catboost_notes_clean = strip_leading_comma(clean_notes_text_series(notes_clean))

    ############################################################################
    # Step 4. Build Output DataFrame
    ############################################################################
    
    print()
    print("Building output DataFrame...")
    
    # Create output with index preserved
    out = pd.DataFrame(
        {"catboost_notes_clean": catboost_notes_clean},
        index=df.index
    )
    # Index name should be preserved, but ensure it
    out.index.name = "event_id_cnty"
    
    ############################################################################
    # Step 5. Remove Duplicate Indices
    ############################################################################
    
    # Drop duplicates if any exist (keeping last)
    n_before = len(out)
    out = out[~out.index.duplicated(keep='last')]
    n_after = len(out)
    if n_before > n_after:
        print(f"  - Removed {n_before - n_after:,} duplicate indices")

    ############################################################################
    # Step 6. Save Processed Data
    ############################################################################
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print()
    print(f"Saving to: {output_path}")
    # Save with index
    out.to_parquet(output_path)
    print(f"Saved: {output_path}")
    print(f"  Final shape: {len(out):,} rows x {out.shape[1]} column(s)")
    print()
    print("Preprocessing complete!")


if __name__ == "__main__":
    app()