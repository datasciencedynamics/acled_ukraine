from pathlib import Path
import pandas as pd
import re


def detect_project_root() -> Path:
    cwd = Path.cwd()
    if cwd.name == "notebooks":
        return cwd.parent
    return cwd


def clean_notes_text_series(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.lower()

    s = s.str.replace(r"https?://\S+|www\.\S+", " ", regex=True)
    s = s.str.replace(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", regex=True)

    s = (
        s.str.replace("’", "'", regex=False)
         .str.replace("“", '"', regex=False)
         .str.replace("”", '"', regex=False)
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
    lead_date_pattern = r"^\s*(On|Around)\s+\d{1,2}\s+[A-Za-z]+\s+\d{4}\s*,?\s*"
    return (
        notes.fillna("").astype(str)
            .str.replace(lead_date_pattern, "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
    )


def main() -> None:
    project_root = detect_project_root()
    data_dir = project_root / "data"
    raw_path = data_dir / "raw" / "acled_ukraine_data_2026_01_02.parquet"
    out_path = data_dir / "text_base.parquet"

    df = pd.read_parquet(raw_path)[["event_id_cnty", "notes"]]

    if not df["event_id_cnty"].is_unique:
        raise ValueError("Primary key event_id_cnty is not unique in the raw file.")

    notes_clean = remove_leading_date_phrase(df["notes"])
    notes_clean_ml = strip_leading_comma(clean_notes_text_series(notes_clean))

    out = (
        pd.DataFrame({"event_id_cnty": df["event_id_cnty"], "notes_clean_ml": notes_clean_ml})
        .drop_duplicates(subset=["event_id_cnty"], keep="last")
    )

    data_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}  rows={len(out)} cols={out.shape[1]}")


if __name__ == "__main__":
    main()
