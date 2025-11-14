
"""
Data preprocessing module for EduRise.

Responsibilities:
- Load raw education datasets (U-DISE+, NAS, Census-like CSVs)
- Clean missing values
- Merge into a single modelling table
"""

import pandas as pd
from pathlib import Path

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_udise(path: Path) -> pd.DataFrame:
    """Load U-DISE+ like CSV file."""
    return pd.read_csv(path)

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Simple cleaning: drop fully empty columns and fill numeric NaNs with median."""
    df = df.copy()
    df = df.dropna(axis=1, how="all")
    for col in df.select_dtypes(include="number").columns:
        median = df[col].median()
        df[col] = df[col].fillna(median)
    return df

def build_modelling_table(udise_file: str) -> pd.DataFrame:
    """Build a single modelling table from a U-DISE-like CSV.

    For demo, we only use one file and apply basic cleaning.
    """
    udise_path = RAW_DATA_DIR / udise_file
    df = load_udise(udise_path)
    df = basic_clean(df)
    # For demo, assume 'enrolment_current' and 'enrolment_next' exist.
    if {"enrolment_current", "enrolment_next"}.issubset(df.columns):
        denom = df["enrolment_current"].replace(0, pd.NA)
        df["enrolment_change_pct"] = (
            (df["enrolment_next"] - df["enrolment_current"]) / denom
        ) * 100
    processed_path = PROCESSED_DATA_DIR / "modelling_table.csv"
    df.to_csv(processed_path, index=False)
    return df

if __name__ == "__main__":
    print("Building modelling table from sample_udise.csv ...")
    table = build_modelling_table("sample_udise.csv")
    print("Processed rows:", len(table))
