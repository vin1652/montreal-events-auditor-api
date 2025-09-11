# agents/cleaner.py
from __future__ import annotations
import pandas as pd
import numpy as np

def clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    steps to clean the raw dataframe:
    - Input: raw dataframe from collector (no filtering, no deduping, no trimming)
      - Parse only the alias datetime columns so time filters work.
      - Do NOT dedupe, do NOT trim strings, do NOT drop rows.
    """
    df = df_raw.copy()

    # Location / coords
    df["lat"]          = pd.to_numeric(df.get("lat"), errors="coerce")
    # Original is "long" in the dataset — create a "lon" alias 
    if "long" in df.columns:
        df["lon"] = pd.to_numeric(df["long"], errors="coerce")
    else:
        df["lon"] = np.nan

    # Datetimes 
    df["start_datetime"] = pd.to_datetime(df.get("date_debut"), errors="coerce")
    df["end_datetime"]   = pd.to_datetime(df.get("date_fin"),   errors="coerce")

    # Simple helper flags used by the newsletter 
    cost_lower = df["cout"].astype(str).str.lower()
    df["is_free"] = cost_lower.str.contains("gratuit", na=False)

    # One display line for venue 
    df["venue_full"] = [
        ", ".join([str(v) for v in [row.get("titre_adresse"), row.get("arrondissement")] if pd.notna(v) and str(v)])
        for _, row in df.iterrows()
    ]

    return df
