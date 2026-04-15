"""
=============================================================
  Retail Sales Forecasting & Inventory Optimization System
  Module: preprocessing.py
  Purpose: Load, clean, validate, and prepare the raw retail
           sales CSV for downstream analysis and modelling.
=============================================================
"""

import pandas as pd
import numpy as np
import os

RAW_PATH    = os.path.join(os.path.dirname(__file__), "..", "data", "retail_sales_data.csv")
CLEAN_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "retail_sales_clean.csv")


# ── 1. Load ───────────────────────────────────────────────
def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    """Load the raw CSV dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Run `python src/data_generator.py` first to create it."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"[✓] Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ── 2. Inspect ────────────────────────────────────────────
def inspect_data(df: pd.DataFrame) -> None:
    """Print a structured data quality report."""
    print("\n" + "=" * 60)
    print("  DATA QUALITY REPORT")
    print("=" * 60)
    print(f"\nShape          : {df.shape}")
    print(f"Date range     : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Stores         : {df['store_id'].nunique()} unique")
    print(f"Products       : {df['product_id'].nunique()} unique")
    print(f"Categories     : {df['category'].nunique()} unique\n")

    print("Missing values per column:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  → None found ✓")
    else:
        print(missing.to_string())

    print("\nDuplicate rows   :", df.duplicated().sum())
    print("\nBasic statistics (numerical):")
    print(df[["units_sold", "unit_price", "revenue"]].describe().round(2).to_string())
    print("=" * 60)


# ── 3. Clean ──────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cleaning steps:
      - Drop exact duplicate rows
      - Fix negative / impossible values
      - Fill missing numerical columns with median
      - Ensure correct dtypes
    """
    initial_rows = len(df)

    # Drop duplicates
    df = df.drop_duplicates()
    print(f"[✓] Removed duplicates  : {initial_rows - len(df)} rows dropped")

    # Clamp negative values to 0
    for col in ["units_sold", "revenue", "unit_price"]:
        neg = (df[col] < 0).sum()
        if neg > 0:
            df[col] = df[col].clip(lower=0)
            print(f"[!] Clamped {neg} negative values in '{col}' to 0")

    # Fill missing numeric columns with column median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"[!] Filled {n_missing} missing values in '{col}' with median")

    # Ensure correct dtypes
    df["date"]          = pd.to_datetime(df["date"])
    df["is_promotion"]  = df["is_promotion"].astype(int)
    df["is_weekend"]    = df["is_weekend"].astype(int)
    df["stockout_flag"] = df["stockout_flag"].astype(int)

    # Recompute revenue where unit_price or units_sold has been corrected
    df["revenue"] = (df["units_sold"] * df["unit_price"]).round(2)

    print(f"[✓] Cleaning complete   : {len(df):,} rows remain")
    return df


# ── 4. Feature Engineering (basic temporal) ───────────────
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add / verify temporal feature columns needed for modelling.
    Advanced feature engineering is in feature_engineering.py.
    """
    df["year"]        = df["date"].dt.year
    df["month"]       = df["date"].dt.month
    df["day"]         = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_number"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"]     = df["date"].dt.quarter
    df["day_of_year"] = df["date"].dt.dayofyear

    # Month name for easier reading
    df["month_name"]  = df["date"].dt.strftime("%b")

    # Is month-start / month-end (useful for bulk buying patterns)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"]   = df["date"].dt.is_month_end.astype(int)

    # Festival flag (Oct / Nov / Dec = True)
    df["is_festival_season"] = df["month"].isin([10, 11, 12]).astype(int)

    print("[✓] Temporal features added")
    return df


# ── 5. Aggregate – product × store × week ─────────────────
def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily rows to weekly level per product/store.
    This smoother granularity is used by the forecasting models.
    """
    df["week_start"] = df["date"] - pd.to_timedelta(df["day_of_week"], unit="D")

    weekly = (
        df.groupby(["store_id", "store_name", "product_id", "product_name",
                    "category", "week_start"])
        .agg(
            units_sold      = ("units_sold",      "sum"),
            revenue         = ("revenue",         "sum"),
            avg_price       = ("unit_price",       "mean"),
            promo_days      = ("is_promotion",     "sum"),
            stockout_days   = ("stockout_flag",    "sum"),
            lead_time_days  = ("lead_time_days",   "first"),
            holding_cost_pct= ("holding_cost_pct","first"),
            base_price      = ("base_price",       "first"),
        )
        .reset_index()
    )

    weekly["week_start"] = pd.to_datetime(weekly["week_start"])
    weekly["year"]       = weekly["week_start"].dt.year
    weekly["month"]      = weekly["week_start"].dt.month
    weekly["week_num"]   = weekly["week_start"].dt.isocalendar().week.astype(int)
    weekly["quarter"]    = weekly["week_start"].dt.quarter
    weekly["is_promo_week"] = (weekly["promo_days"] > 0).astype(int)

    print(f"[✓] Weekly aggregation  : {len(weekly):,} rows")
    return weekly


# ── 6. Save ───────────────────────────────────────────────
def save_clean(df: pd.DataFrame, path: str = CLEAN_PATH) -> None:
    """Save the cleaned dataset."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[✓] Clean data saved   → {path}")


# ── Pipeline entry point ──────────────────────────────────
def run_preprocessing_pipeline(raw_path: str = RAW_PATH,
                                clean_path: str = CLEAN_PATH) -> pd.DataFrame:
    """Execute the full preprocessing pipeline and return the clean daily DF."""
    df = load_data(raw_path)
    inspect_data(df)
    df = clean_data(df)
    df = add_time_features(df)
    save_clean(df, clean_path)
    return df


if __name__ == "__main__":
    df = run_preprocessing_pipeline()
    print("\nPreview of cleaned data:")
    print(df.head(3).to_string(index=False))
