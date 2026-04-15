"""
=============================================================
  Retail Sales Forecasting & Inventory Optimization System
  Module: feature_engineering.py
  Purpose: Transform cleaned weekly data into ML-ready feature
           matrix — lag features, rolling statistics, calendar
           encodings, and interaction terms.
=============================================================
"""

import pandas as pd
import numpy as np
import os

CLEAN_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "retail_sales_clean.csv")
FEATURE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "features.csv")


# ── 1. Aggregate to weekly level before feature engineering ─
def get_weekly_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group the daily clean data to weekly product × store level.
    Each row = one product × one store × one week.
    """
    df["week_start"] = df["date"] - pd.to_timedelta(df["day_of_week"], unit="D")

    weekly = (
        df.groupby(["store_id", "store_name", "product_id", "product_name",
                    "category", "week_start"])
        .agg(
            units_sold       = ("units_sold",       "sum"),
            revenue          = ("revenue",          "sum"),
            avg_price        = ("unit_price",        "mean"),
            promo_days       = ("is_promotion",      "sum"),
            stockout_days    = ("stockout_flag",     "sum"),
            lead_time_days   = ("lead_time_days",    "first"),
            holding_cost_pct = ("holding_cost_pct", "first"),
            base_price       = ("base_price",        "first"),
            is_festival_season=("is_festival_season","max"),
        )
        .reset_index()
    )

    weekly["week_start"]    = pd.to_datetime(weekly["week_start"])
    weekly["year"]          = weekly["week_start"].dt.year
    weekly["month"]         = weekly["week_start"].dt.month
    weekly["week_num"]      = weekly["week_start"].dt.isocalendar().week.astype(int)
    weekly["quarter"]       = weekly["week_start"].dt.quarter
    weekly["is_promo_week"] = (weekly["promo_days"] > 0).astype(int)
    weekly = weekly.sort_values(["store_id", "product_id", "week_start"]).reset_index(drop=True)
    return weekly


# ── 2. Lag Features ───────────────────────────────────────
def add_lag_features(df: pd.DataFrame,
                     lags: list = [1, 2, 3, 4, 8, 12]) -> pd.DataFrame:
    """
    Add lagged units_sold values.
    Lag-1  → last week's sales (strongest predictor)
    Lag-4  → 4 weeks ago (monthly comparison)
    Lag-52 → same week last year (year-over-year)
    """
    grp = df.groupby(["store_id", "product_id"])["units_sold"]
    for lag in lags:
        df[f"lag_{lag}w"] = grp.shift(lag)
    print(f"[✓] Lag features added : {lags}")
    return df


# ── 3. Rolling Window Statistics ──────────────────────────
def add_rolling_features(df: pd.DataFrame,
                         windows: list = [2, 4, 8, 12]) -> pd.DataFrame:
    """
    For each rolling window size:
      - rolling mean (trend / level)
      - rolling std  (volatility)
      - rolling max  (peak demand)
    """
    grp = df.groupby(["store_id", "product_id"])["units_sold"]
    for w in windows:
        df[f"roll_mean_{w}w"]  = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        df[f"roll_std_{w}w"]   = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).std().fillna(0))
        df[f"roll_max_{w}w"]   = grp.transform(lambda x: x.shift(1).rolling(w, min_periods=1).max())
    print(f"[✓] Rolling features  : windows = {windows}")
    return df


# ── 4. Calendar Cyclic Encodings ──────────────────────────
def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode month and week_num as sin/cos pairs so that the model
    understands December and January are close together.
    """
    df["month_sin"]    = np.sin(2 * np.pi * df["month"]    / 12)
    df["month_cos"]    = np.cos(2 * np.pi * df["month"]    / 12)
    df["week_num_sin"] = np.sin(2 * np.pi * df["week_num"] / 52)
    df["week_num_cos"] = np.cos(2 * np.pi * df["week_num"] / 52)
    print("[✓] Cyclic calendar encodings added")
    return df


# ── 5. Category One-Hot Encoding ──────────────────────────
def add_category_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the 'category' column (drop_first to avoid multicollinearity)."""
    dummies = pd.get_dummies(df["category"], prefix="cat", drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    print(f"[✓] Category OHE      : {list(dummies.columns)}")
    return df


# ── 6. Interaction Features ───────────────────────────────
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create meaningful interaction features."""
    df["promo_x_festival"] = df["is_promo_week"] * df["is_festival_season"]
    df["price_x_promo"]    = df["avg_price"] * df["is_promo_week"]

    # Demand stability score (lower = more volatile)
    df["demand_cv_4w"] = df["roll_std_4w"] / (df["roll_mean_4w"] + 1e-6)
    print("[✓] Interaction features added")
    return df


# ── 7. Drop rows that still have NaN lags ─────────────────
def drop_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    After computing lag_52w, the first 52 weeks of each series
    will have NaN.  Drop these so the model trains cleanly.
    """
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"[✓] Dropped NaN rows  : {before - len(df)} rows removed (lag warm-up)")
    return df


# ── Master pipeline ───────────────────────────────────────
def run_feature_engineering(df: pd.DataFrame,
                             save_path: str = FEATURE_PATH) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Input  : cleaned daily DataFrame
    Output : weekly feature-rich DataFrame ready for model training
    """
    print("\n" + "=" * 60)
    print("  Running Feature Engineering …")
    print("=" * 60)

    weekly = get_weekly_series(df)
    weekly = add_lag_features(weekly)
    weekly = add_rolling_features(weekly)
    weekly = add_cyclic_features(weekly)
    weekly = add_category_encoding(weekly)
    weekly = add_interaction_features(weekly)
    weekly = drop_nan_rows(weekly)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    weekly.to_csv(save_path, index=False)
    print(f"\n[✓] Feature matrix saved → {save_path}")
    print(f"    Shape  : {weekly.shape}")
    print(f"    Columns: {list(weekly.columns)}")
    return weekly


if __name__ == "__main__":
    from preprocessing import run_preprocessing_pipeline
    df     = run_preprocessing_pipeline()
    weekly = run_feature_engineering(df)
    print("\nSample feature row:")
    print(weekly.iloc[0])
