"""
=============================================================
  Retail Sales Forecasting & Inventory Optimization System
  Module: trend_decomposition.py
  Future Improvement #6:
    - Prophet-style trend + seasonality decomposition
    - Uses statsmodels STL (Seasonal-Trend-Loess) decomposition
      which is available without complex installation and gives
      the same interpretable components as Facebook Prophet:
        * Trend component
        * Seasonal component (weekly/yearly)
        * Residual component
    - Upgrade path: swap STL with Prophet in production
=============================================================

UPGRADE TO PROPHET:
  pip install prophet
  from prophet import Prophet
  m = Prophet()
  m.fit(df.rename(columns={"week_start":"ds","units_sold":"y"}))
  future = m.make_future_dataframe(periods=12, freq="W")
  forecast = m.predict(future)
  m.plot(forecast); m.plot_components(forecast)
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from sklearn.metrics           import mean_absolute_error

warnings.filterwarnings("ignore")

IMG_DIR    = os.path.join(os.path.dirname(__file__), "..", "images")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(IMG_DIR,    exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

BG    = "#1a1a2e"
FG    = "#e0e0e0"
PANEL = "#16213e"
ACCENT= "#e94560"


def build_weekly_series(features_df: pd.DataFrame,
                         store_id:    str = "S001",
                         product_id:  str = "P006") -> pd.Series:
    """
    Extract a single weekly time series for STL decomposition.
    Uses total units sold across all products to get one clean series.
    """
    df = features_df.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])

    ts = (
        df[(df["store_id"] == store_id) & (df["product_id"] == product_id)]
        .groupby("week_start")["units_sold"]
        .sum()
        .sort_index()
    )
    # Ensure weekly frequency, fill small gaps
    ts = ts.asfreq("W", fill_value=None).interpolate(method="linear")
    ts = ts.ffill().bfill()
    return ts


def run_stl_decomposition(ts: pd.Series, period: int = 52) -> dict:
    """
    Decompose time series into Trend, Seasonal, Residual using STL.
    period=52 captures yearly seasonality in weekly data.
    """
    stl  = STL(ts, period=period, robust=True)
    res  = stl.fit()

    components = pd.DataFrame({
        "date":      ts.index,
        "observed":  ts.values,
        "trend":     res.trend,
        "seasonal":  res.seasonal,
        "residual":  res.resid,
    })

    # Summary statistics
    trend_slope = np.polyfit(range(len(res.trend)), res.trend, 1)[0]
    seasonal_strength = (1 - np.var(res.resid) / np.var(res.seasonal + res.resid))
    trend_strength    = (1 - np.var(res.resid) / np.var(res.trend + res.resid))

    print(f"[+] STL Decomposition Results:")
    print(f"    Trend slope       : {trend_slope:+.2f} units/week")
    print(f"    Trend strength    : {max(0, trend_strength):.3f}  (>0.64 = strong)")
    print(f"    Seasonal strength : {max(0, seasonal_strength):.3f}  (>0.64 = strong)")
    print(f"    Residual std      : {res.resid.std():.2f}")

    return {
        "result":           res,
        "components":       components,
        "trend_slope":      trend_slope,
        "seasonal_strength": max(0, seasonal_strength),
        "trend_strength":    max(0, trend_strength),
    }


def stl_forecast(ts: pd.Series, result: dict,
                 horizon: int = 12) -> pd.Series:
    """
    Simple STL-based forecast:
      future_trend    = last_trend + slope × horizon (linear extrapolation)
      future_seasonal = repeat last full year of seasonal component
      forecast        = future_trend + future_seasonal
    """
    trend    = result["result"].trend
    seasonal = result["result"].seasonal
    slope    = result["trend_slope"]

    last_trend_val = trend.iloc[-1]
    future_index   = pd.date_range(ts.index[-1] + pd.Timedelta(weeks=1),
                                   periods=horizon, freq="W")

    # Extrapolate trend
    future_trend = pd.Series(
        [last_trend_val + slope * (i + 1) for i in range(horizon)],
        index=future_index
    )

    # Repeat seasonal pattern (last 52 weeks or less)
    season_period = min(52, len(seasonal))
    seasonal_cycle = seasonal.iloc[-season_period:].values
    future_seasonal = pd.Series(
        [seasonal_cycle[i % season_period] for i in range(horizon)],
        index=future_index
    )

    forecast = (future_trend + future_seasonal).clip(lower=0)
    return forecast


def compute_stl_for_all_products(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run STL decomposition for every product × store combination.
    Returns a summary DataFrame with trend and seasonality metrics.
    """
    df = features_df.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    summaries = []

    for (sid, sname, pid, pname, cat), grp in df.groupby(
            ["store_id", "store_name", "product_id", "product_name", "category"]):
        ts = (grp.groupby("week_start")["units_sold"]
                  .sum()
                  .sort_index()
                  .asfreq("W")
                  .interpolate(method="linear")
                  .ffill().bfill())

        if len(ts) < 104:    # need at least 2 years
            continue

        try:
            stl     = STL(ts, period=52, robust=True)
            res     = stl.fit()
            slope   = np.polyfit(range(len(res.trend)), res.trend, 1)[0]
            ss      = max(0, 1 - np.var(res.resid)
                             / (np.var(res.seasonal + res.resid) + 1e-6))
            ts_val  = max(0, 1 - np.var(res.resid)
                              / (np.var(res.trend + res.resid) + 1e-6))
            summaries.append({
                "store_id":          sid,
                "store_name":        sname,
                "product_id":        pid,
                "product_name":      pname,
                "category":          cat,
                "trend_slope":       round(slope, 4),
                "trend_strength":    round(ts_val, 4),
                "seasonal_strength": round(ss, 4),
                "residual_std":      round(res.resid.std(), 4),
                "growth_direction":  "Growing" if slope > 0 else "Declining",
            })
        except Exception:
            continue

    summary_df = pd.DataFrame(summaries)
    print(f"[✓] STL decomposed {len(summary_df)} product-store series")
    return summary_df


def plot_stl_decomposition(ts: pd.Series, result: dict,
                            forecast: pd.Series,
                            store_id: str, product_id: str) -> str:
    """Four-panel STL decomposition + forecast chart."""
    plt.style.use("dark_background")
    plt.rcParams["figure.facecolor"] = BG

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=False)
    comp = result["components"]

    # Observed
    axes[0].plot(comp["date"], comp["observed"], color="#00b4d8", linewidth=1.5)
    # Add forecast continuation
    fut_dates = forecast.index
    axes[0].plot(fut_dates, forecast.values, color=ACCENT,
                 linewidth=2, linestyle="--", label="12-week forecast")
    axes[0].axvline(ts.index[-1], color="gray", linewidth=1, linestyle=":")
    axes[0].set_title(f"STL Decomposition — Store {store_id} | Product {product_id}",
                      fontsize=13)
    axes[0].set_ylabel("Observed")
    axes[0].legend(fontsize=9)
    axes[0].set_facecolor(PANEL)
    axes[0].grid(linestyle="--", alpha=0.3)

    # Trend
    axes[1].plot(comp["date"], comp["trend"], color="#06d6a0", linewidth=2)
    axes[1].set_ylabel("Trend")
    axes[1].set_facecolor(PANEL)
    axes[1].grid(linestyle="--", alpha=0.3)

    # Seasonal
    axes[2].fill_between(comp["date"], comp["seasonal"],
                         alpha=0.6, color="#ffd166")
    axes[2].set_ylabel("Seasonal")
    axes[2].set_facecolor(PANEL)
    axes[2].grid(linestyle="--", alpha=0.3)

    # Residual
    axes[3].bar(comp["date"], comp["residual"],
                color=ACCENT, alpha=0.6, width=5)
    axes[3].axhline(0, color="white", linewidth=0.6)
    axes[3].set_ylabel("Residual")
    axes[3].set_xlabel("Date")
    axes[3].set_facecolor(PANEL)
    axes[3].grid(linestyle="--", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(IMG_DIR, "22_stl_decomposition.png")
    fig.savefig(path, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[✓] Saved -> {path}")
    return path


def plot_stl_summary(summary_df: pd.DataFrame) -> str:
    """Heatmap of trend strength × seasonal strength per product."""
    plt.style.use("dark_background")
    plt.rcParams["figure.facecolor"] = BG

    pivot = (summary_df.groupby(["product_name", "store_name"])
             ["seasonal_strength"].mean()
             .unstack("store_name")
             .fillna(0))

    import seaborn as sns
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", annot=True, fmt=".2f",
                linewidths=0.5, cbar_kws={"label": "Seasonal Strength"})
    ax.set_title("Seasonal Strength Heatmap (Product x Store) — STL",
                 fontsize=13, pad=12)
    ax.set_xlabel("Store")
    ax.set_ylabel("Product")
    fig.tight_layout()

    path = os.path.join(IMG_DIR, "23_stl_seasonal_strength.png")
    fig.savefig(path, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[✓] Saved -> {path}")
    return path


def run_trend_decomposition(features_df: pd.DataFrame) -> dict:
    print("\n" + "=" * 60)
    print("  Running Trend + Seasonality Decomposition (STL) ...")
    print("=" * 60)

    # Detailed single-product decomposition
    ts     = build_weekly_series(features_df, "S001", "P006")
    result = run_stl_decomposition(ts)
    fcast  = stl_forecast(ts, result, horizon=12)
    plot_stl_decomposition(ts, result, fcast, "S001", "P006")

    # Summary across all products
    summary_df = compute_stl_for_all_products(features_df)
    plot_stl_summary(summary_df)

    summary_df.to_csv(
        os.path.join(REPORT_DIR, "stl_decomposition_summary.csv"), index=False)
    print("[✓] STL decomposition complete")
    return {"ts": ts, "stl_result": result,
            "forecast": fcast, "summary": summary_df}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.preprocessing     import run_preprocessing_pipeline
    from src.feature_engineering import run_feature_engineering
    df   = run_preprocessing_pipeline()
    feat = run_feature_engineering(df)
    run_trend_decomposition(feat)
