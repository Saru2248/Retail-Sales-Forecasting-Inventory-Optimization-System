"""
=============================================================
  Retail Sales Forecasting & Inventory Optimization System
  Module: weather_demand.py
  Future Improvement #3:
    - Weather-based demand adjustment for seasonal products
    - Synthetic weather generation (replaces OpenWeatherMap
      API when no API key is available)
    - Shows how temperature/rainfall shifts demand
=============================================================

NOTE: In production, replace generate_synthetic_weather()
with a call to the OpenWeatherMap API:
  GET https://api.openweathermap.org/data/2.5/history/city
  API key: stored in environment variable OPENWEATHER_API_KEY
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics       import r2_score

IMG_DIR    = os.path.join(os.path.dirname(__file__), "..", "images")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(IMG_DIR,    exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

BG    = "#1a1a2e"
FG    = "#e0e0e0"
PANEL = "#16213e"
ACCENT= "#e94560"

# Category weather sensitivity
# weather_effect: positive = demand rises with temperature
CATEGORY_WEATHER_SENSITIVITY = {
    "Beverages":    +0.8,    # cold drinks up when hot
    "Dairy":        -0.3,    # milk sales dip slightly in summer
    "Snacks":       +0.2,    # mild positive
    "Staples":       0.0,    # weather-neutral
    "Personal Care": +0.3,   # sunscreen, deodorant up in summer
    "Ready Meals":  -0.4,    # people eat out more in summer
    "Household":     0.0,    # neutral
}

CITY_CLIMATE = {
    "Mumbai Central":   {"base_temp": 28, "temp_amp": 5,  "rain_months": [6,7,8,9]},
    "Delhi North":      {"base_temp": 25, "temp_amp": 15, "rain_months": [7,8]},
    "Bengaluru South":  {"base_temp": 22, "temp_amp": 4,  "rain_months": [5,6,9,10]},
    "Pune East":        {"base_temp": 26, "temp_amp": 7,  "rain_months": [6,7,8,9]},
    "Chennai West":     {"base_temp": 29, "temp_amp": 4,  "rain_months": [10,11,12]},
}


def generate_synthetic_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate realistic synthetic weekly weather (temperature + rainfall)
    for each city based on Indian climate patterns.
    Returns a weekly city × weather DataFrame.
    """
    np.random.seed(99)
    records = []
    df["week_start"] = pd.to_datetime(df["week_start"])

    weeks  = df["week_start"].unique()
    cities = df["store_name"].unique()

    for city in cities:
        climate = CITY_CLIMATE.get(city, {"base_temp": 27, "temp_amp": 8,
                                           "rain_months": [7, 8]})
        for week in weeks:
            month = pd.Timestamp(week).month
            day_of_year = pd.Timestamp(week).day_of_year

            # Sinusoidal temperature model
            temp = (climate["base_temp"]
                    + climate["temp_amp"] * np.sin(2 * np.pi * (day_of_year - 80) / 365)
                    + np.random.normal(0, 2))

            # Rainfall: higher in monsoon months
            is_rain_month = month in climate["rain_months"]
            rainfall_mm   = (np.random.exponential(30) if is_rain_month
                             else np.random.exponential(3))
            is_rainy_week = rainfall_mm > 20

            records.append({
                "week_start":    week,
                "store_name":    city,
                "temperature_c": round(temp, 1),
                "rainfall_mm":   round(rainfall_mm, 1),
                "is_rainy_week": int(is_rainy_week),
            })

    weather_df = pd.DataFrame(records)
    weather_df["week_start"] = pd.to_datetime(weather_df["week_start"])
    return weather_df


def merge_weather_with_demand(features_df: pd.DataFrame,
                               weather_df:  pd.DataFrame) -> pd.DataFrame:
    """Merge weather features into the weekly feature matrix."""
    features_df["week_start"] = pd.to_datetime(features_df["week_start"])
    merged = features_df.merge(
        weather_df, on=["week_start", "store_name"], how="left"
    )
    merged["temperature_c"] = merged["temperature_c"].fillna(27)
    merged["rainfall_mm"]   = merged["rainfall_mm"].fillna(0)
    merged["is_rainy_week"] = merged["is_rainy_week"].fillna(0).astype(int)

    # Weather-adjusted demand multiplier per category
    merged["weather_multiplier"] = merged.apply(
        lambda r: 1.0 + CATEGORY_WEATHER_SENSITIVITY.get(r["category"], 0)
                       * (r["temperature_c"] - 27) / 10,
        axis=1
    )
    merged["weather_adjusted_units"] = (
        merged["units_sold"] * merged["weather_multiplier"]
    ).round(1)

    print(f"[✓] Weather features merged: {merged.shape}")
    return merged


def quantify_weather_impact(merged: pd.DataFrame) -> pd.DataFrame:
    """Regression: how much does temperature explain demand per category?"""
    results = []
    for cat, grp in merged.groupby("category"):
        if len(grp) < 30:
            continue
        X  = grp[["temperature_c", "is_rainy_week"]].fillna(0).values
        y  = grp["units_sold"].values
        lr = LinearRegression()
        lr.fit(X, y)
        r2 = r2_score(y, lr.predict(X))
        results.append({
            "category":         cat,
            "temp_coefficient": round(lr.coef_[0], 4),
            "rain_coefficient": round(lr.coef_[1], 4),
            "r2_weather":       round(r2, 4),
            "sensitivity":      CATEGORY_WEATHER_SENSITIVITY.get(cat, 0),
        })
    impact_df = pd.DataFrame(results).sort_values("temp_coefficient", ascending=False)
    print("\n[+] Weather Impact on Demand by Category:")
    print(impact_df.to_string(index=False))
    return impact_df


def plot_weather_analysis(merged: pd.DataFrame,
                           impact_df: pd.DataFrame) -> str:
    """Three-panel weather analysis plot."""
    plt.style.use("dark_background")
    plt.rcParams["figure.facecolor"] = BG

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 1. Temperature vs Beverage demand scatter ─────────
    bev = merged[merged["category"] == "Beverages"]
    axes[0].scatter(bev["temperature_c"], bev["units_sold"],
                    alpha=0.3, s=8, color="#00b4d8")
    m_lr = LinearRegression()
    X_t  = bev["temperature_c"].values.reshape(-1, 1)
    m_lr.fit(X_t, bev["units_sold"])
    t_range = np.linspace(bev["temperature_c"].min(),
                          bev["temperature_c"].max(), 100)
    axes[0].plot(t_range, m_lr.predict(t_range.reshape(-1, 1)),
                 color=ACCENT, linewidth=2, label="Trend")
    axes[0].set_title("Temperature vs Beverages Demand", fontsize=12)
    axes[0].set_xlabel("Temperature (°C)")
    axes[0].set_ylabel("Units Sold")
    axes[0].legend()
    axes[0].set_facecolor(PANEL)
    axes[0].grid(linestyle="--", alpha=0.3)

    # ── 2. Temperature coefficient bar chart ──────────────
    colors = ["#e94560" if c > 0 else "#06d6a0"
              for c in impact_df["temp_coefficient"]]
    axes[1].barh(impact_df["category"], impact_df["temp_coefficient"],
                 color=colors, edgecolor="none")
    axes[1].axvline(0, color="white", linewidth=0.8, linestyle="--")
    axes[1].set_title("Temperature Effect per Category", fontsize=12)
    axes[1].set_xlabel("Temperature Coefficient (units/°C)")
    axes[1].set_facecolor(PANEL)
    axes[1].grid(axis="x", linestyle="--", alpha=0.3)

    # ── 3. Monthly average temperature by city ───────────
    merged["month"] = pd.to_datetime(merged["week_start"]).dt.month
    city_temp = merged.groupby(["store_name", "month"])["temperature_c"].mean().reset_index()
    for city in city_temp["store_name"].unique():
        sub = city_temp[city_temp["store_name"] == city].sort_values("month")
        axes[2].plot(sub["month"], sub["temperature_c"], marker="o",
                     markersize=4, linewidth=1.5, label=city[:9])
    axes[2].set_title("Monthly Avg Temperature by City", fontsize=12)
    axes[2].set_xlabel("Month")
    axes[2].set_ylabel("Temperature (°C)")
    axes[2].legend(fontsize=7, framealpha=0.2)
    axes[2].set_xticks(range(1, 13))
    axes[2].set_facecolor(PANEL)
    axes[2].grid(linestyle="--", alpha=0.3)

    fig.suptitle("Weather-Based Demand Adjustment Analysis",
                 fontsize=14, color=FG, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(IMG_DIR, "19_weather_demand.png")
    fig.savefig(path, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[✓] Saved -> {path}")
    return path


def run_weather_demand(features_df: pd.DataFrame) -> dict:
    print("\n" + "=" * 60)
    print("  Running Weather-Based Demand Adjustment ...")
    print("=" * 60)

    weather_df = generate_synthetic_weather(features_df)
    weather_df.to_csv(
        os.path.join(REPORT_DIR, "weather_data.csv"), index=False)

    merged    = merge_weather_with_demand(features_df, weather_df)
    impact_df = quantify_weather_impact(merged)
    plot_weather_analysis(merged, impact_df)

    impact_df.to_csv(
        os.path.join(REPORT_DIR, "weather_impact.csv"), index=False)
    merged.to_csv(
        os.path.join(REPORT_DIR, "weather_adjusted_demand.csv"), index=False)

    print("[✓] Weather demand adjustment complete")
    return {"weather_df": weather_df, "merged": merged, "impact": impact_df}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.preprocessing     import run_preprocessing_pipeline
    from src.feature_engineering import run_feature_engineering
    df   = run_preprocessing_pipeline()
    feat = run_feature_engineering(df)
    run_weather_demand(feat)
