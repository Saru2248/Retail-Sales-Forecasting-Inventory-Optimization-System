# -*- coding: utf-8 -*-
"""
=============================================================
  Retail Sales Forecasting & Inventory Optimization System
  Entry Point: main.py  (v2.0 — All Future Improvements Integrated)
  Purpose: Orchestrate the full pipeline including all 10
           future improvement modules.
=============================================================

USAGE:
  python main.py                         # full pipeline (core + extensions)
  python main.py --skip-generate        # use existing dataset
  python main.py --only-forecast        # skip to forecasting step
  python main.py --core-only            # run only the 7 core steps
  python main.py --extensions-only      # run only the 10 extension modules
  streamlit run app/streamlit_dashboard.py   # launch web dashboard
"""

import argparse
import sys
import time
import os

# Force UTF-8 output on Windows terminals
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                   errors="replace")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def banner():
    print("")
    print("=" * 64)
    print("  RETAIL SALES FORECASTING & INVENTORY OPTIMIZATION SYSTEM")
    print("         Industry-Grade ML Pipeline v2.0")
    print("  10 Future Improvements Fully Integrated")
    print("=" * 64)
    print("")


def step(number: int, title: str):
    print(f"\n" + "-" * 62)
    print(f"  STEP {number}: {title}")
    print("-" * 62)


# ══════════════════════════════════════════════════════════
# CORE PIPELINE (Steps 1-7)
# ══════════════════════════════════════════════════════════
def run_core_pipeline(skip_generate: bool = False,
                       only_forecast:  bool = False) -> dict:
    """Run the original 7-step core forecasting pipeline."""

    # ── STEP 1 – Generate Dataset ─────────────────────────
    if not skip_generate and not only_forecast:
        step(1, "GENERATING SYNTHETIC RETAIL DATASET")
        from data_generator import generate_dataset, save_dataset
        df_raw = generate_dataset()
        save_dataset(df_raw)
    else:
        step(1, "SKIPPING DATA GENERATION (flag set)")
        print("  -> Using existing dataset in data/")

    # ── STEP 2 – Preprocessing ────────────────────────────
    if not only_forecast:
        step(2, "PREPROCESSING & CLEANING")
        from preprocessing import run_preprocessing_pipeline
        df_clean = run_preprocessing_pipeline()
    else:
        step(2, "LOADING PREPROCESSED DATA")
        import pandas as pd
        df_clean = pd.read_csv(
            os.path.join("data", "retail_sales_clean.csv"),
            parse_dates=["date"]
        )
        print(f"  -> Loaded {len(df_clean):,} rows")

    # ── STEP 3 – EDA ──────────────────────────────────────
    if not only_forecast:
        step(3, "EXPLORATORY DATA ANALYSIS (8 charts)")
        from eda import run_eda
        run_eda(df_clean)

    # ── STEP 4 – Feature Engineering ─────────────────────
    step(4, "FEATURE ENGINEERING")
    from feature_engineering import run_feature_engineering
    df_features = run_feature_engineering(df_clean)

    # ── STEP 5 – Forecasting ──────────────────────────────
    step(5, "TRAINING FORECASTING MODELS")
    from forecasting import train_global_model, generate_forecasts
    result      = train_global_model(df_features)
    forecast_df = generate_forecasts(df_features, result, forecast_weeks=12)

    # ── STEP 6 – Inventory Optimization ──────────────────
    step(6, "INVENTORY OPTIMIZATION & REORDER ALERTS")
    from inventory_optimizer import run_inventory_optimization
    report, alerts = run_inventory_optimization(df_features, forecast_df)

    # ── STEP 7 – Final Visualizations ────────────────────
    step(7, "GENERATING EXECUTIVE DASHBOARD & REPORTS")
    from visualization import run_visualization
    run_visualization(df_features, forecast_df, report)

    return {
        "df_clean":    df_clean,
        "df_features": df_features,
        "forecast_df": forecast_df,
        "report":      report,
        "alerts":      alerts,
    }


# ══════════════════════════════════════════════════════════
# EXTENSION PIPELINE (Future Improvements 1-10)
# ══════════════════════════════════════════════════════════
def run_extension_pipeline(data: dict) -> None:
    """
    Run all 10 future improvement modules.
    Each module is wrapped in try/except so one failure
    doesn't stop the rest.
    """
    df_features = data["df_features"]
    forecast_df = data["forecast_df"]
    report      = data["report"]
    alerts      = data["alerts"]

    ext_step = 8   # continue numbering from core steps

    # ── EXT 1 & 10: Regional Clustering + Demand Segmentation ─
    step(ext_step, "REGIONAL CLUSTERING & PRODUCT SEGMENTATION (#1, #10)")
    ext_step += 1
    try:
        from regional_clustering import run_regional_clustering
        run_regional_clustering(df_features)
    except Exception as e:
        print(f"  [!] Skipped: {e}")

    # ── EXT 2: Price Elasticity ───────────────────────────
    step(ext_step, "PRICE ELASTICITY MODELING (#2)")
    ext_step += 1
    try:
        from price_elasticity import run_price_elasticity
        run_price_elasticity(df_features)
    except Exception as e:
        print(f"  [!] Skipped: {e}")

    # ── EXT 3: Weather-Based Demand Adjustment ────────────
    step(ext_step, "WEATHER-BASED DEMAND ADJUSTMENT (#3)")
    ext_step += 1
    try:
        from weather_demand import run_weather_demand
        run_weather_demand(df_features)
    except Exception as e:
        print(f"  [!] Skipped: {e}")

    # ── EXT 5: Email Alert System ─────────────────────────
    step(ext_step, "AUTOMATED ALERT SYSTEM — DRY RUN (#5)")
    ext_step += 1
    try:
        from alert_system import run_alert_system
        run_alert_system(inventory_df=report, alerts_df=alerts)
    except Exception as e:
        print(f"  [!] Skipped: {e}")

    # ── EXT 6: STL Trend Decomposition ───────────────────
    step(ext_step, "TREND + SEASONALITY DECOMPOSITION — STL (#6)")
    ext_step += 1
    try:
        from trend_decomposition import run_trend_decomposition
        run_trend_decomposition(df_features)
    except Exception as e:
        print(f"  [!] Skipped: {e}")

    # ── EXT 7: ERP Connector ─────────────────────────────
    step(ext_step, "ERP SYSTEM INTEGRATION SIMULATION (#7)")
    ext_step += 1
    try:
        from erp_connector import simulate_erp_sync
        simulate_erp_sync(alerts)
    except Exception as e:
        print(f"  [!] Skipped: {e}")

    # ── EXT 8: Promotional Modeling + A/B Test ────────────
    step(ext_step, "PROMOTIONAL IMPACT MODELING & A/B TEST (#8)")
    ext_step += 1
    try:
        from promotional_modeling import run_promotional_modeling
        run_promotional_modeling(df_features)
    except Exception as e:
        print(f"  [!] Skipped: {e}")

    # ── EXT 9: Anomaly Detection ───────────────────────────
    step(ext_step, "ANOMALY DETECTION — ISOLATION FOREST + Z-SCORE (#9)")
    ext_step += 1
    try:
        from anomaly_detection import run_anomaly_detection
        run_anomaly_detection(df_features)
    except Exception as e:
        print(f"  [!] Skipped: {e}")


# ══════════════════════════════════════════════════════════
# SUMMARY PRINTER
# ══════════════════════════════════════════════════════════
def print_summary(t_elapsed: float) -> None:
    print("")
    print("=" * 64)
    print(f"  PIPELINE COMPLETE!  ({t_elapsed:.0f}s)")
    print("=" * 64)
    print("")
    print("  CORE OUTPUTS:")
    print("    data/retail_sales_data.csv          -> Raw dataset")
    print("    data/retail_sales_clean.csv         -> Cleaned data")
    print("    data/features.csv                   -> Feature matrix")
    print("    outputs/forecasts.csv               -> 12-week forecasts")
    print("    outputs/inventory_report.csv        -> Inventory status")
    print("    outputs/reorder_alerts.csv          -> Reorder alerts")
    print("    models/best_forecast_model.pkl      -> Saved ML model")
    print("")
    print("  EXTENSION OUTPUTS:")
    print("    images/16_store_regional_clustering.png")
    print("    images/17_product_demand_segments.png")
    print("    images/18_price_elasticity.png")
    print("    images/19_weather_demand.png")
    print("    images/20_anomaly_detection.png")
    print("    images/21_promotional_modeling.png")
    print("    images/22_stl_decomposition.png")
    print("    images/23_stl_seasonal_strength.png")
    print("    reports/store_clusters.csv")
    print("    reports/price_elasticity.csv")
    print("    reports/promo_uplift.csv")
    print("    reports/ab_test_results.csv")
    print("    reports/anomaly_report.csv")
    print("    reports/erp_stock_snapshot.csv")
    print("    reports/alert_logs/alert_<timestamp>.txt")
    print("")
    print("  STREAMLIT DASHBOARD:")
    print("    streamlit run app/streamlit_dashboard.py")
    print("")
    print("=" * 64)
    print("")


# ══════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════
def run_pipeline(skip_generate:     bool = False,
                  only_forecast:     bool = False,
                  core_only:         bool = False,
                  extensions_only:   bool = False):
    banner()
    t_start = time.time()

    if extensions_only:
        # Load pre-existing data and run only extensions
        import pandas as pd
        step(0, "LOADING EXISTING PIPELINE OUTPUTS")
        df_clean    = pd.read_csv(
            os.path.join("data", "retail_sales_clean.csv"), parse_dates=["date"])
        df_features = pd.read_csv(os.path.join("data", "features.csv"))
        forecast_df = pd.read_csv(os.path.join("outputs", "forecasts.csv"))
        report      = pd.read_csv(os.path.join("outputs", "inventory_report.csv"))
        alerts      = pd.read_csv(os.path.join("outputs", "reorder_alerts.csv"))
        df_features["week_start"] = pd.to_datetime(df_features["week_start"])
        forecast_df["week_start"] = pd.to_datetime(forecast_df["week_start"])
        data = {"df_clean": df_clean, "df_features": df_features,
                "forecast_df": forecast_df, "report": report, "alerts": alerts}
        run_extension_pipeline(data)
    else:
        data = run_core_pipeline(skip_generate, only_forecast)
        if not core_only:
            run_extension_pipeline(data)

    print_summary(time.time() - t_start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retail Sales Forecasting & Inventory Optimization System v2.0"
    )
    parser.add_argument("--skip-generate",  action="store_true",
                        help="Skip dataset generation")
    parser.add_argument("--only-forecast",  action="store_true",
                        help="Jump to forecasting (needs preprocessed data)")
    parser.add_argument("--core-only",      action="store_true",
                        help="Run only Steps 1-7 (no extension modules)")
    parser.add_argument("--extensions-only", action="store_true",
                        help="Run only extension modules (needs existing outputs)")
    args = parser.parse_args()

    run_pipeline(
        skip_generate   = args.skip_generate,
        only_forecast   = args.only_forecast,
        core_only       = args.core_only,
        extensions_only = args.extensions_only,
    )
