"""
=============================================================
  Retail Sales Forecasting & Inventory Optimization System
  Module: forecasting.py
  Purpose: Train multiple forecasting models (Random Forest,
           XGBoost, Linear Regression), evaluate them, select
           the best, and produce 12-week ahead forecasts.
=============================================================
"""

import os
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LinearRegression
from sklearn.ensemble      import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics        import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing  import StandardScaler

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[!] XGBoost not installed. XGBRegressor will be skipped.")

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────
FEATURE_PATH  = os.path.join(os.path.dirname(__file__), "..", "data",    "features.csv")
FORECAST_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "forecasts.csv")
MODEL_DIR     = os.path.join(os.path.dirname(__file__), "..", "models")
IMG_DIR       = os.path.join(os.path.dirname(__file__), "..", "images")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMG_DIR,   exist_ok=True)

# ── Feature columns used for training ────────────────────
LAG_COLS     = [f"lag_{w}w" for w in [1, 2, 3, 4, 8, 12]]
ROLL_COLS    = [f"roll_{s}_{w}w" for w in [2, 4, 8, 12] for s in ["mean", "std", "max"]]
CYCLIC_COLS  = ["month_sin", "month_cos", "week_num_sin", "week_num_cos"]
CALENDAR     = ["year", "month", "quarter", "week_num",
                "is_promo_week", "is_festival_season",
                "promo_days", "stockout_days"]
INTERACTION  = ["promo_x_festival", "price_x_promo", "demand_cv_4w", "avg_price"]

TARGET = "units_sold"


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return only feature columns that exist in the DataFrame."""
    candidate = LAG_COLS + ROLL_COLS + CYCLIC_COLS + CALENDAR + INTERACTION
    # also add any one-hot category cols
    cat_cols  = [c for c in df.columns if c.startswith("cat_")]
    candidate = candidate + cat_cols
    return [c for c in candidate if c in df.columns]


# ── Evaluation helper ────────────────────────────────────
def evaluate(y_true, y_pred, model_name: str) -> dict:
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    r2    = r2_score(y_true, y_pred)
    mape  = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    print(f"  [{model_name:30s}]  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}  MAPE={mape:.1f}%")
    return {"model": model_name, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


# ── Train / evaluate on a single product-store series ────
def train_and_evaluate(df: pd.DataFrame) -> dict:
    """
    Train three models using TimeSeriesSplit cross-validation.
    Returns metrics dict and the trained models.
    """
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y = df[TARGET].values

    # --- Time-Series Split (no data leakage) ---
    tscv       = TimeSeriesSplit(n_splits=5)
    models_cfg = {
        "LinearRegression": LinearRegression(),
        "RandomForest":     RandomForestRegressor(n_estimators=150, max_depth=8,
                                                   n_jobs=-1, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=150, max_depth=5,
                                                        learning_rate=0.08, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models_cfg["XGBoost"] = XGBRegressor(n_estimators=150, max_depth=5,
                                              learning_rate=0.08, verbosity=0,
                                              random_state=42)

    best_rmse   = float("inf")
    best_name   = None
    best_model  = None
    all_metrics = []

    scaler = StandardScaler()

    for name, model in models_cfg.items():
        val_preds = np.zeros(len(y))
        val_true  = np.zeros(len(y))
        count     = 0

        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # Scale for linear model; tree models don't need it but it's harmless
            X_tr_s  = scaler.fit_transform(X_tr)  if "Linear" in name else X_tr
            X_val_s = scaler.transform(X_val)      if "Linear" in name else X_val

            model.fit(X_tr_s, y_tr)
            preds = model.predict(X_val_s).clip(min=0)

            val_preds[val_idx] = preds
            val_true[val_idx]  = y_val
            count += len(val_idx)

        metrics = evaluate(val_true[-count:], val_preds[-count:].round(), name)
        all_metrics.append(metrics)

        if metrics["RMSE"] < best_rmse:
            best_rmse  = metrics["RMSE"]
            best_name  = name
            best_model = model

    print(f"\n  ★  Best model: {best_name}  (RMSE = {best_rmse:.2f})\n")
    return {"best_name": best_name, "best_model": best_model,
            "metrics": all_metrics, "feature_cols": feature_cols,
            "scaler": scaler}


# ── Aggregate training across all products (global model) ─
def train_global_model(df: pd.DataFrame) -> dict:
    """
    Train one global Random Forest across ALL product × store series.
    This is the primary model saved and used for batch forecasting.
    """
    print("=" * 60)
    print("  Training Global Forecasting Models …")
    print("=" * 60)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y = df[TARGET].values

    # 80/20 time-based split
    split = int(len(df) * 0.80)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=10,
                                               min_samples_leaf=5,
                                               n_jobs=-1, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, max_depth=6,
                                                       learning_rate=0.07,
                                                       subsample=0.85, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(n_estimators=200, max_depth=6,
                                         learning_rate=0.07, subsample=0.85,
                                         colsample_bytree=0.8, verbosity=0,
                                         random_state=42)

    metrics_list = []
    best_model   = None
    best_rmse    = float("inf")
    best_name    = None

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    for name, model in models.items():
        X_tr_use = X_train_s if "Linear" in name else X_train
        X_te_use = X_test_s  if "Linear" in name else X_test
        model.fit(X_tr_use, y_train)
        preds   = model.predict(X_te_use).clip(min=0).round()
        metrics = evaluate(y_test, preds, name)
        metrics_list.append(metrics)
        if metrics["RMSE"] < best_rmse:
            best_rmse  = metrics["RMSE"]
            best_name  = name
            best_model = model

    print(f"\n  ★  Best global model: {best_name}  (RMSE = {best_rmse:.2f})\n")

    # Save best model
    model_path = os.path.join(MODEL_DIR, "best_forecast_model.pkl")
    joblib.dump({"model": best_model, "scaler": scaler,
                 "feature_cols": feature_cols, "model_name": best_name}, model_path)
    print(f"[✓] Model saved → {model_path}")

    # Save metrics CSV
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(os.path.join(MODEL_DIR, "model_metrics.csv"), index=False)

    # Feature importance plot (RF or GBM)
    if hasattr(best_model, "feature_importances_"):
        _plot_feature_importance(best_model, feature_cols, best_name)

    # Actual vs Predicted plot
    best_use = X_test_s if "Linear" in best_name else X_test
    best_preds = best_model.predict(best_use).clip(min=0).round()
    _plot_actual_vs_predicted(y_test, best_preds, best_name)

    return {
        "best_name":     best_name,
        "best_model":    best_model,
        "scaler":        scaler,
        "feature_cols":  feature_cols,
        "X_test":        X_test,
        "y_test":        y_test,
        "test_preds":    best_preds,
        "metrics":       metrics_list,
        "df_test":       df.iloc[split:].copy().assign(predicted=best_preds),
    }


# ──────────────────────────────────────────────────────────
# Generate 12-week ahead rolling forecasts
# ──────────────────────────────────────────────────────────
def generate_forecasts(df: pd.DataFrame, result: dict,
                        forecast_weeks: int = 12) -> pd.DataFrame:
    """
    For each product × store, generate rolling 12-week forecasts
    by iteratively appending predictions as new lag values.
    """
    print(f"\nGenerating {forecast_weeks}-week ahead forecasts …")

    model        = result["best_model"]
    scaler       = result["scaler"]
    feature_cols = result["feature_cols"]
    best_name    = result["best_name"]

    all_forecasts = []

    for (store_id, product_id), grp in df.groupby(["store_id", "product_id"]):
        grp   = grp.sort_values("week_start").copy()
        last  = grp["week_start"].max()

        # We'll append synthetic future rows iteratively
        future_rows = []
        temp        = grp.copy()

        for horizon in range(1, forecast_weeks + 1):
            future_date = last + pd.Timedelta(weeks=horizon)

            # Build the feature row from the last known state
            last_row  = temp.iloc[-1]
            next_row  = {
                "week_start":          future_date,
                "store_id":            store_id,
                "store_name":          last_row["store_name"],
                "product_id":          product_id,
                "product_name":        last_row["product_name"],
                "category":            last_row["category"],
                "year":                future_date.year,
                "month":               future_date.month,
                "quarter":             (future_date.month - 1) // 3 + 1,
                "week_num":            future_date.isocalendar()[1],
                "avg_price":           last_row["avg_price"],
                "lead_time_days":      last_row["lead_time_days"],
                "holding_cost_pct":    last_row["holding_cost_pct"],
                "base_price":          last_row["base_price"],
                "is_promo_week":       0,       # conservative — no promo assumed
                "promo_days":          0,
                "stockout_days":       0,
                "is_festival_season":  1 if future_date.month in [10, 11, 12] else 0,
                # cyclic
                "month_sin":  np.sin(2 * np.pi * future_date.month / 12),
                "month_cos":  np.cos(2 * np.pi * future_date.month / 12),
                "week_num_sin": np.sin(2 * np.pi * future_date.isocalendar()[1] / 52),
                "week_num_cos": np.cos(2 * np.pi * future_date.isocalendar()[1] / 52),
            }

            # Lag features pulled from the growing `temp` series
            series = temp["units_sold"]
            for lag in [1, 2, 3, 4, 8, 12]:
                idx = len(series) - lag
                next_row[f"lag_{lag}w"] = series.iloc[idx] if idx >= 0 else series.mean()

            # Rolling features
            for w in [2, 4, 8, 12]:
                window = series.iloc[-w:] if len(series) >= w else series
                next_row[f"roll_mean_{w}w"] = window.mean()
                next_row[f"roll_std_{w}w"]  = window.std() if len(window) > 1 else 0
                next_row[f"roll_max_{w}w"]  = window.max()

            # Interaction
            next_row["promo_x_festival"] = next_row["is_promo_week"] * next_row["is_festival_season"]
            next_row["price_x_promo"]    = next_row["avg_price"] * next_row["is_promo_week"]
            next_row["demand_cv_4w"]     = (next_row["roll_std_4w"]
                                            / (next_row["roll_mean_4w"] + 1e-6))

            # One-hot category (same columns as training)
            for col in [c for c in feature_cols if c.startswith("cat_")]:
                cat_name = col.replace("cat_", "")
                next_row[col] = 1 if last_row["category"] == cat_name else 0

            # Predict
            feat_vec  = np.array([[next_row.get(f, 0) for f in feature_cols]])
            if "Linear" in best_name:
                feat_vec = scaler.transform(feat_vec)
            pred_units = max(0, int(round(model.predict(feat_vec)[0])))

            next_row["units_sold"]       = pred_units      # feed back for next lag
            next_row["predicted_units"]  = pred_units
            next_row["horizon_week"]     = horizon

            future_rows.append(next_row)
            temp = pd.concat([temp, pd.DataFrame([next_row])], ignore_index=True)

        all_forecasts.extend(future_rows)

    forecast_df = pd.DataFrame(all_forecasts)
    os.makedirs(os.path.dirname(FORECAST_PATH), exist_ok=True)
    forecast_df.to_csv(FORECAST_PATH, index=False)
    print(f"[✓] Forecasts saved → {FORECAST_PATH}  ({len(forecast_df):,} rows)")
    _plot_forecast_sample(df, forecast_df)
    return forecast_df


# ── Plotting helpers ──────────────────────────────────────
BG = "#1a1a2e"
FG = "#e0e0e0"
ACCENT = "#e94560"

def _plot_feature_importance(model, feature_cols: list, model_name: str) -> None:
    plt.style.use("dark_background")
    plt.rcParams["figure.facecolor"] = BG

    importance = pd.Series(model.feature_importances_, index=feature_cols)
    top15      = importance.nlargest(15).sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = plt.cm.plasma(np.linspace(0.3, 0.9, len(top15)))
    ax.barh(top15.index, top15.values, color=colors, edgecolor="none")
    ax.set_title(f"Feature Importance — {model_name}", fontsize=14, pad=12)
    ax.set_xlabel("Importance Score")
    ax.set_facecolor("#16213e")
    fig.tight_layout()
    path = os.path.join(IMG_DIR, "09_feature_importance.png")
    fig.savefig(path, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[✓] Saved → {path}")


def _plot_actual_vs_predicted(y_true, y_pred, model_name: str) -> None:
    plt.style.use("dark_background")
    plt.rcParams["figure.facecolor"] = BG

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Line comparison (first 200 points)
    n = min(200, len(y_true))
    axes[0].plot(y_true[:n],  label="Actual",    color="#00b4d8", linewidth=1.5)
    axes[0].plot(y_pred[:n],  label="Predicted", color=ACCENT,    linewidth=1.5, alpha=0.85)
    axes[0].set_title(f"Actual vs Predicted — {model_name}", fontsize=13)
    axes[0].set_xlabel("Sample Index")
    axes[0].set_ylabel("Units Sold")
    axes[0].legend()
    axes[0].set_facecolor("#16213e")
    axes[0].grid(linestyle="--", alpha=0.3)

    # Scatter
    axes[1].scatter(y_true, y_pred, alpha=0.3, s=8, color=ACCENT)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[1].plot(lims, lims, "w--", linewidth=1)
    axes[1].set_title("Scatter: Actual vs Predicted", fontsize=13)
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predicted")
    axes[1].set_facecolor("#16213e")
    axes[1].grid(linestyle="--", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(IMG_DIR, "10_actual_vs_predicted.png")
    fig.savefig(path, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[✓] Saved → {path}")


def _plot_forecast_sample(hist_df: pd.DataFrame, forecast_df: pd.DataFrame) -> None:
    """Plot 1-year historical + 12-week forecast for a sample product-store."""
    plt.style.use("dark_background")
    plt.rcParams["figure.facecolor"] = BG

    # Pick the product with highest avg sales
    top_combo = (hist_df.groupby(["store_id", "product_id"])["units_sold"]
                         .mean()
                         .idxmax())
    sid, pid = top_combo

    hist = (hist_df[(hist_df["store_id"] == sid) & (hist_df["product_id"] == pid)]
                .groupby("week_start")["units_sold"].sum()
                .reset_index()
                .tail(52))

    fore = forecast_df[(forecast_df["store_id"] == sid) & (forecast_df["product_id"] == pid)]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(hist["week_start"],         hist["units_sold"],
            label="Historical (1 yr)", color="#00b4d8", linewidth=2)
    ax.plot(fore["week_start"],         fore["predicted_units"],
            label="12-Week Forecast",  color=ACCENT,   linewidth=2,
            linestyle="--", marker="o", markersize=5)

    ax.axvline(hist["week_start"].max(), color="gray", linestyle=":", linewidth=1)
    ax.set_title(f"📊  Sales Forecast — Store {sid} | Product {pid}", fontsize=14, pad=12)
    ax.set_xlabel("Week")
    ax.set_ylabel("Units Sold")
    ax.legend()
    ax.set_facecolor("#16213e")
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(IMG_DIR, "11_sales_forecast.png")
    fig.savefig(path, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[✓] Saved → {path}")


# ── Entry point ───────────────────────────────────────────
if __name__ == "__main__":
    from preprocessing     import run_preprocessing_pipeline
    from feature_engineering import run_feature_engineering

    df_clean   = run_preprocessing_pipeline()
    df_features = run_feature_engineering(df_clean)

    result      = train_global_model(df_features)
    forecast_df = generate_forecasts(df_features, result, forecast_weeks=12)
    print("\nForecast sample (first 5 rows):")
    print(forecast_df[["week_start", "store_id", "product_id",
                         "product_name", "predicted_units"]].head())
