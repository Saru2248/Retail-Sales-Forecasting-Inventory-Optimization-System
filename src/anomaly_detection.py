"""
=============================================================
  Retail Sales Forecasting & Inventory Optimization System
  Module: anomaly_detection.py
  Future Improvement #9:
    - Anomaly detection for unusual demand spikes/drops
    - Using Isolation Forest + Z-Score + IQR methods
    - Flags weeks where sales deviate from forecast
=============================================================
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble       import IsolationForest
from sklearn.preprocessing  import StandardScaler

warnings.filterwarnings("ignore")

IMG_DIR    = os.path.join(os.path.dirname(__file__), "..", "images")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(IMG_DIR,    exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

BG    = "#1a1a2e"
FG    = "#e0e0e0"
PANEL = "#16213e"
ACCENT= "#e94560"


# ── 1. Z-Score detction per product-store series ─────────
def detect_anomalies_zscore(features_df: pd.DataFrame,
                              threshold: float = 3.0) -> pd.DataFrame:
    """
    Mark weeks where |Z-score| > threshold as anomalies.
    Computed per product × store time series.
    """
    df = features_df.copy()

    def zscore_flag(series):
        m    = series.mean()
        s    = series.std() + 1e-6
        z    = (series - m) / s
        return z.abs() > threshold, z

    df["z_score"] = 0.0
    df["anomaly_zscore"] = False

    for (sid, pid), grp in df.groupby(["store_id", "product_id"]):
        idx   = grp.index
        flags, z = zscore_flag(grp["units_sold"])
        df.loc[idx, "z_score"]       = z.values
        df.loc[idx, "anomaly_zscore"] = flags.values

    n_anomalies = df["anomaly_zscore"].sum()
    print(f"[✓] Z-Score anomalies detected: {n_anomalies:,} "
          f"({n_anomalies/len(df)*100:.2f}% of all weeks)")
    return df


# ── 2. Isolation Forest detection ─────────────────────────
def detect_anomalies_isolation_forest(features_df: pd.DataFrame,
                                       contamination: float = 0.05
                                       ) -> pd.DataFrame:
    """
    Apply Isolation Forest on multivariate feature space.
    Features used: units_sold, lag_1w, roll_mean_4w, roll_std_4w,
                   is_festival_season, is_promo_week.
    contamination = 0.05 means top 5% most anomalous are flagged.
    """
    df = features_df.copy()

    feat_cols = ["units_sold", "lag_1w", "roll_mean_4w",
                 "roll_std_4w", "is_festival_season", "is_promo_week"]
    feat_cols = [c for c in feat_cols if c in df.columns]

    X = df[feat_cols].fillna(df[feat_cols].median()).values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(n_estimators=100,
                          contamination=contamination,
                          random_state=42,
                          n_jobs=-1)
    df["anomaly_score"]  = iso.fit_predict(X_scaled)    # -1 = anomaly, 1 = normal
    df["anomaly_if"]     = df["anomaly_score"] == -1
    df["if_score"]       = iso.decision_function(X_scaled)  # lower = more anomalous

    n_anomalies = df["anomaly_if"].sum()
    print(f"[✓] Isolation Forest anomalies: {n_anomalies:,} "
          f"({n_anomalies/len(df)*100:.2f}% of all weeks)")
    return df


# ── 3. Combine both methods ───────────────────────────────
def combine_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consensus anomaly: flagged by EITHER Z-Score OR Isolation Forest.
    Double-flagged = high-confidence anomaly.
    """
    df = df.copy()
    df["anomaly_both"]   = df["anomaly_zscore"] & df["anomaly_if"]
    df["anomaly_any"]    = df["anomaly_zscore"] | df["anomaly_if"]
    df["anomaly_type"] = "Normal"
    df.loc[df["anomaly_both"],              "anomaly_type"] = "HIGH CONFIDENCE"
    df.loc[df["anomaly_zscore"] & ~df["anomaly_both"], "anomaly_type"] = "Z-Score"
    df.loc[df["anomaly_if"]    & ~df["anomaly_both"], "anomaly_type"] = "IsoForest"

    # Classify direction
    df.loc[df["anomaly_any"] & (df["z_score"] > 0), "anomaly_direction"] = "Spike"
    df.loc[df["anomaly_any"] & (df["z_score"] <= 0), "anomaly_direction"] = "Drop"
    df["anomaly_direction"] = df.get("anomaly_direction", "Normal")

    print("\n[+] Anomaly type breakdown:")
    print(df["anomaly_type"].value_counts().to_string())
    return df


# ── 4. Generate anomaly report ────────────────────────────
def build_anomaly_report(df: pd.DataFrame) -> pd.DataFrame:
    """Extract anomalous rows into a prioritised report."""
    anomalies = df[df["anomaly_any"]].copy()

    if anomalies.empty:
        print("[!] No anomalies found.")
        return anomalies

    # Estimated lost/excess revenue
    anomalies["expected_units"] = anomalies["roll_mean_4w"].fillna(
        anomalies["units_sold"].mean()
    )
    anomalies["deviation_units"] = (anomalies["units_sold"]
                                     - anomalies["expected_units"]).round(1)

    report_cols = ["week_start", "store_name", "product_name", "category",
                   "units_sold", "expected_units", "deviation_units",
                   "anomaly_type", "z_score", "if_score"]
    report_cols = [c for c in report_cols if c in anomalies.columns]

    report = (anomalies[report_cols]
              .sort_values("z_score", key=abs, ascending=False)
              .reset_index(drop=True))

    print(f"\n[+] Top 5 anomaly events:")
    print(report.head().to_string(index=False))
    return report


# ── 5. Visualizations ─────────────────────────────────────
def plot_anomalies(df: pd.DataFrame, report: pd.DataFrame) -> str:
    """Three-panel anomaly detection chart."""
    plt.style.use("dark_background")
    plt.rcParams["figure.facecolor"] = BG

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── Panel 1: Time-series with anomaly overlay (sample product) ─
    top = (df.groupby(["store_id", "product_id"])["units_sold"]
              .mean().idxmax())
    sid, pid = top
    ts = df[(df["store_id"] == sid) & (df["store_id"] == sid) &
            (df["product_id"] == pid)].sort_values("week_start")

    if "week_start" in ts.columns:
        ts["week_start"] = pd.to_datetime(ts["week_start"])
        normal   = ts[~ts["anomaly_any"]]
        anomaly  = ts[ts["anomaly_any"]]
        axes[0].plot(ts["week_start"], ts["units_sold"],
                     color="#00b4d8", linewidth=1.2, label="Demand")
        if "roll_mean_4w" in ts.columns:
            axes[0].plot(ts["week_start"], ts["roll_mean_4w"],
                         color=FG, linewidth=0.8, linestyle="--",
                         alpha=0.5, label="Rolling Mean")
        axes[0].scatter(anomaly["week_start"], anomaly["units_sold"],
                        color=ACCENT, s=60, zorder=5, label="Anomaly")
        axes[0].set_title(f"Anomaly Detection — Store {sid} | Prod {pid[:4]}",
                          fontsize=11)
        axes[0].set_xlabel("Week")
        axes[0].set_ylabel("Units Sold")
        axes[0].legend(fontsize=8)
        axes[0].set_facecolor(PANEL)
        axes[0].grid(linestyle="--", alpha=0.3)

    # ── Panel 2: Anomaly type pie ─────────────────────────
    type_counts = df[df["anomaly_any"]]["anomaly_type"].value_counts()
    colors      = [ACCENT, "#ffd166", "#00b4d8", "#06d6a0"][:len(type_counts)]
    axes[1].pie(type_counts, labels=type_counts.index, autopct="%1.0f%%",
                colors=colors, startangle=90,
                textprops={"color": FG, "fontsize": 9},
                wedgeprops={"edgecolor": BG, "linewidth": 2})
    axes[1].set_title("Anomaly Type Distribution", fontsize=12)

    # ── Panel 3: Z-Score distribution ────────────────────
    axes[2].hist(df["z_score"].clip(-5, 5), bins=60,
                 color="#00b4d8", alpha=0.7, edgecolor="none")
    axes[2].axvline(3, color=ACCENT, linewidth=2, linestyle="--", label="+3σ")
    axes[2].axvline(-3, color=ACCENT, linewidth=2, linestyle="--", label="-3σ")
    axes[2].set_title("Z-Score Distribution (demand anomaly threshold)", fontsize=11)
    axes[2].set_xlabel("Z-Score")
    axes[2].set_ylabel("Frequency")
    axes[2].legend(fontsize=8)
    axes[2].set_facecolor(PANEL)
    axes[2].grid(linestyle="--", alpha=0.3)

    fig.suptitle("Anomaly Detection — Isolation Forest + Z-Score",
                 fontsize=14, color=FG, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(IMG_DIR, "20_anomaly_detection.png")
    fig.savefig(path, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[✓] Saved -> {path}")
    return path


# ── Master runner ─────────────────────────────────────────
def run_anomaly_detection(features_df: pd.DataFrame) -> dict:
    print("\n" + "=" * 60)
    print("  Running Anomaly Detection ...")
    print("=" * 60)

    df     = detect_anomalies_zscore(features_df)
    df     = detect_anomalies_isolation_forest(df)
    df     = combine_anomalies(df)
    report = build_anomaly_report(df)
    plot_anomalies(df, report)

    if not report.empty:
        report.to_csv(
            os.path.join(REPORT_DIR, "anomaly_report.csv"), index=False)
        print(f"[✓] Anomaly report saved -> reports/anomaly_report.csv")

    return {"df_with_anomalies": df, "anomaly_report": report}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.preprocessing     import run_preprocessing_pipeline
    from src.feature_engineering import run_feature_engineering
    df   = run_preprocessing_pipeline()
    feat = run_feature_engineering(df)
    run_anomaly_detection(feat)
