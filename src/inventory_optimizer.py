"""
=============================================================
  Retail Sales Forecasting & Inventory Optimization System
  Module: inventory_optimizer.py
  Purpose: Compute safety stock, reorder point, Economic Order
           Quantity (EOQ), and generate actionable reorder
           alerts based on the 12-week forecast.
=============================================================

BUSINESS LOGIC (simplified EOQ model):
──────────────────────────────────────
  Safety Stock   = Z × σ_demand × √(Lead Time)
  Reorder Point  = (Avg Weekly Demand × Lead Time in weeks) + Safety Stock
  EOQ            = √(2 × D × S / H)
    D = Annual demand (units)
    S = Ordering cost per order (₹100 assumed)
    H = Holding cost per unit per year = base_price × holding_cost_pct
  Inventory Status = REORDER / OK / OVERSTOCK based on simulated current stock
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

FORECAST_PATH   = os.path.join(os.path.dirname(__file__), "..", "outputs", "forecasts.csv")
FEATURE_PATH    = os.path.join(os.path.dirname(__file__), "..", "data",    "features.csv")
INVENTORY_PATH  = os.path.join(os.path.dirname(__file__), "..", "outputs", "inventory_report.csv")
ALERT_PATH      = os.path.join(os.path.dirname(__file__), "..", "outputs", "reorder_alerts.csv")
IMG_DIR         = os.path.join(os.path.dirname(__file__), "..", "images")
os.makedirs(IMG_DIR, exist_ok=True)

BG     = "#1a1a2e"
FG     = "#e0e0e0"
ACCENT = "#e94560"
SERVICE_LEVEL_Z = 1.65    # 95% service level ≈ Z = 1.65
ORDERING_COST   = 100.0   # ₹ per order placed


# ── 1. Compute demand statistics from historical features ─
def compute_demand_stats(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each product × store compute:
      - avg_weekly_demand
      - std_weekly_demand
      - annual demand projection
    """
    stats = (
        features_df.groupby(["store_id", "store_name",
                              "product_id", "product_name",
                              "category",
                              "lead_time_days",
                              "holding_cost_pct",
                              "base_price"])
        .agg(
            avg_weekly_demand = ("units_sold", "mean"),
            std_weekly_demand = ("units_sold", "std"),
            obs_weeks         = ("units_sold", "count"),
        )
        .reset_index()
    )
    stats["std_weekly_demand"] = stats["std_weekly_demand"].fillna(0)
    # Annualise demand
    stats["annual_demand"] = stats["avg_weekly_demand"] * 52
    return stats


# ── 2. Compute Safety Stock ───────────────────────────────
def compute_safety_stock(stats: pd.DataFrame) -> pd.DataFrame:
    """
    Safety Stock = Z × σ × √(Lead Time in weeks)
    Lead time is in days → convert to weeks.
    """
    stats = stats.copy()
    stats["lead_time_weeks"] = stats["lead_time_days"] / 7
    stats["safety_stock"] = (
        SERVICE_LEVEL_Z
        * stats["std_weekly_demand"]
        * np.sqrt(stats["lead_time_weeks"])
    ).round(1)
    return stats


# ── 3. Compute Reorder Point ─────────────────────────────
def compute_reorder_point(stats: pd.DataFrame) -> pd.DataFrame:
    """Reorder Point = (avg demand × lead time in weeks) + safety stock"""
    stats = stats.copy()
    stats["reorder_point"] = (
        stats["avg_weekly_demand"] * stats["lead_time_weeks"]
        + stats["safety_stock"]
    ).round(1)
    return stats


# ── 4. Compute EOQ ───────────────────────────────────────
def compute_eoq(stats: pd.DataFrame) -> pd.DataFrame:
    """
    Economic Order Quantity:
      EOQ = √(2 × D × S / H)
      H   = base_price × holding_cost_pct
    """
    stats = stats.copy()
    stats["holding_cost_per_unit"] = (stats["base_price"]
                                       * stats["holding_cost_pct"])
    stats["EOQ"] = np.sqrt(
        2 * stats["annual_demand"] * ORDERING_COST
        / stats["holding_cost_per_unit"].replace(0, np.nan)
    ).fillna(0).round(1)
    return stats


# ── 5. Simulate current stock levels ─────────────────────
def simulate_current_stock(stats: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate a realistic current stock level as a random fraction
    of (EOQ + reorder_point) so we get a mix of REORDER / OK / OVERSTOCK.
    """
    np.random.seed(7)
    stats = stats.copy()
    stock_multiplier = np.random.uniform(0.4, 2.5, size=len(stats))
    stats["current_stock"] = (
        (stats["reorder_point"] + stats["EOQ"] * 0.5) * stock_multiplier
    ).round(0).astype(int)
    return stats


# ── 6. Assign Inventory Status ────────────────────────────
def assign_status(stats: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each product-store row:
      REORDER   → current_stock ≤ reorder_point
      OVERSTOCK → current_stock > (reorder_point + 2 × EOQ)
      OK        → otherwise
    """
    def _status(row):
        if row["current_stock"] <= row["reorder_point"]:
            return "🔴 REORDER"
        elif row["current_stock"] > row["reorder_point"] + 2 * row["EOQ"]:
            return "🟡 OVERSTOCK"
        else:
            return "🟢 OK"

    stats = stats.copy()
    stats["inventory_status"] = stats.apply(_status, axis=1)

    breakdown = stats["inventory_status"].value_counts().to_dict()
    print("  Inventory Status Breakdown:")
    for k, v in breakdown.items():
        print(f"    {k} : {v} product-store combinations")
    return stats


# ── 7. Merge forecast demand into report ─────────────────
def merge_forecast_demand(stats: pd.DataFrame,
                           forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Add forecasted 12-week demand total to the inventory table."""
    forecast_agg = (
        forecast_df.groupby(["store_id", "product_id"])["predicted_units"]
        .sum()
        .reset_index()
        .rename(columns={"predicted_units": "forecast_12w_demand"})
    )
    merged = stats.merge(forecast_agg, on=["store_id", "product_id"], how="left")
    merged["forecast_12w_demand"] = merged["forecast_12w_demand"].fillna(0).round(0).astype(int)
    return merged


# ── 8. Generate Reorder Alerts ────────────────────────────
def generate_reorder_alerts(report: pd.DataFrame) -> pd.DataFrame:
    """Filter to REORDER items and compute recommended order quantity."""
    alerts = report[report["inventory_status"] == "🔴 REORDER"].copy()
    alerts["recommended_order_qty"] = (
        alerts["EOQ"] + alerts["reorder_point"] - alerts["current_stock"]
    ).clip(lower=0).round(0).astype(int)
    alerts["estimated_order_cost"] = (
        alerts["recommended_order_qty"] * alerts["base_price"]
    ).round(2)
    alerts = alerts.sort_values("estimated_order_cost", ascending=False)
    print(f"\n  ⚠ {len(alerts)} products require immediate reorder!")
    return alerts


# ── 9. Visualizations ─────────────────────────────────────
def plot_inventory_status(report: pd.DataFrame) -> None:
    plt.style.use("dark_background")
    plt.rcParams["figure.facecolor"] = BG

    # ── Pie chart of status distribution ──────────────────
    status_counts = report["inventory_status"].value_counts()
    colors = {"🔴 REORDER": "#e94560",
               "🟢 OK":      "#06d6a0",
               "🟡 OVERSTOCK":"#ffd166"}
    c_list = [colors.get(s, "#aaa") for s in status_counts.index]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].pie(status_counts, labels=status_counts.index,
                autopct="%1.1f%%", colors=c_list,
                startangle=90, textprops={"color": FG, "fontsize": 11})
    axes[0].set_title("📦  Inventory Status Distribution", fontsize=14)
    axes[0].set_facecolor(BG)

    # ── Bar: avg safety stock by category ─────────────────
    cat_ss = report.groupby("category")["safety_stock"].mean().sort_values()
    palette = sns.color_palette("Set2", len(cat_ss))
    axes[1].barh(cat_ss.index, cat_ss.values, color=palette, edgecolor="none")
    axes[1].set_title("🛡  Avg Safety Stock by Category", fontsize=14)
    axes[1].set_xlabel("Units")
    axes[1].set_facecolor("#16213e")
    axes[1].grid(axis="x", linestyle="--", alpha=0.4)

    fig.tight_layout()
    path = os.path.join(IMG_DIR, "12_inventory_status.png")
    fig.savefig(path, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[✓] Saved → {path}")


def plot_reorder_alerts(alerts: pd.DataFrame) -> None:
    if alerts.empty:
        return
    plt.style.use("dark_background")
    plt.rcParams["figure.facecolor"] = BG

    top = alerts.head(15)[["product_name", "store_name",
                             "recommended_order_qty"]].copy()
    top["label"] = top["product_name"] + "\n(" + top["store_name"] + ")"

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(top)))
    bars = ax.barh(top["label"], top["recommended_order_qty"],
                   color=colors, edgecolor="none")

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 1, bar.get_y() + bar.get_height() / 2,
                f"{int(w)} units", va="center", fontsize=8, color=FG)

    ax.set_title("🚨  Top 15 Products Requiring Immediate Reorder", fontsize=14, pad=12)
    ax.set_xlabel("Recommended Order Quantity (units)")
    ax.set_facecolor("#16213e")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(IMG_DIR, "13_reorder_alerts.png")
    fig.savefig(path, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[✓] Saved → {path}")


def plot_eoq_vs_stock(report: pd.DataFrame) -> None:
    plt.style.use("dark_background")
    plt.rcParams["figure.facecolor"] = BG

    sample = report.sample(min(60, len(report)), random_state=1)
    colors = sample["inventory_status"].map({
        "🔴 REORDER":   "#e94560",
        "🟢 OK":         "#06d6a0",
        "🟡 OVERSTOCK":  "#ffd166",
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(sample["EOQ"], sample["current_stock"],
               c=colors, s=70, alpha=0.8, edgecolors="white", linewidth=0.3)
    ax.set_xlabel("EOQ (Economic Order Quantity)")
    ax.set_ylabel("Current Stock Level")
    ax.set_title("🔄  EOQ vs Current Stock (coloured by status)", fontsize=14, pad=12)
    ax.set_facecolor("#16213e")
    ax.grid(linestyle="--", alpha=0.3)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor="#e94560", label="REORDER"),
                    Patch(facecolor="#06d6a0", label="OK"),
                    Patch(facecolor="#ffd166", label="OVERSTOCK")]
    ax.legend(handles=legend_elems, framealpha=0.3)
    fig.tight_layout()

    path = os.path.join(IMG_DIR, "14_eoq_vs_stock.png")
    fig.savefig(path, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[✓] Saved → {path}")


# ── Master pipeline ───────────────────────────────────────
def run_inventory_optimization(features_df: pd.DataFrame = None,
                                forecast_df: pd.DataFrame = None) -> tuple:
    """
    Full inventory optimization pipeline.
    Returns (inventory_report, reorder_alerts) as DataFrames.
    """
    print("\n" + "=" * 60)
    print("  Running Inventory Optimization …")
    print("=" * 60)

    if features_df is None:
        features_df = pd.read_csv(FEATURE_PATH)
    if forecast_df is None:
        forecast_df = pd.read_csv(FORECAST_PATH)

    stats  = compute_demand_stats(features_df)
    stats  = compute_safety_stock(stats)
    stats  = compute_reorder_point(stats)
    stats  = compute_eoq(stats)
    stats  = simulate_current_stock(stats)
    stats  = assign_status(stats)
    report = merge_forecast_demand(stats, forecast_df)

    # Save full report
    report.to_csv(INVENTORY_PATH, index=False)
    print(f"\n[✓] Inventory report saved → {INVENTORY_PATH}")

    # Generate alerts
    alerts = generate_reorder_alerts(report)
    alerts.to_csv(ALERT_PATH, index=False)
    print(f"[✓] Reorder alerts saved  → {ALERT_PATH}")

    # Visualizations
    plot_inventory_status(report)
    plot_reorder_alerts(alerts)
    plot_eoq_vs_stock(report)

    return report, alerts


if __name__ == "__main__":
    report, alerts = run_inventory_optimization()
    print("\n--- Inventory Report (top 5) ---")
    cols = ["store_name", "product_name", "current_stock",
            "reorder_point", "safety_stock", "EOQ", "inventory_status"]
    print(report[cols].head().to_string(index=False))
    print("\n--- Reorder Alerts (top 5) ---")
    acols = ["store_name", "product_name", "recommended_order_qty",
             "estimated_order_cost"]
    print(alerts[acols].head().to_string(index=False))
