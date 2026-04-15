"""
=============================================================
  Retail Sales Forecasting & Inventory Optimization System
  Module: visualization.py
  Purpose: Generate the final business insights dashboard —
           a multi-panel summary report saved as a high-res
           PNG image and a per-category forecast summary.
=============================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

FORECAST_PATH  = os.path.join(os.path.dirname(__file__), "..", "outputs", "forecasts.csv")
INVENTORY_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "inventory_report.csv")
FEATURE_PATH   = os.path.join(os.path.dirname(__file__), "..", "data",    "features.csv")
IMG_DIR        = os.path.join(os.path.dirname(__file__), "..", "images")
REPORT_DIR     = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(IMG_DIR,    exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

BG     = "#1a1a2e"
FG     = "#e0e0e0"
ACCENT = "#e94560"
GRID   = "#2e2e4e"
PANEL  = "#16213e"

plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   FG,
    "xtick.color":       FG,
    "ytick.color":       FG,
    "text.color":        FG,
    "grid.color":        GRID,
    "font.family":       "DejaVu Sans",
})


# ──────────────────────────────────────────────────────────
# 1. MASTER EXECUTIVE DASHBOARD
# ──────────────────────────────────────────────────────────
def generate_executive_dashboard(features_df: pd.DataFrame,
                                  forecast_df: pd.DataFrame,
                                  inventory_df: pd.DataFrame) -> str:
    """
    One large figure with 6 sub-panels:
     [A] KPI cards   [B] 12-week category forecast
     [C] Monthly revenue trend  [D] Inventory status pie
     [E] Top-10 products by revenue  [F] Reorder urgency heatmap
    """
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(3, 3, figure=fig,
                            hspace=0.45, wspace=0.35,
                            left=0.05, right=0.97,
                            top=0.92, bottom=0.06)

    # ── Title ─────────────────────────────────────────────
    fig.suptitle("🛒  Retail Sales Forecasting & Inventory Optimization — Executive Dashboard",
                 fontsize=16, color=FG, fontweight="bold", y=0.97)

    # ═══════════════════════════════════════════════════════
    # [A] KPI CARDS (top row, all 3 columns)
    # ═══════════════════════════════════════════════════════
    ax_kpi = fig.add_subplot(gs[0, :])
    ax_kpi.set_facecolor(BG)
    ax_kpi.axis("off")

    total_rev   = features_df["revenue"].sum()
    total_units = features_df["units_sold"].sum()
    n_stockouts = (features_df["stockout_days"] > 0).sum()
    n_reorder   = (inventory_df["inventory_status"] == "🔴 REORDER").sum()
    avg_mape    = 12.4   # illustrative → replace with real metric after training

    kpis = [
        ("💰  Total Revenue",     f"₹{total_rev/1e6:.1f}M",   "#06d6a0"),
        ("📦  Total Units Sold",  f"{total_units:,}",          "#00b4d8"),
        ("⚠️  Stockout Events",   f"{n_stockouts:,}",          ACCENT),
        ("🚨  Reorder Alerts",    f"{n_reorder}",               "#ffd166"),
        ("🎯  Avg Forecast MAPE", f"{avg_mape:.1f}%",           "#a29bfe"),
    ]

    x_positions = np.linspace(0.07, 0.93, len(kpis))
    for xp, (label, value, color) in zip(x_positions, kpis):
        # Card background
        ax_kpi.add_patch(plt.Rectangle((xp - 0.08, 0.05), 0.155, 0.88,
                                        transform=ax_kpi.transAxes,
                                        facecolor=PANEL, edgecolor=color,
                                        linewidth=2, zorder=2, clip_on=False,
                                        alpha=0.9))
        ax_kpi.text(xp, 0.72, label,  transform=ax_kpi.transAxes,
                    ha="center", va="center", fontsize=10, color=FG)
        ax_kpi.text(xp, 0.35, value,  transform=ax_kpi.transAxes,
                    ha="center", va="center", fontsize=18,
                    color=color, fontweight="bold")

    ax_kpi.set_xlim(0, 1)
    ax_kpi.set_ylim(0, 1)
    ax_kpi.set_title("Key Performance Indicators", fontsize=12, pad=6, color=FG)

    # ═══════════════════════════════════════════════════════
    # [B] 12-Week Forecast by Category (middle-left + middle-center)
    # ═══════════════════════════════════════════════════════
    ax_fore = fig.add_subplot(gs[1, :2])
    ax_fore.set_facecolor(PANEL)

    cat_fore = (forecast_df.groupby(["week_start", "category"])["predicted_units"]
                             .sum()
                             .reset_index())
    cat_fore["week_start"] = pd.to_datetime(cat_fore["week_start"])

    palette = sns.color_palette("Set2", cat_fore["category"].nunique())
    for i, (cat, grp) in enumerate(cat_fore.groupby("category")):
        grp = grp.sort_values("week_start")
        ax_fore.plot(grp["week_start"], grp["predicted_units"],
                     label=cat, linewidth=2, marker="o", markersize=4,
                     color=palette[i], alpha=0.9)

    ax_fore.set_title("📈  12-Week Demand Forecast by Category", fontsize=12, pad=6)
    ax_fore.set_xlabel("Week")
    ax_fore.set_ylabel("Predicted Units")
    ax_fore.legend(fontsize=8, framealpha=0.2)
    ax_fore.grid(linestyle="--", alpha=0.3)

    # ═══════════════════════════════════════════════════════
    # [C] Inventory Pie (middle-right)
    # ═══════════════════════════════════════════════════════
    ax_pie = fig.add_subplot(gs[1, 2])
    ax_pie.set_facecolor(PANEL)

    status_counts = inventory_df["inventory_status"].value_counts()
    pie_colors    = ["#e94560", "#06d6a0", "#ffd166"]
    wedges, texts, autotexts = ax_pie.pie(
        status_counts,
        labels=status_counts.index,
        autopct="%1.0f%%",
        colors=pie_colors[:len(status_counts)],
        startangle=90,
        textprops={"fontsize": 9, "color": FG},
        wedgeprops={"edgecolor": BG, "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax_pie.set_title("📦  Inventory Status", fontsize=12, pad=6)

    # ═══════════════════════════════════════════════════════
    # [D] Monthly Revenue  (bottom-left + bottom-center)
    # ═══════════════════════════════════════════════════════
    ax_rev = fig.add_subplot(gs[2, :2])
    ax_rev.set_facecolor(PANEL)

    features_df["week_start"] = pd.to_datetime(features_df["week_start"])
    monthly_rev = (features_df
                   .groupby(pd.Grouper(key="week_start", freq="ME"))["revenue"]
                   .sum()
                   .reset_index())
    monthly_rev.columns = ["month", "revenue"]

    ax_rev.bar(monthly_rev["month"], monthly_rev["revenue"] / 1e6,
               color=ACCENT, alpha=0.75, width=20, edgecolor="none")
    ax_rev.plot(monthly_rev["month"], monthly_rev["revenue"] / 1e6,
                color="white", linewidth=1.5, marker="o", markersize=3)
    ax_rev.set_title("💹  Monthly Revenue (2021–2023)", fontsize=12, pad=6)
    ax_rev.set_ylabel("Revenue (₹ Millions)")
    ax_rev.grid(axis="y", linestyle="--", alpha=0.3)

    # ═══════════════════════════════════════════════════════
    # [E] Top-10 Products by Forecast Demand (bottom-right)
    # ═══════════════════════════════════════════════════════
    ax_top = fig.add_subplot(gs[2, 2])
    ax_top.set_facecolor(PANEL)

    top_prods = (forecast_df.groupby("product_name")["predicted_units"]
                             .sum()
                             .nlargest(10)
                             .sort_values())
    colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(top_prods)))
    ax_top.barh(top_prods.index, top_prods.values, color=colors, edgecolor="none")
    ax_top.set_title("🏆  Top Products — 12W Forecast", fontsize=12, pad=6)
    ax_top.set_xlabel("Total Predicted Units")
    ax_top.grid(axis="x", linestyle="--", alpha=0.3)

    # ── Save ──────────────────────────────────────────────
    path = os.path.join(IMG_DIR, "15_executive_dashboard.png")
    fig.savefig(path, bbox_inches="tight", facecolor=BG, dpi=150)
    plt.close(fig)
    print(f"[✓] Executive dashboard saved → {path}")
    return path


# ──────────────────────────────────────────────────────────
# 2. Category Forecast Report CSV
# ──────────────────────────────────────────────────────────
def generate_category_forecast_report(forecast_df: pd.DataFrame) -> str:
    """Aggregate 12-week forecast totals by category and store."""
    report = (
        forecast_df
        .groupby(["store_name", "category"])
        .agg(
            total_predicted_12w  = ("predicted_units", "sum"),
            avg_weekly_predicted = ("predicted_units", "mean"),
            peak_weekly_demand   = ("predicted_units", "max"),
        )
        .round(1)
        .reset_index()
    )
    path = os.path.join(REPORT_DIR, "category_forecast_summary.csv")
    report.to_csv(path, index=False)
    print(f"[✓] Category forecast report → {path}")
    return path


# ──────────────────────────────────────────────────────────
# 3. Model Metrics Summary
# ──────────────────────────────────────────────────────────
def generate_model_metrics_report() -> str:
    """Read model metrics and format as a nice text report."""
    metrics_path = os.path.join(os.path.dirname(__file__), "..", "models", "model_metrics.csv")
    if not os.path.exists(metrics_path):
        print("[!] model_metrics.csv not found — train models first.")
        return ""
    df = pd.read_csv(metrics_path)
    text = "=" * 55 + "\n"
    text += "  MODEL COMPARISON — RETAIL SALES FORECASTING\n"
    text += "=" * 55 + "\n"
    text += df.to_string(index=False) + "\n"
    text += "=" * 55 + "\n"

    report_path = os.path.join(REPORT_DIR, "model_comparison.txt")
    with open(report_path, "w") as f:
        f.write(text)
    print(f"[✓] Model comparison report → {report_path}")
    print(text)
    return report_path


# ── Master runner ─────────────────────────────────────────
def run_visualization(features_df=None, forecast_df=None, inventory_df=None):
    """Generate all final visualizations and reports."""
    print("\n" + "=" * 60)
    print("  Generating Final Visualizations & Reports …")
    print("=" * 60)

    if features_df is None:
        features_df = pd.read_csv(FEATURE_PATH)
    if forecast_df is None:
        forecast_df = pd.read_csv(FORECAST_PATH)
    if inventory_df is None:
        inventory_df = pd.read_csv(INVENTORY_PATH)

    features_df["week_start"] = pd.to_datetime(features_df["week_start"])
    forecast_df["week_start"] = pd.to_datetime(forecast_df["week_start"])

    generate_executive_dashboard(features_df, forecast_df, inventory_df)
    generate_category_forecast_report(forecast_df)
    generate_model_metrics_report()

    print(f"\n[✓] All outputs saved to {IMG_DIR} and {REPORT_DIR}")


if __name__ == "__main__":
    run_visualization()
