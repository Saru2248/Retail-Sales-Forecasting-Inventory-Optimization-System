"""
=============================================================
  Retail Sales Forecasting & Inventory Optimization System
  Module: eda.py
  Purpose: Exploratory Data Analysis — generate and save all
           key insight plots used in the project report and
           GitHub README.
=============================================================
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (works without display)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Output directory ──────────────────────────────────────
IMG_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
os.makedirs(IMG_DIR, exist_ok=True)

# ── Styling ───────────────────────────────────────────────
PALETTE = "Set2"
BG      = "#1a1a2e"
FG      = "#e0e0e0"
ACCENT  = "#e94560"
GRID    = "#2e2e4e"

def _style() -> None:
    """Apply a dark professional style to all plots."""
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    "#16213e",
        "axes.edgecolor":    GRID,
        "axes.labelcolor":   FG,
        "xtick.color":       FG,
        "ytick.color":       FG,
        "text.color":        FG,
        "grid.color":        GRID,
        "font.family":       "DejaVu Sans",
        "axes.titlesize":    14,
        "axes.labelsize":    11,
        "legend.fontsize":   9,
        "figure.dpi":        120,
    })

def _save(fig: plt.Figure, name: str) -> str:
    path = os.path.join(IMG_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[✓] Saved → {path}")
    return path


# ──────────────────────────────────────────────────────────
# 1. Monthly Sales Trend (all stores combined)
# ──────────────────────────────────────────────────────────
def plot_monthly_sales_trend(df: pd.DataFrame) -> str:
    """Line chart: total monthly revenue across all stores."""
    _style()
    monthly = (df.groupby(["year", "month"])["revenue"]
                 .sum()
                 .reset_index())
    monthly["period"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
    )
    monthly = monthly.sort_values("period")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(monthly["period"], monthly["revenue"] / 1e6,
            color=ACCENT, linewidth=2.2, marker="o", markersize=4)
    ax.fill_between(monthly["period"], monthly["revenue"] / 1e6,
                    alpha=0.15, color=ACCENT)
    ax.set_title("📈  Monthly Revenue Trend (All Stores)", fontsize=15, pad=14)
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue (₹ Millions)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("₹%.1fM"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return _save(fig, "01_monthly_sales_trend.png")


# ──────────────────────────────────────────────────────────
# 2. Category-wise Total Revenue (bar chart)
# ──────────────────────────────────────────────────────────
def plot_category_revenue(df: pd.DataFrame) -> str:
    """Horizontal bar chart of revenue by product category."""
    _style()
    cat_rev = (df.groupby("category")["revenue"]
                 .sum()
                 .sort_values()
                 .reset_index())

    colors = sns.color_palette(PALETTE, len(cat_rev))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(cat_rev["category"], cat_rev["revenue"] / 1e6,
                   color=colors, edgecolor="none")

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height() / 2,
                f"₹{width:.1f}M", va="center", fontsize=9, color=FG)

    ax.set_title("🛒  Total Revenue by Category (3 Years)", fontsize=15, pad=14)
    ax.set_xlabel("Revenue (₹ Millions)")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return _save(fig, "02_category_revenue.png")


# ──────────────────────────────────────────────────────────
# 3. Store-wise Sales Comparison
# ──────────────────────────────────────────────────────────
def plot_store_comparison(df: pd.DataFrame) -> str:
    """Grouped bar chart: annual revenue per store."""
    _style()
    store_yr = (df.groupby(["store_name", "year"])["revenue"]
                  .sum()
                  .reset_index())

    fig, ax = plt.subplots(figsize=(12, 5))
    years   = sorted(store_yr["year"].unique())
    x       = np.arange(store_yr["store_name"].nunique())
    stores  = store_yr["store_name"].unique()
    width   = 0.25
    colors  = sns.color_palette(PALETTE, len(years))

    for i, yr in enumerate(years):
        vals = store_yr[store_yr["year"] == yr].set_index("store_name").reindex(stores)["revenue"] / 1e6
        ax.bar(x + i * width, vals, width, label=str(yr), color=colors[i], alpha=0.88)

    ax.set_xticks(x + width)
    ax.set_xticklabels(stores, rotation=15, ha="right")
    ax.set_title("🏪  Store-wise Annual Revenue Comparison", fontsize=15, pad=14)
    ax.set_ylabel("Revenue (₹ Millions)")
    ax.legend(title="Year")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return _save(fig, "03_store_comparison.png")


# ──────────────────────────────────────────────────────────
# 4. Top 10 Products by Revenue
# ──────────────────────────────────────────────────────────
def plot_top_products(df: pd.DataFrame) -> str:
    """Horizontal bar chart of top 10 products by total revenue."""
    _style()
    prod_rev = (df.groupby("product_name")["revenue"]
                  .sum()
                  .nlargest(10)
                  .reset_index()
                  .sort_values("revenue"))

    colors = sns.color_palette("rocket", len(prod_rev))
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(prod_rev["product_name"], prod_rev["revenue"] / 1e6,
                   color=colors, edgecolor="none")

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2,
                f"₹{w:.1f}M", va="center", fontsize=9, color=FG)

    ax.set_title("🏆  Top 10 Products by Revenue (3 Years)", fontsize=15, pad=14)
    ax.set_xlabel("Revenue (₹ Millions)")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return _save(fig, "04_top_products.png")


# ──────────────────────────────────────────────────────────
# 5. Weekly Sales Seasonality Heatmap
# ──────────────────────────────────────────────────────────
def plot_seasonality_heatmap(df: pd.DataFrame) -> str:
    """Heatmap: avg units sold by month × day-of-week."""
    _style()
    pivot = (df.groupby(["month", "day_of_week"])["units_sold"]
               .mean()
               .reset_index()
               .pivot(index="month", columns="day_of_week", values="units_sold"))
    pivot.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", linewidths=0.5,
                annot=True, fmt=".0f",
                cbar_kws={"label": "Avg Units Sold"})
    ax.set_title("🗓  Seasonality Heatmap — Avg Units Sold (Month × Day)", fontsize=14, pad=14)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Month")
    fig.tight_layout()
    return _save(fig, "05_seasonality_heatmap.png")


# ──────────────────────────────────────────────────────────
# 6. Promotion Impact Boxplot
# ──────────────────────────────────────────────────────────
def plot_promotion_impact(df: pd.DataFrame) -> str:
    """Boxplot comparing units sold on promo vs non-promo days."""
    _style()
    df2 = df.copy()
    df2["Promotion"] = df2["is_promotion"].map({1: "Promo Day", 0: "Normal Day"})

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df2, x="Promotion", y="units_sold", ax=ax,
                palette={"Promo Day": ACCENT, "Normal Day": "#00b4d8"},
                width=0.45, fliersize=2)
    ax.set_title("🎯  Promotion Impact on Units Sold", fontsize=15, pad=14)
    ax.set_ylabel("Units Sold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return _save(fig, "06_promotion_impact.png")


# ──────────────────────────────────────────────────────────
# 7. Stockout Analysis by Category
# ──────────────────────────────────────────────────────────
def plot_stockout_analysis(df: pd.DataFrame) -> str:
    """Bar chart of total stockout days per category."""
    _style()
    so = (df[df["stockout_flag"] == 1]
            .groupby("category")["stockout_flag"]
            .sum()
            .sort_values(ascending=False)
            .reset_index())

    colors = sns.color_palette("flare", len(so))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(so["category"], so["stockout_flag"], color=colors, edgecolor="none")
    ax.set_title("⚠️  Stockout Occurrences by Category", fontsize=15, pad=14)
    ax.set_xlabel("Category")
    ax.set_ylabel("Stockout Events (days)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    return _save(fig, "07_stockout_analysis.png")


# ──────────────────────────────────────────────────────────
# 8. Correlation Heatmap
# ──────────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    """Correlation heatmap of key numeric features."""
    _style()
    cols = ["units_sold", "revenue", "unit_price", "is_promotion",
            "is_weekend", "month", "day_of_week", "is_festival_season"]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(10, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, ax=ax, mask=mask, cmap="coolwarm", center=0,
                annot=True, fmt=".2f", square=True,
                cbar_kws={"shrink": 0.8})
    ax.set_title("🔗  Feature Correlation Heatmap", fontsize=15, pad=14)
    fig.tight_layout()
    return _save(fig, "08_correlation_heatmap.png")


# ──────────────────────────────────────────────────────────
# Master runner
# ──────────────────────────────────────────────────────────
def run_eda(df: pd.DataFrame) -> list:
    """Run all EDA plots and return list of saved image paths."""
    print("\n" + "=" * 60)
    print("  Running Exploratory Data Analysis …")
    print("=" * 60)
    paths = []
    paths.append(plot_monthly_sales_trend(df))
    paths.append(plot_category_revenue(df))
    paths.append(plot_store_comparison(df))
    paths.append(plot_top_products(df))
    paths.append(plot_seasonality_heatmap(df))
    paths.append(plot_promotion_impact(df))
    paths.append(plot_stockout_analysis(df))
    paths.append(plot_correlation_heatmap(df))
    print(f"\n[✓] {len(paths)} EDA charts saved to {IMG_DIR}")
    return paths


if __name__ == "__main__":
    from preprocessing import run_preprocessing_pipeline
    df = run_preprocessing_pipeline()
    run_eda(df)
