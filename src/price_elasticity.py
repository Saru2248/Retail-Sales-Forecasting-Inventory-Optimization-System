"""
=============================================================
  Retail Sales Forecasting & Inventory Optimization System
  Module: price_elasticity.py
  Future Improvement #2:
    - Price elasticity modeling using log-log regression
    - Dynamic pricing recommendations per product
=============================================================

THEORY:
  Price Elasticity of Demand (PED):
    PED = % change in quantity / % change in price

  Using log-log regression (OLS):
    ln(Q) = alpha + beta * ln(P) + controls
    beta IS the elasticity coefficient directly.
    beta < -1 → elastic (price-sensitive)
    -1 < beta < 0 → inelastic (price-insensitive)
    beta > 0 → Giffen good (unusual — check data)
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics        import r2_score

warnings.filterwarnings("ignore")

IMG_DIR    = os.path.join(os.path.dirname(__file__), "..", "images")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(IMG_DIR,    exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

BG    = "#1a1a2e"
FG    = "#e0e0e0"
PANEL = "#16213e"
ACCENT= "#e94560"


def compute_price_elasticity(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate price elasticity for each product using log-log OLS regression.
    Controls: promotion flag, festival season, month, year.
    Returns DataFrame with elasticity per product and interpretation.
    """
    df   = features_df.copy()
    df   = df[df["units_sold"] > 0].copy()      # remove zero-demand rows
    df["ln_quantity"] = np.log(df["units_sold"] + 1)
    df["ln_price"]    = np.log(df["avg_price"]  + 1)

    results = []

    for product_id, grp in df.groupby("product_id"):
        if len(grp) < 20:
            continue

        product_name = grp["product_name"].iloc[0]
        category     = grp["category"].iloc[0]

        # Control variables
        X_cols = ["ln_price", "is_promo_week", "is_festival_season",
                  "month", "year"]
        X_cols = [c for c in X_cols if c in grp.columns]
        X      = grp[X_cols].fillna(0).values
        y      = grp["ln_quantity"].values

        model      = LinearRegression()
        model.fit(X, y)
        y_pred     = model.predict(X)
        r2         = r2_score(y, y_pred)
        elasticity = model.coef_[0]   # coefficient on ln_price = elasticity

        # Pricing recommendation
        if elasticity < -1.5:
            recommendation = "Lower price → Big demand boost (Highly elastic)"
        elif elasticity < -0.5:
            recommendation = "Price reduction moderately increases demand"
        elif elasticity < 0:
            recommendation = "Price-insensitive — maintain current pricing"
        else:
            recommendation = "Unusual — verify data or check for Giffen behavior"

        results.append({
            "product_id":       product_id,
            "product_name":     product_name,
            "category":         category,
            "price_elasticity": round(elasticity, 4),
            "r2_score":         round(r2, 4),
            "interpretation":   recommendation,
        })

    elasticity_df = pd.DataFrame(results).sort_values("price_elasticity")
    print("[+] Price Elasticity Results:")
    print(elasticity_df[["product_name", "price_elasticity",
                          "interpretation"]].to_string(index=False))
    return elasticity_df


def compute_optimal_price(elasticity_df: pd.DataFrame,
                           features_df:  pd.DataFrame) -> pd.DataFrame:
    """
    For each product, compute the price range that maximises revenue:
      Revenue = P * Q(P)
      Using elasticity: Q2/Q1 = (P2/P1)^elasticity
    Suggest -10%, 0%, +10% price scenarios.
    """
    price_pivot = (
        features_df.groupby("product_id")["avg_price"].mean().reset_index()
    )
    demand_pivot = (
        features_df.groupby("product_id")["units_sold"].mean().reset_index()
    )
    base = price_pivot.merge(demand_pivot, on="product_id")
    opt  = elasticity_df.merge(base, on="product_id")

    scenarios = []
    for _, row in opt.iterrows():
        e   = row["price_elasticity"]
        p0  = row["avg_price"]
        q0  = row["units_sold"]
        for delta in [-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15]:
            p1  = p0 * (1 + delta)
            q1  = q0 * ((p1 / p0) ** e) if p0 > 0 else q0
            rev = p1 * q1
            scenarios.append({
                "product_id":   row["product_id"],
                "product_name": row["product_name"],
                "base_price":   round(p0, 2),
                "test_price":   round(p1, 2),
                "price_change_pct": f"{delta*100:+.0f}%",
                "predicted_units": round(max(0, q1), 1),
                "predicted_revenue": round(max(0, rev), 2),
            })

    scenarios_df = pd.DataFrame(scenarios)
    # Find the optimal (max revenue) scenario per product
    optimal = (
        scenarios_df.loc[
            scenarios_df.groupby("product_id")["predicted_revenue"].idxmax()
        ][[  "product_id", "product_name", "base_price",
             "test_price", "price_change_pct", "predicted_revenue"]]
    )
    return optimal


def plot_price_elasticity(elasticity_df: pd.DataFrame,
                           optimal_df:   pd.DataFrame) -> str:
    """Two-panel chart: elasticity bar chart + revenue optimisation."""
    plt.style.use("dark_background")
    plt.rcParams["figure.facecolor"] = BG

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Elasticity bar chart ──────────────────────────────
    colors = [ACCENT if e < -1 else "#ffd166" if e < 0 else "#06d6a0"
              for e in elasticity_df["price_elasticity"]]
    bars = axes[0].barh(elasticity_df["product_name"],
                         elasticity_df["price_elasticity"],
                         color=colors, edgecolor="none")
    axes[0].axvline(0, color="white", linewidth=0.8, linestyle="--")
    axes[0].axvline(-1, color="#ffd166", linewidth=0.8,
                    linestyle=":", label="Elastic threshold (< -1)")
    axes[0].set_title("Price Elasticity by Product", fontsize=13)
    axes[0].set_xlabel("Elasticity Coefficient (log-log OLS)")
    axes[0].set_facecolor(PANEL)
    axes[0].grid(axis="x", linestyle="--", alpha=0.3)
    axes[0].legend(fontsize=8)

    # ── Optimal price change ─────────────────────────────
    opt = optimal_df.copy()
    opt["price_chg_num"] = opt["price_change_pct"].str.replace("%", "").astype(float)
    price_colors = ["#06d6a0" if x < 0 else ACCENT if x > 0 else "#ffd166"
                    for x in opt["price_chg_num"]]
    axes[1].barh(opt["product_name"], opt["price_chg_num"],
                  color=price_colors, edgecolor="none")
    axes[1].axvline(0, color="white", linewidth=0.8, linestyle="--")
    axes[1].set_title("Optimal Price Change Recommendation", fontsize=13)
    axes[1].set_xlabel("Recommended Price Change (%)")
    axes[1].set_facecolor(PANEL)
    axes[1].grid(axis="x", linestyle="--", alpha=0.3)

    fig.suptitle("Price Elasticity Analysis & Dynamic Pricing Strategy",
                 fontsize=14, color=FG, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(IMG_DIR, "18_price_elasticity.png")
    fig.savefig(path, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[✓] Saved -> {path}")
    return path


def run_price_elasticity(features_df: pd.DataFrame) -> dict:
    print("\n" + "=" * 60)
    print("  Running Price Elasticity Analysis ...")
    print("=" * 60)

    elasticity_df = compute_price_elasticity(features_df)
    optimal_df    = compute_optimal_price(elasticity_df, features_df)
    plot_price_elasticity(elasticity_df, optimal_df)

    elasticity_df.to_csv(
        os.path.join(REPORT_DIR, "price_elasticity.csv"), index=False)
    optimal_df.to_csv(
        os.path.join(REPORT_DIR, "optimal_pricing.csv"), index=False)

    print(f"[✓] Elasticity report saved -> reports/price_elasticity.csv")
    print(f"[✓] Pricing report saved    -> reports/optimal_pricing.csv")
    return {"elasticity": elasticity_df, "optimal_pricing": optimal_df}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.preprocessing     import run_preprocessing_pipeline
    from src.feature_engineering import run_feature_engineering
    df   = run_preprocessing_pipeline()
    feat = run_feature_engineering(df)
    run_price_elasticity(feat)
