"""
=============================================================
  Retail Sales Forecasting & Inventory Optimization System
  Module: regional_clustering.py
  Future Improvement #1 + #10:
    - Multi-store regional forecasting with K-Means clustering
    - Region-wise demand segmentation by product demand pattern
=============================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster      import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics       import silhouette_score

IMG_DIR    = os.path.join(os.path.dirname(__file__), "..", "images")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(IMG_DIR,    exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

BG     = "#1a1a2e"
FG     = "#e0e0e0"
ACCENT = "#e94560"
PANEL  = "#16213e"

# ── Geographic region mapping ─────────────────────────────
STORE_REGION = {
    "S001": "West",
    "S002": "North",
    "S003": "South",
    "S004": "West",
    "S005": "South",
}
REGION_COLOR = {
    "North": "#00b4d8",
    "South": "#06d6a0",
    "West":  "#ffd166",
    "East":  "#e94560",
}


# ──────────────────────────────────────────────────────────
# PART A: STORE-LEVEL REGIONAL CLUSTERING
# ──────────────────────────────────────────────────────────

def build_store_feature_matrix(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a store-level feature matrix for clustering.
    Features: avg demand, demand volatility, promo sensitivity,
              stockout rate, revenue per week, seasonal variance.
    """
    # Map geographic region
    features_df = features_df.copy()
    features_df["region"] = features_df["store_id"].map(STORE_REGION)

    agg = (
        features_df.groupby(["store_id", "store_name", "region"])
        .agg(
            avg_weekly_demand     = ("units_sold",      "mean"),
            demand_std            = ("units_sold",      "std"),
            avg_revenue_per_week  = ("revenue",         "mean"),
            promo_sensitivity     = ("is_promo_week",   "mean"),
            stockout_rate         = ("stockout_days",   "mean"),
            demand_cv             = ("demand_cv_4w",    "mean"),
            n_products            = ("product_id",      "nunique"),
        )
        .reset_index()
    )
    agg["demand_std"] = agg["demand_std"].fillna(0)
    return agg


def cluster_stores(store_features: pd.DataFrame,
                   n_clusters: int = 3) -> pd.DataFrame:
    """K-Means cluster stores by their demand behaviour."""
    feature_cols = ["avg_weekly_demand", "demand_std", "avg_revenue_per_week",
                    "promo_sensitivity", "stockout_rate", "demand_cv"]
    X = store_features[feature_cols].fillna(0).values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Find optimal k with silhouette score (k=2..5)
    best_k, best_score = 2, -1
    for k in range(2, min(6, len(store_features))):
        km      = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels  = km.fit_predict(X_scaled)
        if len(set(labels)) > 1:
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_k, best_score = k, score

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    store_features = store_features.copy()
    store_features["demand_cluster"]       = km.fit_predict(X_scaled)
    store_features["cluster_label"]        = store_features["demand_cluster"].map(
        lambda c: f"Cluster-{c+1}"
    )
    store_features["silhouette_score"]     = best_score
    print(f"[✓] Store clustering: k={best_k}, silhouette={best_score:.3f}")
    return store_features


def plot_store_clusters(store_features: pd.DataFrame) -> str:
    """Plot store clusters using PCA 2D projection."""
    plt.style.use("dark_background")
    plt.rcParams["figure.facecolor"] = BG

    feature_cols = ["avg_weekly_demand", "demand_std", "avg_revenue_per_week",
                    "promo_sensitivity", "stockout_rate", "demand_cv"]
    X       = store_features[feature_cols].fillna(0).values
    scaler  = StandardScaler()
    X_s     = scaler.fit_transform(X)
    pca     = PCA(n_components=2)
    X_2d    = pca.fit_transform(X_s)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: PCA scatter – demand clusters
    palette = plt.cm.Set2(np.linspace(0, 0.8, store_features["demand_cluster"].nunique()))
    for i, (cl, grp) in enumerate(store_features.groupby("demand_cluster")):
        idx = store_features["demand_cluster"] == cl
        axes[0].scatter(X_2d[idx, 0], X_2d[idx, 1],
                        label=f"Cluster {cl+1}", s=200,
                        color=palette[i], edgecolors="white", linewidth=1.5)
        for _, row in grp.iterrows():
            j = store_features.index.get_loc(row.name)
            axes[0].annotate(row["store_name"], (X_2d[j, 0], X_2d[j, 1]),
                             textcoords="offset points", xytext=(6, 4),
                             fontsize=8, color=FG)
    axes[0].set_title("Store Demand Clusters (PCA 2D)", fontsize=13)
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}% var)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}% var)")
    axes[0].legend(fontsize=9)
    axes[0].set_facecolor(PANEL)
    axes[0].grid(linestyle="--", alpha=0.3)

    # Right: Geographic region bar chart of avg revenue
    region_rev = store_features.groupby("region")["avg_revenue_per_week"].mean().sort_values()
    colors = [REGION_COLOR.get(r, ACCENT) for r in region_rev.index]
    axes[1].barh(region_rev.index, region_rev.values, color=colors, edgecolor="none")
    axes[1].set_title("Avg Weekly Revenue by Region", fontsize=13)
    axes[1].set_xlabel("Avg Revenue (Rs)")
    axes[1].set_facecolor(PANEL)
    axes[1].grid(axis="x", linestyle="--", alpha=0.3)

    fig.suptitle("Multi-Store Regional Clustering Analysis", fontsize=14,
                 color=FG, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(IMG_DIR, "16_store_regional_clustering.png")
    fig.savefig(path, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[✓] Saved -> {path}")
    return path


# ──────────────────────────────────────────────────────────
# PART B: PRODUCT DEMAND SEGMENTATION
# ──────────────────────────────────────────────────────────

def build_product_feature_matrix(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a product-level feature matrix for demand segmentation.
    Segments: high-velocity, slow-moving, seasonal, volatile.
    """
    agg = (
        features_df.groupby(["product_id", "product_name", "category"])
        .agg(
            avg_demand       = ("units_sold",      "mean"),
            demand_std       = ("units_sold",      "std"),
            demand_cv        = ("demand_cv_4w",    "mean"),
            promo_lift       = ("is_promo_week",   "mean"),
            stockout_rate    = ("stockout_days",   "mean"),
            avg_revenue      = ("revenue",         "mean"),
        )
        .reset_index()
    )
    agg["demand_std"] = agg["demand_std"].fillna(0)
    agg["demand_cv"]  = agg["demand_cv"].fillna(0)
    return agg


def segment_products(product_features: pd.DataFrame,
                     n_segments: int = 4) -> pd.DataFrame:
    """Cluster products by demand pattern into 4 strategic segments."""
    feature_cols = ["avg_demand", "demand_std", "demand_cv",
                    "promo_lift", "stockout_rate"]
    X = product_features[feature_cols].fillna(0).values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_segments, random_state=42, n_init=10)
    product_features = product_features.copy()
    product_features["segment"] = km.fit_predict(X_scaled)

    # Name segments based on demand profile
    seg_profiles = product_features.groupby("segment")["avg_demand"].mean().sort_values()
    seg_map = {}
    labels  = ["Slow Movers", "Stable Core", "High Velocity", "Star Products"]
    for i, seg_id in enumerate(seg_profiles.index):
        seg_map[seg_id] = labels[i]
    product_features["segment_name"] = product_features["segment"].map(seg_map)

    print("[✓] Product demand segments:")
    print(product_features.groupby("segment_name")[["product_name", "avg_demand"]]
          .agg({"product_name": "count", "avg_demand": "mean"})
          .rename(columns={"product_name": "count"})
          .round(1).to_string())
    return product_features


def plot_product_segments(product_features: pd.DataFrame) -> str:
    """Visualize product segments as a scatter and a grouped bar."""
    plt.style.use("dark_background")
    plt.rcParams["figure.facecolor"] = BG

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    palette = {"Slow Movers":   "#ffd166",
               "Stable Core":   "#06d6a0",
               "High Velocity": "#00b4d8",
               "Star Products": ACCENT}

    # Left: avg_demand vs demand_cv scatter
    for seg, grp in product_features.groupby("segment_name"):
        axes[0].scatter(grp["avg_demand"], grp["demand_cv"],
                        label=seg, color=palette.get(seg, "white"),
                        s=120, edgecolors="white", linewidth=0.8, alpha=0.9)
        for _, row in grp.iterrows():
            axes[0].annotate(row["product_name"][:10],
                             (row["avg_demand"], row["demand_cv"]),
                             textcoords="offset points", xytext=(4, 3),
                             fontsize=6, color=FG, alpha=0.7)

    axes[0].set_title("Demand Segmentation Map", fontsize=13)
    axes[0].set_xlabel("Avg Weekly Demand (units)")
    axes[0].set_ylabel("Demand Coefficient of Variation (CV)")
    axes[0].legend(fontsize=8, framealpha=0.2)
    axes[0].set_facecolor(PANEL)
    axes[0].grid(linestyle="--", alpha=0.3)

    # Right: avg revenue per segment
    seg_rev = product_features.groupby("segment_name")["avg_revenue"].mean().sort_values()
    colors  = [palette.get(s, ACCENT) for s in seg_rev.index]
    axes[1].barh(seg_rev.index, seg_rev.values, color=colors, edgecolor="none")
    axes[1].set_title("Avg Revenue per Demand Segment", fontsize=13)
    axes[1].set_xlabel("Avg Weekly Revenue (Rs)")
    axes[1].set_facecolor(PANEL)
    axes[1].grid(axis="x", linestyle="--", alpha=0.3)

    fig.suptitle("Product Demand Segmentation (K-Means)", fontsize=14,
                 color=FG, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(IMG_DIR, "17_product_demand_segments.png")
    fig.savefig(path, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[✓] Saved -> {path}")
    return path


# ── Master runner ─────────────────────────────────────────
def run_regional_clustering(features_df: pd.DataFrame) -> dict:
    print("\n" + "=" * 60)
    print("  Running Regional Clustering & Demand Segmentation ...")
    print("=" * 60)

    # Store clustering
    store_feats    = build_store_feature_matrix(features_df)
    store_clusters = cluster_stores(store_feats)
    plot_store_clusters(store_clusters)
    store_clusters.to_csv(
        os.path.join(REPORT_DIR, "store_clusters.csv"), index=False)

    # Product segmentation
    prod_feats    = build_product_feature_matrix(features_df)
    prod_segments = segment_products(prod_feats)
    plot_product_segments(prod_segments)
    prod_segments.to_csv(
        os.path.join(REPORT_DIR, "product_segments.csv"), index=False)

    print("[✓] Regional clustering & segmentation complete")
    return {"store_clusters": store_clusters, "product_segments": prod_segments}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.preprocessing import run_preprocessing_pipeline
    from src.feature_engineering import run_feature_engineering
    df = run_preprocessing_pipeline()
    feat = run_feature_engineering(df)
    run_regional_clustering(feat)
