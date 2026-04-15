"""
=============================================================
  Retail Sales Forecasting & Inventory Optimization System
  Module: app/streamlit_dashboard.py  (v2.0 — Single Page)
  All sections displayed on one scrollable page.
=============================================================

RUN WITH:
    venv\Scripts\streamlit.exe run app/streamlit_dashboard.py
"""

def _rerun():
    """Version-safe rerun — works on Streamlit 1.18+ and 1.27+"""
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

import os
import sys
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────
DATA_DIR    = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, "outputs")
REPORT_DIR  = os.path.join(PROJECT_ROOT, "reports")

CLEAN_PATH     = os.path.join(DATA_DIR,   "retail_sales_clean.csv")
FEATURE_PATH   = os.path.join(DATA_DIR,   "features.csv")
FORECAST_PATH  = os.path.join(OUTPUT_DIR, "forecasts.csv")
INVENTORY_PATH = os.path.join(OUTPUT_DIR, "inventory_report.csv")
ALERT_PATH     = os.path.join(OUTPUT_DIR, "reorder_alerts.csv")
ELASTICITY_PATH= os.path.join(REPORT_DIR, "price_elasticity.csv")
AB_PATH        = os.path.join(REPORT_DIR, "ab_test_results.csv")
ANOMALY_PATH   = os.path.join(REPORT_DIR, "anomaly_report.csv")
CLUSTER_PATH   = os.path.join(REPORT_DIR, "store_clusters.csv")
SEGMENT_PATH   = os.path.join(REPORT_DIR, "product_segments.csv")
STL_PATH       = os.path.join(REPORT_DIR, "stl_decomposition_summary.csv")
WEATHER_PATH   = os.path.join(REPORT_DIR, "weather_impact.csv")

# ── Palette ───────────────────────────────────────────────
ACCENT = "#e94560"
BLUE   = "#00b4d8"
GREEN  = "#06d6a0"
YELLOW = "#ffd166"
PANEL  = "#16213e"
BG     = "#0f1117"

LAYOUT = dict(plot_bgcolor=PANEL, paper_bgcolor=BG,
              font_color="white", margin=dict(l=0, r=0, t=30, b=0))


# ── Cached loaders ────────────────────────────────────────
@st.cache_data
def load_csv(path, **kwargs):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, **kwargs)


# ── Divider helper ────────────────────────────────────────
def section(title: str, icon: str = ""):
    st.markdown(f"""
    <div style="margin-top:40px;margin-bottom:8px;padding:12px 18px;
                background:linear-gradient(90deg,#e94560 0%,#16213e 100%);
                border-radius:8px;">
      <h2 style="color:white;margin:0;font-size:20px;">{icon} {title}</h2>
    </div>""", unsafe_allow_html=True)


def kpi(col, label, value, color=BLUE):
    col.markdown(f"""
    <div style="background:#1e2130;border-radius:10px;padding:18px 12px;
                text-align:center;border-top:3px solid {color};">
      <div style="font-size:26px;font-weight:700;color:{color};">{value}</div>
      <div style="font-size:12px;color:#aaa;margin-top:4px;">{label}</div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="Retail Intelligence Dashboard",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Global CSS ────────────────────────────────────────
    st.markdown("""
    <style>
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0f1117; padding-top: 0px; }
    .block-container { padding-top: 1rem; }
    h1 { color: #e94560 !important; }
    .stDataFrame thead th { background: #1a1a2e !important; color: white; }
    .stTabs [data-baseweb="tab"] { color: #aaa; }
    .stTabs [aria-selected="true"] { color: #e94560 !important; }
    hr { border-color: #2a2a4a; }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap"
          rel="stylesheet">
    """, unsafe_allow_html=True)

    # ── Sidebar – Filters only ────────────────────────────
    with st.sidebar:
        st.markdown("## 🛒 Retail Intelligence")
        st.markdown("*All sections visible below*")
        st.markdown("---")
        st.markdown("**Global Filters**")

        clean_df = load_csv(CLEAN_PATH, parse_dates=["date"])
        if clean_df is not None:
            stores = ["All"] + sorted(clean_df["store_name"].unique().tolist())
            cats   = ["All"] + sorted(clean_df["category"].unique().tolist())
            min_d  = clean_df["date"].min().date()
            max_d  = clean_df["date"].max().date()
        else:
            stores, cats = ["All"], ["All"]
            min_d = max_d = None

        sel_store = st.selectbox("🏪 Store", stores)
        sel_cat   = st.selectbox("🗂 Category", cats)

        if min_d and max_d:
            date_range = st.date_input("📅 Date Range", (min_d, max_d))
        else:
            date_range = None

        st.markdown("---")
        if st.button("🔄 Refresh All Data"):
            st.cache_data.clear()
            _rerun()

        st.markdown("---")
        st.markdown("**Quick Links**")
        st.markdown("""
- [📊 KPIs](#executive-kpis)
- [📈 Sales](#sales-trends)
- [🔮 Forecast](#demand-forecast)
- [📦 Inventory](#inventory-status)
- [🚨 Alerts](#reorder-alerts)
- [💰 Pricing](#price-elasticity)
- [🎯 Promotions](#promotional-analysis)
- [🌦 Weather](#weather-demand)
- [🔍 Anomalies](#anomaly-detection)
- [🔬 Decomposition](#trend-decomposition)
- [🏪 Clustering](#store-clustering)
        """)

    # ── Verify data exists ────────────────────────────────
    if clean_df is None:
        st.error("❌ No data found. Please run `python main.py` first.")
        st.code("venv\\Scripts\\python.exe main.py", language="bash")
        st.stop()

    # Apply filters
    df = clean_df.copy()
    if sel_store != "All":
        df = df[df["store_name"] == sel_store]
    if sel_cat != "All":
        df = df[df["category"] == sel_cat]
    if date_range and len(date_range) == 2:
        df = df[(df["date"] >= pd.Timestamp(date_range[0])) &
                (df["date"] <= pd.Timestamp(date_range[1]))]

    # Load all supporting data
    forecast_df  = load_csv(FORECAST_PATH,  parse_dates=["week_start"])
    inventory_df = load_csv(INVENTORY_PATH)
    _alerts_raw  = load_csv(ALERT_PATH)
    alerts_df    = _alerts_raw if _alerts_raw is not None else pd.DataFrame()
    elasticity_df= load_csv(ELASTICITY_PATH)
    ab_df        = load_csv(AB_PATH)
    anomaly_df   = load_csv(ANOMALY_PATH)
    cluster_df   = load_csv(CLUSTER_PATH)
    segment_df   = load_csv(SEGMENT_PATH)
    stl_df       = load_csv(STL_PATH)
    weather_df   = load_csv(WEATHER_PATH)

    # ── PAGE HEADER ───────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a1a2e,#e94560 150%);
                padding:28px 32px;border-radius:12px;margin-bottom:8px;">
      <h1 style="margin:0;font-size:28px;color:white;">
        🛒 Retail Sales Forecasting & Inventory Optimization
      </h1>
      <p style="margin:6px 0 0;color:#ddd;font-size:14px;">
        End-to-end ML pipeline · 5 Stores · 15 Products · 3 Years ·
        XGBoost R²=0.988 · All 10 Enhancements Active
      </p>
    </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    # SECTION 1 – EXECUTIVE KPIs
    # ══════════════════════════════════════════════════════
    section("Executive KPIs", "📊")
    st.markdown('<a name="executive-kpis"></a>', unsafe_allow_html=True)

    total_rev   = df["revenue"].sum()
    total_units = df["units_sold"].sum()
    n_stockout  = (df["stockout_flag"] == 1).sum() if "stockout_flag" in df.columns else 0
    n_reorder   = (inventory_df["inventory_status"] == "REORDER").sum() \
                   if inventory_df is not None and "inventory_status" in inventory_df.columns else 0
    n_alerts    = len(alerts_df) if not alerts_df.empty else 0
    n_anomalies = len(anomaly_df) if anomaly_df is not None else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpi(c1, "Total Revenue",       f"₹{total_rev/1e6:.1f}M", ACCENT)
    kpi(c2, "Units Sold",          f"{total_units:,.0f}",     BLUE)
    kpi(c3, "Stockout Events",     f"{n_stockout:,}",         YELLOW)
    kpi(c4, "Reorder Alerts",      f"{n_reorder}",            ACCENT)
    kpi(c5, "Anomalies Detected",  f"{n_anomalies}",          "#bd00ff")
    kpi(c6, "Forecast MAPE",       "~8.5%",                   GREEN)

    # ══════════════════════════════════════════════════════
    # SECTION 2 – SALES TRENDS
    # ══════════════════════════════════════════════════════
    section("Sales Trends", "📈")
    st.markdown('<a name="sales-trends"></a>', unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        monthly = (df.groupby(df["date"].dt.to_period("M"))["revenue"]
                     .sum().reset_index())
        monthly["date"] = monthly["date"].astype(str)
        fig = px.area(monthly, x="date", y="revenue",
                      title="Monthly Revenue Trend",
                      labels={"revenue": "Revenue (₹)", "date": "Month"},
                      color_discrete_sequence=[ACCENT])
        fig.update_traces(fillcolor="rgba(233,69,96,0.15)", line_color=ACCENT)
        fig.update_layout(**LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        cat_rev = df.groupby("category")["revenue"].sum().reset_index()
        fig = px.pie(cat_rev, names="category", values="revenue",
                     title="Revenue by Category", hole=0.45,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(**LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        store_rev = (df.groupby("store_name")["revenue"]
                       .sum().sort_values(ascending=False).reset_index())
        
        if len(store_rev) > 50:
            st.markdown(f"<div style='font-size:13px; color:#aaa; margin-bottom:5px;'>Showing Top 50 of {len(store_rev)} Stores</div>", unsafe_allow_html=True)
            store_rev = store_rev.head(50)
        
        store_rev = store_rev.sort_values("revenue", ascending=True)
        
        chart_height = max(350, len(store_rev) * 24)
        
        if len(store_rev) > 1:
            fig = px.bar(store_rev, x="revenue", y="store_name",
                         orientation="h", title="Total Revenue by Store",
                         color="revenue", color_continuous_scale="Viridis",
                         labels={"revenue": "Revenue (₹)"})
        else:
            fig = px.bar(store_rev, x="revenue", y="store_name",
                         orientation="h", title="Total Revenue by Store",
                         color_discrete_sequence=[BLUE],
                         labels={"revenue": "Revenue (₹)"})
            
        fig.update_layout(**LAYOUT, showlegend=False, coloraxis_showscale=False,
                          height=chart_height)
        fig.update_yaxes(autorange="reversed" if len(store_rev) == 1 else True)
        
        # Put inside a scrollable container
        with st.container():
            st.plotly_chart(fig, use_container_width=True)

    with col_r2:
        if "month" in df.columns and "day_of_week" in df.columns:
            heat = (df.groupby(["month", "day_of_week"])["units_sold"]
                      .mean().reset_index()
                      .pivot(index="month", columns="day_of_week",
                             values="units_sold"))
            heat.columns = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            fig = px.imshow(heat, color_continuous_scale="YlOrRd",
                            title="Seasonality — Month × Day of Week",
                            text_auto=".0f")
            fig.update_layout(**LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════
    # SECTION 3 – DEMAND FORECAST
    # ══════════════════════════════════════════════════════
    section("12-Week Demand Forecast", "🔮")
    st.markdown('<a name="demand-forecast"></a>', unsafe_allow_html=True)

    if forecast_df is not None:
        fore = forecast_df.copy()
        if sel_store != "All" and "store_name" in fore.columns:
            fore = fore[fore["store_name"] == sel_store]
        if sel_cat != "All" and "category" in fore.columns:
            fore = fore[fore["category"] == sel_cat]

        col_l, col_r = st.columns([3, 2])
        with col_l:
            cat_fore = (fore.groupby(["week_start", "category"])["predicted_units"]
                            .sum().reset_index())
            fig = px.line(cat_fore, x="week_start", y="predicted_units",
                          color="category", markers=True,
                          title="12-Week Forecast by Category",
                          labels={"predicted_units": "Predicted Units",
                                  "week_start": "Week"})
            fig.update_layout(**LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            top_prods = (fore.groupby("product_name")["predicted_units"]
                             .sum().nlargest(10).reset_index()
                             .sort_values("predicted_units"))
            fig = px.bar(top_prods, x="predicted_units", y="product_name",
                         orientation="h",
                         title="Top 10 Products (12W Total Forecast)",
                         color="predicted_units",
                         color_continuous_scale="Blues",
                         labels={"predicted_units": "Predicted Units"})
            fig.update_layout(**LAYOUT, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Forecast data not available. Run `python main.py` to generate.")

    # ══════════════════════════════════════════════════════
    # SECTION 4 – INVENTORY STATUS
    # ══════════════════════════════════════════════════════
    section("Inventory Status", "📦")
    st.markdown('<a name="inventory-status"></a>', unsafe_allow_html=True)

    if inventory_df is not None:
        inv = inventory_df.copy()
        if sel_store != "All" and "store_name" in inv.columns:
            inv = inv[inv["store_name"] == sel_store]
        if sel_cat != "All" and "category" in inv.columns:
            inv = inv[inv["category"] == sel_cat]

        col_a, col_b, col_c = st.columns([1, 2, 2])
        with col_a:
            status_counts = inv["inventory_status"].value_counts()
            fig = px.pie(values=status_counts.values,
                         names=status_counts.index,
                         title="Inventory Status Mix", hole=0.5,
                         color_discrete_map={"REORDER": ACCENT,
                                             "OK": GREEN,
                                             "OVERSTOCK": YELLOW})
            fig.update_layout(**LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            if "EOQ" in inv.columns and "current_stock" in inv.columns:
                fig = px.scatter(inv, x="EOQ", y="current_stock",
                                 color="inventory_status",
                                 hover_data=["product_name", "store_name"],
                                 title="EOQ vs Current Stock",
                                 color_discrete_map={"REORDER": ACCENT,
                                                     "OK": GREEN,
                                                     "OVERSTOCK": YELLOW})
                fig.update_layout(**LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

        with col_c:
            if "safety_stock" in inv.columns and "reorder_point" in inv.columns:
                cat_inv = inv.groupby("category").agg(
                    avg_safety_stock = ("safety_stock", "mean"),
                    avg_reorder_pt   = ("reorder_point","mean"),
                    avg_current_stock= ("current_stock","mean"),
                ).reset_index()
                fig = go.Figure()
                fig.add_bar(name="Avg Current Stock",
                            x=cat_inv["category"],
                            y=cat_inv["avg_current_stock"],
                            marker_color=BLUE)
                fig.add_bar(name="Avg Reorder Point",
                            x=cat_inv["category"],
                            y=cat_inv["avg_reorder_pt"],
                            marker_color=YELLOW)
                fig.add_bar(name="Avg Safety Stock",
                            x=cat_inv["category"],
                            y=cat_inv["avg_safety_stock"],
                            marker_color=GREEN)
                fig.update_layout(**LAYOUT, barmode="group",
                                  title="Stock Levels by Category",
                                  xaxis_tickangle=-20)
                st.plotly_chart(fig, use_container_width=True)

        disp = ["store_name","product_name","category","current_stock",
                "reorder_point","safety_stock","EOQ","inventory_status"]
        disp = [c for c in disp if c in inv.columns]
        st.dataframe(inv[disp].reset_index(drop=True), use_container_width=True)
    else:
        st.info("Inventory data not available. Run `python main.py` to generate.")

    # ══════════════════════════════════════════════════════
    # SECTION 5 – REORDER ALERTS
    # ══════════════════════════════════════════════════════
    section("Reorder Alerts", "🚨")
    st.markdown('<a name="reorder-alerts"></a>', unsafe_allow_html=True)

    if not alerts_df.empty:
        alrt = alerts_df.copy()
        if sel_store != "All" and "store_name" in alrt.columns:
            alrt = alrt[alrt["store_name"] == sel_store]

        c1, c2, c3 = st.columns(3)
        kpi(c1, "Products to Reorder", str(len(alrt)), ACCENT)
        ord_qty = alrt.get("recommended_order_qty", pd.Series([0])).sum()
        ord_val = alrt.get("estimated_order_cost",  pd.Series([0])).sum()
        kpi(c2, "Total Order Qty",   f"{ord_qty:,.0f}", YELLOW)
        kpi(c3, "Total Order Value", f"₹{ord_val:,.0f}", GREEN)
        st.markdown("<br>", unsafe_allow_html=True)

        col_l, col_r = st.columns(2)
        with col_l:
            if "recommended_order_qty" in alrt.columns:
                top15 = alrt.nlargest(15, "recommended_order_qty")
                fig = px.bar(top15, x="recommended_order_qty",
                             y="product_name", orientation="h",
                             color="estimated_order_cost",
                             color_continuous_scale="Reds",
                             title="Priority Reorder List (Top 15)")
                fig.update_layout(**LAYOUT, yaxis={"autorange": "reversed"},
                                  coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

        with col_r:
            if "category" in alrt.columns:
                cat_alrt = alrt.groupby("category").size().reset_index(name="count")
                fig = px.pie(cat_alrt, names="category", values="count",
                             title="Reorder Alerts by Category", hole=0.4,
                             color_discrete_sequence=px.colors.qualitative.Set1)
                fig.update_layout(**LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

        st.dataframe(alrt.reset_index(drop=True), use_container_width=True)
    else:
        st.success("✅ All inventory levels are healthy — no reorders needed!")

    # ══════════════════════════════════════════════════════
    # SECTION 6 – PRICE ELASTICITY
    # ══════════════════════════════════════════════════════
    section("Price Elasticity Analysis", "💰")
    st.markdown('<a name="price-elasticity"></a>', unsafe_allow_html=True)

    if elasticity_df is not None:
        col_l, col_r = st.columns(2)
        with col_l:
            colors = [ACCENT if e < -1 else YELLOW if e < 0 else GREEN
                      for e in elasticity_df["price_elasticity"]]
            fig = go.Figure(go.Bar(
                x=elasticity_df["price_elasticity"],
                y=elasticity_df["product_name"],
                orientation="h",
                marker_color=colors,
            ))
            fig.add_vline(x=0,  line_color="white", line_dash="dash")
            fig.add_vline(x=-1, line_color=YELLOW,  line_dash="dot",
                          annotation_text="Elastic threshold")
            fig.update_layout(**LAYOUT, title="Price Elasticity by Product",
                              xaxis_title="Elasticity Coefficient")
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown("**Elasticity Interpretation**")
            st.markdown("""
| Elasticity | Label | Pricing action |
|---|---|---|
| < -1.5 | Highly elastic | Lower price → big demand gain |
| -1 to -1.5 | Elastic | Small discount worthwhile |
| -0.5 to -1 | Inelastic | Hold price; margin matters |
| > 0 | Giffen / unusual | Investigate data |
            """)
            st.dataframe(
                elasticity_df[["product_name", "category",
                               "price_elasticity", "r2_score",
                               "interpretation"]].reset_index(drop=True),
                use_container_width=True,
            )
    else:
        st.info("Run `python main.py` to generate price elasticity analysis.")

    # ══════════════════════════════════════════════════════
    # SECTION 7 – PROMOTIONAL ANALYSIS
    # ══════════════════════════════════════════════════════
    section("Promotional Impact & A/B Test", "🎯")
    st.markdown('<a name="promotional-analysis"></a>', unsafe_allow_html=True)

    if ab_df is not None:
        col_l, col_r = st.columns(2)
        with col_l:
            colors = ["#06d6a0" if s else "#444"
                      for s in ab_df.get("significant", [False]*len(ab_df))]
            fig = go.Figure(go.Bar(
                x=ab_df.get("ATE_pct", []),
                y=ab_df.get("product_name", []),
                orientation="h",
                marker_color=colors,
            ))
            fig.add_vline(x=0, line_color="white", line_dash="dash")
            fig.update_layout(**LAYOUT,
                              title="A/B Test — Average Treatment Effect (%)",
                              xaxis_title="ATE (%)")
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            promo_uplift = load_csv(os.path.join(REPORT_DIR, "promo_uplift.csv"))
            if promo_uplift is not None and "uplift_pct" in promo_uplift.columns:
                fig = px.histogram(
                    promo_uplift, x=promo_uplift["uplift_pct"].clip(-50, 100),
                    nbins=40, color_discrete_sequence=[ACCENT],
                    title="Promotion Uplift Distribution",
                    labels={"x": "Uplift (%)"},
                )
                fig.add_vline(x=0, line_color="white", line_dash="dash")
                fig.add_vline(x=promo_uplift["uplift_pct"].mean(),
                              line_color=YELLOW, line_dash="solid",
                              annotation_text=f'Mean={promo_uplift["uplift_pct"].mean():.1f}%')
                fig.update_layout(**LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

        promo_roi = load_csv(os.path.join(REPORT_DIR, "promo_roi.csv"))
        if promo_roi is not None and "ROI_pct" in promo_roi.columns:
            roi_colors = [GREEN if r > 0 else ACCENT for r in promo_roi["ROI_pct"]]
            fig = go.Figure(go.Bar(
                x=promo_roi["category"], y=promo_roi["ROI_pct"],
                marker_color=roi_colors,
            ))
            fig.add_hline(y=0, line_color="white", line_dash="dash")
            fig.update_layout(**LAYOUT, title="Promotional ROI by Category (%)",
                              yaxis_title="ROI (%)")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run `python main.py` to generate promotional analysis.")

    # ══════════════════════════════════════════════════════
    # SECTION 8 – WEATHER DEMAND
    # ══════════════════════════════════════════════════════
    section("Weather-Based Demand Adjustment", "🌦")
    st.markdown('<a name="weather-demand"></a>', unsafe_allow_html=True)

    if weather_df is not None:
        col_l, col_r = st.columns(2)
        with col_l:
            pos = weather_df[weather_df["temp_coefficient"] >= 0]
            neg = weather_df[weather_df["temp_coefficient"] < 0]
            fig = go.Figure()
            fig.add_bar(x=pos["category"], y=pos["temp_coefficient"],
                        name="Positive (hot weather ↑ demand)",
                        marker_color=ACCENT)
            fig.add_bar(x=neg["category"], y=neg["temp_coefficient"],
                        name="Negative (hot weather ↓ demand)",
                        marker_color=BLUE)
            fig.add_hline(y=0, line_color="white", line_dash="dash")
            fig.update_layout(**LAYOUT,
                              title="Temperature Impact on Demand by Category",
                              yaxis_title="Temp Coefficient (units/°C)",
                              xaxis_tickangle=-20, barmode="relative")
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            if "r2_weather" in weather_df.columns:
                fig = px.bar(weather_df.sort_values("r2_weather"),
                             x="r2_weather", y="category", orientation="h",
                             color="r2_weather", color_continuous_scale="Greens",
                             title="R² — How much temperature explains demand",
                             labels={"r2_weather": "R² Score"})
                fig.update_layout(**LAYOUT, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

        st.dataframe(weather_df.reset_index(drop=True), use_container_width=True)
    else:
        st.info("Run `python main.py` to generate weather analysis.")

    # ══════════════════════════════════════════════════════
    # SECTION 9 – ANOMALY DETECTION
    # ══════════════════════════════════════════════════════
    section("Anomaly Detection", "🔍")
    st.markdown('<a name="anomaly-detection"></a>', unsafe_allow_html=True)

    if anomaly_df is not None and not anomaly_df.empty:
        col_a, col_b, col_c = st.columns(3)
        kpi(col_a, "Total Anomalies",       str(len(anomaly_df)),          ACCENT)
        hc = (anomaly_df["anomaly_type"] == "HIGH CONFIDENCE").sum() \
             if "anomaly_type" in anomaly_df.columns else 0
        kpi(col_b, "High Confidence",       str(hc),                       "#bd00ff")
        kpi(col_c, "Detection Methods",     "IsoForest + Z-Score",         BLUE)
        st.markdown("<br>", unsafe_allow_html=True)

        col_l, col_r = st.columns(2)
        with col_l:
            if "anomaly_type" in anomaly_df.columns:
                type_counts = anomaly_df["anomaly_type"].value_counts()
                fig = px.pie(values=type_counts.values,
                             names=type_counts.index,
                             title="Anomaly Type Breakdown", hole=0.4,
                             color_discrete_map={
                                 "HIGH CONFIDENCE": ACCENT,
                                 "Z-Score":         YELLOW,
                                 "IsoForest":        BLUE,
                             })
                fig.update_layout(**LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

        with col_r:
            if "category" in anomaly_df.columns:
                cat_anom = anomaly_df.groupby("category").size().reset_index(name="count")
                fig = px.bar(cat_anom.sort_values("count"),
                             x="count", y="category", orientation="h",
                             color="count", color_continuous_scale="Reds",
                             title="Anomalies by Category")
                fig.update_layout(**LAYOUT, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

        disp_c = [c for c in ["week_start","store_name","product_name","category",
                               "units_sold","expected_units","deviation_units",
                               "anomaly_type","z_score"]
                  if c in anomaly_df.columns]
        st.dataframe(anomaly_df[disp_c].head(50).reset_index(drop=True),
                     use_container_width=True)
    else:
        st.info("Run `python main.py` to generate anomaly detection results.")

    # ══════════════════════════════════════════════════════
    # SECTION 10 – STL DECOMPOSITION
    # ══════════════════════════════════════════════════════
    section("Trend & Seasonality Decomposition (STL)", "🔬")
    st.markdown('<a name="trend-decomposition"></a>', unsafe_allow_html=True)

    if stl_df is not None:
        col_l, col_r = st.columns(2)
        with col_l:
            fig = px.box(stl_df, x="category", y="seasonal_strength",
                         color="category",
                         title="Seasonal Strength by Category",
                         labels={"seasonal_strength": "Seasonal Strength"})
            fig.add_hline(y=0.64, line_color=ACCENT, line_dash="dash",
                          annotation_text="Strong threshold (0.64)")
            fig.update_layout(**LAYOUT, showlegend=False, xaxis_tickangle=-20)
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            fig = px.scatter(stl_df, x="trend_slope", y="seasonal_strength",
                             color="growth_direction",
                             hover_data=["product_name", "store_name"],
                             title="Trend Slope vs Seasonal Strength",
                             color_discrete_map={"Growing": GREEN,
                                                 "Declining": ACCENT},
                             labels={"trend_slope": "Trend Slope (units/wk)"})
            fig.add_vline(x=0, line_color="white", line_dash="dash")
            fig.update_layout(**LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            stl_df[["product_name","store_name","category",
                     "trend_slope","trend_strength",
                     "seasonal_strength","growth_direction"]]
            .sort_values("seasonal_strength", ascending=False)
            .reset_index(drop=True),
            use_container_width=True,
        )
    else:
        st.info("Run `python main.py` to generate STL decomposition.")

    # ══════════════════════════════════════════════════════
    # SECTION 11 – STORE CLUSTERING + PRODUCT SEGMENTS
    # ══════════════════════════════════════════════════════
    section("Store Clustering & Product Segmentation", "🏪")
    st.markdown('<a name="store-clustering"></a>', unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        if cluster_df is not None:
            chart_height = max(400, len(cluster_df) * 22)
            fig = px.bar(cluster_df.sort_values("avg_revenue_per_week"),
                         x="avg_revenue_per_week", y="store_name",
                         color="cluster_label", orientation="h",
                         title="Store Clusters — Avg Weekly Revenue",
                         labels={"avg_revenue_per_week": "Avg Revenue (₹/wk)"})
            fig.update_layout(**LAYOUT, height=chart_height)
            with st.container():
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Store cluster data not available.")

    with col_r:
        if segment_df is not None and "segment_name" in segment_df.columns:
            fig = px.scatter(segment_df,
                             x="avg_demand", y="demand_cv",
                             color="segment_name", size="avg_revenue",
                             hover_data=["product_name", "category"],
                             title="Product Demand Segments",
                             color_discrete_map={
                                 "Star Products":  ACCENT,
                                 "High Velocity":  BLUE,
                                 "Stable Core":    GREEN,
                                 "Slow Movers":    YELLOW,
                             },
                             labels={"avg_demand": "Avg Weekly Demand",
                                     "demand_cv": "Demand CV (Volatility)"})
            fig.update_layout(**LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Product segment data not available.")

    if segment_df is not None:
        seg_summary = (segment_df.groupby("segment_name")
                                  .agg(products=("product_name","count"),
                                       avg_demand=("avg_demand","mean"),
                                       avg_revenue=("avg_revenue","mean"))
                                  .round(1).reset_index())
        st.dataframe(seg_summary, use_container_width=True)

    # ── Footer ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#555;font-size:12px;padding:16px;">
      Retail Sales Forecasting & Inventory Optimization System v2.0 ·
      Built with Python · Scikit-learn · XGBoost · Streamlit · Plotly
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
