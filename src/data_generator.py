"""
=============================================================
  Retail Sales Forecasting & Inventory Optimization System
  Module: data_generator.py
  Purpose: Generate a realistic synthetic retail sales dataset
           covering 3 years, multiple stores, products, and
           categories — with embedded seasonality, trends,
           promotions, and noise.
=============================================================
"""

import pandas as pd
import numpy as np
import os

# ── Reproducibility ───────────────────────────────────────
np.random.seed(42)

# ── Configuration ─────────────────────────────────────────
START_DATE  = "2023-01-01"
END_DATE    = "2026-04-14"
DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "retail_sales_data.csv")

# ── Master reference data ─────────────────────────────────
STATES_AND_CITIES = {
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik"],
    "Delhi": ["New Delhi", "Dwarka", "Rohini", "Saket"],
    "Karnataka": ["Bengaluru", "Mysuru", "Mangaluru"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Noida", "Agra"],
    "West Bengal": ["Kolkata", "Howrah", "Siliguri"],
    "Telangana": ["Hyderabad", "Warangal"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur"],
    "Kerala": ["Kochi", "Thiruvananthapuram", "Kozhikode"]
}

STORES = {}
# Generate exactly 100 stores programmatically
_store_counter = 1
for state, cities in STATES_AND_CITIES.items():
    # 10 states, 10 stores per state = 100 total stores
    for city in cities:
        stores_per_city = 10 // len(cities)
        # handle remainder so it perfectly loops
        stores_to_make = stores_per_city + (1 if _store_counter % 2 == 0 and len(cities) % 2 != 0 else 0)
        # force 10 stores per state simply:
        pass
    
    # Simple forced distribution: 10 per state evenly distributed across its cities
    for i in range(10):
        city = cities[i % len(cities)]
        store_id = f"S{_store_counter:03d}"
        STORES[store_id] = (f"{city} Branch {i+1}", state)
        _store_counter += 1

PRODUCTS = [
    # (product_id, name, category, base_price, base_demand, lead_time_days, holding_cost_pct)
    ("P001", "Whole Milk 1L",        "Dairy",       55,  80,  2, 0.25),
    ("P002", "Basmati Rice 5Kg",     "Staples",     320, 50,  5, 0.20),
    ("P003", "Sunflower Oil 1L",     "Staples",     130, 60,  4, 0.20),
    ("P004", "Colgate Toothpaste",   "Personal Care",85, 40,  7, 0.15),
    ("P005", "Lay's Chips 100g",     "Snacks",      20,  120, 3, 0.30),
    ("P006", "Parle-G Biscuits",     "Snacks",      10,  200, 3, 0.30),
    ("P007", "Dove Soap 3-Pack",     "Personal Care",150, 35, 7, 0.15),
    ("P008", "Tata Tea Premium 500g","Beverages",   250, 45,  5, 0.20),
    ("P009", "Coca-Cola 2L",         "Beverages",   90,  75,  3, 0.25),
    ("P010", "Maggi Noodles 12-Pack","Ready Meals", 148, 90,  4, 0.20),
    ("P011", "Amul Butter 500g",     "Dairy",       280, 30,  2, 0.25),
    ("P012", "Dettol Hand Wash 500ml","Personal Care",120,50, 7, 0.15),
    ("P013", "Kurkure Snacks 200g",  "Snacks",      30,  110, 3, 0.30),
    ("P014", "Haldiram Namkeen 400g","Snacks",      80,  70,  3, 0.30),
    ("P015", "Rin Detergent 1Kg",    "Household",   80,  55,  6, 0.18),
]

# Indian festival months (high-demand boosts)
FESTIVAL_BOOSTS = {
    10: 1.35,   # Dussehra / Navratri
    11: 1.40,   # Diwali season
    12: 1.20,   # Christmas / New Year
    1:  1.15,   # New Year tail
    8:  1.10,   # Raksha Bandhan / Independence Day
    4:  1.10,   # Gudi Padwa / Baisakhi
}

# Category-level seasonal multipliers per month (1 = normal)
CATEGORY_SEASON = {
    "Dairy":        [1.0, 0.95, 0.95, 1.0, 1.05, 1.1, 1.1, 1.05, 1.0, 1.0, 1.05, 1.1],
    "Staples":      [1.05, 1.0, 1.0, 1.05, 1.0, 1.0, 1.0, 1.0, 1.0, 1.05, 1.1, 1.15],
    "Personal Care":[1.0, 1.0, 1.05, 1.05, 1.0, 0.95, 0.95, 1.0, 1.05, 1.1, 1.15, 1.05],
    "Snacks":       [1.0, 0.95, 0.95, 1.0, 1.0, 1.05, 1.1, 1.1, 1.05, 1.15, 1.2, 1.15],
    "Beverages":    [0.95, 0.95, 1.0, 1.1, 1.2, 1.35, 1.35, 1.25, 1.1, 1.0, 0.95, 1.0],
    "Ready Meals":  [1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.15, 1.0, 1.05, 1.1, 1.1],
    "Household":    [1.05, 1.0, 1.05, 1.05, 1.0, 0.95, 0.95, 1.0, 1.0, 1.05, 1.15, 1.1],
}


def generate_dataset() -> pd.DataFrame:
    """Generate synthetic retail sales data and return as DataFrame."""
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    records = []

    for date in dates:
        month       = date.month
        day_of_week = date.dayofweek        # 0 = Monday
        year        = date.year
        week_num    = date.isocalendar()[1]

        # Yearly growth trend (small uplift each year)
        year_growth = 1.0 + 0.07 * (year - 2021)

        # Weekend boost: Fri=1.15, Sat=1.25, Sun=1.20
        weekend_mult = {4: 1.15, 5: 1.25, 6: 1.20}.get(day_of_week, 1.0)

        # Festival boost
        festival_mult = FESTIVAL_BOOSTS.get(month, 1.0)

        # Promotion day (~12% of days)
        is_promo    = 1 if np.random.random() < 0.12 else 0
        promo_boost = 1.25 if is_promo else 1.0

        for pid, pname, category, base_price, base_demand, lead_time, holding_cost in PRODUCTS:
            for sid, (store_name, state) in STORES.items():

                # Base demand with seasonal adjustment
                cat_mult   = CATEGORY_SEASON[category][month - 1]
                store_mult = np.random.uniform(0.85, 1.15)   # store-level variation

                # Compute expected units sold
                expected = (base_demand
                            * cat_mult
                            * weekend_mult
                            * festival_mult
                            * promo_boost
                            * year_growth
                            * store_mult)

                # Add stochastic Poisson noise
                units_sold = max(0, int(np.random.poisson(expected)))

                # Simulate occasional stockout (0=available, 1=stockout)
                stockout_flag = 1 if np.random.random() < 0.04 else 0
                if stockout_flag:
                    units_sold = 0

                # Price with small random discount on promo days
                discount = np.random.uniform(0.05, 0.20) if is_promo else 0.0
                sale_price = round(base_price * (1 - discount), 2)

                # Revenue
                revenue = round(units_sold * sale_price, 2)

                records.append({
                    "date":           date,
                    "store_id":       sid,
                    "store_name":     store_name,
                    "state":          state,
                    "product_id":     pid,
                    "product_name":   pname,
                    "category":       category,
                    "units_sold":     units_sold,
                    "unit_price":     sale_price,
                    "revenue":        revenue,
                    "is_promotion":   is_promo,
                    "is_weekend":     1 if day_of_week >= 5 else 0,
                    "month":          month,
                    "year":           year,
                    "day_of_week":    day_of_week,
                    "week_number":    week_num,
                    "stockout_flag":  stockout_flag,
                    "lead_time_days": lead_time,
                    "holding_cost_pct": holding_cost,
                    "base_price":     base_price,
                })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


def save_dataset(df: pd.DataFrame, path: str = DATA_PATH) -> str:
    """Save the DataFrame to CSV and return the path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[✓] Dataset saved → {path}")
    print(f"    Rows   : {len(df):,}")
    print(f"    Columns: {len(df.columns)}")
    print(f"    Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    return path


if __name__ == "__main__":
    print("=" * 60)
    print("  Generating Retail Sales Dataset …")
    print("=" * 60)
    df = generate_dataset()
    save_dataset(df)
    print("\nSample rows:")
    print(df.head(3).to_string(index=False))
