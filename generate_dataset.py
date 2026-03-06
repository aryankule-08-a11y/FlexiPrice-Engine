"""
generate_dataset.py
-------------------
Generates a realistic synthetic e-commerce dataset for dynamic pricing.
Run this file once to produce `dataset.csv`.
"""

import pandas as pd
import numpy as np

def generate_dataset(n_rows: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Create a synthetic dataset with features relevant to dynamic pricing.

    Columns
    -------
    product_id        : Unique product identifier
    product_category  : Category of the product
    base_price        : Original listed price ($)
    competitor_price  : Current competitor price ($)
    demand_level      : Demand index (0–100)
    inventory_level   : Units in stock
    customer_traffic  : Daily visitor count
    time_of_day       : Hour bucket label (Morning / Afternoon / Evening / Night)
    day_of_week       : Day name (Monday–Sunday)
    season            : Season label
    is_peak           : 1 if peak period else 0
    discount_pct      : Applied discount percentage
    units_sold        : Units sold that day
    revenue           : Revenue generated ($)
    optimal_price     : Target variable — the "best" price (simulated)
    """
    rng = np.random.default_rng(seed)

    # Product categories and their base-price ranges
    categories = {
        "Electronics":    (200, 1200),
        "Fashion":        (20,  250),
        "Home & Kitchen": (30,  500),
        "Sports":         (25,  400),
        "Books":          (5,   60),
        "Beauty":         (10,  150),
    }

    cat_names = list(categories.keys())
    cat_choice = rng.choice(cat_names, size=n_rows)

    # Base prices drawn from category-specific ranges
    base_prices = np.array([
        rng.uniform(*categories[c]) for c in cat_choice
    ])

    # Competitor prices: base ± random noise
    competitor_prices = base_prices * rng.uniform(0.85, 1.15, size=n_rows)

    # Demand index 0–100
    demand = rng.integers(5, 100, size=n_rows).astype(float)

    # Inventory
    inventory = rng.integers(1, 500, size=n_rows).astype(float)

    # Customer traffic
    traffic = rng.integers(50, 5000, size=n_rows).astype(float)

    # Time of day
    tod_labels = ["Morning", "Afternoon", "Evening", "Night"]
    time_of_day = rng.choice(tod_labels, size=n_rows)

    # Day of week
    days = ["Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"]
    day_of_week = rng.choice(days, size=n_rows)

    # Season
    seasons = ["Spring", "Summer", "Autumn", "Winter"]
    season = rng.choice(seasons, size=n_rows)

    # Peak flag — weekends + evenings + holiday seasons
    is_peak = (
        np.isin(day_of_week, ["Saturday", "Sunday"]).astype(int) |
        np.isin(time_of_day, ["Evening"]).astype(int) |
        np.isin(season, ["Winter", "Summer"]).astype(int)
    )

    # Discount percentage
    discount_pct = rng.uniform(0, 35, size=n_rows)

    # --- Simulate optimal price using a realistic formula ---
    # Higher demand → higher price
    # Lower inventory → higher price (scarcity)
    # Higher traffic → higher price
    # Peak → price premium
    # Competitor undercut → constrained
    demand_factor    = 1 + 0.003 * demand           # up to +30 %
    scarcity_factor  = 1 + 0.0008 * (500 - inventory) # up to +40 %
    traffic_factor   = 1 + 0.00005 * traffic         # up to +25 %
    peak_factor      = 1 + 0.08 * is_peak            # +8 % during peak
    comp_factor      = 0.5 + 0.5 * (competitor_prices / base_prices)

    optimal_price = (
        base_prices
        * demand_factor
        * scarcity_factor
        * traffic_factor
        * peak_factor
        * comp_factor
    )
    # Add small noise
    optimal_price *= rng.normal(1, 0.03, size=n_rows)
    optimal_price = np.round(np.clip(optimal_price, 1, None), 2)

    # Units sold — inversely related to optimal price ratio
    price_ratio = optimal_price / base_prices
    units_sold = np.round(
        demand * (1.2 - 0.3 * price_ratio) * rng.uniform(0.8, 1.2, size=n_rows)
    ).astype(int)
    units_sold = np.clip(units_sold, 0, None)

    # Revenue
    revenue = np.round(optimal_price * units_sold, 2)

    # Product IDs
    product_ids = [f"PROD-{str(i).zfill(5)}" for i in range(1, n_rows + 1)]

    df = pd.DataFrame({
        "product_id":       product_ids,
        "product_category": cat_choice,
        "base_price":       np.round(base_prices, 2),
        "competitor_price": np.round(competitor_prices, 2),
        "demand_level":     demand,
        "inventory_level":  inventory,
        "customer_traffic": traffic,
        "time_of_day":      time_of_day,
        "day_of_week":      day_of_week,
        "season":           season,
        "is_peak":          is_peak,
        "discount_pct":     np.round(discount_pct, 2),
        "units_sold":       units_sold,
        "revenue":          revenue,
        "optimal_price":    optimal_price,
    })
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("dataset.csv", index=False)
    print(f"[OK] Dataset generated: {df.shape[0]} rows x {df.shape[1]} columns")
    print(df.head())
