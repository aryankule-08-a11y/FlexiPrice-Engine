"""
model.py
--------
Machine-learning module for the Dynamic Price Optimizer.

Models
------
1. Random Forest Regressor  (primary — non-linear pricing patterns)
2. Linear Regression         (secondary — baseline comparison)

Both models predict the **optimal_price** given features such as
demand, inventory, competitor price, traffic, season, and time.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from typing import Tuple, Dict, Any


# ── Feature Engineering ─────────────────────────────────────────────

# The columns we will use as model inputs
FEATURE_COLS = [
    "base_price",
    "competitor_price",
    "demand_level",
    "inventory_level",
    "customer_traffic",
    "is_peak",
    "discount_pct",
    "time_of_day_enc",
    "day_of_week_enc",
    "season_enc",
    "product_category_enc",
    "price_ratio",           # competitor / base price
    "demand_inv_ratio",      # demand / inventory
]

TARGET_COL = "optimal_price"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns and engineer derived features.
    Returns a copy of the dataframe with new columns.
    """
    df = df.copy()

    # Label-encode categoricals
    le_map: Dict[str, LabelEncoder] = {}
    for col in ["time_of_day", "day_of_week", "season", "product_category"]:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
        le_map[col] = le

    # Derived features
    df["price_ratio"] = df["competitor_price"] / df["base_price"].replace(0, 1)
    df["demand_inv_ratio"] = df["demand_level"] / df["inventory_level"].replace(0, 1)

    return df, le_map


# ── Model Training ──────────────────────────────────────────────────

def train_models(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[RandomForestRegressor, LinearRegression, Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Train both models and return them along with evaluation metrics.

    Returns
    -------
    rf_model      : Fitted RandomForestRegressor
    lr_model      : Fitted LinearRegression
    metrics       : Dict of evaluation metrics for both models
    le_map        : Dict of LabelEncoders for categorical columns
    X_test, y_test: Test split for downstream visualisation
    """
    df_feat, le_map = prepare_features(df)

    X = df_feat[FEATURE_COLS]
    y = df_feat[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # ---- Random Forest ----
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    # ---- Linear Regression ----
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)

    # ---- Metrics ----
    metrics = {
        "rf": {
            "MAE":  round(mean_absolute_error(y_test, rf_preds), 2),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, rf_preds)), 2),
            "R2":   round(r2_score(y_test, rf_preds), 4),
        },
        "lr": {
            "MAE":  round(mean_absolute_error(y_test, lr_preds), 2),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, lr_preds)), 2),
            "R2":   round(r2_score(y_test, lr_preds), 4),
        },
    }

    # Feature importances (Random Forest)
    fi = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Importance": rf.feature_importances_,
    }).sort_values("Importance", ascending=False)
    metrics["feature_importance"] = fi

    # Predictions for test set (used in charts)
    test_results = pd.DataFrame({
        "Actual": y_test.values,
        "RF_Predicted": rf_preds,
        "LR_Predicted": lr_preds,
    })

    return rf, lr, metrics, le_map, test_results


# ── Single-Row Prediction ──────────────────────────────────────────

def predict_price(
    model,
    base_price: float,
    competitor_price: float,
    demand_level: float,
    inventory_level: float,
    customer_traffic: float,
    is_peak: int,
    discount_pct: float,
    time_of_day: str,
    day_of_week: str,
    season: str,
    product_category: str,
    le_map: Dict[str, Any],
) -> float:
    """
    Predict the optimal price for a single observation.
    Uses the same feature engineering pipeline as training.
    """
    # Encode categoricals safely (unseen labels default to 0)
    def safe_encode(le: Any, value: str) -> int:
        try:
            return int(le.transform([value])[0])
        except ValueError:
            return 0

    tod_enc  = safe_encode(le_map["time_of_day"], time_of_day)
    dow_enc  = safe_encode(le_map["day_of_week"], day_of_week)
    sea_enc  = safe_encode(le_map["season"], season)
    cat_enc  = safe_encode(le_map["product_category"], product_category)

    price_ratio     = competitor_price / max(base_price, 1)
    demand_inv_ratio = demand_level / max(inventory_level, 1)

    row = pd.DataFrame([{
        "base_price":          base_price,
        "competitor_price":    competitor_price,
        "demand_level":        demand_level,
        "inventory_level":     inventory_level,
        "customer_traffic":    customer_traffic,
        "is_peak":             is_peak,
        "discount_pct":        discount_pct,
        "time_of_day_enc":     tod_enc,
        "day_of_week_enc":     dow_enc,
        "season_enc":          sea_enc,
        "product_category_enc": cat_enc,
        "price_ratio":         price_ratio,
        "demand_inv_ratio":    demand_inv_ratio,
    }])

    prediction = model.predict(row)[0]
    return round(max(prediction, 0), 2)
