"""
Demand Prediction Regressor — AdvantAge OS Inventory Module
===========================================================

Purpose:
    Predict the total number of units a product will sell during the
    NEXT 7 DAYS, using historical sales data.

Model:
    MLPRegressor (scikit-learn) with StandardScaler pipeline.

    NOTE: The original design called for GradientBoostingRegressor, but
    sklearn's tree-based C extensions (_splitter.pyd) are currently blocked
    by a Windows Application Control / DLL loading policy on this machine.
    MLPRegressor uses only pure-Python / BLAS backends that are unaffected
    and provides non-linear capacity that Ridge lacks.
    When the policy issue is resolved, GradientBoosting can be tested as
    an alternative.

Target:
    future_demand_7d -- sum of potential_demand (true underlying demand,
    unconstrained by stock) over the next 7 calendar days

Features (fixed order):
    1. avg_daily_sales_7d   — rolling 7-day mean of units_sold
    2. sales_std_7d         — rolling 7-day std of units_sold
    3. current_stock        — stock_after on the observation day
    4. trend_score          — trend boost indicator from sales_log

Data split:
    Chronological — first 80 % of dates for training, last 20 % for testing.

NOTE:
    Prototype trained on synthetic data. NOT production-ready.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Paths (relative to project root: C:\major project)
# ---------------------------------------------------------------------------
SALES_PATH = "inventory/data/sales_log.csv"
CATALOG_PATH = "inventory/data/product_catalog.csv"
MODEL_SAVE_PATH = "inventory/models/demand_regressor.pkl"

# Feature column order — MUST be identical at training and prediction time.
FEATURE_COLS = [
    "avg_daily_sales_7d",
    "sales_std_7d",
    "current_stock",
    "trend_score",
]

DEMAND_HORIZON_DAYS = 7  # predict total sales over next N days
HISTORY_WINDOW = 7       # days of history required for rolling features


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------
def validate_data():
    """Check that required files and columns exist."""
    print("=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)

    for path, label in [(SALES_PATH, "sales_log.csv"),
                        (CATALOG_PATH, "product_catalog.csv")]:
        if not os.path.exists(path):
            print(f"FAIL: {label} not found at {path}")
            sys.exit(1)
        print(f"  OK: {label} found")

    sales = pd.read_csv(SALES_PATH)

    required_cols = ["product_id", "date", "potential_demand",
                     "units_sold", "stock_after", "trend_score"]
    for col in required_cols:
        assert col in sales.columns, f"Missing column in sales_log: {col}"

    assert sales.isnull().sum().sum() == 0, "Unexpected nulls in sales_log"

    sales["date"] = pd.to_datetime(sales["date"])

    n_products = sales["product_id"].nunique()
    n_rows = len(sales)
    date_min = sales["date"].min().strftime("%Y-%m-%d")
    date_max = sales["date"].max().strftime("%Y-%m-%d")

    print(f"  OK: {n_rows} rows, {n_products} products")
    print(f"  OK: Date range {date_min} to {date_max}")
    print(f"  OK: No missing values")
    print("=" * 60)
    print()

    return sales


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def build_features(sales: pd.DataFrame) -> pd.DataFrame:
    """
    For each product, create sliding-window samples:
      - Features from the previous 7 days (backward-looking only)
      - Target = sum of potential_demand over the NEXT 7 days

    No target leakage: input features use only past/present observed data
    (units_sold, stock_after). The target uses future potential_demand
    which represents true underlying demand unconstrained by stock.
    """
    sales = sales.copy()
    sales["date"] = pd.to_datetime(sales["date"])
    sales.sort_values(["product_id", "date"], inplace=True)

    frames = []
    for pid, grp in sales.groupby("product_id"):
        grp = grp.sort_values("date").reset_index(drop=True)

        # Rolling features -- strictly backward-looking (min_periods=7)
        # Uses observed units_sold (stock-constrained), which is the
        # information actually available to the inventory system.
        grp["avg_daily_sales_7d"] = (
            grp["units_sold"].rolling(HISTORY_WINDOW, min_periods=HISTORY_WINDOW).mean()
        )
        grp["sales_std_7d"] = (
            grp["units_sold"].rolling(HISTORY_WINDOW, min_periods=HISTORY_WINDOW).std()
        )

        # Current stock at observation time
        grp["current_stock"] = grp["stock_after"]

        # Target: sum of POTENTIAL_DEMAND over the next DEMAND_HORIZON_DAYS
        # This is the true underlying demand, not stock-limited sales.
        future_demand = []
        for i in range(len(grp)):
            future_window = grp["potential_demand"].iloc[i + 1: i + 1 + DEMAND_HORIZON_DAYS]
            if len(future_window) < DEMAND_HORIZON_DAYS:
                future_demand.append(np.nan)  # not enough future data
            else:
                future_demand.append(future_window.sum())
        grp["future_demand_7d"] = future_demand

        frames.append(grp)

    df = pd.concat(frames, ignore_index=True)

    # Drop rows where features or target couldn't be computed
    df.dropna(subset=FEATURE_COLS + ["future_demand_7d"], inplace=True)
    df["future_demand_7d"] = df["future_demand_7d"].astype(int)

    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_demand_model():
    """Train an MLPRegressor pipeline for 7-day demand prediction."""
    sales = validate_data()
    df = build_features(sales)

    print("DEMAND PREDICTION REGRESSOR - TRAINING")
    print("=" * 60)
    print(f"Model: MLPRegressor (with StandardScaler)")
    print(f"Total usable samples: {len(df)}")
    print(f"Target stats:\n{df['future_demand_7d'].describe().to_string()}")
    print()

    # ------------------------------------------------------------------
    # Chronological split — first 80 % of dates → train, last 20 % → test
    # ------------------------------------------------------------------
    unique_dates = sorted(df["date"].unique())
    split_idx = int(len(unique_dates) * 0.8)
    train_dates = set(unique_dates[:split_idx])
    test_dates = set(unique_dates[split_idx:])

    train_df = df[df["date"].isin(train_dates)]
    test_df = df[df["date"].isin(test_dates)]

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["future_demand_7d"]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["future_demand_7d"]

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples:     {len(X_test)}")
    print(f"Train date range: {min(train_dates).strftime('%Y-%m-%d')} - "
          f"{max(train_dates).strftime('%Y-%m-%d')}")
    print(f"Test date range:  {min(test_dates).strftime('%Y-%m-%d')} - "
          f"{max(test_dates).strftime('%Y-%m-%d')}")
    print()

    # ------------------------------------------------------------------
    # Model -- Pipeline: StandardScaler -> MLPRegressor
    # When tree DLL issue is resolved, GradientBoosting can be tested.
    # ------------------------------------------------------------------
    reg = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=5000,
            alpha=0.0001,
            early_stopping=False,
            random_state=42,
        )),
    ])
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("REGRESSION METRICS")
    print("-" * 60)
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")
    print()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(reg, MODEL_SAVE_PATH)
    print(f"Model saved -> {MODEL_SAVE_PATH}")
    print("=" * 60)

    return reg


# ---------------------------------------------------------------------------
# Reusable prediction function
# ---------------------------------------------------------------------------
def predict_future_demand(current_stock: float,
                          avg_daily_sales_7d: float,
                          sales_std_7d: float,
                          trend_score: float) -> int:
    """
    Load the trained demand model and predict 7-day demand for a single
    observation.

    Parameters
    ----------
    current_stock : float
    avg_daily_sales_7d : float
    sales_std_7d : float
    trend_score : float

    Returns
    -------
    int — predicted total units sold over the next 7 days (>= 0)
    """
    reg = joblib.load(MODEL_SAVE_PATH)

    features = pd.DataFrame(
        [[avg_daily_sales_7d, sales_std_7d, current_stock, trend_score]],
        columns=FEATURE_COLS,
    )

    raw_pred = reg.predict(features)[0]
    prediction = max(0, round(raw_pred))

    return int(prediction)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_demand_model()
