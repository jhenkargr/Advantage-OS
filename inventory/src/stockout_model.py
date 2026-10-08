"""
Stockout Risk Classifier — AdvantAge OS Inventory Module
========================================================

Purpose:
    Predict whether a product is likely to stock out (stock reaches 0)
    within the next 3 days, using historical sales data.

Model:
    LogisticRegression (scikit-learn) with StandardScaler pipeline.

    NOTE: The original design called for GradientBoostingClassifier, but
    sklearn's tree-based C extensions (_splitter.pyd) are currently blocked
    by a Windows Application Control / DLL loading policy on this machine.
    LogisticRegression uses only pure-Python / BLAS backends that are
    unaffected. When the policy issue is resolved, swap in:
        from sklearn.ensemble import GradientBoostingClassifier
    with minimal code changes.

Target:
    stockout_within_3d  (1 = stockout likely, 0 = safe)

Features (fixed order):
    1. avg_daily_sales_7d   — rolling 7-day mean of units_sold
    2. sales_std_7d         — rolling 7-day std of units_sold
    3. current_stock        — stock_after on the observation day
    4. trend_score          — trend boost indicator from sales_log

Data split:
    Chronological — first 80 % of dates for training, last 20 % for testing.
    This prevents future information from leaking into training.

NOTE:
    This is a prototype trained on synthetic data. It is NOT production-ready.
    Future improvements include additional lag features, holiday/promo flags,
    and cross-validation on a larger, real-world dataset.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

# ---------------------------------------------------------------------------
# Paths (relative to project root: C:\major project)
# ---------------------------------------------------------------------------
SALES_PATH = "inventory/data/sales_log.csv"
CATALOG_PATH = "inventory/data/product_catalog.csv"
MODEL_SAVE_PATH = "inventory/models/stockout_model.pkl"

# Feature column order — MUST be identical at training and prediction time.
FEATURE_COLS = [
    "avg_daily_sales_7d",
    "sales_std_7d",
    "current_stock",
    "trend_score",
]

STOCKOUT_HORIZON_DAYS = 3  # predict stockout within next N days


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
    catalog = pd.read_csv(CATALOG_PATH)

    required_sales_cols = ["product_id", "date", "units_sold",
                           "stock_after", "trend_score"]
    required_catalog_cols = ["product_id", "product_name", "category",
                             "current_stock", "low_stock_threshold"]

    for col in required_sales_cols:
        assert col in sales.columns, f"Missing column in sales_log: {col}"
    for col in required_catalog_cols:
        assert col in catalog.columns, f"Missing column in catalog: {col}"

    assert sales.isnull().sum().sum() == 0, "Unexpected nulls in sales_log"

    sales["date"] = pd.to_datetime(sales["date"])

    n_products = sales["product_id"].nunique()
    n_rows = len(sales)
    date_min = sales["date"].min().strftime("%Y-%m-%d")
    date_max = sales["date"].max().strftime("%Y-%m-%d")

    print(f"  OK: {n_rows} rows, {n_products} products")
    print(f"  OK: Date range {date_min} to {date_max}")
    print(f"  OK: No missing values")
    print(f"  OK: Catalog has {len(catalog)} products")
    print("=" * 60)
    print()

    return sales


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def build_features(sales: pd.DataFrame) -> pd.DataFrame:
    """
    For each product-day row, compute rolling features and the binary
    stockout target.  Only uses past/present information for features.
    """
    sales = sales.copy()
    sales["date"] = pd.to_datetime(sales["date"])
    sales.sort_values(["product_id", "date"], inplace=True)

    frames = []
    for pid, grp in sales.groupby("product_id"):
        grp = grp.sort_values("date").reset_index(drop=True)

        # Rolling features — strictly backward-looking (min_periods=7)
        grp["avg_daily_sales_7d"] = (
            grp["units_sold"].rolling(7, min_periods=7).mean()
        )
        grp["sales_std_7d"] = (
            grp["units_sold"].rolling(7, min_periods=7).std()
        )

        # Current stock available at observation time
        grp["current_stock"] = grp["stock_after"]

        # Target: will stock reach 0 within next STOCKOUT_HORIZON_DAYS days?
        # We look ahead at stock_after for the next N days.
        stockout_flags = []
        for i in range(len(grp)):
            future_window = grp["stock_after"].iloc[i + 1: i + 1 + STOCKOUT_HORIZON_DAYS]
            if len(future_window) < STOCKOUT_HORIZON_DAYS:
                stockout_flags.append(np.nan)  # not enough future data
            elif (future_window <= 0).any():
                stockout_flags.append(1)
            else:
                stockout_flags.append(0)
        grp["stockout_within_3d"] = stockout_flags

        frames.append(grp)

    df = pd.concat(frames, ignore_index=True)

    # Drop rows where rolling features or target couldn't be computed
    df.dropna(subset=FEATURE_COLS + ["stockout_within_3d"], inplace=True)
    df["stockout_within_3d"] = df["stockout_within_3d"].astype(int)

    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_stockout_model():
    """Train a LogisticRegression pipeline for stockout prediction."""
    sales = validate_data()
    df = build_features(sales)

    print("STOCKOUT RISK CLASSIFIER — TRAINING")
    print("=" * 60)
    print(f"Model: LogisticRegression (with StandardScaler)")
    print(f"Total usable samples: {len(df)}")
    print(f"Class distribution:\n{df['stockout_within_3d'].value_counts().to_string()}")
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
    y_train = train_df["stockout_within_3d"]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["stockout_within_3d"]

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples:     {len(X_test)}")
    print(f"Train date range: {min(train_dates).strftime('%Y-%m-%d')} - "
          f"{max(train_dates).strftime('%Y-%m-%d')}")
    print(f"Test date range:  {min(test_dates).strftime('%Y-%m-%d')} - "
          f"{max(test_dates).strftime('%Y-%m-%d')}")
    print()

    # ------------------------------------------------------------------
    # Model — Pipeline: StandardScaler → LogisticRegression
    # When tree DLL issue is resolved, replace with:
    #   GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    # ------------------------------------------------------------------
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",   # helps with imbalanced stockout class
            random_state=42,
        )),
    ])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    print("CLASSIFICATION REPORT")
    print("-" * 60)
    print(classification_report(y_test, y_pred, zero_division=0))

    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"Stockout-class (1) - Precision: {prec:.4f}  "
          f"Recall: {rec:.4f}  F1: {f1:.4f}")
    print()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_SAVE_PATH)
    print(f"Model saved -> {MODEL_SAVE_PATH}")
    print("=" * 60)

    return clf


# ---------------------------------------------------------------------------
# Reusable prediction function
# ---------------------------------------------------------------------------
def predict_stockout(current_stock: float,
                     avg_daily_sales_7d: float,
                     sales_std_7d: float,
                     trend_score: float) -> dict:
    """
    Load the trained stockout model and predict for a single observation.

    Parameters
    ----------
    current_stock : float
    avg_daily_sales_7d : float
    sales_std_7d : float
    trend_score : float

    Returns
    -------
    dict with keys:
        stockout_prediction  — 0 or 1
        stockout_probability — probability of class 1
    """
    clf = joblib.load(MODEL_SAVE_PATH)

    features = pd.DataFrame(
        [[avg_daily_sales_7d, sales_std_7d, current_stock, trend_score]],
        columns=FEATURE_COLS,
    )

    pred = int(clf.predict(features)[0])
    proba = float(clf.predict_proba(features)[0][1])

    return {
        "stockout_prediction": pred,
        "stockout_probability": round(proba, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_stockout_model()
