"""
Inventory Orchestrator - AdvantAge OS
=====================================

Phase-1 inventory decision/orchestration layer.

Pipeline:
    product_id
        -> read catalog + sales history
        -> calculate latest features
        -> call stockout classifier
        -> call demand model
        -> combine results
        -> calculate safety stock
        -> calculate recommended reorder quantity
        -> produce inventory recommendation

Usage:
    python inventory/src/orchestrator.py              # all products
    python inventory/src/orchestrator.py SKU001       # single product

Works from any working directory (paths are resolved relative to this file).

Does NOT retrain models. Does NOT modify data files or model .pkl files.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Paths — resolved relative to THIS file, not the working directory
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # inventory/

CATALOG_PATH = BASE_DIR / "data" / "product_catalog.csv"
SALES_LOG_PATH = BASE_DIR / "data" / "sales_log.csv"
STOCKOUT_MODEL_PATH = BASE_DIR / "models" / "stockout_model.pkl"
DEMAND_MODEL_PATH = BASE_DIR / "models" / "demand_regressor.pkl"

# Feature column order — MUST match training order exactly.
FEATURE_COLS = [
    "avg_daily_sales_7d",
    "sales_std_7d",
    "current_stock",
    "trend_score",
]

# Simple Phase-1 replenishment parameters.
# This is a baseline business rule, NOT an advanced inventory optimization.
SAFETY_STOCK_DAYS = 2
STOCKOUT_PROB_THRESHOLD = 0.50


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _validate_paths():
    """Check that all required data and model files exist."""
    for path, label in [
        (CATALOG_PATH, "product_catalog.csv"),
        (SALES_LOG_PATH, "sales_log.csv"),
        (STOCKOUT_MODEL_PATH, "stockout_model.pkl"),
        (DEMAND_MODEL_PATH, "demand_regressor.pkl"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found at {path}")


def _load_catalog() -> pd.DataFrame:
    """Load and validate product catalog."""
    catalog = pd.read_csv(CATALOG_PATH)
    required = ["product_id", "product_name", "category",
                "current_stock", "price"]
    missing = [c for c in required if c not in catalog.columns]
    if missing:
        raise ValueError(f"Catalog missing columns: {missing}")
    return catalog


def _load_sales_log() -> pd.DataFrame:
    """Load and validate sales log."""
    sales = pd.read_csv(SALES_LOG_PATH)
    required = ["product_id", "date", "units_sold",
                "stock_after", "trend_score"]
    missing = [c for c in required if c not in sales.columns]
    if missing:
        raise ValueError(f"Sales log missing columns: {missing}")
    sales["date"] = pd.to_datetime(sales["date"])
    return sales


# ---------------------------------------------------------------------------
# Feature calculation
# ---------------------------------------------------------------------------
def compute_latest_features(product_id: str,
                            sales_log_df: pd.DataFrame,
                            trend_score_override: float | None = None) -> dict:
    """
    Compute the four model input features from the latest 7 observations
    of a given product.

    Uses only past/present observed data (units_sold, stock_after).
    Does NOT use future potential_demand as an input feature.

    Parameters
    ----------
    product_id : str
    sales_log_df : pd.DataFrame  (must contain date, units_sold,
                                   stock_after, trend_score)
    trend_score_override : float, optional
        Custom trend score to override historical trend indicator.

    Returns
    -------
    dict with keys matching FEATURE_COLS
    """
    prod = sales_log_df[sales_log_df["product_id"] == product_id].copy()

    if prod.empty:
        raise ValueError(f"Product {product_id} not found in sales log")

    prod = prod.sort_values("date")

    if len(prod) < 7:
        raise ValueError(
            f"Product {product_id}: only {len(prod)} records, need >= 7"
        )

    latest_7 = prod.tail(7)

    avg_daily_sales_7d = float(latest_7["units_sold"].mean())
    sales_std_7d = float(latest_7["units_sold"].std())
    if np.isnan(sales_std_7d):
        sales_std_7d = 0.0

    latest_row = prod.iloc[-1]
    current_stock = float(latest_row["stock_after"])
    trend_score = float(trend_score_override) if trend_score_override is not None else float(latest_row["trend_score"])

    return {
        "avg_daily_sales_7d": avg_daily_sales_7d,
        "sales_std_7d": sales_std_7d,
        "current_stock": current_stock,
        "trend_score": trend_score,
    }


# ---------------------------------------------------------------------------
# Main recommendation function
# ---------------------------------------------------------------------------
def get_replenishment_recommendation(product_id: str, trend_score_override: float | None = None) -> dict:
    """
    Produce a complete inventory recommendation for a single product.

    Loads catalog, sales history, and both pre-trained models.
    Does NOT retrain or modify any model files.

    Parameters
    ----------
    product_id : str   e.g. "SKU001"
    trend_score_override : float, optional
        Custom trend score to override historical trend indicator.

    Returns
    -------
    dict with product info, predictions, and reorder recommendation
    """
    _validate_paths()

    # --- Catalog lookup ---
    catalog = _load_catalog()
    cat_row = catalog[catalog["product_id"] == product_id]
    if cat_row.empty:
        raise ValueError(f"Product {product_id} not found in catalog")
    cat_row = cat_row.iloc[0]

    product_name = cat_row["product_name"]
    category = cat_row["category"]
    catalog_current_stock = int(cat_row["current_stock"])
    price = float(cat_row["price"]) if "price" in cat_row and pd.notna(cat_row["price"]) else 0.0

    # --- Feature calculation ---
    sales = _load_sales_log()
    features = compute_latest_features(product_id, sales, trend_score_override=trend_score_override)

    model_current_stock = features["current_stock"]
    avg_daily_sales_7d = features["avg_daily_sales_7d"]
    sales_std_7d = features["sales_std_7d"]
    trend_score = features["trend_score"]

    # --- Build feature DataFrame (exact training column order) ---
    X = pd.DataFrame([features], columns=FEATURE_COLS)

    # --- Load models (read-only, never saved back) ---
    stockout_model = joblib.load(STOCKOUT_MODEL_PATH)
    demand_model = joblib.load(DEMAND_MODEL_PATH)

    # --- Stockout prediction ---
    stockout_prediction = int(stockout_model.predict(X)[0])
    stockout_probability = float(stockout_model.predict_proba(X)[0][1])

    # --- Demand prediction ---
    raw_demand = float(demand_model.predict(X)[0])
    predicted_7_day_demand = max(0, round(raw_demand))

    # --- Reorder logic ---
    safety_stock = round(avg_daily_sales_7d * SAFETY_STOCK_DAYS)
    low_stock_limit = (
        int(cat_row["low_stock_threshold"])
        if "low_stock_threshold" in cat_row and pd.notna(cat_row["low_stock_threshold"])
        else safety_stock
    )

    # Required stocks to maintain to cover expected demand plus safety buffer
    required_stock_to_maintain = predicted_7_day_demand + safety_stock

    # Recommended order = deficit to reach required stock level
    recommended_order_quantity = max(
        0,
        required_stock_to_maintain - int(model_current_stock),
    )

    # --- Alert logic ---
    alert_needed = (
        stockout_probability >= STOCKOUT_PROB_THRESHOLD
        or model_current_stock <= low_stock_limit
    )

    if stockout_probability >= STOCKOUT_PROB_THRESHOLD:
        alert_level = "HIGH"
    elif model_current_stock <= low_stock_limit:
        alert_level = "MEDIUM"
    elif stockout_probability >= 0.30:
        alert_level = "LOW"
    else:
        alert_level = "NONE"

    return {
        "product_id": product_id,
        "product_name": product_name,
        "category": category,
        "price": price,
        "catalog_current_stock": catalog_current_stock,
        "model_current_stock": int(model_current_stock),
        "avg_daily_sales_7d": round(avg_daily_sales_7d, 2),
        "sales_std_7d": round(sales_std_7d, 2),
        "trend_score": trend_score,
        "predicted_7_day_demand": predicted_7_day_demand,
        "stockout_probability": round(stockout_probability, 4),
        "stockout_prediction": stockout_prediction,
        "safety_stock": safety_stock,
        "required_stock_to_maintain": required_stock_to_maintain,
        "recommended_order_quantity": recommended_order_quantity,
        "alert_needed": alert_needed,
        "alert_level": alert_level,
    }


# ---------------------------------------------------------------------------
# All-product sweep
# ---------------------------------------------------------------------------
def run_for_all_products(trend_score_override: float | None = None) -> pd.DataFrame:
    """
    Run get_replenishment_recommendation() for every product in the catalog.

    Parameters
    ----------
    trend_score_override : float, optional
        Custom trend score to apply across all products.

    Returns
    -------
    pd.DataFrame — one row per successfully processed product
    """
    _validate_paths()
    catalog = _load_catalog()
    product_ids = catalog["product_id"].tolist()

    results = []
    for pid in product_ids:
        try:
            rec = get_replenishment_recommendation(pid, trend_score_override=trend_score_override)
            results.append(rec)
        except ValueError as e:
            print(f"  Skipped {pid}: {e}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    """Command-line interface: run all products or a single product."""
    SEP = "=" * 72

    # Optional single-product argument
    single_product = sys.argv[1] if len(sys.argv) > 1 else None

    if single_product:
        # ------ Single product mode ------
        print(SEP)
        print(f"  INVENTORY RECOMMENDATION: {single_product}")
        print(SEP)
        try:
            rec = get_replenishment_recommendation(single_product)
        except (ValueError, FileNotFoundError) as e:
            print(f"  ERROR: {e}")
            sys.exit(1)

        for key, val in rec.items():
            print(f"  {key:30s}: {val}")
        print(SEP)
        return

    # ------ All products mode ------
    print()
    print(SEP)
    print("  ADVANTAGE OS - INVENTORY RECOMMENDATIONS")
    print(SEP)
    print()

    df = run_for_all_products()

    if df.empty:
        print("  No products could be processed.")
        sys.exit(1)

    # --- Summary table ---
    display_cols = [
        "product_id",
        "product_name",
        "model_current_stock",
        "predicted_7_day_demand",
        "stockout_probability",
        "recommended_order_quantity",
        "alert_level",
    ]
    table = df[display_cols].copy()
    table.columns = [
        "SKU", "Product", "Stock", "Demand(7d)",
        "Stockout%", "Order Qty", "Alert",
    ]
    table["Stockout%"] = (table["Stockout%"] * 100).round(1).astype(str) + "%"

    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 120)
    pd.set_option("display.max_colwidth", 22)
    print(table.to_string(index=False))
    print()

    # --- Summary ---
    print("-" * 72)
    total = len(df)
    attention = df["alert_needed"].sum()
    print(f"  Total products processed      : {total}")
    print(f"  Products requiring attention  : {attention}")
    print("-" * 72)

    # --- Products needing attention ---
    if attention > 0:
        print()
        print("  PRODUCTS REQUIRING ATTENTION:")
        print()
        alerts = df[df["alert_needed"]]
        for _, row in alerts.iterrows():
            print(f"  [{row['alert_level']:6s}]  {row['product_id']}  "
                  f"{row['product_name']:26s}  "
                  f"stock={row['model_current_stock']:3d}  "
                  f"demand={row['predicted_7_day_demand']:3d}  "
                  f"order={row['recommended_order_quantity']:3d}  "
                  f"stockout={row['stockout_probability']:.0%}")

    print()
    print(SEP)
    print("  NOTE: Models loaded from disk (not retrained).")
    print("  No data or model files were modified.")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
