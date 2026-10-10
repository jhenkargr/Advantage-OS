"""Public inference service for the expanded inventory models."""

from functools import lru_cache
from pathlib import Path
import sys
import types

import joblib
import pandas as pd

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

if __package__:
    from .orchestrator import FEATURE_COLS, compute_latest_features
else:
    from orchestrator import FEATURE_COLS, compute_latest_features


INVENTORY_DIR = Path(__file__).resolve().parent.parent
CATALOG_PATH = INVENTORY_DIR / "data" / "product_catalog.csv"
SALES_LOG_PATH = INVENTORY_DIR / "data" / "sales_log.csv"
DEMAND_MODEL_PATH = INVENTORY_DIR / "models" / "demand_regressor_expanded.pkl"
STOCKOUT_MODEL_PATH = INVENTORY_DIR / "models" / "stockout_model_expanded.pkl"

REQUIRED_CATALOG_COLUMNS = {
    "product_id",
    "product_name",
    "category",
    "price",
}
REQUIRED_SALES_COLUMNS = {
    "product_id",
    "date",
    "units_sold",
    "stock_after",
    "trend_score",
}


@lru_cache(maxsize=1)
def _load_models():
    """Load and cache only the approved expanded models."""
    for path in (DEMAND_MODEL_PATH, STOCKOUT_MODEL_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required model file not found: {path}")

    return joblib.load(DEMAND_MODEL_PATH), joblib.load(STOCKOUT_MODEL_PATH)


def _read_csv(path: Path, required_columns: set[str], label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} file not found: {path}")

    frame = pd.read_csv(path)
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")
    return frame


EXPANDED_MODEL_METRICS = {
    "demand_model": {
        "model_name": "Multi-Layer Perceptron Regressor (MLPRegressor)",
        "loss_function": "Squared Error Loss (with L2 regularization, alpha=0.0001)",
        "architecture": "Hidden layers (64, 32), Adam optimizer, max_iter=5000",
        "horizon_days": 7,
        "test_mae": 14.97,
        "test_rmse": 23.40,
        "test_r2": 0.4293,
        "val_mae": 18.73,
        "val_rmse": 25.47,
        "val_r2": 0.3967,
    },
    "stockout_model": {
        "model_name": "Logistic Regression Classifier (StandardScaler pipeline)",
        "loss_function": "Binary Cross-Entropy / Log Loss (balanced class weights)",
        "horizon_days": 3,
        "test_accuracy": 0.8104,
        "test_precision": 0.6103,
        "test_recall": 0.8881,
        "test_f1": 0.7234,
        "val_accuracy": 0.7955,
        "val_f1": 0.7039,
        "confusion_matrix": {"TN": 270, "FP": 76, "FN": 15, "TP": 119},
    },
}


def get_inventory_prediction(product_id: str, trend_score: float | None = None) -> dict:
    """
    Return demand, stockout risk, required stock to maintain, and alert for a SKU.

    Parameters
    ----------
    product_id : str
        Catalog SKU identifier (e.g. "SKU001").
    trend_score : float, optional
        Custom trend score override. If None, uses the latest trend score from historical data.
    """
    catalog = _read_csv(CATALOG_PATH, REQUIRED_CATALOG_COLUMNS, "product catalog")
    matches = catalog[catalog["product_id"] == product_id]
    if matches.empty:
        raise ValueError(f"Unknown product_id: {product_id}")

    sales = _read_csv(SALES_LOG_PATH, REQUIRED_SALES_COLUMNS, "sales log")
    sales["date"] = pd.to_datetime(sales["date"])
    features = compute_latest_features(product_id, sales)

    # Apply custom trend override if supplied
    effective_trend = float(trend_score) if trend_score is not None else float(features["trend_score"])
    features["trend_score"] = effective_trend

    feature_frame = pd.DataFrame([features], columns=FEATURE_COLS)

    demand_model, stockout_model = _load_models()
    predicted_demand = max(0, round(float(demand_model.predict(feature_frame)[0])))
    stockout_probability = float(stockout_model.predict_proba(feature_frame)[0][1])

    catalog_row = matches.iloc[0]
    current_stock = int(features["current_stock"])
    safety_stock = round(features["avg_daily_sales_7d"] * 2)
    low_stock_limit = (
        int(catalog_row["low_stock_threshold"])
        if "low_stock_threshold" in catalog_row and pd.notna(catalog_row["low_stock_threshold"])
        else safety_stock
    )

    # Required stocks to maintain: expected 7-day demand + safety buffer
    required_stock_to_maintain = predicted_demand + safety_stock

    # Recommended reorder quantity: deficit to reach the required stock level
    recommended_order_quantity = max(
        0,
        required_stock_to_maintain - current_stock,
    )

    alert_needed = (
        stockout_probability >= 0.50
        or current_stock <= low_stock_limit
    )
    if stockout_probability >= 0.50:
        alert_level = "HIGH"
    elif current_stock <= low_stock_limit:
        alert_level = "MEDIUM"
    elif stockout_probability >= 0.30:
        alert_level = "LOW"
    else:
        alert_level = "NONE"

    # Confidence interval around predicted demand based on Test MAE (+/- ~15 units)
    mae = EXPANDED_MODEL_METRICS["demand_model"]["test_mae"]
    confidence_interval = {
        "min_demand": max(0, round(predicted_demand - mae)),
        "max_demand": round(predicted_demand + mae),
    }

    return {
        "product_id": str(product_id),
        "product_name": str(catalog_row["product_name"]),
        "category": str(catalog_row["category"]),
        "price": float(catalog_row["price"]) if "price" in catalog_row else 0.0,
        "current_stock": current_stock,
        "trend_score": round(effective_trend, 2),
        "avg_daily_sales_7d": round(features["avg_daily_sales_7d"], 2),
        "safety_stock": safety_stock,
        "predicted_7_day_demand": int(predicted_demand),
        "confidence_interval_demand": confidence_interval,
        "required_stock_to_maintain": int(required_stock_to_maintain),
        "recommended_order_quantity": int(recommended_order_quantity),
        "stockout_probability": round(stockout_probability, 4),
        "alert_needed": bool(alert_needed),
        "alert_level": alert_level,
        "model_metrics": EXPANDED_MODEL_METRICS,
    }


def get_all_inventory_predictions(
    default_trend_score: float | None = None,
    trend_overrides: dict[str, float] | None = None,
) -> dict:
    """
    Run demand forecasting and stock requirement evaluation for each and every product in catalog.

    Parameters
    ----------
    default_trend_score : float, optional
        A global trend score applied to all products unless overridden.
    trend_overrides : dict, optional
        Mapping of specific product_id to custom trend score, e.g. {"SKU001": 0.8}.
    """
    catalog = _read_csv(CATALOG_PATH, REQUIRED_CATALOG_COLUMNS, "product catalog")
    trend_map = trend_overrides or {}

    predictions = []
    total_stock = 0
    total_demand = 0
    total_required_stock = 0
    total_reorder_units = 0
    high_risk_count = 0
    medium_risk_count = 0

    for _, row in catalog.iterrows():
        pid = str(row["product_id"])
        # Determine trend score for this SKU: SKU override > default trend > historical
        sku_trend = trend_map.get(pid, default_trend_score)
        pred = get_inventory_prediction(pid, trend_score=sku_trend)
        predictions.append(pred)

        total_stock += pred["current_stock"]
        total_demand += pred["predicted_7_day_demand"]
        total_required_stock += pred["required_stock_to_maintain"]
        total_reorder_units += pred["recommended_order_quantity"]
        if pred["alert_level"] == "HIGH":
            high_risk_count += 1
        elif pred["alert_level"] == "MEDIUM":
            medium_risk_count += 1

    return {
        "total_skus": len(predictions),
        "total_units_in_stock": total_stock,
        "total_predicted_7_day_demand": total_demand,
        "total_required_stock": total_required_stock,
        "total_reorder_units": total_reorder_units,
        "high_risk_count": high_risk_count,
        "medium_risk_count": medium_risk_count,
        "model_metrics": EXPANDED_MODEL_METRICS,
        "predictions": predictions,
    }

