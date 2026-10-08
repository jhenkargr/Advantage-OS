"""Public inference service for the expanded inventory models."""

from functools import lru_cache
from pathlib import Path
import sys
import types

import joblib
import pandas as pd

# This Windows environment blocks sklearn's OpenMP extension; register its
# single-threaded fallback before sklearn is imported while unpickling models.
_openmp_helpers = types.ModuleType("sklearn.utils._openmp_helpers")
_openmp_helpers._openmp_parallelism_enabled = lambda: False
_openmp_helpers._openmp_effective_n_threads = lambda n=None: 1
sys.modules.setdefault("sklearn.utils._openmp_helpers", _openmp_helpers)

if __package__:
    from . import _sklearn_shim
    from .orchestrator import FEATURE_COLS, compute_latest_features
else:
    import _sklearn_shim
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
    "low_stock_threshold",
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


def get_inventory_prediction(product_id: str) -> dict:
    """Return demand, stockout risk, reorder quantity, and alert for a SKU."""
    catalog = _read_csv(CATALOG_PATH, REQUIRED_CATALOG_COLUMNS, "product catalog")
    matches = catalog[catalog["product_id"] == product_id]
    if matches.empty:
        raise ValueError(f"Unknown product_id: {product_id}")

    sales = _read_csv(SALES_LOG_PATH, REQUIRED_SALES_COLUMNS, "sales log")
    sales["date"] = pd.to_datetime(sales["date"])
    features = compute_latest_features(product_id, sales)
    feature_frame = pd.DataFrame([features], columns=FEATURE_COLS)

    demand_model, stockout_model = _load_models()
    predicted_demand = max(0, round(float(demand_model.predict(feature_frame)[0])))
    stockout_probability = float(stockout_model.predict_proba(feature_frame)[0][1])

    catalog_row = matches.iloc[0]
    current_stock = int(features["current_stock"])
    low_stock_threshold = int(catalog_row["low_stock_threshold"])
    safety_stock = round(features["avg_daily_sales_7d"] * 2)
    recommended_order_quantity = max(
        0,
        predicted_demand - current_stock + safety_stock,
    )

    alert_needed = (
        stockout_probability >= 0.50
        or current_stock <= low_stock_threshold
    )
    if stockout_probability >= 0.50:
        alert_level = "HIGH"
    elif current_stock <= low_stock_threshold:
        alert_level = "MEDIUM"
    else:
        alert_level = "NONE"

    return {
        "product_id": str(product_id),
        "product_name": str(catalog_row["product_name"]),
        "category": str(catalog_row["category"]),
        "current_stock": current_stock,
        "predicted_7_day_demand": int(predicted_demand),
        "stockout_probability": round(stockout_probability, 4),
        "recommended_order_quantity": int(recommended_order_quantity),
        "alert_needed": bool(alert_needed),
        "alert_level": alert_level,
    }
