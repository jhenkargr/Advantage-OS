"""Inventory and ML Demand/Stockout Prediction Service for AdVantage OS."""

from functools import lru_cache
import os
from pathlib import Path
import sys
import threading
import types
import joblib
import numpy as np
import pandas as pd

from app.config import (
    CATALOG_PATH,
    SALES_LOG_PATH,
    DEMAND_MODEL_PATH,
    STOCKOUT_MODEL_PATH,
)
from app.models.schemas import (
    ProductItem,
    ProductCreate,
    InventoryPredictionResponse,
    InventorySummaryResponse,
    SalesHistoryPoint,
)

# ---------------------------------------------------------------------------
# Windows WDAC & Sklearn OpenMP Protection Shim
# ---------------------------------------------------------------------------
_openmp_helpers = types.ModuleType("sklearn.utils._openmp_helpers")
_openmp_helpers._openmp_parallelism_enabled = lambda: False
_openmp_helpers._openmp_effective_n_threads = lambda n=None: 1
sys.modules.setdefault("sklearn.utils._openmp_helpers", _openmp_helpers)

# Import workspace's _sklearn_shim if available
_shim_path = Path(__file__).resolve().parent.parent.parent.parent / "inventory" / "src"
if str(_shim_path) not in sys.path:
    sys.path.insert(0, str(_shim_path))

try:
    import _sklearn_shim
except ImportError:
    pass

# Lock for catalog CSV concurrent writes
_CATALOG_LOCK = threading.Lock()

FEATURE_COLS = [
    "avg_daily_sales_7d",
    "sales_std_7d",
    "current_stock",
    "trend_score",
]

REQUIRED_CATALOG_COLUMNS = {
    "product_id",
    "product_name",
    "category",
    "current_stock",
    "low_stock_threshold",
}


@lru_cache(maxsize=1)
def load_ml_models():
    """Load and cache demand regressor and stockout classifier."""
    if not DEMAND_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Demand model not found at: {DEMAND_MODEL_PATH}")
    if not STOCKOUT_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Stockout model not found at: {STOCKOUT_MODEL_PATH}")

    demand_model = joblib.load(DEMAND_MODEL_PATH)
    stockout_model = joblib.load(STOCKOUT_MODEL_PATH)
    return demand_model, stockout_model


class InventoryService:
    @staticmethod
    def get_catalog_df() -> pd.DataFrame:
        if not CATALOG_PATH.is_file():
            raise FileNotFoundError(f"Catalog file not found: {CATALOG_PATH}")
        df = pd.read_csv(CATALOG_PATH)
        missing = REQUIRED_CATALOG_COLUMNS.difference(df.columns)
        if missing:
            raise ValueError(f"Catalog is missing columns: {sorted(missing)}")
        return df

    @staticmethod
    def get_sales_df() -> pd.DataFrame:
        if not SALES_LOG_PATH.is_file():
            raise FileNotFoundError(f"Sales log file not found: {SALES_LOG_PATH}")
        df = pd.read_csv(SALES_LOG_PATH)
        df["date"] = pd.to_datetime(df["date"])
        return df

    @classmethod
    def list_all_products(cls) -> list[ProductItem]:
        catalog_df = cls.get_catalog_df()
        products = []
        for _, row in catalog_df.iterrows():
            products.append(
                ProductItem(
                    product_id=str(row["product_id"]),
                    product_name=str(row["product_name"]),
                    category=str(row["category"]),
                    current_stock=int(row["current_stock"]),
                    low_stock_threshold=int(row["low_stock_threshold"]),
                )
            )
        return products

    @classmethod
    def get_product(cls, product_id: str) -> ProductItem:
        catalog_df = cls.get_catalog_df()
        match = catalog_df[catalog_df["product_id"] == product_id]
        if match.empty:
            raise ValueError(f"Product '{product_id}' not found in catalog.")
        row = match.iloc[0]
        return ProductItem(
            product_id=str(row["product_id"]),
            product_name=str(row["product_name"]),
            category=str(row["category"]),
            current_stock=int(row["current_stock"]),
            low_stock_threshold=int(row["low_stock_threshold"]),
        )

    @classmethod
    def create_product(cls, data: ProductCreate) -> ProductItem:
        with _CATALOG_LOCK:
            df = cls.get_catalog_df()
            if data.product_id in df["product_id"].values:
                raise ValueError(f"Product ID '{data.product_id}' already exists.")

            new_row = pd.DataFrame(
                [
                    {
                        "product_id": data.product_id,
                        "product_name": data.product_name,
                        "category": data.category,
                        "current_stock": data.current_stock,
                        "low_stock_threshold": data.low_stock_threshold,
                    }
                ]
            )
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(CATALOG_PATH, index=False)

        return ProductItem(
            product_id=data.product_id,
            product_name=data.product_name,
            category=data.category,
            current_stock=data.current_stock,
            low_stock_threshold=data.low_stock_threshold,
        )

    @classmethod
    def update_stock(cls, product_id: str, new_stock: int) -> ProductItem:
        with _CATALOG_LOCK:
            df = cls.get_catalog_df()
            idx = df.index[df["product_id"] == product_id].tolist()
            if not idx:
                raise ValueError(f"Product '{product_id}' not found.")
            df.loc[idx[0], "current_stock"] = new_stock
            df.to_csv(CATALOG_PATH, index=False)
            row = df.loc[idx[0]]

        return ProductItem(
            product_id=str(row["product_id"]),
            product_name=str(row["product_name"]),
            category=str(row["category"]),
            current_stock=int(row["current_stock"]),
            low_stock_threshold=int(row["low_stock_threshold"]),
        )

    @classmethod
    def adjust_stock_delta(cls, product_id: str, delta: int) -> ProductItem:
        with _CATALOG_LOCK:
            df = cls.get_catalog_df()
            idx = df.index[df["product_id"] == product_id].tolist()
            if not idx:
                raise ValueError(f"Product '{product_id}' not found.")
            current = int(df.loc[idx[0], "current_stock"])
            updated = max(0, current + delta)
            df.loc[idx[0], "current_stock"] = updated
            df.to_csv(CATALOG_PATH, index=False)
            row = df.loc[idx[0]]

        return ProductItem(
            product_id=str(row["product_id"]),
            product_name=str(row["product_name"]),
            category=str(row["category"]),
            current_stock=int(row["current_stock"]),
            low_stock_threshold=int(row["low_stock_threshold"]),
        )

    @classmethod
    def compute_features(cls, product_id: str, sales_df: pd.DataFrame) -> dict:
        prod = sales_df[sales_df["product_id"] == product_id].copy()
        if prod.empty:
            raise ValueError(f"Product '{product_id}' not found in sales history.")

        prod = prod.sort_values("date")
        if len(prod) < 7:
            # Fallback if product has few records
            avg_daily_sales = float(prod["units_sold"].mean()) if not prod.empty else 1.0
            sales_std = float(prod["units_sold"].std()) if len(prod) > 1 else 0.0
            latest_row = prod.iloc[-1]
            current_stock = float(latest_row.get("stock_after", 10.0))
            trend_score = float(latest_row.get("trend_score", 1.0))
        else:
            latest_7 = prod.tail(7)
            avg_daily_sales = float(latest_7["units_sold"].mean())
            sales_std = float(latest_7["units_sold"].std())
            if np.isnan(sales_std):
                sales_std = 0.0

            latest_row = prod.iloc[-1]
            current_stock = float(latest_row["stock_after"])
            trend_score = float(latest_row["trend_score"])

        return {
            "avg_daily_sales_7d": avg_daily_sales,
            "sales_std_7d": sales_std,
            "current_stock": current_stock,
            "trend_score": trend_score,
        }

    @classmethod
    def predict_sku(cls, product_id: str) -> InventoryPredictionResponse:
        catalog_df = cls.get_catalog_df()
        match = catalog_df[catalog_df["product_id"] == product_id]
        if match.empty:
            raise ValueError(f"Unknown product_id: '{product_id}'")

        catalog_row = match.iloc[0]
        sales_df = cls.get_sales_df()
        features = cls.compute_features(product_id, sales_df)

        feature_frame = pd.DataFrame([features], columns=FEATURE_COLS)
        demand_model, stockout_model = load_ml_models()

        predicted_demand = max(0, round(float(demand_model.predict(feature_frame)[0])))
        stockout_prob = float(stockout_model.predict_proba(feature_frame)[0][1])

        # Current stock from latest observation / catalog
        current_stock = int(features["current_stock"])
        low_stock_threshold = int(catalog_row["low_stock_threshold"])
        safety_stock = round(features["avg_daily_sales_7d"] * 2)

        recommended_order_quantity = max(
            0,
            predicted_demand - current_stock + safety_stock,
        )

        alert_needed = (stockout_prob >= 0.50) or (current_stock <= low_stock_threshold)
        if stockout_prob >= 0.50:
            alert_level = "HIGH"
        elif current_stock <= low_stock_threshold:
            alert_level = "MEDIUM"
        elif stockout_prob >= 0.30:
            alert_level = "LOW"
        else:
            alert_level = "NONE"

        return InventoryPredictionResponse(
            product_id=str(product_id),
            product_name=str(catalog_row["product_name"]),
            category=str(catalog_row["category"]),
            current_stock=current_stock,
            predicted_7_day_demand=int(predicted_demand),
            stockout_probability=round(stockout_prob, 4),
            recommended_order_quantity=int(recommended_order_quantity),
            alert_needed=bool(alert_needed),
            alert_level=alert_level,
            safety_stock=safety_stock,
            avg_daily_sales=round(features["avg_daily_sales_7d"], 2),
            trend_score=round(features["trend_score"], 2),
        )

    @classmethod
    def predict_all(cls) -> InventorySummaryResponse:
        catalog_df = cls.get_catalog_df()
        predictions = []
        high_risk = 0
        med_risk = 0
        total_reorder = 0
        total_stock = 0

        for _, row in catalog_df.iterrows():
            pid = str(row["product_id"])
            try:
                pred = cls.predict_sku(pid)
                predictions.append(pred)
                total_stock += pred.current_stock
                total_reorder += pred.recommended_order_quantity
                if pred.alert_level == "HIGH":
                    high_risk += 1
                elif pred.alert_level == "MEDIUM":
                    med_risk += 1
            except Exception as e:
                # In case of missing sales history for a newly added SKU
                current_stock = int(row["current_stock"])
                total_stock += current_stock
                predictions.append(
                    InventoryPredictionResponse(
                        product_id=pid,
                        product_name=str(row["product_name"]),
                        category=str(row["category"]),
                        current_stock=current_stock,
                        predicted_7_day_demand=0,
                        stockout_probability=0.0,
                        recommended_order_quantity=0,
                        alert_needed=current_stock <= int(row["low_stock_threshold"]),
                        alert_level="MEDIUM" if current_stock <= int(row["low_stock_threshold"]) else "NONE",
                    )
                )

        return InventorySummaryResponse(
            total_skus=len(predictions),
            total_units_in_stock=total_stock,
            high_risk_count=high_risk,
            medium_risk_count=med_risk,
            total_reorder_units=total_reorder,
            predictions=predictions,
        )

    @classmethod
    def get_sales_history(cls, product_id: str, limit: int = 30) -> list[SalesHistoryPoint]:
        sales_df = cls.get_sales_df()
        prod = sales_df[sales_df["product_id"] == product_id].sort_values("date")
        if prod.empty:
            raise ValueError(f"No sales history found for product '{product_id}'")

        history = []
        for _, row in prod.tail(limit).iterrows():
            history.append(
                SalesHistoryPoint(
                    date=row["date"].strftime("%Y-%m-%d"),
                    units_sold=int(row["units_sold"]),
                    stock_after=int(row["stock_after"]),
                    trend_score=round(float(row["trend_score"]), 2),
                )
            )
        return history
