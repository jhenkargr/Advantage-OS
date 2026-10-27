"""Inventory and ML Demand/Stockout Prediction Service for AdVantage OS."""

from datetime import datetime, timedelta
from functools import lru_cache
import os
from pathlib import Path
import sys
import threading
import types
import joblib
import numpy as np
import pandas as pd

from inventory.config import (
    CATALOG_PATH,
    SALES_LOG_PATH,
    DEMAND_MODEL_PATH,
    STOCKOUT_MODEL_PATH,
)
from typing import Any, Optional, Union
from inventory.models.schemas import (
    ProductItem,
    ProductCreate,
    ProductUpdate,
    InventoryPredictionResponse,
    InventorySummaryResponse,
    SalesHistoryPoint,
    DailySalesDataPoint,
    TimelineChartDataset,
    TimelineChart,
    CumulativeSalesChart,
    ChannelBreakdownItem,
    ChannelBreakdownChart,
    StockBenchmarkChart,
    ProjectedDepletionPoint,
    ProjectedDepletionChart,
    ProductChartsRepresentation,
    ProductSalesSummary,
    ProductDetailStockPlan,
    MLForecastInsights,
    ProductFullAnalyticsResponse,
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


class FallbackDemandModel:
    """Heuristic fallback when pretrained ML model file is not available or incompatible."""
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        results = []
        for _, row in X.iterrows():
            avg_daily = float(row.get("avg_daily_sales_7d", 1.5) or 1.5)
            trend = float(row.get("trend_score", 1.0) or 1.0)
            demand_7d = max(1, round(avg_daily * 7 * max(0.5, trend)))
            results.append(demand_7d)
        return np.array(results)


class FallbackStockoutModel:
    """Heuristic fallback for stockout probability."""
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = []
        for _, row in X.iterrows():
            stock = float(row.get("current_stock", 10) or 10)
            avg_daily = float(row.get("avg_daily_sales_7d", 1.5) or 1.5)
            days_left = stock / max(0.1, avg_daily)
            if stock <= 0:
                p_out = 0.95
            elif days_left < 3:
                p_out = 0.75
            elif days_left < 7:
                p_out = 0.55
            elif days_left < 14:
                p_out = 0.25
            else:
                p_out = 0.05
            probs.append([1.0 - p_out, p_out])
        return np.array(probs)


@lru_cache(maxsize=1)
def load_ml_models():
    """Load and cache demand regressor and stockout classifier with resilient fallback."""
    demand_model = None
    stockout_model = None

    if DEMAND_MODEL_PATH.is_file():
        try:
            demand_model = joblib.load(DEMAND_MODEL_PATH)
        except Exception as e:
            print(f"⚠️ [InventoryService] Could not load demand model from {DEMAND_MODEL_PATH}: {e}")

    if STOCKOUT_MODEL_PATH.is_file():
        try:
            stockout_model = joblib.load(STOCKOUT_MODEL_PATH)
        except Exception as e:
            print(f"⚠️ [InventoryService] Could not load stockout model from {STOCKOUT_MODEL_PATH}: {e}")

    if demand_model is None:
        print("ℹ️ [InventoryService] Using FallbackDemandModel heuristic engine.")
        demand_model = FallbackDemandModel()

    if stockout_model is None:
        print("ℹ️ [InventoryService] Using FallbackStockoutModel heuristic engine.")
        stockout_model = FallbackStockoutModel()

    return demand_model, stockout_model


class InventoryService:
    _CATALOG_CACHE: Optional[pd.DataFrame] = None
    _CATALOG_CACHE_TIME: float = 0.0
    _SALES_CACHE: Optional[pd.DataFrame] = None
    _SALES_CACHE_TIME: float = 0.0
    _CACHE_TTL: float = 120.0  # Cache for 2 minutes to eliminate repeated remote DB latency

    @classmethod
    def invalidate_cache(cls):
        """Invalidate cached DataFrames when new catalog items or sales are written."""
        cls._CATALOG_CACHE = None
        cls._CATALOG_CACHE_TIME = 0.0
        cls._SALES_CACHE = None
        cls._SALES_CACHE_TIME = 0.0

    @classmethod
    def get_catalog_df(cls, force_refresh: bool = False) -> pd.DataFrame:
        import time
        if not force_refresh and cls._CATALOG_CACHE is not None:
            if (time.time() - cls._CATALOG_CACHE_TIME) < cls._CACHE_TTL:
                return cls._CATALOG_CACHE.copy()

        df = None
        try:
            from inventory.db import SessionLocal, Product, ensure_tables
            db = SessionLocal()
            ensure_tables(db.get_bind())
            prods = db.query(Product).all()
            db.close()
            if prods:
                records = [
                    {
                        "product_id": str(p.product_id),
                        "product_name": str(p.product_name),
                        "category": str(p.category),
                        "current_stock": int(p.current_stock or 0),
                        "low_stock_threshold": int(p.low_stock_threshold or 10),
                    }
                    for p in prods
                ]
                df = pd.DataFrame(records)
        except Exception:
            df = None

        if df is None or df.empty:
            if CATALOG_PATH.is_file():
                df = pd.read_csv(CATALOG_PATH)
                missing = REQUIRED_CATALOG_COLUMNS.difference(df.columns)
                if missing:
                    raise ValueError(f"Catalog is missing columns: {sorted(missing)}")
            else:
                print(f"ℹ️ [InventoryService] Catalog file not found at {CATALOG_PATH}. Using starter catalog.")
                df = pd.DataFrame([
                    {"product_id": "P001", "product_name": "Espresso Blend Coffee Beans", "category": "Beverages", "current_stock": 25, "low_stock_threshold": 10},
                    {"product_id": "P002", "product_name": "Matcha Green Tea Latte Mix", "category": "Beverages", "current_stock": 8, "low_stock_threshold": 10},
                    {"product_id": "P003", "product_name": "Cold Brew Coffee Bottle 250ml", "category": "Beverages", "current_stock": 3, "low_stock_threshold": 10},
                    {"product_id": "P005", "product_name": "Brown Sugar Boba Milk Tea", "category": "Beverages", "current_stock": 0, "low_stock_threshold": 10},
                    {"product_id": "P009", "product_name": "Hot Chocolate", "category": "Beverages", "current_stock": 0, "low_stock_threshold": 10},
                    {"product_id": "P014", "product_name": "Mango Cheesecake Slice", "category": "Bakery & Desserts", "current_stock": 2, "low_stock_threshold": 8},
                    {"product_id": "P016", "product_name": "Sourdough Loaf", "category": "Bakery & Desserts", "current_stock": 0, "low_stock_threshold": 5},
                ])

        cls._CATALOG_CACHE = df.copy()
        cls._CATALOG_CACHE_TIME = time.time()
        return df

    @classmethod
    def get_sales_df(cls, db: Optional[Any] = None, force_refresh: bool = False) -> pd.DataFrame:
        import time
        if not force_refresh and db is None and cls._SALES_CACHE is not None:
            if (time.time() - cls._SALES_CACHE_TIME) < cls._CACHE_TTL:
                return cls._SALES_CACHE.copy()

        close_session = False
        if db is None:
            try:
                from inventory.db import SessionLocal
                db = SessionLocal()
                close_session = True
            except Exception:
                db = None

        df = None
        if db is not None:
            try:
                from inventory.db import SalesLog, ensure_tables
                ensure_tables(db.get_bind())
                logs = db.query(SalesLog).all()
                if logs:
                    records = [
                        {
                            "product_id": str(l.product_id),
                            "date": pd.to_datetime(l.date),
                            "units_sold": int(l.units_sold or 0),
                            "price": float(l.price or 0.0),
                            "channel": str(l.channel or "POS_STORE"),
                            "stock_after": None,
                            "trend_score": 1.0,
                        }
                        for l in logs
                    ]
                    df = pd.DataFrame(records)
            except Exception as exc:
                print(f"[InventoryService] DB get_sales_df notice: {exc}")
            finally:
                if close_session:
                    db.close()

        if df is None or df.empty:
            # Check candidate paths
            for candidate in [
                SALES_LOG_PATH,
                _HERE / "sales_history.csv",
                _HERE / "inventory" / "data" / "sales_history.csv",
            ]:
                if candidate.is_file():
                    cand_df = pd.read_csv(candidate)
                    cand_df["date"] = pd.to_datetime(cand_df["date"])
                    if "units_sold" not in cand_df.columns and "item_sold" in cand_df.columns:
                        cand_df["units_sold"] = cand_df["item_sold"]
                    df = cand_df
                    break

        if df is None or df.empty:
            if SALES_LOG_PATH.is_file():
                df = pd.read_csv(SALES_LOG_PATH)
                df["date"] = pd.to_datetime(df["date"])
            else:
                print(f"ℹ️ [InventoryService] Sales log file not found at {SALES_LOG_PATH}. Using fallback sales frame.")
                df = pd.DataFrame(columns=["product_id", "date", "units_sold", "price", "channel", "stock_after", "trend_score"])

        if db is None and df is not None:
            cls._SALES_CACHE = df.copy()
            cls._SALES_CACHE_TIME = time.time()

        return df


    @classmethod
    def _sync_to_csv(cls, payload: dict):
        """Keep CATALOG_PATH and root products.csv in sync with database updates."""
        with _CATALOG_LOCK:
            try:
                pid = str(payload["product_id"]).strip()
                if CATALOG_PATH.is_file():
                    df = pd.read_csv(CATALOG_PATH)
                else:
                    df = pd.DataFrame(columns=list(REQUIRED_CATALOG_COLUMNS))

                idx = df.index[df["product_id"].astype(str) == pid].tolist()
                if idx:
                    target_idx = idx[0]
                    if payload.get("product_name") is not None:
                        df.loc[target_idx, "product_name"] = str(payload["product_name"]).strip()
                    if payload.get("category") is not None:
                        df.loc[target_idx, "category"] = str(payload["category"]).strip()
                    if payload.get("current_stock") is not None:
                        df.loc[target_idx, "current_stock"] = int(payload["current_stock"])
                    if payload.get("low_stock_threshold") is not None:
                        df.loc[target_idx, "low_stock_threshold"] = int(payload["low_stock_threshold"])
                else:
                    new_row = pd.DataFrame(
                        [
                            {
                                "product_id": pid,
                                "product_name": str(payload.get("product_name") or pid).strip(),
                                "category": str(payload.get("category") or "General").strip(),
                                "current_stock": int(payload.get("current_stock", 0) or 0),
                                "low_stock_threshold": int(payload.get("low_stock_threshold", 10) or 10),
                            }
                        ]
                    )
                    df = pd.concat([df, new_row], ignore_index=True)

                CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(CATALOG_PATH, index=False)
            except Exception as exc:
                print(f"[InventoryService] CSV sync notice: {exc}")

            # Also update products.csv in virenv/Scripts if present
            try:
                candidate_csv = Path(r"D:\major project\Backend\virenv\Scripts\products.csv")
                if candidate_csv.is_file():
                    p_df = pd.read_csv(candidate_csv)
                    p_idx = p_df.index[p_df["product_id"].astype(str) == pid].tolist()
                    if p_idx:
                        t_idx = p_idx[0]
                        if payload.get("product_name") is not None:
                            p_df.loc[t_idx, "product_name"] = str(payload["product_name"]).strip()
                        if payload.get("category") is not None:
                            p_df.loc[t_idx, "category"] = str(payload["category"]).strip()
                        if payload.get("current_stock") is not None and "stock_available" in p_df.columns:
                            p_df.loc[t_idx, "stock_available"] = int(payload["current_stock"])
                        if payload.get("unit_price") is not None and "current_price" in p_df.columns:
                            p_df.loc[t_idx, "current_price"] = float(payload["unit_price"])
                        p_df.to_csv(candidate_csv, index=False)
            except Exception as exc:
                print(f"[InventoryService] Root products.csv sync notice: {exc}")

    @classmethod
    def list_all_products(cls, db: Optional[Any] = None) -> list[ProductItem]:
        """List all available products in the SME catalog with current stock levels."""
        close_session = False
        if db is None:
            try:
                from inventory.db import SessionLocal
                db = SessionLocal()
                close_session = True
            except Exception:
                db = None

        if db is not None:
            try:
                from inventory.db import Product, ensure_tables
                ensure_tables(db.get_bind())
                db_prods = db.query(Product).order_by(Product.product_id).all()
                if db_prods:
                    items = [
                        ProductItem(
                            product_id=str(p.product_id),
                            product_name=str(p.product_name),
                            category=str(p.category),
                            current_stock=int(p.current_stock or 0),
                            low_stock_threshold=int(p.low_stock_threshold or 10),
                            unit_price=float(p.unit_price or 0.0),
                            supplier_lead_time_days=int(p.supplier_lead_time_days or 3),
                            reorder_quantity=int(p.reorder_quantity or 20),
                        )
                        for p in db_prods
                    ]
                    return items
            except Exception as exc:
                print(f"[InventoryService] DB fetch error, falling back to CSV: {exc}")
            finally:
                if close_session:
                    db.close()

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
                    unit_price=float(row.get("unit_price", 0.0) or 0.0) if "unit_price" in row else 0.0,
                    supplier_lead_time_days=int(row.get("supplier_lead_time_days", 3) or 3) if "supplier_lead_time_days" in row else 3,
                    reorder_quantity=int(row.get("reorder_quantity", 20) or 20) if "reorder_quantity" in row else 20,
                )
            )
        return products

    @classmethod
    def get_product(cls, product_id: str, db: Optional[Any] = None) -> ProductItem:
        """Retrieve catalog item details by product SKU ID."""
        pid = str(product_id).strip()
        close_session = False
        if db is None:
            try:
                from inventory.db import SessionLocal
                db = SessionLocal()
                close_session = True
            except Exception:
                db = None

        if db is not None:
            try:
                from inventory.db import Product, ensure_tables
                ensure_tables(db.get_bind())
                p = db.query(Product).filter(Product.product_id == pid).first()
                if p:
                    return ProductItem(
                        product_id=str(p.product_id),
                        product_name=str(p.product_name),
                        category=str(p.category),
                        current_stock=int(p.current_stock or 0),
                        low_stock_threshold=int(p.low_stock_threshold or 10),
                        unit_price=float(p.unit_price or 0.0),
                        supplier_lead_time_days=int(p.supplier_lead_time_days or 3),
                        reorder_quantity=int(p.reorder_quantity or 20),
                    )
            except Exception as exc:
                print(f"[InventoryService] DB get_product error: {exc}")
            finally:
                if close_session:
                    db.close()

        catalog_df = cls.get_catalog_df()
        match = catalog_df[catalog_df["product_id"].astype(str) == pid]
        if match.empty:
            raise ValueError(f"Product '{pid}' not found in catalog.")
        row = match.iloc[0]
        return ProductItem(
            product_id=str(row["product_id"]),
            product_name=str(row["product_name"]),
            category=str(row["category"]),
            current_stock=int(row["current_stock"]),
            low_stock_threshold=int(row["low_stock_threshold"]),
            unit_price=float(row.get("unit_price", 0.0) or 0.0) if "unit_price" in row else 0.0,
            supplier_lead_time_days=int(row.get("supplier_lead_time_days", 3) or 3) if "supplier_lead_time_days" in row else 3,
            reorder_quantity=int(row.get("reorder_quantity", 20) or 20) if "reorder_quantity" in row else 20,
        )

    @classmethod
    def create_or_update_product(cls, data: Any, db: Optional[Any] = None) -> ProductItem:
        """Add new product or update existing product in catalog (both database and CSV)."""
        if hasattr(data, "model_dump"):
            payload = data.model_dump(exclude_unset=True)
        elif hasattr(data, "dict"):
            payload = data.dict(exclude_unset=True)
        elif isinstance(data, dict):
            payload = dict(data)
        else:
            raise ValueError("Invalid product data format")

        pid = str(payload.get("product_id", "")).strip()
        if not pid:
            raise ValueError("product_id is required")

        close_session = False
        if db is None:
            try:
                from inventory.db import SessionLocal
                db = SessionLocal()
                close_session = True
            except Exception:
                db = None

        prod_record = None
        if db is not None:
            try:
                from inventory.db import Product, ensure_tables
                ensure_tables(db.get_bind())
                existing = db.query(Product).filter(Product.product_id == pid).first()
                if existing:
                    if "product_name" in payload and payload["product_name"] is not None:
                        existing.product_name = str(payload["product_name"]).strip()
                    if "category" in payload and payload["category"] is not None:
                        existing.category = str(payload["category"]).strip()
                    if "current_stock" in payload and payload["current_stock"] is not None:
                        existing.current_stock = int(payload["current_stock"])
                    if "low_stock_threshold" in payload and payload["low_stock_threshold"] is not None:
                        existing.low_stock_threshold = int(payload["low_stock_threshold"])
                    if "unit_price" in payload and payload["unit_price"] is not None:
                        existing.unit_price = float(payload["unit_price"])
                    if "supplier_lead_time_days" in payload and payload["supplier_lead_time_days"] is not None:
                        existing.supplier_lead_time_days = int(payload["supplier_lead_time_days"])
                    if "reorder_quantity" in payload and payload["reorder_quantity"] is not None:
                        existing.reorder_quantity = int(payload["reorder_quantity"])
                    db.commit()
                    db.refresh(existing)
                    prod_record = existing
                else:
                    new_prod = Product(
                        product_id=pid,
                        product_name=str(payload.get("product_name") or f"Product {pid}").strip(),
                        category=str(payload.get("category") or "General").strip(),
                        current_stock=int(payload.get("current_stock", 0) or 0),
                        low_stock_threshold=int(payload.get("low_stock_threshold", 10) or 10),
                        unit_price=float(payload.get("unit_price", 0.0) or 0.0),
                        supplier_lead_time_days=int(payload.get("supplier_lead_time_days", 3) or 3),
                        reorder_quantity=int(payload.get("reorder_quantity", 20) or 20),
                    )
                    db.add(new_prod)
                    db.commit()
                    db.refresh(new_prod)
                    prod_record = new_prod
            except Exception as exc:
                if db:
                    db.rollback()
                print(f"[InventoryService] Database save error: {exc}")
            finally:
                if close_session:
                    db.close()

        # Update CSV files to guarantee sync across local file stores
        cls._sync_to_csv(payload)

        if prod_record:
            return ProductItem(
                product_id=str(prod_record.product_id),
                product_name=str(prod_record.product_name),
                category=str(prod_record.category),
                current_stock=int(prod_record.current_stock),
                low_stock_threshold=int(prod_record.low_stock_threshold),
                unit_price=float(prod_record.unit_price or 0.0),
                supplier_lead_time_days=int(prod_record.supplier_lead_time_days or 3),
                reorder_quantity=int(prod_record.reorder_quantity or 20),
            )
        else:
            return cls.get_product(pid)

    @classmethod
    def create_product(cls, data: Union[ProductCreate, ProductUpdate, dict], db: Optional[Any] = None) -> ProductItem:
        """Alias for create_or_update_product."""
        return cls.create_or_update_product(data, db=db)

    @classmethod
    def update_stock(cls, product_id: str, new_stock: int, db: Optional[Any] = None) -> ProductItem:
        """Manually update the current stock level for a SKU."""
        return cls.create_or_update_product({"product_id": product_id, "current_stock": new_stock}, db=db)

    @classmethod
    def adjust_stock_delta(cls, product_id: str, delta: int, db: Optional[Any] = None) -> ProductItem:
        """Adjust stock by a positive or negative quantity delta."""
        curr = cls.get_product(product_id, db=db)
        updated = max(0, curr.current_stock + delta)
        return cls.create_or_update_product({"product_id": product_id, "current_stock": updated}, db=db)

    @classmethod
    def compute_features(
        cls,
        product_id: str,
        sales_df: pd.DataFrame,
        catalog_stock: Optional[float] = None,
        prod_sales_df: Optional[pd.DataFrame] = None,
    ) -> dict:
        pid = str(product_id).strip()
        if prod_sales_df is not None:
            prod = prod_sales_df
        else:
            prod = sales_df[sales_df["product_id"].astype(str).str.strip() == pid]

        if catalog_stock is None:
            # Fallback only if not provided
            catalog_stock = 10.0
            try:
                cat_item = cls.get_product(pid)
                catalog_stock = float(cat_item.current_stock or 0.0)
            except Exception:
                pass

        if prod is None or prod.empty:
            return {
                "avg_daily_sales_7d": 1.5,
                "sales_std_7d": 0.5,
                "current_stock": catalog_stock,
                "trend_score": 1.0,
            }

        if not prod.index.is_monotonic_increasing:
            prod = prod.sort_values("date")

        if len(prod) < 7:
            # Fallback if product has few records
            avg_daily_sales = float(prod["units_sold"].mean()) if not prod.empty else 1.0
            sales_std = float(prod["units_sold"].std()) if len(prod) > 1 else 0.0
            latest_row = prod.iloc[-1]
            current_stock = float(latest_row.get("stock_after")) if pd.notna(latest_row.get("stock_after")) else catalog_stock
            trend_score = float(latest_row.get("trend_score")) if pd.notna(latest_row.get("trend_score")) else 1.0
        else:
            latest_7 = prod.tail(7)
            avg_daily_sales = float(latest_7["units_sold"].mean())
            sales_std = float(latest_7["units_sold"].std())
            if np.isnan(sales_std):
                sales_std = 0.0

            latest_row = prod.iloc[-1]
            current_stock = float(latest_row.get("stock_after")) if pd.notna(latest_row.get("stock_after")) else catalog_stock
            trend_score = float(latest_row.get("trend_score")) if pd.notna(latest_row.get("trend_score")) else 1.0

        return {
            "avg_daily_sales_7d": round(avg_daily_sales, 2),
            "sales_std_7d": round(sales_std, 2),
            "current_stock": current_stock,
            "trend_score": trend_score,
        }

    @classmethod
    def predict_sku(
        cls,
        product_id: str,
        catalog_df: Optional[pd.DataFrame] = None,
        sales_df: Optional[pd.DataFrame] = None,
        catalog_row: Optional[pd.Series] = None,
        prod_sales_df: Optional[pd.DataFrame] = None,
    ) -> InventoryPredictionResponse:
        pid = str(product_id).strip()
        if catalog_row is None:
            if catalog_df is None:
                catalog_df = cls.get_catalog_df()
            match = catalog_df[catalog_df["product_id"].astype(str).str.strip() == pid]
            if match.empty:
                raise ValueError(f"Unknown product_id: '{pid}'")
            catalog_row = match.iloc[0]

        if sales_df is None and prod_sales_df is None:
            sales_df = cls.get_sales_df()

        cat_stock = float(catalog_row.get("current_stock", 10) or 10)
        features = cls.compute_features(
            pid,
            sales_df if sales_df is not None else pd.DataFrame(),
            catalog_stock=cat_stock,
            prod_sales_df=prod_sales_df,
        )

        feature_frame = pd.DataFrame([features], columns=FEATURE_COLS)
        demand_model, stockout_model = load_ml_models()

        predicted_demand = max(0, round(float(demand_model.predict(feature_frame)[0])))
        stockout_prob = float(stockout_model.predict_proba(feature_frame)[0][1])

        # Current stock from latest observation / catalog
        current_stock = int(features["current_stock"])
        low_stock_threshold = int(catalog_row.get("low_stock_threshold", 10) or 10)
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
            product_id=pid,
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
    def predict_all(cls, force_refresh_data: bool = False) -> InventorySummaryResponse:
        catalog_df = cls.get_catalog_df(force_refresh=force_refresh_data)
        if catalog_df.empty:
            return InventorySummaryResponse(
                total_skus=0,
                total_units_in_stock=0,
                high_risk_count=0,
                medium_risk_count=0,
                total_reorder_units=0,
                predictions=[],
            )

        sales_df = cls.get_sales_df(force_refresh=force_refresh_data)

        # Pre-group sales by product_id once for O(1) lookups
        sales_by_pid = {}
        if not sales_df.empty and "product_id" in sales_df.columns:
            for pid_val, group in sales_df.groupby("product_id"):
                sales_by_pid[str(pid_val).strip()] = group

        features_list = []
        rows_data = []

        for _, row in catalog_df.iterrows():
            pid = str(row["product_id"]).strip()
            cat_stock = float(row.get("current_stock", 0) or 0)
            prod_sales = sales_by_pid.get(pid)
            feat = cls.compute_features(
                product_id=pid,
                sales_df=sales_df,
                catalog_stock=cat_stock,
                prod_sales_df=prod_sales,
            )
            features_list.append(feat)
            rows_data.append((pid, row, feat))

        # Vectorized batch prediction with ML models in a single call (< 5ms)
        feature_frame = pd.DataFrame(features_list, columns=FEATURE_COLS)
        demand_model, stockout_model = load_ml_models()

        pred_demands = np.maximum(0, np.round(demand_model.predict(feature_frame)).astype(int))
        stockout_probs = stockout_model.predict_proba(feature_frame)[:, 1]

        predictions = []
        high_risk = 0
        med_risk = 0
        total_reorder = 0
        total_stock = 0

        for i, (pid, row, feat) in enumerate(rows_data):
            predicted_demand = int(pred_demands[i])
            stockout_prob = float(stockout_probs[i])
            current_stock = int(feat["current_stock"])
            low_stock_threshold = int(row.get("low_stock_threshold", 10) or 10)
            safety_stock = round(feat["avg_daily_sales_7d"] * 2)

            recommended_order_quantity = max(
                0,
                predicted_demand - current_stock + safety_stock,
            )

            alert_needed = (stockout_prob >= 0.50) or (current_stock <= low_stock_threshold)
            if stockout_prob >= 0.50:
                alert_level = "HIGH"
                high_risk += 1
            elif current_stock <= low_stock_threshold:
                alert_level = "MEDIUM"
                med_risk += 1
            elif stockout_prob >= 0.30:
                alert_level = "LOW"
            else:
                alert_level = "NONE"

            total_stock += current_stock
            total_reorder += recommended_order_quantity

            predictions.append(
                InventoryPredictionResponse(
                    product_id=pid,
                    product_name=str(row["product_name"]),
                    category=str(row["category"]),
                    current_stock=current_stock,
                    predicted_7_day_demand=int(predicted_demand),
                    stockout_probability=round(stockout_prob, 4),
                    recommended_order_quantity=int(recommended_order_quantity),
                    alert_needed=bool(alert_needed),
                    alert_level=alert_level,
                    safety_stock=safety_stock,
                    avg_daily_sales=round(feat["avg_daily_sales_7d"], 2),
                    trend_score=round(feat["trend_score"], 2),
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

    @classmethod
    def get_product_full_analytics(cls, product_id: str, db: Optional[Any] = None) -> ProductFullAnalyticsResponse:
        """
        Consolidates complete information for a product across:
        1. Product catalog details (products table / CSV)
        2. Sales logs (sales_logs table / sales_log.csv / sales_history.csv)
        3. Stock replenishment plan (stock_plans table / stock_plan.csv)
        And synthesizes rich visual graph representations (Timeline, Cumulative, Channels, Benchmarks, 30-day projection).
        """
        pid = str(product_id).strip()
        close_session = False
        if db is None:
            try:
                from inventory.db import SessionLocal
                db = SessionLocal()
                close_session = True
            except Exception:
                db = None

        try:
            # 1. Fetch Product
            product = cls.get_product(pid, db=db)
            unit_price = float(product.unit_price or 0.0)

            # 2. Fetch Sales Logs
            raw_sales = []
            if db is not None:
                try:
                    from inventory.db import SalesLog, ensure_tables
                    ensure_tables(db.get_bind())
                    db_logs = db.query(SalesLog).filter(SalesLog.product_id == pid).order_by(SalesLog.date).all()
                    for log in db_logs:
                        raw_sales.append({
                            "date": str(log.date),
                            "units_sold": int(log.units_sold or 0),
                            "price": float(log.price or unit_price),
                            "channel": str(log.channel or "POS_STORE"),
                            "promotion_applied": bool(log.promotion_applied or False),
                            "stock_after": None,
                            "trend_score": None,
                        })
                except Exception as exc:
                    print(f"[get_product_full_analytics] DB sales query note: {exc}")

            # Fallback/supplement from CSV if no DB sales logs found
            if not raw_sales:
                try:
                    if SALES_LOG_PATH.is_file():
                        sdf = pd.read_csv(SALES_LOG_PATH)
                        match_s = sdf[sdf["product_id"].astype(str) == pid].sort_values("date")
                        for _, row in match_s.iterrows():
                            raw_sales.append({
                                "date": str(row["date"]),
                                "units_sold": int(row.get("units_sold", 0)),
                                "price": unit_price,
                                "channel": "POS_STORE",
                                "promotion_applied": False,
                                "stock_after": int(row["stock_after"]) if pd.notna(row.get("stock_after")) else None,
                                "trend_score": float(row["trend_score"]) if pd.notna(row.get("trend_score")) else None,
                            })
                except Exception as exc:
                    print(f"[get_product_full_analytics] CSV sales_log error: {exc}")

                if not raw_sales:
                    try:
                        sh_path = Path(r"D:\major project\Backend\virenv\Scripts\sales_history.csv")
                        if sh_path.is_file():
                            shdf = pd.read_csv(sh_path)
                            qty_c = "units_sold" if "units_sold" in shdf.columns else ("item_sold" if "item_sold" in shdf.columns else None)
                            match_sh = shdf[shdf["product_id"].astype(str) == pid].sort_values("date")
                            for _, row in match_sh.iterrows():
                                units = int(row[qty_c]) if (qty_c and pd.notna(row.get(qty_c))) else 0
                                p_val = float(row.get("price", row.get("unit_price", unit_price))) if pd.notna(row.get("price")) else unit_price
                                raw_sales.append({
                                    "date": str(row["date"]),
                                    "units_sold": units,
                                    "price": p_val,
                                    "channel": str(row.get("channel", "POS_STORE")),
                                    "promotion_applied": bool(row.get("promotion_applied", False)),
                                    "stock_after": None,
                                    "trend_score": None,
                                })
                    except Exception as exc:
                        print(f"[get_product_full_analytics] CSV sales_history error: {exc}")

            # Sort chronological
            raw_sales.sort(key=lambda x: x["date"])

            # 3. Fetch Stock Plan
            plan_obj = None
            if db is not None:
                try:
                    from inventory.db import StockPlan, ensure_tables
                    ensure_tables(db.get_bind())
                    plan_obj = db.query(StockPlan).filter(StockPlan.product_id == pid).first()
                except Exception as exc:
                    print(f"[get_product_full_analytics] DB stock plan query note: {exc}")

            # Fallback to stock_plan.csv if not found in DB
            if plan_obj is None:
                try:
                    sp_csv = Path(r"D:\major project\Backend\virenv\Scripts\stock_plan.csv")
                    if sp_csv.is_file():
                        spdf = pd.read_csv(sp_csv)
                        match_sp = spdf[spdf["Product ID"].astype(str) == pid] if "Product ID" in spdf.columns else spdf[spdf["product_id"].astype(str) == pid]
                        if not match_sp.empty:
                            srow = match_sp.iloc[0]
                            def _c_int(v):
                                try:
                                    return int(float(v)) if pd.notna(v) and str(v).lower() not in ("nan", "none", "<na>") else None
                                except: return None
                            def _c_flt(v):
                                try:
                                    return float(v) if pd.notna(v) and str(v).lower() not in ("nan", "none", "<na>") else 0.0
                                except: return 0.0
                            p_avg = _c_flt(srow.get("Avg daily demand", srow.get("avg_daily_demand", 0.0)))
                            p_forecast = _c_flt(srow.get("30-day forecast", srow.get("forecast_30d", 0.0)))
                            p_safety = _c_int(srow.get("Safety stock", srow.get("safety_stock")))
                            p_reorder = _c_int(srow.get("Reorder point", srow.get("reorder_point")))
                            p_target = _c_int(srow.get("Target stock", srow.get("target_stock")))
                            p_order = _c_int(srow.get("Order qty now", srow.get("order_qty_now")))
                            p_status = str(srow.get("Status", srow.get("status", "OK")))
                            p_note = str(srow.get("Note", srow.get("note", ""))) if pd.notna(srow.get("Note", srow.get("note"))) else None
                            plan_obj = "CSV_FOUND"
                except Exception as exc:
                    print(f"[get_product_full_analytics] CSV stock plan note: {exc}")

            total_units = sum(r["units_sold"] for r in raw_sales)
            avg_daily_calc = round(total_units / max(len(raw_sales), 1), 2) if raw_sales else 1.0

            if plan_obj is not None and plan_obj != "CSV_FOUND":
                def _safe_int(v):
                    try: return int(float(v))
                    except: return None
                p_avg = float(plan_obj.avg_daily_demand or avg_daily_calc)
                p_forecast = float(plan_obj.forecast_30d or round(p_avg * 30, 1))
                p_safety = _safe_int(plan_obj.safety_stock) or max(5, round(p_avg * 2))
                p_reorder = _safe_int(plan_obj.reorder_point) or (round(p_avg * (product.supplier_lead_time_days or 3)) + p_safety)
                p_target = _safe_int(plan_obj.target_stock) or (round(p_avg * 14) + p_safety)
                p_order = _safe_int(plan_obj.order_qty_now)
                if p_order is None:
                    p_order = max(0, p_target - product.current_stock)
                p_status = str(plan_obj.status or ("OK" if product.current_stock > p_reorder else "REORDER_NOW"))
                p_note = plan_obj.note
            elif plan_obj == "CSV_FOUND":
                pass
            else:
                p_avg = avg_daily_calc
                p_forecast = round(p_avg * 30, 1)
                p_safety = max(5, round(p_avg * 2))
                p_reorder = round(p_avg * (product.supplier_lead_time_days or 3)) + p_safety
                p_target = round(p_avg * 14) + p_safety
                p_order = max(0, p_target - product.current_stock)
                p_status = "OK" if product.current_stock > p_reorder else ("CRITICAL" if product.current_stock == 0 else "REORDER_NOW")
                p_note = f"Dynamic stock replenishment plan generated from historical demand ({p_avg} units/day)."

            # Reorder urgency
            if product.current_stock == 0:
                urgency = "CRITICAL"
            elif product.current_stock <= (p_safety or product.low_stock_threshold):
                urgency = "HIGH"
            elif product.current_stock <= (p_reorder or product.low_stock_threshold * 1.5):
                urgency = "MEDIUM"
            else:
                urgency = "LOW" if p_target and product.current_stock < p_target else "OK"

            stock_plan_dto = ProductDetailStockPlan(
                avg_daily_demand=round(p_avg, 2),
                forecast_30d=round(p_forecast, 2),
                current_stock=product.current_stock,
                safety_stock=p_safety,
                reorder_point=p_reorder,
                target_stock=p_target,
                order_qty_now=p_order,
                status=p_status,
                note=p_note,
                reorder_urgency=urgency,
            )

            # 4. Compute Sales Summary KPIs
            total_revenue = sum(r["units_sold"] * (r["price"] if r["price"] > 0 else unit_price) for r in raw_sales)
            avg_daily_sales = round(total_units / max(len(raw_sales), 1), 2) if raw_sales else 0.0

            last_7 = raw_sales[-7:] if len(raw_sales) >= 7 else raw_sales
            last_30 = raw_sales[-30:] if len(raw_sales) >= 30 else raw_sales

            units_7d = sum(r["units_sold"] for r in last_7)
            rev_7d = sum(r["units_sold"] * (r["price"] if r["price"] > 0 else unit_price) for r in last_7)
            units_30d = sum(r["units_sold"] for r in last_30)
            rev_30d = sum(r["units_sold"] * (r["price"] if r["price"] > 0 else unit_price) for r in last_30)

            min_units = min((r["units_sold"] for r in raw_sales), default=0)
            max_units = max((r["units_sold"] for r in raw_sales), default=0)

            peak_date = None
            if raw_sales:
                peak_item = max(raw_sales, key=lambda x: x["units_sold"])
                peak_date = peak_item["date"]

            days_remaining = None
            effective_daily = p_avg if p_avg > 0 else avg_daily_sales
            if effective_daily > 0:
                days_remaining = round(product.current_stock / effective_daily, 1)

            sales_summary = ProductSalesSummary(
                total_records=len(raw_sales),
                total_units_sold=total_units,
                total_revenue=round(total_revenue, 2),
                avg_daily_sales=avg_daily_sales,
                sales_velocity_7d=round(units_7d / max(len(last_7), 1), 2),
                revenue_7d=round(rev_7d, 2),
                sales_velocity_30d=round(units_30d / max(len(last_30), 1), 2),
                revenue_30d=round(rev_30d, 2),
                min_daily_units=min_units,
                max_daily_units=max_units,
                peak_sales_date=peak_date,
                days_of_inventory_remaining=days_remaining,
            )

            # 5. Build Graph / Chart Datasets
            timeline_labels = [r["date"] for r in raw_sales]
            timeline_units = [r["units_sold"] for r in raw_sales]
            timeline_rev = [round(r["units_sold"] * (r["price"] if r["price"] > 0 else unit_price), 2) for r in raw_sales]
            timeline_stocks = [r.get("stock_after") for r in raw_sales]
            timeline_trends = [r.get("trend_score") for r in raw_sales]
            data_points = [
                DailySalesDataPoint(
                    date=r["date"],
                    units_sold=r["units_sold"],
                    revenue=round(r["units_sold"] * (r["price"] if r["price"] > 0 else unit_price), 2),
                    price=r["price"],
                    stock_after=r.get("stock_after"),
                    channel=r.get("channel", "POS_STORE"),
                    promotion_applied=r.get("promotion_applied", False),
                    trend_score=r.get("trend_score"),
                )
                for r in raw_sales
            ]

            timeline_chart = TimelineChart(
                title=f"Daily Sales & Revenue Timeline for {product.product_name}",
                chart_type="line",
                labels=timeline_labels,
                datasets=TimelineChartDataset(
                    units_sold=timeline_units,
                    revenue=timeline_rev,
                    stock_levels=timeline_stocks,
                    trend_scores=timeline_trends,
                ),
                data_points=data_points,
            )

            cum_u = []
            cum_r = []
            curr_u = 0
            curr_r = 0.0
            for u, r in zip(timeline_units, timeline_rev):
                curr_u += u
                curr_r += r
                cum_u.append(curr_u)
                cum_r.append(round(curr_r, 2))

            cumulative_chart = CumulativeSalesChart(
                title=f"Cumulative Growth for {product.product_name}",
                chart_type="area",
                labels=timeline_labels,
                cumulative_units=cum_u,
                cumulative_revenue=cum_r,
            )

            channel_stats = {}
            for r in raw_sales:
                ch = r.get("channel") or "POS_STORE"
                if ch not in channel_stats:
                    channel_stats[ch] = {"units": 0, "rev": 0.0}
                channel_stats[ch]["units"] += r["units_sold"]
                channel_stats[ch]["rev"] += r["units_sold"] * (r["price"] if r["price"] > 0 else unit_price)

            channel_items = []
            for ch, st in channel_stats.items():
                pct = round((st["units"] / total_units * 100), 1) if total_units > 0 else 0.0
                channel_items.append(
                    ChannelBreakdownItem(
                        channel=ch,
                        units_sold=st["units"],
                        revenue=round(st["rev"], 2),
                        percentage=pct,
                    )
                )
            if not channel_items:
                channel_items.append(
                    ChannelBreakdownItem(channel="POS_STORE", units=0, revenue=0.0, percentage=100.0)
                )

            channel_chart = ChannelBreakdownChart(
                title="Sales Breakdown by Channel",
                chart_type="pie",
                channels=channel_items,
            )

            benchmark_chart = StockBenchmarkChart(
                title=f"Inventory Benchmarks ({product.product_id})",
                chart_type="bar",
                categories=[
                    "Current Stock",
                    "Low Stock Threshold",
                    "Safety Stock",
                    "Reorder Point",
                    "Target Stock",
                ],
                values=[
                    product.current_stock,
                    product.low_stock_threshold,
                    p_safety,
                    p_reorder,
                    p_target,
                ],
            )

            run_rate = p_avg if p_avg > 0 else (avg_daily_sales if avg_daily_sales > 0 else 1.0)
            depletion_points = []
            stockout_day = None
            start_date = datetime.utcnow()
            for day_idx in range(1, 31):
                proj_date = (start_date + timedelta(days=day_idx)).strftime("%Y-%m-%d")
                stock_left = max(0.0, round(product.current_stock - (day_idx * run_rate), 1))
                if stock_left == 0.0 and stockout_day is None and product.current_stock > 0:
                    stockout_day = day_idx
                depletion_points.append(
                    ProjectedDepletionPoint(
                        day=day_idx,
                        date=proj_date,
                        projected_stock=stock_left,
                        safety_stock=p_safety,
                        reorder_point=p_reorder,
                    )
                )

            depletion_chart = ProjectedDepletionChart(
                title="30-Day Stock Depletion Projection",
                chart_type="line",
                days_until_stockout=stockout_day,
                daily_run_rate=round(run_rate, 2),
                projection=depletion_points,
            )

            charts = ProductChartsRepresentation(
                timeline_chart=timeline_chart,
                cumulative_sales_chart=cumulative_chart,
                channel_breakdown_chart=channel_chart,
                stock_benchmark_chart=benchmark_chart,
                projected_depletion_chart=depletion_chart,
            )

            # 6. ML Forecast Insights
            ml_insights = MLForecastInsights(available=False)
            try:
                pred_res = cls.predict_sku(pid)
                ml_insights = MLForecastInsights(
                    available=True,
                    predicted_7_day_demand=pred_res.predicted_7_day_demand,
                    stockout_probability=pred_res.stockout_probability,
                    recommended_order_quantity=pred_res.recommended_order_quantity,
                    alert_level=pred_res.alert_level,
                )
            except Exception:
                pass

            # 7. Stock status
            if product.current_stock == 0:
                stock_status = "OUT_OF_STOCK"
            elif product.current_stock <= product.low_stock_threshold:
                stock_status = "LOW_STOCK"
            elif p_target and product.current_stock > p_target * 1.5:
                stock_status = "OVERSTOCKED"
            else:
                stock_status = "HEALTHY"

            return ProductFullAnalyticsResponse(
                product=product,
                stock_status=stock_status,
                stock_plan=stock_plan_dto,
                sales_summary=sales_summary,
                charts=charts,
                ml_forecast=ml_insights,
            )

        finally:
            if close_session and db is not None:
                db.close()
