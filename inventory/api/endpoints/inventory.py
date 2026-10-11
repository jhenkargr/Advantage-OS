"""Inventory and ML Forecasting API Endpoints."""

import io
import os
from pathlib import Path
from typing import Optional, List
import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from inventory.db import Base, Product, SalesLog, StockPlan, get_db, save_stock_plan_to_db, ensure_tables
from inventory.models.schemas import (
    ProductItem,
    ProductCreate,
    ProductUpdate,
    StockUpdate,
    InventoryPredictionResponse,
    InventorySummaryResponse,
    SalesHistoryPoint,
    ProductUploadResponse,
    SalesHistoryUploadResponse,
    StockPlanItem,
    StockPlanResponse,
    ProductFullAnalyticsResponse,
)
from inventory.services.inventory_service import InventoryService

router = APIRouter(prefix="/inventory", tags=["Inventory & Demand Prediction"])


@router.get("/catalog", response_model=list[ProductItem], summary="Get product catalog")
def get_catalog(db: Session = Depends(get_db)):
    """List all available products in the SME catalog with current stock levels."""
    try:
        return InventoryService.list_all_products(db=db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/catalog/{product_id}", response_model=ProductItem, summary="Get single product")
def get_product(product_id: str, db: Session = Depends(get_db)):
    """Retrieve catalog item details by product SKU ID."""
    try:
        return InventoryService.get_product(product_id, db=db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/catalog/{product_id}/analytics",
    response_model=ProductFullAnalyticsResponse,
    summary="Get complete product analytics, sales history, stock plan, and graph representations",
)
def get_product_analytics(product_id: str, db: Session = Depends(get_db)):
    """
    Fetch comprehensive product intelligence combining:
    - Base product catalog record (products table / CSV)
    - Complete sales history & logs (sales_logs table / sales_history.csv / sales_log.csv)
    - Stock replenishment & forecasting plan (stock_plans table / stock_plan.csv)
    - Pre-structured datasets formatted for UI charts (Timeline, Cumulative, Channel distribution, Benchmarks, 30-Day depletion curve)
    - ML inference predictions (Demand & stockout risk)
    """
    try:
        return InventoryService.get_product_full_analytics(product_id, db=db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/catalog/{product_id}/details",
    response_model=ProductFullAnalyticsResponse,
    summary="Get comprehensive product details and charts (alias for /analytics)",
)
def get_product_details(product_id: str, db: Session = Depends(get_db)):
    """Alias for /catalog/{product_id}/analytics."""
    return get_product_analytics(product_id, db=db)


@router.get(
    "/catalog/{product_id}/insights",
    response_model=ProductFullAnalyticsResponse,
    summary="Get product insights, sales logs, and stock plan graphs (alias for /analytics)",
)
def get_product_insights(product_id: str, db: Session = Depends(get_db)):
    """Alias for /catalog/{product_id}/analytics."""
    return get_product_analytics(product_id, db=db)


@router.post("/catalog", response_model=ProductItem, summary="Add or update product in catalog")
def create_or_update_product(product: ProductCreate, db: Session = Depends(get_db)):
    """Add a new product or update an existing product in the catalog."""
    try:
        return InventoryService.create_or_update_product(product, db=db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/catalog/{product_id}", response_model=ProductItem, summary="Update product in catalog")
def update_product_by_id(product_id: str, product_update: ProductUpdate, db: Session = Depends(get_db)):
    """Update an existing product's details in the catalog by SKU ID."""
    try:
        payload = product_update.model_dump(exclude_unset=True) if hasattr(product_update, "model_dump") else product_update.dict(exclude_unset=True)
        payload["product_id"] = product_id
        return InventoryService.create_or_update_product(payload, db=db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/catalog", response_model=ProductItem, summary="Update product in catalog")
def update_product(product_update: ProductUpdate, db: Session = Depends(get_db)):
    """Update an existing product's details in the catalog."""
    if not product_update.product_id:
        raise HTTPException(status_code=400, detail="product_id is required in request body")
    try:
        return InventoryService.create_or_update_product(product_update, db=db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/catalog/{product_id}/stock", response_model=ProductItem, summary="Update product stock")
def update_stock(product_id: str, stock_update: StockUpdate, db: Session = Depends(get_db)):
    """Manually update the current stock level for a SKU."""
    try:
        return InventoryService.update_stock(product_id, stock_update.new_stock, db=db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predict/{product_id}", response_model=InventoryPredictionResponse, summary="Predict demand & stockout for SKU")
def predict_product_inventory(product_id: str):
    """
    Run ML inference models for a SKU:
    - 7-day future demand forecast (Demand Regressor)
    - 3-day stockout probability (Stockout Classifier)
    - Safety stock and recommended reorder quantity
    - Dynamic alert level (HIGH, MEDIUM, LOW, NONE)
    """
    try:
        return InventoryService.predict_sku(product_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML Prediction Error: {e}")


# ── CSV Upload & Stock Plan Utilities ──────────────────────────────────────────

_HERE_PATH = Path(__file__).resolve()
_CANDIDATE_DIRS = [
    _HERE_PATH.parents[4] if len(_HERE_PATH.parents) >= 5 else None,
    _HERE_PATH.parents[3] if len(_HERE_PATH.parents) >= 4 else None,
    Path(r"D:\major project\Backend\virenv\Scripts"),
    Path("."),
]
_SCRIPTS_DIR = next((d for d in _CANDIDATE_DIRS if d and (d / "products.csv").exists()), _HERE_PATH.parents[4])

DEFAULT_PRODUCTS_CSV = _SCRIPTS_DIR / "products.csv"
DEFAULT_SALES_CSV = _SCRIPTS_DIR / "sales_history.csv"
DEFAULT_STOCK_PLAN_CSV = _SCRIPTS_DIR / "stock_plan.csv"


def _run_demand_model_and_save(
    db: Session,
    sales_path: Path = DEFAULT_SALES_CSV,
    products_path: Path = DEFAULT_PRODUCTS_CSV,
    out_path: Path = DEFAULT_STOCK_PLAN_CSV,
) -> tuple[int, list[StockPlanItem]]:
    """Runs demand.py ML model and saves output to stock_plan.csv and Neon/SQLite database."""
    try:
        from demand import run as run_demand_forecast
    except ImportError:
        import sys
        adv_dir = Path(__file__).resolve().parent.parent.parent
        if str(adv_dir) not in sys.path:
            sys.path.insert(0, str(adv_dir))
        from demand import run as run_demand_forecast

    out_base = str(out_path.parent / out_path.stem)
    df_plan = run_demand_forecast(
        sales=str(sales_path),
        products=str(products_path),
        out=out_base,
    )

    save_stock_plan_to_db(df_plan, db=db)

    plan_records = db.query(StockPlan).all()
    items = [
        StockPlanItem(
            product_id=p.product_id,
            product_name=p.product_name,
            avg_daily_demand=p.avg_daily_demand,
            forecast_30d=p.forecast_30d,
            current_stock=p.current_stock,
            safety_stock=p.safety_stock,
            reorder_point=p.reorder_point,
            target_stock=p.target_stock,
            order_qty_now=p.order_qty_now,
            status=p.status,
            note=p.note,
        )
        for p in plan_records
    ]
    return len(items), items


async def upload_products(
    file: Optional[UploadFile] = File(None, description="CSV file containing products (e.g. products.csv)"),
    replace_all: bool = Form(False, description="Set True to wipe existing products before importing"),
    recalculate_stock_plan: bool = Form(True, description="Automatically recalculate and save stock plan to database and stock_plan.csv"),
    db: Session = Depends(get_db),
) -> ProductUploadResponse:
    dest_path = DEFAULT_PRODUCTS_CSV
    filename = file.filename if file and file.filename else dest_path.name

    if file is not None and file.filename:
        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Invalid file format. Please upload a .csv file.")
        content = await file.read()
        if not content.strip():
            raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {e}")
        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(content)
        except Exception as e:
            print(f"[upload-products] Warning saving to {dest_path}: {e}")
    else:
        if not dest_path.is_file():
            raise HTTPException(status_code=400, detail=f"Default file {dest_path} not found. Please upload a CSV file.")
        df = pd.read_csv(dest_path)

    if "product_id" not in df.columns:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required column 'product_id'. Detected columns: {list(df.columns)}",
        )

    # Guarantee schema and tables exist in database (PostgreSQL / SQLite)
    try:
        ensure_tables(db.get_bind())
    except Exception as exc:
        print(f"[upload-products] Table ensure notice: {exc}")

    if replace_all:
        try:
            db.query(Product).delete()
            db.commit()
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to clear products: {e}")

    inserted_count = 0
    updated_count = 0
    sample_prods = []

    try:
        existing_map = {p.product_id: p for p in db.query(Product).all()}
        new_objects = []

        for _, row in df.iterrows():
            pid = str(row["product_id"]).strip()
            if not pid or pid.lower() == "nan":
                continue

            pname = str(row.get("product_name", pid)).strip()
            cat = str(row.get("category", "General")).strip()

            raw_stock = row.get("stock_available", row.get("current_stock", 0))
            stock_val = int(raw_stock) if pd.notna(raw_stock) else 0

            raw_price = row.get("current_price", row.get("unit_price", 0.0))
            price_val = float(raw_price) if pd.notna(raw_price) else 0.0

            low_stock = int(row.get("low_stock_threshold", 10)) if pd.notna(row.get("low_stock_threshold")) else 10
            lead_time = int(row.get("supplier_lead_time_days", 3)) if pd.notna(row.get("supplier_lead_time_days")) else 3
            reorder_qty = int(row.get("reorder_quantity", 20)) if pd.notna(row.get("reorder_quantity")) else 20

            if pid in existing_map:
                p = existing_map[pid]
                p.product_name = pname
                p.category = cat
                p.current_stock = stock_val
                p.unit_price = price_val
                p.low_stock_threshold = low_stock
                p.supplier_lead_time_days = lead_time
                p.reorder_quantity = reorder_qty
                updated_count += 1
            else:
                prod = Product(
                    product_id=pid,
                    product_name=pname,
                    category=cat,
                    current_stock=stock_val,
                    unit_price=price_val,
                    low_stock_threshold=low_stock,
                    supplier_lead_time_days=lead_time,
                    reorder_quantity=reorder_qty,
                )
                new_objects.append(prod)
                existing_map[pid] = prod
                inserted_count += 1

            if len(sample_prods) < 5:
                sample_prods.append({
                    "product_id": pid,
                    "product_name": pname,
                    "category": cat,
                    "stock": stock_val,
                    "price": price_val,
                })

        if new_objects:
            db.bulk_save_objects(new_objects)
        db.commit()

        sp_generated = False
        sp_rows = 0
        if recalculate_stock_plan and DEFAULT_SALES_CSV.is_file():
            try:
                sp_rows, _ = _run_demand_model_and_save(
                    db=db,
                    sales_path=DEFAULT_SALES_CSV,
                    products_path=dest_path,
                    out_path=DEFAULT_STOCK_PLAN_CSV,
                )
                sp_generated = True
            except Exception as exc:
                print(f"[upload-products] Demand forecasting warning: {exc}")

        total_in_db = db.query(Product).count()

        return ProductUploadResponse(
            status="success",
            message=f"Processed {len(df)} products: {inserted_count} added, {updated_count} updated in database.",
            filename=filename,
            total_rows_parsed=len(df),
            inserted_count=inserted_count,
            updated_count=updated_count,
            total_products_in_db=total_in_db,
            stock_plan_generated=sp_generated,
            stock_plan_rows=sp_rows,
            stock_plan_csv_path=str(DEFAULT_STOCK_PLAN_CSV) if sp_generated else None,
            sample_products=sample_prods,
        )

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error inserting products into database: {e}")


async def upload_sales_history(
    file: Optional[UploadFile] = File(None, description="CSV file containing sales history (e.g. sales_history.csv)"),
    replace_all: bool = Form(False, description="Set True to clear existing sales logs before importing"),
    recalculate_stock_plan: bool = Form(True, description="Automatically calculate demand forecast and save stock plan to database and stock_plan.csv"),
    db: Session = Depends(get_db),
) -> SalesHistoryUploadResponse:
    dest_path = DEFAULT_SALES_CSV
    filename = file.filename if file and file.filename else dest_path.name

    if file is not None and file.filename:
        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Invalid file format. Please upload a .csv file.")
        content = await file.read()
        if not content.strip():
            raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {e}")
        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(content)
        except Exception as e:
            print(f"[upload-sales-history] Warning saving to {dest_path}: {e}")
    else:
        if not dest_path.is_file():
            raise HTTPException(status_code=400, detail=f"Default file {dest_path} not found. Please upload a CSV file.")
        df = pd.read_csv(dest_path)

    required_cols = {"date", "product_id"}
    if not required_cols.issubset(df.columns):
        raise HTTPException(
            status_code=422,
            detail=f"Missing required columns {required_cols}. Detected: {list(df.columns)}",
        )

    qty_col = "units_sold" if "units_sold" in df.columns else ("item_sold" if "item_sold" in df.columns else None)
    if not qty_col:
        raise HTTPException(status_code=422, detail="Missing quantity column ('units_sold' or 'item_sold').")

    # Guarantee schema and tables exist in database (PostgreSQL / SQLite)
    try:
        ensure_tables(db.get_bind())
    except Exception as exc:
        print(f"[upload-sales-history] Table ensure notice: {exc}")

    if replace_all:
        try:
            db.query(SalesLog).delete()
            db.commit()
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to clear sales logs: {e}")

    try:
        existing_pids = {p.product_id for p in db.query(Product.product_id).all()}
        unique_sales_pids = set(df["product_id"].astype(str).str.strip())
        missing_pids = unique_sales_pids - existing_pids
        if missing_pids:
            new_stubs = [
                Product(
                    product_id=mpid,
                    product_name=f"Product {mpid}",
                    category="General",
                    current_stock=0,
                    unit_price=0.0,
                )
                for mpid in missing_pids
            ]
            db.bulk_save_objects(new_stubs)
            db.commit()

        sales_records = []
        for _, row in df.iterrows():
            pid = str(row["product_id"]).strip()
            date_str = str(row["date"]).strip()
            raw_qty = row[qty_col]
            units = int(raw_qty) if pd.notna(raw_qty) else 0

            raw_price = row.get("price", row.get("unit_price", 0.0))
            price_val = float(raw_price) if pd.notna(raw_price) else 0.0

            channel_val = str(row.get("channel", "POS_STORE")).strip()
            promo_val = bool(row.get("promotion_applied", False)) if pd.notna(row.get("promotion_applied")) else False
            disc_val = float(row.get("discount_percent", 0.0)) if pd.notna(row.get("discount_percent")) else 0.0

            sales_records.append(
                SalesLog(
                    product_id=pid,
                    date=date_str,
                    units_sold=units,
                    price=price_val,
                    channel=channel_val,
                    promotion_applied=promo_val,
                    discount_percent=disc_val,
                )
            )

        if sales_records:
            db.bulk_save_objects(sales_records)
            db.commit()

        sp_generated = False
        sp_rows = 0
        summary_items: list[StockPlanItem] = []
        if recalculate_stock_plan and DEFAULT_PRODUCTS_CSV.is_file():
            try:
                sp_rows, summary_items = _run_demand_model_and_save(
                    db=db,
                    sales_path=dest_path,
                    products_path=DEFAULT_PRODUCTS_CSV,
                    out_path=DEFAULT_STOCK_PLAN_CSV,
                )
                sp_generated = True
            except Exception as exc:
                print(f"[upload-sales-history] Demand forecasting warning: {exc}")

        total_sales_in_db = db.query(SalesLog).count()

        return SalesHistoryUploadResponse(
            status="success",
            message=f"Successfully imported {len(sales_records)} sales records into database.",
            filename=filename,
            total_rows_parsed=len(df),
            inserted_count=len(sales_records),
            total_sales_logs_in_db=total_sales_in_db,
            stock_plan_generated=sp_generated,
            stock_plan_rows=sp_rows,
            stock_plan_csv_path=str(DEFAULT_STOCK_PLAN_CSV) if sp_generated else None,
            stock_plan_summary=summary_items[:10],
        )

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error importing sales history into database: {e}")

