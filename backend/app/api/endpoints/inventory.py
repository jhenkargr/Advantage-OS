"""Inventory and ML Forecasting API Endpoints."""

from fastapi import APIRouter, HTTPException, Query
from app.models.schemas import (
    ProductItem,
    ProductCreate,
    StockUpdate,
    InventoryPredictionResponse,
    InventorySummaryResponse,
    SalesHistoryPoint,
)
from app.services.inventory_service import InventoryService

router = APIRouter(prefix="/inventory", tags=["Inventory & Demand Prediction"])


@router.get("/catalog", response_model=list[ProductItem], summary="Get product catalog")
def get_catalog():
    """List all available products in the SME catalog with current stock levels."""
    try:
        return InventoryService.list_all_products()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/catalog/{product_id}", response_model=ProductItem, summary="Get single product")
def get_product(product_id: str):
    """Retrieve catalog item details by product SKU ID."""
    try:
        return InventoryService.get_product(product_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/catalog", response_model=ProductItem, status_code=201, summary="Add new product to catalog")
def create_product(product: ProductCreate):
    """Add a new product to the catalog."""
    try:
        return InventoryService.create_product(product)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/catalog/{product_id}/stock", response_model=ProductItem, summary="Update product stock")
def update_stock(product_id: str, stock_update: StockUpdate):
    """Manually update the current stock level for a SKU."""
    try:
        return InventoryService.update_stock(product_id, stock_update.new_stock)
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


@router.get("/predict-all", response_model=InventorySummaryResponse, summary="Batch ML predictions across all catalog SKUs")
def predict_all_products():
    """
    Batch evaluate all products in the catalog through the demand regressor and stockout models.
    Returns high-risk count, medium-risk count, and replenishment plan.
    """
    try:
        return InventoryService.predict_all()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sales-history/{product_id}", response_model=list[SalesHistoryPoint], summary="Get recent sales history")
def get_sales_history(product_id: str, limit: int = Query(30, ge=7, le=90)):
    """Retrieve historical daily sales velocity, stock levels, and trend scores for charting."""
    try:
        return InventoryService.get_sales_history(product_id, limit)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
