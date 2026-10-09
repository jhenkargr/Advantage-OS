"""Application configuration for AdVantage OS Backend."""

import os
from pathlib import Path
from pydantic import BaseModel

# Project directory paths
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
INVENTORY_DIR = WORKSPACE_ROOT / "inventory"
DATA_DIR = INVENTORY_DIR / "data"
MODELS_DIR = INVENTORY_DIR / "models"
CATALOG_PATH = DATA_DIR / "product_catalog.csv"
SALES_LOG_PATH = DATA_DIR / "sales_log.csv"
DEMAND_MODEL_PATH = MODELS_DIR / "demand_regressor_expanded.pkl"
STOCKOUT_MODEL_PATH = MODELS_DIR / "stockout_model_expanded.pkl"

# Receipts and uploads directory
RECEIPTS_DIR = DATA_DIR / "receipts"
RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)

# Tesseract executable configuration
DEFAULT_TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CMD = os.environ.get("TESSERACT_CMD", DEFAULT_TESSERACT_CMD)


class Settings(BaseModel):
    app_name: str = "AdVantage OS API"
    app_version: str = "1.0.0"
    app_description: str = (
        "AI-Driven Local Growth Platform for SMEs — FastAPI Backend powering "
        "Trend Analysis, Influencer Matching, Creative Studio, OCR Receipt Ingestion, "
        "and ML-Driven Inventory Demand & Stockout Forecasting."
    )
    api_prefix: str = "/api"
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "*",
    ]


settings = Settings()
