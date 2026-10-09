"""Main FastAPI application entry point for AdVantage OS."""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import (
    settings,
    CATALOG_PATH,
    DEMAND_MODEL_PATH,
    STOCKOUT_MODEL_PATH,
    TESSERACT_CMD,
)
from app.api.api_router import api_router
from app.services.inventory_service import load_ml_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle handler."""
    print("=" * 60)
    print(f"🚀 Starting {settings.app_name} v{settings.app_version}")
    print(f"📂 Catalog path: {CATALOG_PATH} (Exists: {CATALOG_PATH.exists()})")
    print(f"🤖 Demand model: {DEMAND_MODEL_PATH} (Exists: {DEMAND_MODEL_PATH.exists()})")
    print(f"🤖 Stockout model: {STOCKOUT_MODEL_PATH} (Exists: {STOCKOUT_MODEL_PATH.exists()})")
    print(f"👁️ Tesseract OCR: {TESSERACT_CMD} (Exists: {Path(TESSERACT_CMD).exists()})")

    # Warm up ML models
    try:
        load_ml_models()
        print("✅ Pre-trained ML inference models successfully loaded into memory.")
    except Exception as e:
        print(f"⚠️ Warning during model preload: {e}")

    print(f"📡 API documentation available at: http://{settings.host}:{settings.port}/docs")
    print("=" * 60)
    yield
    print(f"🛑 Shutting down {settings.app_name}")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=settings.app_description,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS Middleware for Frontend Access (React / Vite / Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all feature routes
app.include_router(api_router, prefix=settings.api_prefix)


@app.get("/", tags=["System"])
def root_endpoint():
    """Welcome and system discovery endpoint."""
    return {
        "system": "AdVantage OS",
        "description": "An AI-Driven Local Growth Platform for SMEs",
        "version": settings.app_version,
        "status": "online",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "architecture_layers": {
            "Module 1": "Data Ingestion & Trend Engine (/api/trends)",
            "Module 2": "Campaign Creation & Influencer Matchmaker (/api/influencers, /api/creatives)",
            "Module 3": "Retail Operations & OCR Inventory Edge (/api/ocr)",
            "Module 4": "Analytics, ML Demand Forecasting & WhatsApp Alerts (/api/inventory, /api/alerts)",
            "Overview": "Consolidated Executive Dashboard (/api/dashboard/stats)",
        },
    }


@app.get("/health", tags=["System"])
def health_check():
    """System health check verifying data files, ML models, and OCR dependencies."""
    return JSONResponse(
        content={
            "status": "healthy",
            "version": settings.app_version,
            "catalog_ready": CATALOG_PATH.is_file(),
            "models_ready": DEMAND_MODEL_PATH.is_file() and STOCKOUT_MODEL_PATH.is_file(),
            "ocr_engine_ready": Path(TESSERACT_CMD).exists(),
        }
    )
