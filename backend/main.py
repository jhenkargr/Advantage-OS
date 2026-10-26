"""
main.py
-------
AdVantage OS — Modular Unified FastAPI Backend for SMEs.

Modular Architecture:
  • Creative Studio        (creative_studio/)  → /image_generation/*, /generate/*
  • Trend Prediction       (trend_predictor/)  → /predict, /predict/upload-model, /predict/model-status,
                                                 /api/trends/analyze/{sme}, /api/keywords/related
  • Core Inventory & Ops   (inventory/)        → /api/inventory/*, /api/demand/stock-plan,
                                                 /api/dashboard/*, /api/influencers/*,
                                                 /api/products/upload-csv, /api/sales-history/upload-csv

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Path setup ─────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv(dotenv_path=_HERE / ".env")

# ── Modular Domain Routers ─────────────────────────────────────────────────────
from inventory.config import (
    settings,
    CATALOG_PATH,
    DEMAND_MODEL_PATH,
    STOCKOUT_MODEL_PATH,
    TESSERACT_CMD,
)
from inventory.api.api_router import api_router
from inventory.services.inventory_service import load_ml_models
from inventory.db import init_db, seed_data_from_csv

import creative_studio.nvidia_client as nv
from creative_studio import (
    router as creative_router,
    EXPOSED_HEADERS as CREATIVE_EXPOSED_HEADERS,
)
from trend_predictor import (
    predict_router,
    trends_router,
    keywords_router,
)
from inventory.demand import router as demand_router


# ── Lifespan (Startup / Shutdown Orchestrator) ─────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 65)
    print(f"🚀 Starting AdVantage OS v{settings.app_version}")
    print(f"📂 Catalog      : {CATALOG_PATH} (exists={CATALOG_PATH.exists()})")
    print(f"🤖 Demand model : {DEMAND_MODEL_PATH} (exists={DEMAND_MODEL_PATH.exists()})")
    print(f"🤖 Stockout mdl : {STOCKOUT_MODEL_PATH} (exists={STOCKOUT_MODEL_PATH.exists()})")
    print(f"👁️  Tesseract    : {TESSERACT_CMD} (exists={Path(TESSERACT_CMD).exists()})")

    # Initialize & seed database
    try:
        init_db()
        seed_data_from_csv()
        print("🐘 Database connection and tables verified.")
    except Exception as e:
        print(f"⚠️  Database connection warning: {e}")

    # Preload ML models
    try:
        load_ml_models()
        print("✅ ML inventory models loaded.")
    except Exception as e:
        print(f"⚠️  Model preload warning: {e}")

    # Pre-warm stockout alerts in the background so client requests load instantly (<5ms)
    def _warmup_alerts_cache():
        try:
            from inventory.services.alert_service import AlertService
            AlertService.get_all_alerts(force_refresh=True)
            print("⚡ Stockout alerts cache pre-warmed & ready.")
        except Exception as err:
            print(f"⚠️ Stockout alerts pre-warm notice: {err}")

    import threading
    threading.Thread(target=_warmup_alerts_cache, daemon=True).start()

    print("📡 Docs: http://localhost:8000/docs")
    print("=" * 65)
    yield
    print("🛑 Shutting down AdVantage OS")


# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AdVantage OS — Unified API",
    description=(
        "AI-Driven Local Growth Platform for SMEs.\n\n"
        "Combines trend prediction (Reddit + X scraping), creative image generation "
        "(NVIDIA API Catalog / Groq / Pollinations), inventory forecasting (ML demand + stockout), "
        "OCR receipt ingestion, influencer matchmaking, and WhatsApp alerts."
    ),
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=CREATIVE_EXPOSED_HEADERS,
)


# ── Mount Feature Routers ──────────────────────────────────────────────────────
# 1. Core Inventory, Dashboard, Influencers & CSV Uploads (/api/...)
app.include_router(api_router, prefix=settings.api_prefix)

# 2. Keyword Intelligence (/api/keywords/...)
app.include_router(keywords_router, prefix=settings.api_prefix)

# 3. Creative Studio (/image_generation/* and /generate/*)
app.include_router(creative_router)

# 4. Trend Prediction & Model DB Persistence (/predict/*)
app.include_router(predict_router)

# 5. SME Trend Analysis (/api/trends/analyze/*)
app.include_router(trends_router)

# 6. Demand Forecasting & Stock Planning (/api/demand/*)
app.include_router(demand_router)


# ── System Discovery & Health Endpoints ─────────────────────────────────────────
@app.get("/", tags=["System"])
def root():
    """Welcome and API discovery endpoint."""
    return {
        "system": "AdVantage OS",
        "description": "AI-Driven Local Growth Platform for SMEs",
        "version": settings.app_version,
        "status": "online",
        "docs": "/docs",
        "modules": {
            "Keyword Expansion (Cohere)": "GET /api/keywords/related",
            "SME Trend Analysis (Cohere)": "GET /api/trends/analyze/{sme}",
            "Trend Prediction":    "POST /predict  |  POST /predict/upload-model  |  GET /predict/model-status",
            "Image Generation":    "POST /image_generation/text2img  |  /img2img  |  /caption",
            "Demand Stock Plan":   "GET  /api/demand/stock-plan",
            "Inventory Catalog":   "GET  /api/inventory/catalog  |  GET /api/inventory/catalog/{id}/analytics",
            "CSV Uploads":         "POST /api/products/upload-csv  |  POST /api/sales-history/upload-csv",
            "Influencers":         "POST /api/influencers/upload-csv  |  GET /api/influencers/cluster/{cluster_id}",
            "Dashboard":           "GET  /api/dashboard/stats",
        },
    }


@app.get("/health", tags=["System"])
def health_check():
    """Unified health check — covers database, data files, ML models, OCR, and Creative Studio."""
    db_ok = False
    try:
        from inventory.db import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            db_ok = True
    except Exception:
        db_ok = False

    return JSONResponse(content={
        "status": "healthy",
        "version": settings.app_version,
        "database_ready": db_ok,
        "database_type": "Neon PostgreSQL",
        "catalog_ready":  CATALOG_PATH.is_file(),
        "models_ready":   DEMAND_MODEL_PATH.is_file() and STOCKOUT_MODEL_PATH.is_file(),
        "ocr_ready":      Path(TESSERACT_CMD).exists(),
        "creative_studio_ready": bool(nv.API_KEY),
        "creative_providers": "NVIDIA API Catalog + Groq + Pollinations",
    })