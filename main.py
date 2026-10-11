"""
main.py
-------
AdVantage OS — Unified FastAPI backend.

This file is the single entry point. It wires together:
  • Trend prediction engine  (Reddit + X scraping, sentiment, 30-day forecast)
  • Image generation studio  (text2img, img2img, ad caption via HuggingFace)
  • Full backend/app module  (inventory, OCR, trends, influencers, creatives,
                              alerts, executive dashboard)

All backend/app routes are mounted under the /api prefix (mirrors backend/run.py).

Endpoints
---------
GET  /                          Root discovery
GET  /health                    Unified health check
POST /predict                   Trend forecast (Reddit + X)
GET  /image_generation/health   Creative studio status
POST /image_generation/text2img Text → PNG image
POST /image_generation/img2img  Image + prompt → PNG image
POST /image_generation/caption  Description → ad caption
GET  /api/keywords/related      Get related keywords for SME trend prediction (Cohere)

--- Mounted from backend/app (prefix: /api) ---
GET  /api/dashboard/stats
GET  /api/trends/active
GET  /api/trends/{trend_id}
POST /api/trends/refresh
POST /api/influencers/upload-csv            Upload influencer clusters CSV to database
GET  /api/influencers/cluster/{cluster_id}  Get influencers by cluster ID
POST /api/creatives/generate
POST /api/creatives/score-ctr
GET  /api/creatives/campaigns
POST /api/creatives/campaigns
GET  /api/inventory/catalog
POST /api/inventory/catalog
PUT  /api/inventory/catalog/{product_id}
GET  /api/inventory/catalog/{product_id}
GET  /api/inventory/catalog/{product_id}/analytics
PUT  /api/inventory/catalog/{product_id}/stock
GET  /api/inventory/predict/{product_id}
GET  /api/demand/stock-plan                 Demand forecast directly from database tables & update stock_plans

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Path setup ─────────────────────────────────────────────────────────────────
# advantage/ directory  →  resolves predict, creative, reddit_scrap, x_scrapper, inventory
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from typing import Optional
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

load_dotenv(dotenv_path=_HERE / ".env")

# ── Advantage-module imports ───────────────────────────────────────────────────
from predict import (                       # noqa: E402
    run_prediction,
    save_model_bytes_to_db,
    get_prediction_model,
    MODEL_PATH,
)
from inventory.db import get_db, get_ml_model_from_db  # noqa: E402
from creative import (                      # noqa: E402
    router as creative_router,
    EXPOSED_HEADERS as CREATIVE_EXPOSED_HEADERS,
)
import nvidia_client as nv                  # noqa: E402
from trend_engine import discover_trends, SME_PROFILES  # noqa: E402
from trend_analyzer_cohere import analyze_trends       # noqa: E402
from trend_db import get_cached_trends, save_trends_to_db  # noqa: E402
from keywords import (                                     # noqa: E402
    router as keywords_router,
    get_related_keywords,
    expand_keywords,
)
from demand import run_from_database                       # noqa: E402

# ── Inventory module imports ───────────────────────────────────────────────────
from inventory.config import (              # noqa: E402
    settings,
    CATALOG_PATH,
    DEMAND_MODEL_PATH,
    STOCKOUT_MODEL_PATH,
    TESSERACT_CMD,
)
from inventory.api.api_router import api_router   # noqa: E402
from inventory.services.inventory_service import load_ml_models  # noqa: E402
from inventory.db import init_db, seed_data_from_csv             # noqa: E402


# ── Lifespan (startup / shutdown) ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 65)
    print(f"🚀 Starting AdVantage OS v{settings.app_version}")
    print(f"📂 Catalog      : {CATALOG_PATH} (exists={CATALOG_PATH.exists()})")
    print(f"🤖 Demand model : {DEMAND_MODEL_PATH} (exists={DEMAND_MODEL_PATH.exists()})")
    print(f"🤖 Stockout mdl : {STOCKOUT_MODEL_PATH} (exists={STOCKOUT_MODEL_PATH.exists()})")
    print(f"👁️  Tesseract    : {TESSERACT_CMD} (exists={Path(TESSERACT_CMD).exists()})")
    
    # Initialize & seed Neon PostgreSQL
    try:
        init_db()
        seed_data_from_csv()
        print("🐘 Neon PostgreSQL database connected and verified.")
    except Exception as e:
        print(f"⚠️  Database connection warning: {e}")

    try:
        load_ml_models()
        print("✅ ML inventory models loaded.")
    except Exception as e:
        print(f"⚠️  Model preload warning: {e}")
    print(f"📡 Docs: http://localhost:8000/docs")
    print("=" * 65)
    yield
    print("🛑 Shutting down AdVantage OS")


# ── App setup ──────────────────────────────────────────────────────────────────
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

# Mount the full backend/app router under /api
app.include_router(api_router, prefix=settings.api_prefix)

# Mount keyword intelligence router (Cohere) under /api
app.include_router(keywords_router, prefix=settings.api_prefix)

# Mount Creative Studio router (NVIDIA API Catalog & Groq)
app.include_router(creative_router)


# ── Schemas ────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    keywords: list[str] = Field(
        ...,
        min_length=1,
        example=["artificial intelligence", "machine learning"],
        description="One or more search keywords to analyse.",
    )
    use_transformer: bool = Field(
        default=True,
        description=(
            "Set to false to use only VADER (faster). "
            "True = DistilBERT + VADER ensemble (more accurate, slower)."
        ),
    )


# ── System routes ──────────────────────────────────────────────────────────────

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


# ── Trend Prediction routes ────────────────────────────────────────────────────

@app.post("/predict/upload-model", tags=["Trend Prediction"])
async def upload_prediction_model(
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
):
    """
    Upload `model_test_analysis.pkl` into the database (Neon PostgreSQL `ml_models` table).
    If no file is attached in the request, automatically uploads and syncs the existing local
    `model_test_analysis.pkl` file into the database.
    """
    if file is not None and file.filename:
        raw_bytes = await file.read()
        filename = file.filename
    else:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=400,
                detail=f"No file uploaded and local file {MODEL_PATH} does not exist.",
            )
        with open(MODEL_PATH, "rb") as f:
            raw_bytes = f.read()
        filename = MODEL_PATH.name

    try:
        result = save_model_bytes_to_db(raw_bytes, filename=filename, db=db)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process and store model in database: {e}",
        )


@app.get("/predict/model-status", tags=["Trend Prediction"])
def prediction_model_status(db: Session = Depends(get_db)):
    """Check whether model_test_analysis is stored in the database and active in memory."""
    from predict import _MODEL_CACHE, _MODEL_CACHE_SOURCE
    record = get_ml_model_from_db("model_test_analysis", db=db)
    local_exists = MODEL_PATH.exists()

    return {
        "model_name": "model_test_analysis",
        "in_database": record is not None,
        "database_storage": "Neon PostgreSQL (table: ml_models)",
        "database_file_size_bytes": record.file_size_bytes if record else None,
        "database_version": record.version if record else None,
        "database_updated_at": str(record.updated_at) if record else None,
        "local_file_exists": local_exists,
        "local_file_size_bytes": MODEL_PATH.stat().st_size if local_exists else None,
        "active_cache_source": _MODEL_CACHE_SOURCE,
        "in_memory_cached": _MODEL_CACHE is not None,
    }


@app.post("/predict", tags=["Trend Prediction"])
def predict(request: PredictRequest, db: Session = Depends(get_db)):
    """
    Scrapes Reddit + X for the given keywords, runs ensemble sentiment
    analysis (VADER + DistilBERT), and returns a 30-day trend forecast
    using the model retrieved from the database.
    """
    keywords = [kw.strip() for kw in request.keywords if kw.strip()]
    if not keywords:
        raise HTTPException(status_code=422, detail="'keywords' list must not be empty.")

    result = run_prediction(keywords, use_transformer=request.use_transformer, db=db)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result


# ── AI-Powered SME Trend Analysis routes (trend_engine + Cohere) ───────────────

def _process_trend_analysis(
    sme: str,
    city: str = "Bengaluru",
    days: int = 90,
    topics: int = 6,
    demo: bool = False,
    sources: list[str] = None,
    refresh: bool = False,
):
    if sources is None:
        sources = ["reddit", "youtube", "trends"]
    sme_clean = sme.strip().lower()
    if sme_clean not in SME_PROFILES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown SME '{sme}'. Allowed categories: {list(SME_PROFILES.keys())}",
        )

    # ── 1. Check Database First ({sme}_trends table) ──────────────────────────
    if not refresh:
        cached_trends = get_cached_trends(sme=sme_clean, city=city)
        if cached_trends is not None:
            return cached_trends

        # Check if an existing output file already exists for this SME
        for out_dir in [_HERE / "trend_output", _HERE.parent / "trend_output", Path("trend_output")]:
            out_file = out_dir / "trending_items.json"
            if out_file.is_file():
                try:
                    import json
                    with open(out_file, "r", encoding="utf-8") as f:
                        file_data = json.load(f)
                    if file_data.get("sme", "").lower() == sme_clean:
                        final_payload = {
                            "status": "success",
                            "sme": sme_clean,
                            "city": city,
                            "generated": file_data.get("generated"),
                            "total_documents_analyzed": file_data.get("total_documents_analyzed", 0),
                            "total_trending_items": len(file_data.get("items", [])),
                            "trending_items": file_data.get("items", []),
                            "topic_clusters": [],
                            "rising_searches": [],
                            "from_file": True,
                        }
                        record_id = save_trends_to_db(sme=sme_clean, city=city, trend_output=final_payload)
                        final_payload["database_saved"] = bool(record_id)
                        final_payload["database_table"] = f"{sme_clean}_trends"
                        final_payload["database_record_id"] = record_id
                        print(f"[trend_db] Imported '{out_file}' into '{sme_clean}_trends' table (ID: {record_id})")
                        return final_payload
                except Exception as err:
                    print(f"[trend_db] Error checking {out_file}: {err}")

    # ── 2. Run Trend Engine in-memory (zero CSV/JSON saved to disk) ───────────
    try:
        trends_raw = discover_trends(
            sme=sme_clean,
            city=city,
            days=days,
            sources=tuple(sources),
            topics=topics,
            demo=demo,
            save_files=False,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Trend engine failed: {exc}")

    # ── 3. Feed in-memory results directly to Cohere LLM ───────────────────────
    try:
        trending_items = analyze_trends(
            trends_data=trends_raw,
            rising_queries=trends_raw.get("rising_queries", []),
        )
    except Exception as exc:
        # Graceful fallback: return raw topic clusters if Cohere experiences an API issue
        return {
            "status": "partial_success",
            "sme": sme_clean,
            "city": city,
            "error_cohere": str(exc),
            "total_documents_analyzed": trends_raw.get("total_documents", 0),
            "trending_items": [],
            "topic_clusters": trends_raw.get("trends", []),
            "rising_searches": trends_raw.get("rising_queries", []),
        }

    # ── 4. Build output payload ────────────────────────────────────────────────
    final_output = {
        "status": "success",
        "sme": sme_clean,
        "city": city,
        "generated": trends_raw.get("generated"),
        "total_documents_analyzed": trends_raw.get("total_documents", 0),
        "total_trending_items": len(trending_items),
        "trending_items": trending_items,
        "topic_clusters": trends_raw.get("trends", []),
        "rising_searches": trends_raw.get("rising_queries", []),
    }

    # ── 5. Save output to database table {sme}_trends ──────────────────────────
    record_id = save_trends_to_db(sme=sme_clean, city=city, trend_output=final_output)
    final_output["database_saved"] = bool(record_id)
    final_output["database_table"] = f"{sme_clean}_trends"
    final_output["database_record_id"] = record_id

    return final_output


@app.get("/api/trends/analyze/{sme}", tags=["Trend Analysis (AI-Powered)"])
def analyze_sme_trends(
    sme: str,
    city: str = "Bengaluru",
    days: int = 90,
    topics: int = 6,
    demo: bool = False,
    refresh: bool = False,
):
    """
    AI-Powered SME Trend Intelligence:
      1. First checks if output is already found in database table '{sme}_trends'.
      2. If found, returns the cached output directly from the database.
      3. If not found (or refresh=True), scrapes and runs Cohere analysis in-memory,
         saves the output to '{sme}_trends', and returns the JSON response.
    """
    return _process_trend_analysis(
        sme=sme, city=city, days=days, topics=topics, demo=demo, refresh=refresh
    )


@app.get("/trends/analyze/{sme}", include_in_schema=False)
def analyze_sme_trends_alias(
    sme: str,
    city: str = "Bengaluru",
    days: int = 90,
    topics: int = 6,
    demo: bool = False,
    refresh: bool = False,
):
    return analyze_sme_trends(
        sme=sme, city=city, days=days, topics=topics, demo=demo, refresh=refresh
    )


# ── Image Generation routes (/image_generation & /generate) ───────────────────
# Mounted above via `app.include_router(creative_router)` from creative.py,
# powered by nvidia_client.py (NVIDIA API Catalog + Groq + Pollinations fallback).


# ── Inventory CSV Upload Convenience Routes ────────────────────────────────────
from typing import Optional
from fastapi import Depends
from sqlalchemy.orm import Session
from inventory.db import get_db
from inventory.models.schemas import ProductUploadResponse, SalesHistoryUploadResponse
from inventory.api.endpoints.inventory import upload_products, upload_sales_history


@app.post(
    "/api/products/upload-csv",
    response_model=ProductUploadResponse,
    tags=["Inventory & Demand Prediction"],
    summary="Upload products CSV to database (Direct Alias)",
    description="Upload products.csv directly into database, save to disk, and update stock plan.",
)
async def api_products_upload_csv(
    file: Optional[UploadFile] = File(None, description="CSV file containing products"),
    replace_all: bool = Form(False, description="Set True to wipe existing products"),
    recalculate_stock_plan: bool = Form(True, description="Recalculate stock plan and save to DB + stock_plan.csv"),
    db: Session = Depends(get_db),
):
    return await upload_products(file, replace_all, recalculate_stock_plan, db)


@app.post(
    "/api/sales-history/upload-csv",
    response_model=SalesHistoryUploadResponse,
    tags=["Inventory & Demand Prediction"],
    summary="Upload sales history CSV to database (Direct Alias)",
    description="Upload sales_history.csv directly into database, save to disk, recalculate demand forecast, and save to database & stock_plan.csv.",
)
async def api_sales_history_upload_csv(
    file: Optional[UploadFile] = File(None, description="CSV file containing sales history"),
    replace_all: bool = Form(False, description="Set True to wipe existing sales logs"),
    recalculate_stock_plan: bool = Form(True, description="Recalculate stock plan and save to DB + stock_plan.csv"),
    db: Session = Depends(get_db),
):
    return await upload_sales_history(file, replace_all, recalculate_stock_plan, db)


# ── Demand Forecasting & Stock Plan Routes (demand.py) ─────────────────────────
from inventory.db import StockPlan
from inventory.models.schemas import StockPlanItem, StockPlanResponse


@app.get(
    "/api/demand/stock-plan",
    response_model=StockPlanResponse,
    tags=["Inventory & Demand Prediction"],
    summary="Compute demand forecast from database and update stock plan",
    description=(
        "Executes demand forecasting (demand.py) taking inputs directly from database tables "
        "'products' and 'sales_logs', saves/updates the output into database table 'stock_plans' "
        "(and writes to stock_plan.csv), and returns the calculated stock plan data."
    ),
)
@app.get(
    "/demand/stock-plan",
    response_model=StockPlanResponse,
    tags=["Inventory & Demand Prediction"],
    include_in_schema=False,
)
def get_demand_stock_plan_from_db(
    lead_time: float = Query(7.0, description="Days from order to delivery"),
    service_level: float = Query(0.95, description="Target service level (e.g. 0.95)"),
    horizon: int = Query(30, description="Forecast review period in days"),
    growth: float = Query(0.03, description="Assumed weekly demand growth (0.03 = 3%)"),
    lag: float = Query(14.0, description="Days growth applied from before window"),
    uplift: float = Query(0.0, description="Demand uplift factor (e.g. 0.0)"),
    service_ids: str = Query("", description="Comma-separated product IDs that are services"),
    status: Optional[str] = Query(None, description="Optional filter by status (Understocked, OK, Overstock, etc.)"),
    db: Session = Depends(get_db),
):
    """
    Connects to demand.py, reads inputs from database tables 'products' and 'sales_logs',
    runs the gradient-boosting + TSB demand forecasting model, updates table 'stock_plans',
    and returns the resulting stock plan.
    """
    # Unwrap default values if called directly in Python instead of via FastAPI
    from fastapi.params import Query as QueryParam
    _lead_time = lead_time.default if isinstance(lead_time, QueryParam) else lead_time
    _service_level = service_level.default if isinstance(service_level, QueryParam) else service_level
    _horizon = horizon.default if isinstance(horizon, QueryParam) else horizon
    _growth = growth.default if isinstance(growth, QueryParam) else growth
    _lag = lag.default if isinstance(lag, QueryParam) else lag
    _uplift = uplift.default if isinstance(uplift, QueryParam) else uplift
    _service_ids = service_ids.default if isinstance(service_ids, QueryParam) else service_ids
    _status = status.default if isinstance(status, QueryParam) else status

    try:
        run_from_database(
            db=db,
            lead_time=_lead_time,
            service_level=_service_level,
            horizon=_horizon,
            growth=_growth,
            lag=_lag,
            uplift=_uplift,
            service_ids=_service_ids,
        )
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error executing demand model from database: {exc}",
        )

    query = db.query(StockPlan)
    if _status:
        query = query.filter(StockPlan.status.ilike(f"%{_status.strip()}%"))
    records = query.all()

    items = [
        StockPlanItem(
            product_id=r.product_id,
            product_name=r.product_name,
            avg_daily_demand=r.avg_daily_demand,
            forecast_30d=r.forecast_30d,
            current_stock=r.current_stock,
            safety_stock=r.safety_stock,
            reorder_point=r.reorder_point,
            target_stock=r.target_stock,
            order_qty_now=r.order_qty_now,
            status=r.status,
            note=r.note,
        )
        for r in records
    ]

    understocked = sum(1 for i in items if i.status == "Understocked")
    overstock = sum(1 for i in items if i.status == "Overstock")
    ok_count = sum(1 for i in items if i.status == "OK")

    scripts_dir = Path(__file__).resolve().parent.parent
    csv_file = str(scripts_dir / "stock_plan.csv")

    return StockPlanResponse(
        status="success",
        total_items=len(items),
        understocked_count=understocked,
        overstock_count=overstock,
        ok_count=ok_count,
        csv_path=csv_file,
        items=items,
        data=items,
    )


#uvicorn main:app --reload --host 0.0.0.0 --port 8000