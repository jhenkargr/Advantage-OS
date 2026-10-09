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

--- Mounted from backend/app (prefix: /api) ---
GET  /api/dashboard/stats
GET  /api/trends/active
GET  /api/trends/{trend_id}
POST /api/trends/refresh
GET  /api/influencers
GET  /api/influencers/match/{trend_id}
POST /api/creatives/generate
POST /api/creatives/score-ctr
GET  /api/creatives/campaigns
POST /api/creatives/campaigns
GET  /api/inventory/catalog
POST /api/inventory/catalog
GET  /api/inventory/catalog/{product_id}
PUT  /api/inventory/catalog/{product_id}/stock
GET  /api/inventory/predict/{product_id}
GET  /api/inventory/predict-all
GET  /api/inventory/sales-history/{product_id}
POST /api/ocr/scan-receipt
POST /api/ocr/scan-sample
POST /api/ocr/apply-receipt
GET  /api/alerts
PUT  /api/alerts/{alert_id}/acknowledge
POST /api/alerts/send-whatsapp

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
# advantage/ directory  →  resolves predict, creative, reddit_scrap, x_scrapper
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# backend/ directory  →  resolves app.* imports
_BACKEND_DIR = _HERE / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

load_dotenv(dotenv_path=_HERE / ".env")

# ── Advantage-module imports ───────────────────────────────────────────────────
from predict import run_prediction          # noqa: E402
from creative import (                      # noqa: E402
    client,
    TEXT2IMG_MODEL,
    IMG2IMG_MODEL,
    CAPTION_MODEL,
    image_to_response,
)

# ── Backend/app imports ────────────────────────────────────────────────────────
from app.config import (                    # noqa: E402
    settings,
    CATALOG_PATH,
    DEMAND_MODEL_PATH,
    STOCKOUT_MODEL_PATH,
    TESSERACT_CMD,
)
from app.api.api_router import api_router   # noqa: E402
from app.services.inventory_service import load_ml_models  # noqa: E402


# ── Lifespan (startup / shutdown) ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 65)
    print(f"🚀 Starting AdVantage OS v{settings.app_version}")
    print(f"📂 Catalog      : {CATALOG_PATH} (exists={CATALOG_PATH.exists()})")
    print(f"🤖 Demand model : {DEMAND_MODEL_PATH} (exists={DEMAND_MODEL_PATH.exists()})")
    print(f"🤖 Stockout mdl : {STOCKOUT_MODEL_PATH} (exists={STOCKOUT_MODEL_PATH.exists()})")
    print(f"👁️  Tesseract    : {TESSERACT_CMD} (exists={Path(TESSERACT_CMD).exists()})")
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
        "(HuggingFace FLUX / Llama), inventory forecasting (ML demand + stockout), "
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
)

# Mount the full backend/app router under /api
app.include_router(api_router, prefix=settings.api_prefix)


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
            "Trend Prediction":    "POST /predict",
            "Image Generation":    "POST /image_generation/text2img  |  /img2img  |  /caption",
            "Trends Engine":       "GET  /api/trends/active",
            "Influencers":         "GET  /api/influencers",
            "Creative Campaigns":  "POST /api/creatives/generate",
            "Inventory + ML":      "GET  /api/inventory/predict-all",
            "OCR Receipts":        "POST /api/ocr/scan-receipt",
            "Alerts + WhatsApp":   "GET  /api/alerts",
            "Dashboard":           "GET  /api/dashboard/stats",
        },
    }


@app.get("/health", tags=["System"])
def health_check():
    """Unified health check — covers data files, ML models, and OCR."""
    return JSONResponse(content={
        "status": "healthy",
        "version": settings.app_version,
        "catalog_ready":  CATALOG_PATH.is_file(),
        "models_ready":   DEMAND_MODEL_PATH.is_file() and STOCKOUT_MODEL_PATH.is_file(),
        "ocr_ready":      Path(TESSERACT_CMD).exists(),
    })


# ── Trend Prediction routes ────────────────────────────────────────────────────

@app.post("/predict", tags=["Trend Prediction"])
def predict(request: PredictRequest):
    """
    Scrapes Reddit + X for the given keywords, runs ensemble sentiment
    analysis (VADER + DistilBERT), and returns a 30-day trend forecast.
    """
    keywords = [kw.strip() for kw in request.keywords if kw.strip()]
    if not keywords:
        raise HTTPException(status_code=422, detail="'keywords' list must not be empty.")

    result = run_prediction(keywords, use_transformer=request.use_transformer)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result


# ── Image Generation routes (/image_generation) ────────────────────────────────

@app.get("/image_generation/health", tags=["Image Generation"])
def image_generation_health():
    """Creative studio status and active model names."""
    return {
        "status": "ok",
        "text2img_model": TEXT2IMG_MODEL,
        "img2img_model":  IMG2IMG_MODEL,
    }


@app.post("/image_generation/text2img", tags=["Image Generation"])
def text_to_image(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
):
    """
    Text → PNG image via FLUX.1-schnell.
    `negative_prompt` is kept for compatibility but may be ignored by the model.
    """
    try:
        image = client.text_to_image(prompt, model=TEXT2IMG_MODEL)
    except Exception as e:
        raise HTTPException(502, f"Image generation failed: {e}")
    return image_to_response(image)


@app.post("/image_generation/img2img", tags=["Image Generation"])
async def image_to_image(
    prompt: str = Form(...),
    reference_image: UploadFile = File(...),
):
    """
    Image + prompt → transformed PNG image via FLUX.1-Kontext-dev.
    Upload a reference image and describe the desired transformation.
    """
    raw = await reference_image.read()
    try:
        image = client.image_to_image(raw, prompt=prompt, model=IMG2IMG_MODEL)
    except Exception as e:
        raise HTTPException(502, f"Image-to-image generation failed: {e}")
    return image_to_response(image)


@app.post("/image_generation/caption", tags=["Image Generation"])
def generate_caption(
    description: str = Form(...),
    tone: str = Form("Playful"),
):
    """
    Description → short Instagram ad caption via Llama-3.1-8B.
    `tone` options: Playful, Professional, Bold, Minimalist, Urgent.
    """
    try:
        completion = client.chat.completions.create(
            model=CAPTION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert social media advertising copywriter. "
                        "Generate concise, engaging Instagram ad captions."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Write ONE short {tone.lower()}-tone Instagram ad caption "
                        f"for: {description}. "
                        "Maximum 20 words. "
                        "Include exactly one relevant emoji. "
                        "Return ONLY the caption text."
                    ),
                },
            ],
            max_tokens=60,
            temperature=0.8,
        )

        caption = completion.choices[0].message.content
        if not caption:
            raise RuntimeError("The caption model returned no text.")

        return {"caption": caption.strip().strip('"')}

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Caption generation failed: {e}")


#uvicorn main:app --reload --host 0.0.0.0 --port 8000