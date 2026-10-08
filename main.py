"""
main.py
-------
FastAPI + Uvicorn backend.

Endpoints
---------
POST /predict
    Body: { "keywords": ["bitcoin", "crypto"] }
    Returns: trend forecast summary dict

GET  /health
    Returns: { "status": "ok" }

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import sys

# Ensure the advantage package directory is importable when running from any CWD
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv(dotenv_path=os.path.join(_HERE, ".env"))

from predict import run_prediction   # noqa: E402  (import after sys.path patch)

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Trend Prediction API",
    description=(
        "Scrapes Reddit & X (Twitter) for the given keywords, runs ensemble "
        "sentiment analysis, and forecasts the trend score for the next 30 days."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Utility"])
def health_check():
    return {"status": "ok"}


@app.post("/predict", tags=["Prediction"])
def predict(request: PredictRequest):
    """
    Accepts a list of keywords, scrapes Reddit + X, runs sentiment
    analysis, and returns a 30-day trend forecast.
    """
    keywords = [kw.strip() for kw in request.keywords if kw.strip()]
    if not keywords:
        raise HTTPException(status_code=422, detail="'keywords' list must not be empty.")

    result = run_prediction(keywords, use_transformer=request.use_transformer)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result
