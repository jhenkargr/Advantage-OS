"""
trend_predictor
===============
AdVantage OS — AI-driven SME Trend Forecasting and Social Listening Subsystem.

Exposes:
  - predict_router: Router for /predict, /predict/upload-model, /predict/model-status
  - trends_router: Router for /api/trends/analyze/{sme}
  - keywords_router: Router for /api/keywords/related
"""

from .predict import router as predict_router, run_prediction
from .trend_engine import router as trends_router, process_trend_analysis, discover_trends, SME_PROFILES
from .keywords import router as keywords_router, get_related_keywords, expand_keywords
from .trend_db import get_cached_trends, save_trends_to_db
from .trend_analyzer_cohere import analyze_trends

__all__ = [
    "predict_router",
    "trends_router",
    "keywords_router",
    "run_prediction",
    "process_trend_analysis",
    "discover_trends",
    "SME_PROFILES",
    "get_related_keywords",
    "expand_keywords",
    "get_cached_trends",
    "save_trends_to_db",
    "analyze_trends",
]
