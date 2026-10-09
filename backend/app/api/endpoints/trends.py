"""Trend Engine API Endpoints."""

from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    TrendItem,
    TrendDetailResponse,
    TrendRefreshResponse,
)
from app.services.trend_service import TrendService

router = APIRouter(prefix="/trends", tags=["Trend Engine & Ingestion"])


@router.get("/active", response_model=list[TrendItem], summary="List active local trends")
def list_trends():
    """List detected local trends across Bengaluru hubs with opportunity scores, lifecycle stages, and linked SKUs."""
    try:
        return TrendService.get_all_trends()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{trend_id}", response_model=TrendDetailResponse, summary="Get trend details & LSTM forecast")
def get_trend_detail(trend_id: str):
    """Retrieve detailed trend metrics, historical post volume time-series, and 7-day predicted trend curve."""
    try:
        return TrendService.get_trend_by_id(trend_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refresh", response_model=TrendRefreshResponse, summary="Trigger social signal ingestion refresh")
def refresh_trends():
    """Trigger background crawler/refresh for social signals (Reddit PRAW, YouTube Data API)."""
    try:
        return TrendService.refresh_trends()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
