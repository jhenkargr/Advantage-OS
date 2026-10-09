"""Master API router aggregating all AdVantage OS feature modules."""

from fastapi import APIRouter
from app.api.endpoints import (
    dashboard,
    inventory,
    ocr,
    trends,
    influencers,
    creatives,
    alerts,
)

api_router = APIRouter()

api_router.include_router(dashboard.router)
api_router.include_router(inventory.router)
api_router.include_router(ocr.router)
api_router.include_router(trends.router)
api_router.include_router(influencers.router)
api_router.include_router(creatives.router)
api_router.include_router(alerts.router)
