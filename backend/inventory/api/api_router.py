"""Master API router aggregating all AdVantage OS feature modules."""

from fastapi import APIRouter
from inventory.api.endpoints import (
    dashboard,
    inventory,
    influencers,
    alerts,
)

api_router = APIRouter()

api_router.include_router(dashboard.router)
api_router.include_router(inventory.router)
api_router.include_router(inventory.csv_router)
api_router.include_router(influencers.router)
api_router.include_router(alerts.router)

