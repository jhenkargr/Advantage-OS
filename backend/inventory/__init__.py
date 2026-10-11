"""AdVantage OS Inventory & Demand Planning Package."""

from inventory import demand
from inventory.demand import router as demand_router

__all__ = ["demand", "demand_router"]
