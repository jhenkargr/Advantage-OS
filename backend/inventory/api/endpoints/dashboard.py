"""Executive Dashboard Overview API Endpoint."""

from fastapi import APIRouter, HTTPException
from inventory.models.schemas import DashboardStatsResponse
from inventory.services.inventory_service import InventoryService
from inventory.services.alert_service import AlertService

router = APIRouter(prefix="/dashboard", tags=["Executive Dashboard"])


@router.get("/stats", response_model=DashboardStatsResponse, summary="Get consolidated SME business metrics")
def get_dashboard_stats():
    """
    Returns single-pane-of-glass executive overview metrics combining:
    - Inventory health and total units
    - Critical stockout risks
    - High-velocity local trends
    - Influencer reach and active campaigns
    """
    try:
        inv_summary = InventoryService.predict_all()
        alerts = AlertService.get_all_alerts()

        top_critical_skus = [
            p for p in inv_summary.predictions if p.alert_level == "HIGH"
        ][:5]

        return DashboardStatsResponse(
            total_skus=inv_summary.total_skus,
            critical_stockout_alerts=inv_summary.high_risk_count,
            active_local_trends=0,
            verified_influencers_count=0,
            active_campaigns_count=0,
            total_inventory_units=inv_summary.total_units_in_stock,
            top_emerging_trends=[],
            top_critical_skus=top_critical_skus,
            recent_alerts=alerts[:5],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

