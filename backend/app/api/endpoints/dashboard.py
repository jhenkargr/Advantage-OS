"""Executive Dashboard Overview API Endpoint."""

from fastapi import APIRouter, HTTPException
from app.models.schemas import DashboardStatsResponse
from app.services.inventory_service import InventoryService
from app.services.trend_service import TrendService
from app.services.influencer_service import InfluencerService
from app.services.creative_service import CreativeService
from app.services.alert_service import AlertService

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
        trends = TrendService.get_all_trends()
        influencers = InfluencerService.list_all_influencers()
        campaigns = CreativeService.list_campaigns()
        alerts = AlertService.get_all_alerts()

        top_critical_skus = [
            p for p in inv_summary.predictions if p.alert_level == "HIGH"
        ][:5]

        top_emerging_trends = sorted(
            trends, key=lambda t: t.opportunity_score, reverse=True
        )[:3]

        return DashboardStatsResponse(
            total_skus=inv_summary.total_skus,
            critical_stockout_alerts=inv_summary.high_risk_count,
            active_local_trends=len(trends),
            verified_influencers_count=len(influencers),
            active_campaigns_count=len([c for c in campaigns if c.status == "Active"]),
            total_inventory_units=inv_summary.total_units_in_stock,
            top_emerging_trends=top_emerging_trends,
            top_critical_skus=top_critical_skus,
            recent_alerts=alerts[:5],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
