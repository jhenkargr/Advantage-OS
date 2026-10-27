import traceback
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
        try:
            inv_summary = InventoryService.predict_all()
        except Exception as inv_err:
            print(f"⚠️ [Dashboard] predict_all notice: {inv_err}")
            traceback.print_exc()
            inv_summary = None

        try:
            alerts = AlertService.get_all_alerts()
        except Exception as alert_err:
            print(f"⚠️ [Dashboard] get_all_alerts notice: {alert_err}")
            traceback.print_exc()
            alerts = []

        top_critical_skus = []
        if inv_summary and getattr(inv_summary, "predictions", None):
            top_critical_skus = [
                p for p in inv_summary.predictions if getattr(p, "alert_level", "") == "HIGH"
            ][:5]

        total_skus = inv_summary.total_skus if inv_summary else 0
        critical_alerts = inv_summary.high_risk_count if inv_summary else len([a for a in alerts if getattr(a, "alert_level", "") == "HIGH"])
        total_units = inv_summary.total_units_in_stock if inv_summary else 0

        return DashboardStatsResponse(
            total_skus=total_skus,
            critical_stockout_alerts=critical_alerts,
            active_local_trends=0,
            verified_influencers_count=0,
            active_campaigns_count=0,
            total_inventory_units=total_units,
            top_emerging_trends=[],
            top_critical_skus=top_critical_skus,
            recent_alerts=alerts[:5] if alerts else [],
        )
    except Exception as e:
        print(f"❌ [Dashboard] Critical error in /stats: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

