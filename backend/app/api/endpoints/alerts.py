"""Proactive Alerts and WhatsApp Notification API Endpoints."""

from fastapi import APIRouter, HTTPException, Query
from app.models.schemas import (
    AlertItem,
    SendWhatsAppRequest,
    SendWhatsAppResponse,
    AcknowledgeAlertRequest,
)
from app.services.alert_service import AlertService

router = APIRouter(prefix="/alerts", tags=["Alerts & WhatsApp Notifications"])


@router.get("", response_model=list[AlertItem], summary="Get active stockout and trend alerts")
def get_alerts(refresh: bool = Query(False, description="Re-evaluate all ML models immediately")):
    """
    Retrieve real-time prioritized alerts:
    - HIGH: Severe stockout probability (>= 50%)
    - MEDIUM: Current stock below minimum threshold
    - Includes recommended reorder quantities and the trend driving demand.
    """
    try:
        return AlertService.get_all_alerts(force_refresh=refresh)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{alert_id}/acknowledge", response_model=AlertItem, summary="Acknowledge or update alert status")
def acknowledge_alert(alert_id: str, request: AcknowledgeAlertRequest):
    """Update alert status (e.g. REORDER_PLACED, ACKNOWLEDGED, DISMISSED) for closed-loop auditing."""
    try:
        return AlertService.acknowledge_alert(alert_id, request)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/send-whatsapp", response_model=SendWhatsAppResponse, summary="Send WhatsApp alert to SME owner")
def send_whatsapp(request: SendWhatsAppRequest):
    """
    Dispatch automated WhatsApp alert to the store owner's mobile phone via Twilio API format.
    Delivers product name, stockout probability, suggested reorder qty, and driving trend.
    """
    try:
        return AlertService.send_whatsapp_alert(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
