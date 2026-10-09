"""Proactive Alert Service and Automated WhatsApp Notification Dispatcher."""

from datetime import datetime
import uuid
from app.models.schemas import (
    AlertItem,
    SendWhatsAppRequest,
    SendWhatsAppResponse,
    AcknowledgeAlertRequest,
)
from app.services.inventory_service import InventoryService


# Simulated persistent alert store
_ALERTS_STORE: list[AlertItem] = []


class AlertService:
    @classmethod
    def get_all_alerts(cls, force_refresh: bool = False) -> list[AlertItem]:
        """Aggregate high and medium risk inventory items into actionable alerts."""
        global _ALERTS_STORE
        if not _ALERTS_STORE or force_refresh:
            predictions_summary = InventoryService.predict_all()
            new_alerts: list[AlertItem] = []
            for pred in predictions_summary.predictions:
                if pred.alert_needed:
                    driving_trend = "Local High Velocity Demand"
                    if pred.category in ["Beverages", "F&B"]:
                        driving_trend = "Boba & Cold Brew Trend Surge (Bengaluru)"
                    elif pred.category == "Wellness":
                        driving_trend = "Clean Living & K-Beauty Craze (HSR/Koramangala)"
                    elif pred.category == "Fashion":
                        driving_trend = "Oversized & Streetwear Trend (Church Street)"

                    new_alerts.append(
                        AlertItem(
                            alert_id=f"ALT-{uuid.uuid4().hex[:6].upper()}",
                            product_id=pred.product_id,
                            product_name=pred.product_name,
                            alert_level=pred.alert_level if pred.alert_level != "NONE" else "LOW",
                            stockout_probability=pred.stockout_probability,
                            current_stock=pred.current_stock,
                            predicted_demand_7d=pred.predicted_7_day_demand,
                            recommended_order_quantity=pred.recommended_order_quantity,
                            driving_trend=driving_trend,
                            message=(
                                f"⚠️ Stockout Risk Alert for {pred.product_name} ({pred.product_id}). "
                                f"Current stock: {pred.current_stock} units. "
                                f"Predicted 7-day demand: {pred.predicted_7_day_demand} units. "
                                f"Recommended reorder: {pred.recommended_order_quantity} units."
                            ),
                            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            status="PENDING",
                        )
                    )
            _ALERTS_STORE = sorted(new_alerts, key=lambda a: (a.alert_level != "HIGH", -a.stockout_probability))

        return _ALERTS_STORE

    @classmethod
    def acknowledge_alert(cls, alert_id: str, req: AcknowledgeAlertRequest) -> AlertItem:
        for alert in _ALERTS_STORE:
            if alert.alert_id == alert_id:
                alert.status = req.action
                return alert
        raise ValueError(f"Alert '{alert_id}' not found.")

    @classmethod
    def send_whatsapp_alert(cls, req: SendWhatsAppRequest) -> SendWhatsAppResponse:
        """Dispatch or simulate automated WhatsApp alert via Twilio messaging format."""
        body = req.custom_message
        if not body and req.alert_id:
            match = next((a for a in _ALERTS_STORE if a.alert_id == req.alert_id), None)
            if match:
                body = (
                    f"🔔 *AdVantage OS | Smart Retail Alert*\n\n"
                    f"Product: *{match.product_name}* ({match.product_id})\n"
                    f"Risk Level: *{match.alert_level}* (Stockout Prob: {int(match.stockout_probability * 100)}%)\n"
                    f"Current Stock: {match.current_stock} units\n"
                    f"Predicted 7-Day Demand: {match.predicted_demand_7d} units\n"
                    f"Recommended Reorder: *{match.recommended_order_quantity} units*\n"
                    f"Trend Driver: {match.driving_trend or 'Retail Traffic'}\n\n"
                    f"Reply *REORDER* to auto-dispatch PO or open your AdVantage OS Dashboard."
                )

        if not body:
            body = (
                "🔔 *AdVantage OS Broadcast*\n\n"
                "Weekly retail inventory & trend report ready for your review."
            )

        # Return simulated delivery payload with standard Twilio-compatible SID
        msg_sid = f"SM{uuid.uuid4().hex}"
        return SendWhatsAppResponse(
            status="SENT",
            recipient_phone=req.recipient_phone,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            message_sid=msg_sid,
            body=body,
        )
