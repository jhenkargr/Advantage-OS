"""Integration and functional test suite for the AdVantage OS FastAPI Backend."""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add backend directory to path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_and_health():
    """Verify system root and healthcheck endpoints."""
    res_root = client.get("/")
    assert res_root.status_code == 200
    data_root = res_root.json()
    assert data_root["system"] == "AdVantage OS"
    assert data_root["status"] == "online"

    res_health = client.get("/health")
    assert res_health.status_code == 200
    data_health = res_health.json()
    assert data_health["status"] == "healthy"
    assert data_health["catalog_ready"] is True
    assert data_health["models_ready"] is True
    print("✅ Root and health checks passed.")


def test_executive_dashboard():
    """Verify consolidated executive dashboard statistics."""
    res = client.get("/api/dashboard/stats")
    assert res.status_code == 200
    data = res.json()
    assert data["total_skus"] >= 40
    assert data["active_local_trends"] >= 5
    assert data["verified_influencers_count"] >= 5
    assert len(data["top_emerging_trends"]) > 0
    print(f"✅ Dashboard stats verified: {data['total_skus']} SKUs, {data['active_local_trends']} trends.")


def test_inventory_and_ml_predictions():
    """Verify catalog, sales history, single prediction, and batch prediction."""
    # 1. Catalog listing
    res_cat = client.get("/api/inventory/catalog")
    assert res_cat.status_code == 200
    catalog = res_cat.json()
    assert len(catalog) >= 40
    assert catalog[0]["product_id"] == "SKU001"

    # 2. Single product detail
    res_sku = client.get("/api/inventory/catalog/SKU001")
    assert res_sku.status_code == 200
    assert res_sku.json()["product_name"] == "Amul Butter 500g"

    # 3. Single ML prediction
    res_pred = client.get("/api/inventory/predict/SKU001")
    assert res_pred.status_code == 200
    pred = res_pred.json()
    assert pred["product_id"] == "SKU001"
    assert "predicted_7_day_demand" in pred
    assert 0.0 <= pred["stockout_probability"] <= 1.0
    assert pred["alert_level"] in ["HIGH", "MEDIUM", "LOW", "NONE"]

    # 4. Batch ML prediction across all SKUs
    res_batch = client.get("/api/inventory/predict-all")
    assert res_batch.status_code == 200
    batch = res_batch.json()
    assert batch["total_skus"] == len(catalog)
    assert batch["total_reorder_units"] >= 0

    # 5. Sales history time-series
    res_hist = client.get("/api/inventory/sales-history/SKU001?limit=14")
    assert res_hist.status_code == 200
    hist = res_hist.json()
    assert len(hist) == 14
    assert "units_sold" in hist[0]
    print("✅ Inventory and ML prediction endpoints passed.")


def test_ocr_sample_receipt():
    """Verify receipt OCR processing with Tesseract on sample receipt image."""
    res = client.post("/api/ocr/scan-sample")
    assert res.status_code == 200
    data = res.json()
    assert "receipt_id" in data
    assert len(data["raw_ocr_text"]) > 0
    assert data["total_items_detected"] >= 1
    print(f"✅ OCR test passed. Extracted store: '{data['store_name']}' with {data['total_items_detected']} detected items.")


def test_trends_engine():
    """Verify trend listing, trend details with forecast, and ingestion refresh."""
    res_list = client.get("/api/trends/active")
    assert res_list.status_code == 200
    trends = res_list.json()
    assert len(trends) >= 6

    # Trend detail
    res_detail = client.get("/api/trends/TRD-001")
    assert res_detail.status_code == 200
    detail = res_detail.json()
    assert detail["trend"]["trend_id"] == "TRD-001"
    assert len(detail["historical_volume"]) == 14
    assert len(detail["predicted_forecast_7d"]) == 7

    # Trend refresh
    res_ref = client.post("/api/trends/refresh")
    assert res_ref.status_code == 200
    print("✅ Trend Engine endpoints passed.")


def test_influencer_matchmaker():
    """Verify influencer discovery and trend matching."""
    res_inf = client.get("/api/influencers")
    assert res_inf.status_code == 200
    influencers = res_inf.json()
    assert len(influencers) >= 5
    for inf in influencers:
        assert inf["authenticity_score"] >= 0.70  # bot filtered

    # Match influencers for Boba trend
    res_match = client.get("/api/influencers/match/TRD-001")
    assert res_match.status_code == 200
    matched = res_match.json()
    assert matched["trend_id"] == "TRD-001"
    assert len(matched["recommended_influencers"]) >= 1
    print("✅ Influencer Matchmaker endpoints passed.")


def test_creative_studio_and_campaigns():
    """Verify creative generation, CTR scoring, and campaign tracking."""
    # 1. Creative generation
    res_gen = client.post(
        "/api/creatives/generate",
        json={"product_id": "SKU016", "trend_id": "TRD-001", "tone": "Aesthetic"},
    )
    assert res_gen.status_code == 200
    gen_data = res_gen.json()
    assert len(gen_data["options"]) == 3
    assert gen_data["options"][0]["estimated_ctr"] > 0

    # 2. CTR Scoring
    res_ctr = client.post(
        "/api/creatives/score-ctr",
        json={
            "headline": "Limited Weekend Boba Drop in Koramangala",
            "caption": "Get fresh handcrafted tapioca pearls brewed daily. Order now before batches run out!",
            "call_to_action": "Order Now",
        },
    )
    assert res_ctr.status_code == 200
    ctr_data = res_ctr.json()
    assert ctr_data["estimated_ctr"] > 0
    assert ctr_data["ctr_grade"] in ["A+", "A", "B+", "B", "C"]

    # 3. Create campaign
    res_cmp = client.post(
        "/api/creatives/campaigns",
        json={
            "campaign_name": "Summer Boba Fest",
            "product_id": "SKU016",
            "trend_id": "TRD-001",
            "influencer_ids": ["@bengaluru_boba_diaries"],
            "headline": "Fresh Boba Weekend Sensation!",
            "caption": "Order now for quick local delivery.",
            "budget_inr": 6000.0,
            "start_date": "2026-05-10",
            "end_date": "2026-05-17",
        },
    )
    assert res_cmp.status_code == 201
    assert res_cmp.json()["campaign_id"].startswith("CMP-")
    print("✅ Creative Studio and Campaign endpoints passed.")


def test_alerts_and_whatsapp():
    """Verify alert generation, WhatsApp simulation, and alert acknowledgment."""
    res_alerts = client.get("/api/alerts")
    assert res_alerts.status_code == 200
    alerts = res_alerts.json()
    assert len(alerts) > 0
    first_alert = alerts[0]

    # Send WhatsApp notification
    res_wa = client.post(
        "/api/alerts/send-whatsapp",
        json={
            "recipient_phone": "+919876543210",
            "alert_id": first_alert["alert_id"],
        },
    )
    assert res_wa.status_code == 200
    wa_data = res_wa.json()
    assert wa_data["status"] == "SENT"
    assert "message_sid" in wa_data
    assert "AdVantage OS" in wa_data["body"]

    # Acknowledge alert
    res_ack = client.put(
        f"/api/alerts/{first_alert['alert_id']}/acknowledge",
        json={"action": "REORDER_PLACED", "notes": "PO sent to supplier"},
    )
    assert res_ack.status_code == 200
    assert res_ack.json()["status"] == "REORDER_PLACED"
    print("✅ Alerts and WhatsApp notification endpoints passed.")


if __name__ == "__main__":
    test_root_and_health()
    test_executive_dashboard()
    test_inventory_and_ml_predictions()
    test_ocr_sample_receipt()
    test_trends_engine()
    test_influencer_matchmaker()
    test_creative_studio_and_campaigns()
    test_alerts_and_whatsapp()
    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
