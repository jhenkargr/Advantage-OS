"""Pydantic schemas and data transfer models for AdVantage OS API."""

from typing import Optional, Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Inventory Schemas
# ---------------------------------------------------------------------------
class ProductItem(BaseModel):
    product_id: str
    product_name: str
    category: str
    current_stock: int
    low_stock_threshold: int


class ProductCreate(BaseModel):
    product_id: str = Field(..., example="SKU041")
    product_name: str = Field(..., example="Avocado Toast Spread 200g")
    category: str = Field(..., example="F&B")
    current_stock: int = Field(..., ge=0, example=30)
    low_stock_threshold: int = Field(..., ge=0, example=10)


class StockUpdate(BaseModel):
    new_stock: int = Field(..., ge=0, description="New absolute stock value")
    reason: Optional[str] = Field("Manual adjustment", description="Audit note")


class InventoryPredictionResponse(BaseModel):
    product_id: str
    product_name: str
    category: str
    current_stock: int
    predicted_7_day_demand: int
    stockout_probability: float
    recommended_order_quantity: int
    alert_needed: bool
    alert_level: Literal["HIGH", "MEDIUM", "LOW", "NONE"]
    safety_stock: Optional[int] = None
    avg_daily_sales: Optional[float] = None
    trend_score: Optional[float] = None


class InventorySummaryResponse(BaseModel):
    total_skus: int
    total_units_in_stock: int
    high_risk_count: int
    medium_risk_count: int
    total_reorder_units: int
    predictions: list[InventoryPredictionResponse]


class SalesHistoryPoint(BaseModel):
    date: str
    units_sold: int
    stock_after: int
    trend_score: float


# ---------------------------------------------------------------------------
# OCR & Receipt Schemas
# ---------------------------------------------------------------------------
class ScannedReceiptItem(BaseModel):
    raw_text: str
    matched_product_id: Optional[str] = None
    matched_product_name: Optional[str] = None
    confidence_score: float
    quantity: int = 1
    unit_price: Optional[float] = None
    line_total: Optional[float] = None


class ScannedReceiptResponse(BaseModel):
    receipt_id: str
    raw_ocr_text: str
    store_name: Optional[str] = None
    receipt_date: Optional[str] = None
    invoice_number: Optional[str] = None
    subtotal: Optional[float] = None
    total_items_detected: int
    items: list[ScannedReceiptItem]


class ReceiptItemApply(BaseModel):
    product_id: str
    quantity: int
    transaction_type: Literal["STOCK_IN", "STOCK_OUT"] = "STOCK_IN"


class ApplyReceiptRequest(BaseModel):
    receipt_id: str
    items: list[ReceiptItemApply]
    notes: Optional[str] = "OCR Receipt Batch Processing"


class ApplyReceiptResponse(BaseModel):
    success: bool
    updated_items_count: int
    message: str
    updated_catalog: list[ProductItem]


# ---------------------------------------------------------------------------
# Trend Engine Schemas
# ---------------------------------------------------------------------------
class TrendItem(BaseModel):
    trend_id: str
    topic_name: str
    category: str
    locality: str
    lifecycle_stage: Literal["Emerging", "Peaking", "Sustained", "Declining"]
    opportunity_score: int = Field(..., ge=0, le=100)
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    weekly_growth_rate: float
    post_volume: int
    sources: list[str]
    sample_keywords: list[str]
    linked_skus: list[str]


class TrendDetailResponse(BaseModel):
    trend: TrendItem
    historical_volume: list[dict]
    predicted_forecast_7d: list[dict]
    actionable_recommendation: str


class TrendRefreshResponse(BaseModel):
    refreshed_at: str
    total_trends_identified: int
    top_emerging_trend: str
    message: str


# ---------------------------------------------------------------------------
# Influencer Matchmaker Schemas
# ---------------------------------------------------------------------------
class InfluencerItem(BaseModel):
    influencer_id: str
    handle: str
    display_name: str
    niche: str
    locality: str
    follower_count: int
    engagement_rate: float
    authenticity_score: float = Field(..., ge=0.0, le=1.0, description="Bot filter score")
    is_verified: bool
    primary_platform: Literal["Instagram", "YouTube"]
    match_score: Optional[float] = None
    recommended_for_trends: list[str] = []


class InfluencerMatchResponse(BaseModel):
    trend_id: str
    trend_name: str
    recommended_influencers: list[InfluencerItem]


# ---------------------------------------------------------------------------
# Creative Studio Schemas
# ---------------------------------------------------------------------------
class CreativeGenerateRequest(BaseModel):
    product_id: str
    trend_id: Optional[str] = None
    target_audience: str = "Local young adults in Bengaluru"
    tone: Literal["Energetic", "Aesthetic", "Informative", "Playful", "Urgent"] = "Aesthetic"


class CreativeOption(BaseModel):
    headline: str
    caption: str
    call_to_action: str
    hashtags: list[str]
    image_prompt: str
    estimated_ctr: float
    ctr_grade: Literal["A+", "A", "B+", "B", "C"]


class CreativeGenerateResponse(BaseModel):
    product_name: str
    trend_topic: Optional[str] = None
    options: list[CreativeOption]


class CTRScoreRequest(BaseModel):
    headline: str
    caption: str
    call_to_action: str


class CTRScoreResponse(BaseModel):
    estimated_ctr: float
    ctr_grade: Literal["A+", "A", "B+", "B", "C"]
    hook_strength: float
    clarity_score: float
    urgency_score: float
    recommendations: list[str]


class CampaignCreate(BaseModel):
    campaign_name: str
    product_id: str
    trend_id: Optional[str] = None
    influencer_ids: list[str] = []
    headline: str
    caption: str
    budget_inr: float = 5000.0
    start_date: str
    end_date: str


class CampaignItem(BaseModel):
    campaign_id: str
    campaign_name: str
    product_id: str
    product_name: str
    trend_topic: Optional[str] = None
    influencers_tagged: list[str]
    headline: str
    status: Literal["Draft", "Active", "Paused", "Completed"]
    budget_inr: float
    estimated_reach: int
    created_at: str


# ---------------------------------------------------------------------------
# Alert Schemas
# ---------------------------------------------------------------------------
class AlertItem(BaseModel):
    alert_id: str
    product_id: str
    product_name: str
    alert_level: Literal["HIGH", "MEDIUM", "LOW"]
    stockout_probability: float
    current_stock: int
    predicted_demand_7d: int
    recommended_order_quantity: int
    driving_trend: Optional[str] = None
    message: str
    created_at: str
    status: Literal["PENDING", "ACKNOWLEDGED", "REORDER_PLACED", "DISMISSED"]


class SendWhatsAppRequest(BaseModel):
    recipient_phone: str = Field("+919876543210", example="+919876543210")
    alert_id: Optional[str] = None
    custom_message: Optional[str] = None


class SendWhatsAppResponse(BaseModel):
    status: Literal["SENT", "SIMULATED", "FAILED"]
    recipient_phone: str
    timestamp: str
    message_sid: str
    body: str


class AcknowledgeAlertRequest(BaseModel):
    action: Literal["ACKNOWLEDGED", "REORDER_PLACED", "DISMISSED"]
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Executive Dashboard Schema
# ---------------------------------------------------------------------------
class DashboardStatsResponse(BaseModel):
    total_skus: int
    critical_stockout_alerts: int
    active_local_trends: int
    verified_influencers_count: int
    active_campaigns_count: int
    total_inventory_units: int
    top_emerging_trends: list[TrendItem]
    top_critical_skus: list[InventoryPredictionResponse]
    recent_alerts: list[AlertItem]
