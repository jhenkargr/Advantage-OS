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
    unit_price: Optional[float] = 0.0
    supplier_lead_time_days: Optional[int] = 3
    reorder_quantity: Optional[int] = 20


class ProductCreate(BaseModel):
    product_id: str = Field(..., example="SKU041", description="SKU identifier")
    product_name: Optional[str] = Field(None, example="Avocado Toast Spread 200g", description="Product title")
    category: Optional[str] = Field(None, example="F&B", description="Product category")
    current_stock: Optional[int] = Field(None, ge=0, example=30, description="Available stock quantity")
    low_stock_threshold: Optional[int] = Field(None, ge=0, example=10, description="Minimum threshold for low stock alert")
    unit_price: Optional[float] = Field(None, ge=0.0, example=15.99, description="Retail selling price")
    supplier_lead_time_days: Optional[int] = Field(None, ge=0, example=3, description="Supplier lead time in days")
    reorder_quantity: Optional[int] = Field(None, ge=0, example=20, description="Standard reorder batch size")


class ProductUpdate(BaseModel):
    product_id: Optional[str] = Field(None, example="SKU041", description="SKU identifier (optional if provided in URL path)")
    product_name: Optional[str] = Field(None, example="Avocado Toast Spread 200g", description="Updated product title")
    category: Optional[str] = Field(None, example="F&B", description="Updated category")
    current_stock: Optional[int] = Field(None, ge=0, example=30, description="Updated stock quantity")
    low_stock_threshold: Optional[int] = Field(None, ge=0, example=10, description="Updated low stock alert threshold")
    unit_price: Optional[float] = Field(None, ge=0.0, example=15.99, description="Updated unit price")
    supplier_lead_time_days: Optional[int] = Field(None, ge=0, example=3, description="Updated supplier lead time in days")
    reorder_quantity: Optional[int] = Field(None, ge=0, example=20, description="Updated reorder quantity")


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


# ---------------------------------------------------------------------------
# CSV Upload & Stock Plan Schemas
# ---------------------------------------------------------------------------
class StockPlanItem(BaseModel):
    product_id: str
    product_name: str
    avg_daily_demand: float
    forecast_30d: float
    current_stock: str
    safety_stock: Optional[int] = None
    reorder_point: Optional[int] = None
    target_stock: Optional[int] = None
    order_qty_now: Optional[int] = None
    status: str
    note: Optional[str] = None


class StockPlanResponse(BaseModel):
    status: str
    total_items: int
    understocked_count: int = 0
    overstock_count: int = 0
    ok_count: int = 0
    csv_path: str
    items: list[StockPlanItem] = []
    data: list[StockPlanItem] = []


class ProductUploadResponse(BaseModel):
    status: str
    message: str
    filename: str
    total_rows_parsed: int
    inserted_count: int
    updated_count: int
    total_products_in_db: int
    stock_plan_generated: bool = False
    stock_plan_rows: int = 0
    stock_plan_csv_path: Optional[str] = None
    sample_products: list[dict] = []


class SalesHistoryUploadResponse(BaseModel):
    status: str
    message: str
    filename: str
    total_rows_parsed: int
    inserted_count: int
    total_sales_logs_in_db: int
    stock_plan_generated: bool = False
    stock_plan_rows: int = 0
    stock_plan_csv_path: Optional[str] = None
    stock_plan_summary: list[StockPlanItem] = []


# ---------------------------------------------------------------------------
# Comprehensive Product Analytics & Visual Representations (Graphs)
# ---------------------------------------------------------------------------

class DailySalesDataPoint(BaseModel):
    date: str
    units_sold: int
    revenue: float = 0.0
    price: Optional[float] = None
    stock_after: Optional[int] = None
    channel: Optional[str] = "POS_STORE"
    promotion_applied: Optional[bool] = False
    trend_score: Optional[float] = None


class TimelineChartDataset(BaseModel):
    units_sold: list[int] = []
    revenue: list[float] = []
    stock_levels: list[Optional[int]] = []
    trend_scores: list[Optional[float]] = []


class TimelineChart(BaseModel):
    title: str = "Daily Sales & Revenue Timeline"
    chart_type: str = "line"
    labels: list[str] = []
    datasets: TimelineChartDataset = TimelineChartDataset()
    data_points: list[DailySalesDataPoint] = []


class CumulativeSalesChart(BaseModel):
    title: str = "Cumulative Units & Revenue Growth"
    chart_type: str = "area"
    labels: list[str] = []
    cumulative_units: list[int] = []
    cumulative_revenue: list[float] = []


class ChannelBreakdownItem(BaseModel):
    channel: str
    units_sold: int
    revenue: float
    percentage: float


class ChannelBreakdownChart(BaseModel):
    title: str = "Sales Distribution by Channel"
    chart_type: str = "pie"
    channels: list[ChannelBreakdownItem] = []


class StockBenchmarkChart(BaseModel):
    title: str = "Inventory Level vs Replenishment Benchmarks"
    chart_type: str = "bar"
    categories: list[str] = [
        "Current Stock",
        "Low Stock Threshold",
        "Safety Stock",
        "Reorder Point",
        "Target Stock",
    ]
    values: list[Optional[int]] = []


class ProjectedDepletionPoint(BaseModel):
    day: int
    date: str
    projected_stock: float
    safety_stock: Optional[int] = None
    reorder_point: Optional[int] = None


class ProjectedDepletionChart(BaseModel):
    title: str = "30-Day Forward Stock Depletion Projection"
    chart_type: str = "line"
    days_until_stockout: Optional[int] = None
    daily_run_rate: float = 0.0
    projection: list[ProjectedDepletionPoint] = []


class ProductChartsRepresentation(BaseModel):
    timeline_chart: TimelineChart
    cumulative_sales_chart: CumulativeSalesChart
    channel_breakdown_chart: ChannelBreakdownChart
    stock_benchmark_chart: StockBenchmarkChart
    projected_depletion_chart: ProjectedDepletionChart


class ProductSalesSummary(BaseModel):
    total_records: int = 0
    total_units_sold: int = 0
    total_revenue: float = 0.0
    avg_daily_sales: float = 0.0
    sales_velocity_7d: float = 0.0
    revenue_7d: float = 0.0
    sales_velocity_30d: float = 0.0
    revenue_30d: float = 0.0
    min_daily_units: int = 0
    max_daily_units: int = 0
    peak_sales_date: Optional[str] = None
    days_of_inventory_remaining: Optional[float] = None


class ProductDetailStockPlan(BaseModel):
    avg_daily_demand: float = 0.0
    forecast_30d: float = 0.0
    current_stock: int = 0
    safety_stock: Optional[int] = None
    reorder_point: Optional[int] = None
    target_stock: Optional[int] = None
    order_qty_now: Optional[int] = None
    status: str = "OK"
    note: Optional[str] = None
    reorder_urgency: str = "OK"  # "CRITICAL", "HIGH", "MEDIUM", "LOW", "OK"


class MLForecastInsights(BaseModel):
    available: bool = False
    predicted_7_day_demand: Optional[int] = None
    stockout_probability: Optional[float] = None
    recommended_order_quantity: Optional[int] = None
    alert_level: Optional[str] = None


class ProductFullAnalyticsResponse(BaseModel):
    product: ProductItem
    stock_status: str  # "OUT_OF_STOCK", "LOW_STOCK", "HEALTHY", "OVERSTOCKED"
    stock_plan: ProductDetailStockPlan
    sales_summary: ProductSalesSummary
    charts: ProductChartsRepresentation
    ml_forecast: MLForecastInsights

