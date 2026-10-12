/**
 * API Data Models & TypeScript Types for AdVantage OS
 */

// ── System & Health ──────────────────────────────────────────────────────────
export interface HealthStatus {
  status: string;
  version: string;
  database_ready: boolean;
  database_type?: string;
  catalog_ready?: boolean;
  models_ready?: boolean;
  ocr_ready?: boolean;
  creative_studio_ready?: boolean;
  creative_providers?: string;
}

// ── Dashboard Metrics ────────────────────────────────────────────────────────
export interface DashboardStats {
  total_skus: number;
  critical_stockout_alerts: number;
  active_local_trends: number;
  verified_influencers_count: number;
  active_campaigns_count: number;
  total_inventory_units: number;
  top_emerging_trends: Array<{
    trend_name: string;
    growth_rate: number;
    volume: number;
  }>;
  top_critical_skus: Array<{
    product_id: string;
    predicted_7_day_demand: number;
    stockout_probability: number;
    recommended_order_quantity: number;
    alert_level: 'HIGH' | 'MEDIUM' | 'LOW';
  }>;
  recent_alerts: Array<{
    id: string;
    type: string;
    message: string;
    timestamp: string;
  }>;
}

// ── Trend Predictor ──────────────────────────────────────────────────────────
export interface TrendingItem {
  id?: string | number;
  name?: string;
  item_name?: string;
  title?: string;
  product?: string;
  trend_score?: number;
  growth_rate?: string | number;
  velocity?: string | number;
  score?: number;
  description?: string;
  category?: string;
  confidence?: number;
  sentiment?: string;
  source?: string;
  [key: string]: unknown;
}

export interface SMETrendAnalysisResponse {
  status: string;
  sme: string;
  city: string;
  generated?: string;
  total_documents_analyzed?: number;
  total_trending_items?: number;
  trending_items: TrendingItem[];
  topic_clusters?: Array<{
    topic_id?: number;
    label?: string;
    keywords?: string[];
    weight?: number;
    growth?: number;
  }>;
  rising_searches?: Array<{
    query: string;
    value?: number | string;
    breakout?: boolean;
  }>;
  from_database?: boolean;
  database_table?: string;
  database_record_id?: number;
  database_saved_at?: string;
  error_cohere?: string;
}

export interface KeywordResponse {
  keyword: string;
  sme_type: string;
  related_keywords: string[];
  total_generated: number;
  source?: string;
}

export interface ForecastDay {
  day: number;
  date: string;
  predicted_score: number;
  predicted_trend_score?: number;
  confidence_lower?: number;
  confidence_upper?: number;
  lower_bound?: number;
  upper_bound?: number;
}

export interface ModelMetadata {
  model_name?: string;
  model_type?: string;
  version?: string;
  metrics?: {
    model_mae?: number;
    model_r2?: number;
    naive_mae?: number;
    naive_r2?: number;
    improvement_pct?: number;
  };
}

export interface AggregationSummary {
  posts_count: number;
  observed_days: number;
  start_date: string;
  end_date: string;
  continuous_days: number;
  line1?: string;
  line2?: string;
}

export interface ObservedTrendPoint {
  date: string;
  observed_trend_score: number;
  trend_score?: number;
}

export interface TrendSummaryTableRow {
  topic: string;
  current_trend_score: number;
  predicted_in_15_days: number;
  'change_15d_%'?: number;
  change_15d_pct?: number;
  indication_15d: string;
  predicted_in_30_days: number;
  'change_30d_%'?: number;
  change_30d_pct?: number;
  indication_30d: string;
}

export interface TrendPredictResult {
  keywords: string[];
  topic?: string;
  rows_collected?: number;
  current_trend_score: number;
  predicted_in_15_days: number;
  change_15d_pct: number;
  'change_15d_%'?: number;
  indication_15d: string;
  predicted_in_30_days: number;
  change_30d_pct: number;
  'change_30d_%'?: number;
  indication_30d: string;
  model_source?: string;
  model_metadata?: ModelMetadata;
  forecast?: ForecastDay[];
  observed?: ObservedTrendPoint[];
  aggregation_summary?: AggregationSummary;
  summary_table?: TrendSummaryTableRow[];
  plot_image?: string;
  error?: string;
}

// ── Influencer Matching ──────────────────────────────────────────────────────
export interface InfluencerItem {
  channel_id: string;
  channel_name: string;
  country?: string;
  published_at?: string;
  subscriber_count: number;
  video_count: number;
  total_view_count: number;
  avg_views_per_video: number;
  engagement_score: number;
  influencer_tier?: 'nano' | 'micro' | 'mid' | 'macro' | 'mega' | string;
  bot_flag: boolean;
  cluster_id?: number;
  cluster_label?: string;
  description?: string;
  keywords?: string;
}

export interface InfluencerClusterResponse {
  cluster_id: number;
  cluster_label?: string;
  count: number;
  influencer_tier_breakdown: Record<string, number>;
  influencers: InfluencerItem[];
}

// ── Creative Studio ──────────────────────────────────────────────────────────
export interface CreativeHealth {
  status: string;
  providers: string;
  key_configured: boolean;
  text2img_models?: string[];
  edit_model?: string;
  img2img_mode?: string;
  [key: string]: unknown;
}

export interface CaptionResponse {
  caption: string;
  source?: string;
  model?: string;
  warning?: string;
}

export interface GeneratedImageMeta {
  url: string;
  blob: Blob;
  modelUsed?: string;
  seedUsed?: string | number;
  prompt: string;
  type: 'text2img' | 'img2img';
  timestamp: string;
}

// ── Inventory & Product Analytics ────────────────────────────────────────────
export interface ProductItem {
  product_id: string;
  product_name: string;
  category: string;
  current_price?: number;
  unit_price?: number;
  stock_available?: number;
  current_stock?: number;
  min_reorder_level?: number;
  low_stock_threshold?: number;
  lead_time_days?: number;
  supplier_lead_time_days?: number;
  supplier_name?: string;
  reorder_quantity?: number;
}

export interface ChartDataPoint {
  label: string;
  value: number;
  secondary_value?: number;
  date?: string;
}

export interface ProductFullAnalyticsResponse {
  product: ProductItem;
  stock_status: string | {
    status?: string;
    stock_ratio?: number;
    days_of_inventory?: number;
    health_score?: number;
  };
  stock_plan?: {
    product_id?: string;
    avg_daily_demand?: number;
    forecast_30_days?: number;
    forecast_30d?: number;
    current_stock?: number;
    safety_stock?: number;
    reorder_point?: number;
    target_stock?: number;
    recommended_order_quantity?: number;
    order_qty_now?: number;
    status?: string;
    reorder_urgency?: string;
    note?: string;
  };
  sales_summary?: {
    total_quantity_sold?: number;
    total_units_sold?: number;
    total_revenue?: number;
    avg_daily_sales?: number;
    avg_price?: number;
    transactions_count?: number;
    total_records?: number;
    sales_velocity_7d?: number;
    revenue_7d?: number;
    sales_velocity_30d?: number;
    revenue_30d?: number;
    days_of_inventory_remaining?: number;
    min_daily_units?: number;
    max_daily_units?: number;
    peak_sales_date?: string;
    first_sale_date?: string;
    last_sale_date?: string;
  };
  charts: {
    timeline_chart?: {
      title?: string;
      unit?: string;
      labels?: string[];
      points?: Array<{ date: string; quantity: number; revenue?: number }>;
      data_points?: Array<{ date: string; units_sold: number; revenue?: number }>;
      datasets?: {
        labels?: string[];
        units_sold?: number[];
        revenue?: number[];
      };
    };
    cumulative_sales_chart?: {
      title?: string;
      labels?: string[];
      cumulative_units?: number[];
      cumulative_revenue?: number[];
      points?: Array<{ date: string; cumulative_quantity: number }>;
    };
    channel_breakdown_chart?: {
      title?: string;
      channels?: Array<{ channel: string; units_sold: number; revenue?: number; percentage: number }>;
      breakdown?: Array<{ channel: string; quantity: number; percentage: number }>;
    };
    stock_benchmark_chart?: {
      title?: string;
      categories?: string[];
      values?: Array<number | null>;
      metrics?: Array<{ label: string; current: number; benchmark: number }>;
    };
    projected_depletion_chart?: {
      title?: string;
      days_until_stockout?: number | null;
      daily_run_rate?: number;
      projection?: Array<{ day: number; date?: string; projected_stock: number; safety_stock?: number; reorder_point?: number }>;
      points?: Array<{ day: number; projected_stock: number; runout_risk?: boolean }>;
    };
  };
  ml_forecast?: {
    available: boolean;
    predicted_7_day_demand?: number | null;
    stockout_probability?: number | null;
    recommended_order_quantity?: number | null;
    alert_level?: 'HIGH' | 'MEDIUM' | 'LOW' | null;
  };
}

export interface AlertItem {
  alert_id: string;
  product_id: string;
  product_name: string;
  alert_level: 'HIGH' | 'MEDIUM' | 'LOW';
  stockout_probability: number;
  current_stock: number;
  predicted_demand_7d: number;
  recommended_order_quantity: number;
  driving_trend?: string;
  message: string;
  created_at: string;
  status: 'PENDING' | 'ACKNOWLEDGED' | 'REORDER_PLACED' | 'DISMISSED';
}

export interface InventoryPredictionResponse {
  product_id: string;
  product_name: string;
  category: string;
  current_stock: number;
  predicted_7_day_demand: number;
  stockout_probability: number;
  recommended_order_quantity: number;
  alert_needed: boolean;
  alert_level: 'HIGH' | 'MEDIUM' | 'LOW' | 'NONE';
  safety_stock?: number;
  avg_daily_sales?: number;
  trend_score?: number;
}

