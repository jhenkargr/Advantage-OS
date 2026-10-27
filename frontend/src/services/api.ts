/**
 * AdVantage OS - API Service Layer
 * Clean, type-safe API client functions communicating with FastAPI backend.
 */

import type {
  AlertItem,
  CaptionResponse,
  CreativeHealth,
  DashboardStats,
  HealthStatus,
  InfluencerClusterResponse,
  InventoryPredictionResponse,
  KeywordResponse,
  ProductFullAnalyticsResponse,
  ProductItem,
  SMETrendAnalysisResponse,
  TrendPredictResult,
} from '../types/api';

// Use environment variable if set (VITE_API_BASE_URL or VITE_BACKEND_URL), otherwise relative path for Vite dev proxy
const RAW_BASE_URL = 
  import.meta.env.VITE_API_BASE_URL ?? 
  import.meta.env.VITE_BACKEND_URL ?? 
  import.meta.env.VITE_API_URL ?? 
  '';

// Normalize BASE_URL by stripping any trailing slash
export const BASE_URL = RAW_BASE_URL.replace(/\/+$/, '');

/**
 * Builds a clean, fully-qualified API URL using the configured backend base URL
 */
export function buildApiUrl(endpoint: string): string {
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  return `${BASE_URL}${cleanEndpoint}`;
}

/**
 * Custom API Error containing status code and backend error details
 */
export class ApiError extends Error {
  status: number;
  details?: unknown;

  constructor(message: string, status: number, details?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.details = details;
  }
}

/**
 * Generic request helper with timeout and JSON parsing
 */
async function request<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = buildApiUrl(endpoint);
  const defaultHeaders: Record<string, string> = {
    Accept: 'application/json',
  };

  // Only set Content-Type to JSON if body is not FormData
  if (!(options.body instanceof FormData)) {
    defaultHeaders['Content-Type'] = 'application/json';
  }

  const response = await fetch(url, {
    ...options,
    headers: {
      ...defaultHeaders,
      ...(options.headers as Record<string, string>),
    },
  });

  if (!response.ok) {
    let errorMessage = `HTTP Error ${response.status}: ${response.statusText}`;
    let details: unknown = undefined;
    try {
      const errorJson = await response.json();
      details = errorJson;
      if (errorJson.detail) {
        errorMessage = typeof errorJson.detail === 'string' 
          ? errorJson.detail 
          : JSON.stringify(errorJson.detail);
      } else if (errorJson.message) {
        errorMessage = errorJson.message;
      }
    } catch {
      // Body not JSON
    }
    throw new ApiError(errorMessage, response.status, details);
  }

  return response.json() as Promise<T>;
}

// ── 1. System Health & Dashboard ─────────────────────────────────────────────

export async function fetchHealth(): Promise<HealthStatus> {
  return request<HealthStatus>('/health');
}

export async function fetchDashboardStats(): Promise<DashboardStats> {
  return request<DashboardStats>('/api/dashboard/stats');
}

// ── 2. Trend Prediction & Cohere Keyword Intelligence ─────────────────────────

export interface TrendAnalysisParams {
  city?: string;
  days?: number;
  topics?: number;
  demo?: boolean;
  refresh?: boolean;
}

/**
 * Fetches trending products and topics for a given SME category.
 * Route: GET /api/trends/analyze/{sme}
 */
export async function fetchSMETrends(
  smeCategory: string,
  params: TrendAnalysisParams = {}
): Promise<SMETrendAnalysisResponse> {
  const query = new URLSearchParams();
  if (params.city) query.set('city', params.city);
  if (params.days) query.set('days', String(params.days));
  if (params.topics) query.set('topics', String(params.topics));
  if (params.demo !== undefined) query.set('demo', String(params.demo));
  if (params.refresh) query.set('refresh', 'true');

  const queryString = query.toString() ? `?${query.toString()}` : '';
  const encodedSme = encodeURIComponent(smeCategory.toLowerCase().trim());
  return request<SMETrendAnalysisResponse>(`/api/trends/analyze/${encodedSme}${queryString}`);
}

/**
 * Expands a seed keyword into related trending terms using Cohere LLM.
 * Route: GET /api/keywords/related?keyword={keyword}&sme_type={sme_type}
 */
export async function fetchRelatedKeywords(
  keyword: string,
  smeType: string = 'cafe',
  limit: number = 15
): Promise<KeywordResponse> {
  const query = new URLSearchParams({
    keyword: keyword.trim(),
    sme_type: smeType.trim(),
    limit: String(limit),
  });
  return request<KeywordResponse>(`/api/keywords/related?${query.toString()}`);
}

/**
 * Runs 30-day trend forecasting using the machine learning ensemble model.
 * Route: POST /predict
 */
export async function predictTrend(
  keywords: string[],
  useTransformer: boolean = false
): Promise<TrendPredictResult> {
  return request<TrendPredictResult>('/predict', {
    method: 'POST',
    body: JSON.stringify({
      keywords,
      use_transformer: useTransformer,
    }),
  });
}

// ── 3. Influencer Matchmaking ────────────────────────────────────────────────

export const SME_CLUSTER_MAPPING: Record<string, number> = {
  'cafes': 10,
  'cafe': 10,
  'cafes_bakeries': 10,
  'bakeries': 2,
  'bakery': 2,
  'restaurants': 1,
  'restaurant': 1,
  'boutiques': 8,
  'boutique': 8,
  'salons': 9,
  'salon': 9,
  'gyms': 10,
  'gym': 10,
  'gift_decor': 4,
  'gift_shop': 4,
  'electronics': 3,
  'd2c': 4,
  'small_d2c': 4,
};

/**
 * Fetches matching influencers for a chosen SME category.
 * Calls /api/influencers/sme/{sme_type} with automatic cluster fallback.
 */
export async function fetchInfluencersBySME(
  smeType: string,
  botFreeOnly: boolean = true,
  limit: number = 80
): Promise<InfluencerClusterResponse> {
  const query = new URLSearchParams({
    bot_free_only: String(botFreeOnly),
    limit: String(limit),
  });

  const normalizedKey = smeType.toLowerCase().replace(/[^a-z0-9]/g, '_');
  try {
    return await request<InfluencerClusterResponse>(
      `/api/influencers/sme/${encodeURIComponent(normalizedKey)}?${query.toString()}`
    );
  } catch (err) {
    // Fallback directly to cluster ID if endpoint alias differs
    const clusterId = SME_CLUSTER_MAPPING[normalizedKey] ?? 10;
    return fetchInfluencersByCluster(clusterId, botFreeOnly, limit);
  }
}

/**
 * Fetches influencers in a specific cluster ID.
 * Route: GET /api/influencers/cluster/{cluster_id}
 */
export async function fetchInfluencersByCluster(
  clusterId: number,
  botFreeOnly: boolean = true,
  limit: number = 80
): Promise<InfluencerClusterResponse> {
  const query = new URLSearchParams({
    bot_free_only: String(botFreeOnly),
    limit: String(limit),
  });
  return request<InfluencerClusterResponse>(
    `/api/influencers/cluster/${clusterId}?${query.toString()}`
  );
}

// ── 4. Creative Studio ───────────────────────────────────────────────────────

export async function fetchCreativeHealth(): Promise<CreativeHealth> {
  return request<CreativeHealth>('/image_generation/health');
}

export interface Text2ImgOptions {
  width?: number;
  height?: number;
  seed?: number;
  negative_prompt?: string;
}

export interface GeneratedImageResult {
  blob: Blob;
  objectUrl: string;
  modelUsed?: string;
  seedUsed?: string;
  finalPrompt?: string;
}

/**
 * Generates an image from a text prompt.
 * Route: POST /image_generation/text2img
 */
export async function generateTextToImage(
  prompt: string,
  options: Text2ImgOptions = {}
): Promise<GeneratedImageResult> {
  const formData = new FormData();
  formData.append('prompt', prompt);
  if (options.width) formData.append('width', String(options.width));
  if (options.height) formData.append('height', String(options.height));
  if (options.seed !== undefined) formData.append('seed', String(options.seed));
  if (options.negative_prompt) formData.append('negative_prompt', options.negative_prompt);

  const url = buildApiUrl('/image_generation/text2img');
  const response = await fetch(url, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    let msg = `Image generation failed (${response.status})`;
    try {
      const err = await response.json();
      msg = err.detail || msg;
    } catch {
      // not JSON
    }
    throw new ApiError(msg, response.status);
  }

  const modelUsed = response.headers.get('X-Model-Used') || undefined;
  const seedUsed = response.headers.get('X-Seed-Used') || undefined;
  const blob = await response.blob();
  const objectUrl = URL.createObjectURL(blob);

  return { blob, objectUrl, modelUsed, seedUsed };
}

/**
 * Transforms a reference image using a creative prompt.
 * Route: POST /image_generation/img2img
 */
export async function generateImageToImage(
  prompt: string,
  referenceImage: File | Blob,
  seed: number = -1
): Promise<GeneratedImageResult> {
  const formData = new FormData();
  formData.append('prompt', prompt);
  formData.append('reference_image', referenceImage, 'reference.png');
  formData.append('seed', String(seed));

  const url = buildApiUrl('/image_generation/img2img');
  const response = await fetch(url, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    let msg = `Image transformation failed (${response.status})`;
    try {
      const err = await response.json();
      msg = err.detail || msg;
    } catch {
      // not JSON
    }
    throw new ApiError(msg, response.status);
  }

  const modelUsed = response.headers.get('X-Model-Used') || undefined;
  const seedUsed = response.headers.get('X-Seed-Used') || undefined;
  const finalPrompt = response.headers.get('X-Final-Prompt') || undefined;
  const blob = await response.blob();
  const objectUrl = URL.createObjectURL(blob);

  return { blob, objectUrl, modelUsed, seedUsed, finalPrompt };
}

/**
 * Generates an advertising caption & social copy from product description.
 * Route: POST /image_generation/caption
 */
export async function generateCaption(
  description: string,
  tone: string = 'Playful'
): Promise<CaptionResponse> {
  const formData = new FormData();
  formData.append('description', description);
  formData.append('tone', tone);

  const url = buildApiUrl('/image_generation/caption');
  const response = await fetch(url, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    let msg = `Caption generation failed (${response.status})`;
    try {
      const err = await response.json();
      msg = err.detail || msg;
    } catch {
      // not JSON
    }
    throw new ApiError(msg, response.status);
  }

  return response.json();
}

// ── 5. Inventory & Product Analytics ─────────────────────────────────────────

/**
 * Fetches all products currently in the SME inventory catalog.
 * Route: GET /api/inventory/catalog
 */
export async function fetchInventoryCatalog(): Promise<ProductItem[]> {
  return request<ProductItem[]>('/api/inventory/catalog');
}

/**
 * Fetches detailed multi-dimensional analytics and chart datasets for a specific SKU.
 * Route: GET /api/inventory/catalog/{productId}/analytics
 */
export async function fetchProductAnalytics(
  productId: string
): Promise<ProductFullAnalyticsResponse> {
  return request<ProductFullAnalyticsResponse>(
    `/api/inventory/catalog/${encodeURIComponent(productId)}/analytics`
  );
}

/**
 * Fetches the global inventory demand & stock replenishment plan.
 * Route: GET /api/demand/stock-plan
 */
export async function fetchStockPlan(): Promise<unknown[]> {
  return request<unknown[]>('/api/demand/stock-plan');
}

/**
 * Fetches active stockout and critical inventory alerts.
 * Route: GET /api/alerts
 */
export async function fetchAlerts(refresh: boolean = false): Promise<AlertItem[]> {
  const query = refresh ? '?refresh=true' : '';
  return request<AlertItem[]>(`/api/alerts${query}`);
}

/**
 * Runs the ML demand regressor & stockout classifier models for a specific product SKU.
 * Route: GET /api/inventory/predict/{productId}
 */
export async function predictProductInventory(
  productId: string
): Promise<InventoryPredictionResponse> {
  return request<InventoryPredictionResponse>(
    `/api/inventory/predict/${encodeURIComponent(productId)}`
  );
}

/**
 * Acknowledges or dismisses an active inventory alert.
 * Route: PUT /api/alerts/{alertId}/acknowledge
 */
export async function acknowledgeAlert(
  alertId: string,
  action: 'ACKNOWLEDGED' | 'REORDER_PLACED' | 'DISMISSED',
  notes?: string
): Promise<AlertItem> {
  return request<AlertItem>(`/api/alerts/${encodeURIComponent(alertId)}/acknowledge`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action, notes }),
  });
}

// ── 6. CSV Uploads & Model Operations ────────────────────────────────────────

/**
 * Uploads products catalog CSV directly to backend database.
 * Route: POST /api/products/upload-csv
 */
export async function uploadProductsCsv(
  file: File,
  replaceAll: boolean = false,
  recalculateStockPlan: boolean = true
): Promise<unknown> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('replace_all', String(replaceAll));
  formData.append('recalculate_stock_plan', String(recalculateStockPlan));
  return request<unknown>('/api/products/upload-csv', {
    method: 'POST',
    body: formData,
  });
}

/**
 * Uploads sales transaction history CSV directly to backend database.
 * Route: POST /api/sales-history/upload-csv
 */
export async function uploadSalesHistoryCsv(
  file: File,
  replaceAll: boolean = false,
  recalculateStockPlan: boolean = true
): Promise<unknown> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('replace_all', String(replaceAll));
  formData.append('recalculate_stock_plan', String(recalculateStockPlan));
  return request<unknown>('/api/sales-history/upload-csv', {
    method: 'POST',
    body: formData,
  });
}

/**
 * Uploads influencers dataset CSV.
 * Route: POST /api/influencers/upload-csv
 */
export async function uploadInfluencersCsv(
  file: File,
  replaceAll: boolean = false
): Promise<unknown> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('replace_all', String(replaceAll));
  return request<unknown>('/api/influencers/upload-csv', {
    method: 'POST',
    body: formData,
  });
}

/**
 * Checks the status of the ML prediction models.
 * Route: GET /predict/model-status
 */
export async function fetchModelStatus(): Promise<unknown> {
  return request<unknown>('/predict/model-status');
}


