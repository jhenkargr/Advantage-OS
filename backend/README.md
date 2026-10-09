# AdVantage OS — FastAPI Backend

**Advantage OS: An AI-Driven Local Growth Platform for SMEs**

FastAPI + Uvicorn backend integrating hyper-local social trend intelligence, micro-influencer discovery, AI-assisted creative studio, receipt-level OCR inventory tracking, machine learning demand & stockout prediction, and automated WhatsApp alerting.

---

## Architecture Overview

```
AdVantage OS Architecture
├── Layer 1: Data Sources (Reddit PRAW, YouTube Data API, Receipt Images)
├── Layer 2: Data Ingestion & Preprocessing (PostgreSQL / CSV storage)
├── Layer 3: AI & Analytics Engine (Demand Regressor, Stockout Classifier, LDA, Bot Filter)
├── Layer 4: Application & API Layer (FastAPI + Uvicorn) <--- [THIS BACKEND]
├── Layer 5: User Interface (React + Tailwind CSS Dashboard)
└── Layer 6: Output & Integrations (Twilio WhatsApp Alerts, PO Generation)
```

---

## Directory Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── config.py                 # Paths, settings, CORS origins
│   ├── main.py                   # FastAPI initialization, CORS, Lifespan
│   ├── api/
│   │   ├── __init__.py
│   │   ├── api_router.py         # Master router aggregating all modules
│   │   └── endpoints/
│   │       ├── dashboard.py      # /api/dashboard - Consolidated metrics
│   │       ├── inventory.py      # /api/inventory - Catalog, stock, ML inference
│   │       ├── ocr.py            # /api/ocr - Receipt OCR & fuzzy catalog matching
│   │       ├── trends.py         # /api/trends - Local trends & LSTM forecasting
│   │       ├── influencers.py   # /api/influencers - Clustering & bot filter
│   │       ├── creatives.py      # /api/creatives - Ad copy, image prompts, CTR
│   │       └── alerts.py         # /api/alerts - Stockout alerts & WhatsApp
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py            # Pydantic request/response models
│   └── services/
│       ├── __init__.py
│       ├── inventory_service.py  # ML inference & catalog state management
│       ├── ocr_service.py        # Tesseract OCR & fuzzy SKU matching
│       ├── trend_service.py      # Trend detection & lifecycle projection
│       ├── influencer_service.py # Niche clustering & bot scoring
│       ├── creative_service.py   # Multimodal copy & CTR heuristic engine
│       └── alert_service.py      # WhatsApp dispatch & alert tracker
├── tests/
│   └── test_api.py               # End-to-end integration test suite
├── run.py                        # Uvicorn runner
└── README.md
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10+ (active virtual environment in `.venv`)
- Tesseract-OCR installed (default: `C:\Program Files\Tesseract-OCR\tesseract.exe`)

### 2. Start the Backend Server

You can start the server using any of the following commands:

**Option A: Root Launcher**
```bash
python run.py
```

**Option B: Backend Launcher**
```bash
python backend/run.py
```

**Option C: Direct Uvicorn Command**
```bash
uvicorn app.main:app --reload --port 8000 --app-dir backend
```

Once started:
- **Interactive Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc UI**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## API Endpoints Reference

### 1. Executive Dashboard (`/api/dashboard`)
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/dashboard/stats` | Consolidated metrics (active trends, critical SKUs, inventory total, alerts) |

### 2. Inventory & ML Demand Prediction (`/api/inventory`)
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/inventory/catalog` | List all 40 products with current stock and threshold |
| `GET` | `/api/inventory/catalog/{product_id}` | Get specific SKU details |
| `POST` | `/api/inventory/catalog` | Add a new product to the catalog |
| `PUT` | `/api/inventory/catalog/{product_id}/stock` | Update stock manually |
| `GET` | `/api/inventory/predict/{product_id}` | Run ML inference (7-day demand, stockout risk, reorder qty) |
| `GET` | `/api/inventory/predict-all` | Batch ML predictions across all catalog SKUs |
| `GET` | `/api/inventory/sales-history/{product_id}` | Get recent daily sales history for charting |

### 3. Retail Operations & OCR Ingestion (`/api/ocr`)
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/ocr/scan-receipt` | Upload image file (multipart/form-data) for OCR extraction and fuzzy matching |
| `POST` | `/api/ocr/scan-sample` | Scan bundled `ocr_samp1.jpeg` sample receipt |
| `POST` | `/api/ocr/apply-receipt` | Apply scanned items to update catalog stock (STOCK_IN / STOCK_OUT) |

### 4. Trend Engine (`/api/trends`)
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/trends/active` | List local social trends with opportunity scores & linked SKUs |
| `GET` | `/api/trends/{trend_id}` | Detailed trend metrics with 14-day history and 7-day forecast |
| `POST` | `/api/trends/refresh` | Trigger social signal ingestion refresh |

### 5. Influencer Matchmaker (`/api/influencers`)
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/influencers` | List micro-influencers filtered for authentic engagement (bot score >= 0.70) |
| `GET` | `/api/influencers/match/{trend_id}` | Match and score top influencers for a selected trend |

### 6. Creative Studio & Campaigns (`/api/creatives`)
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/creatives/generate` | Generate AI ad headlines, captions, hashtags, and image prompts |
| `POST` | `/api/creatives/score-ctr` | Predict Click-Through-Rate (CTR) and copy effectiveness |
| `GET` | `/api/creatives/campaigns` | List created marketing campaigns |
| `POST` | `/api/creatives/campaigns` | Create and launch a new campaign |

### 7. Alerts & Automated WhatsApp (`/api/alerts`)
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/alerts` | List active stockout & trend alerts |
| `PUT` | `/api/alerts/{alert_id}/acknowledge` | Acknowledge alert / record reorder action |
| `POST` | `/api/alerts/send-whatsapp` | Dispatch WhatsApp alert in Twilio message format |

---

## Running the Automated Test Suite

```bash
python backend/tests/test_api.py
```
Validates all endpoints end-to-end, including ML inference, OCR parsing, trend forecasting, and alert delivery.
