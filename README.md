# 🚀 AdVantage OS

> **AI-Driven Local Growth & Autonomous Retail Intelligence Platform for SMEs**

AdVantage OS is an enterprise-grade backend platform built with **FastAPI**, **Neon PostgreSQL**, and state-of-the-art **AI/ML pipelines**. It empowers Small and Medium Enterprises (SMEs) with real-time social listening, predictive demand planning, automated generative marketing campaigns, and intelligent stock management.

---

## 🏗️ System Architecture & Modular Layout

AdVantage OS follows a clean, modular architecture where domain features are encapsulated into decoupled routers mounted onto the unified FastAPI orchestrator:

```
advantage/
├── main.py                     # Unified API Gateway & Lifespan Orchestrator
├── requirements.txt            # Production dependencies
├── .env.example                # Environment configuration template
├── .gitignore                  # GitHub ignore rules (secrets, models, DBs)
│
├── creative_studio/            # Creative Studio Module
│   ├── __init__.py             # Exports router & nvidia_client
│   ├── creative.py             # Dual-mode API router (/image_generation & /generate)
│   └── nvidia_client.py        # NVIDIA API Catalog + Groq + Pollinations AI client
│
├── trend_predictor/            # Trend Prediction & Market Intelligence Module
│   ├── __init__.py             # Exports predict_router, trends_router, keywords_router
│   ├── predict.py              # Trend Forecast Router & Model DB Persistence (/predict)
│   ├── model_test_analysis.pkl # Pretrained Trend Forecast Model (synced to Neon DB)
│   ├── reddit_scrap.py         # Reddit social listener & data extraction
│   ├── x_scrapper.py           # Twitter / X keyword listening
│   ├── trend_engine.py         # SME Trend Discovery Router (/api/trends/analyze/{sme})
│   ├── trend_analyzer_cohere.py# Cohere LLM Trend Item Analysis & Noise Filter
│   ├── trend_db.py             # Neon DB caching for SME trend clusters
│   └── keywords.py             # Cohere Keyword Expansion Router (/api/keywords)
│
└── inventory/                  # Core Retail & Inventory Subsystem
    ├── __init__.py
    ├── demand.py               # Demand Forecasting & Stock Planning Router (/api/demand)
    ├── config.py               # Application settings & environment loader
    ├── db.py                   # SQLAlchemy engine, Neon Postgres models & migrations
    ├── api/
    │   ├── api_router.py       # Master Inventory router aggregator
    │   └── endpoints/
    │       ├── inventory.py    # Catalog CRUD, analytics, CSV batch upload endpoints
    │       ├── dashboard.py    # Executive KPI metrics & financial summaries
    │       └── influencers.py  # Influencer matchmaking & clustering
    ├── models/
    │   └── schemas.py          # Pydantic v2 schemas for request/response contracts
    └── services/
        └── inventory_service.py # Core business logic & inventory analytics
```

---

## ✨ Core Feature Modules

### 1. 🎨 Creative Studio (`creative_studio/`)
- **Dual-Mode API Support**: Seamlessly exposes both `/image_generation/*` and `/generate/*` routes.
- **Multi-Provider AI Fallback**:
  - **NVIDIA API Catalog** (SDXL / Flux high-fidelity image generation)
  - **Groq LLaMA-3.3-70b-versatile** (Ultra-fast marketing copy and ad captions)
  - **Pollinations AI** (Zero-configuration fallback provider)

### 2. 📈 Trend Prediction & Database Model Persistence (`trend_predictor/`)
- **Live Social Listening**: Scrapes Reddit and X in real-time for target keywords.
- **Dual Ensemble Sentiment**: Combines VADER lexical sentiment with transformer-based DistilBERT models.
- **Database Model Storage**: Stores model binaries (`model_test_analysis.pkl`) directly inside Neon PostgreSQL (`ml_models` table) with automatic caching and zero-disk reload support.
- **Endpoints**:
  - `POST /predict`: Scrape & generate 30-day recursive trend forecasts.
  - `POST /predict/upload-model`: Upload or sync model weights into Neon PostgreSQL.
  - `GET /predict/model-status`: Real-time health check on active model weights.

### 3. 🔍 SME Trend Discovery & Cohere LLM Analysis (`trend_predictor/`)
- **Multi-Source Scraping**: Pulls signals across Reddit, YouTube, and Google Trends.
- **Unsupervised Topic Discovery**: Discovers latent consumer themes using NMF / LDA clustering.
- **Cohere LLM Filtration**: Employs `command-r-plus` to strip conversational noise and output structured, commercial trend recommendations stored in `{sme}_trends` database tables.

### 4. 📦 Demand Forecasting & Automated Stock Planning (`inventory/demand.py`)
- **Blended ML Engine**: 50/50 blend of **HistGradientBoostingRegressor** (Poisson loss) and **TSB** (Teunter-Syntetos-Babai) intermittent demand models.
- **Dynamic Inventory Parameters**: Computes safety stock, reorder points, and recommended order quantities based on lead times and service levels.
- **Database Integration**: Directly updates table `stock_plans` in Neon PostgreSQL and synchronizes with local CSV exports.

### 5. 🏷️ Product Catalog & Analytics (`inventory/`)
- Full CRUD catalog management with SKU tracking.
- Multi-channel sales history and margin analytics.
- Bulk CSV onboarding for products (`/api/products/upload-csv`) and sales logs (`/api/sales-history/upload-csv`).
- Micro-influencer matchmaking by demographic clusters.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Neon PostgreSQL database (or local PostgreSQL / SQLite)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/advantage-backend.git
   cd advantage-backend
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   Copy the example environment file and add your credentials:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with your API keys:
   - `DATABASE_URL`: Neon PostgreSQL connection string
   - `COHERE_API_KEY`: Cohere platform key
   - `NVIDIA_API_KEY`: NVIDIA API catalog key
   - `GROQ_API_KEY`: Groq Cloud key

---

## 🏃 Running the Application

Start the unified FastAPI server:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Once running:
- **API Base URL**: `http://localhost:8000`
- **Interactive API Docs (Swagger UI)**: `http://localhost:8000/docs`
- **Alternative Docs (ReDoc)**: `http://localhost:8000/redoc`
- **System Health Check**: `http://localhost:8000/health`

---

## 📡 API Reference Overview

| Module | Method | Endpoint | Description |
|---|---|---|---|
| **System** | `GET` | `/` | API discovery and available feature modules |
| **System** | `GET` | `/health` | Unified system, DB, ML, and OCR health status |
| **Creative** | `POST` | `/image_generation/text2img` | Generate marketing images from prompt |
| **Creative** | `POST` | `/image_generation/img2img` | Transform existing product visuals |
| **Creative** | `POST` | `/image_generation/caption` | Generate marketing ad copy via Groq LLaMA |
| **Prediction** | `POST` | `/predict` | Scrape Reddit+X & predict 30-day trend |
| **Prediction** | `POST` | `/predict/upload-model` | Upload / sync ML model into Neon DB |
| **Prediction** | `GET` | `/predict/model-status` | Inspect model cache & DB status |
| **Trends** | `GET` | `/api/trends/analyze/{sme}` | Cohere-powered SME trend extraction |
| **Keywords** | `GET` | `/api/keywords/related` | Semantic keyword expansion |
| **Demand** | `GET` | `/api/demand/stock-plan` | Calculate demand forecast & update stock plan |
| **Inventory** | `GET` | `/api/inventory/catalog` | List all SME products |
| **Inventory** | `GET` | `/api/inventory/catalog/{id}/analytics` | Complete product analytics with graph data |
| **Inventory** | `POST` | `/api/products/upload-csv` | Bulk upload products CSV into database |
| **Inventory** | `POST` | `/api/sales-history/upload-csv` | Bulk upload sales history CSV into database |
| **Dashboard** | `GET` | `/api/dashboard/stats` | Executive dashboard KPI summary |

---

## 🛡️ License & Contributions

Developed for the AdVantage OS project. Built for high reliability, modular scalability, and production readiness.
