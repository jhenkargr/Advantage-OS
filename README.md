# 🚀 AdVantage OS

> **AI-Driven Local Growth & Autonomous Retail Intelligence Platform for SMEs**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19.2-61DAFB.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-8.3-646CFF.svg)](https://vitejs.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6.svg)](https://www.typescriptlang.org/)
[![Neon Database](https://img.shields.io/badge/Neon-PostgreSQL-00E599.svg)](https://neon.tech/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

AdVantage OS is an end-to-end, full-stack enterprise retail operating system engineered for Small and Medium Enterprises (SMEs). It bridges the gap between high-level data science and everyday business operations by combining **real-time social listening**, **stationary rate-of-change ML trend forecasting**, **automated Poisson/TSB demand planning**, **generative multi-model creative marketing**, and an **interactive, modern React 19 dashboard**.

---

## 📋 Table of Contents

- [Overview & Problem Statement](#-overview--problem-statement)
- [System Architecture](#-system-architecture)
- [Key Features & Modules](#-key-features--modules)
  - [1. Market Intelligence & Trend Predictor](#1-market-intelligence--trend-predictor)
  - [2. Autonomous Demand & Inventory Engine](#2-autonomous-demand--inventory-engine)
  - [3. Generative Creative Studio](#3-generative-creative-studio)
  - [4. Influencer Discovery & Matchmaking](#4-influencer-discovery--matchmaking)
  - [5. Executive React 19 Dashboard](#5-executive-react-19-dashboard)
- [Project Directory Structure](#-project-directory-structure)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
  - [1. Clone / Import Repository](#1-clone--import-repository)
  - [2. Backend Setup (FastAPI & ML)](#2-backend-setup-fastapi--ml)
  - [3. Environment Variables Configuration](#3-environment-variables-configuration)
  - [4. Database Initialization (Neon PostgreSQL)](#4-database-initialization-neon-postgresql)
  - [5. Frontend Setup (React + Vite)](#5-frontend-setup-react--vite)
- [Running the Application](#-running-the-application)
- [API Reference & Documentation](#-api-reference--documentation)
- [Environment Variables Reference](#-environment-variables-reference)
- [Testing & Model Verification](#-testing--model-verification)
- [Contributing & License](#-contributing--license)

---

## 💡 Overview & Problem Statement

Small and medium retailers face four critical hurdles:
1. **Unpredictable Consumer Micro-Trends:** Consumer tastes shift rapidly on social media (TikTok, Instagram, Reddit, X), leaving traditional retailers weeks behind the curve.
2. **Stockouts vs. Dead Inventory:** Traditional inventory systems rely on basic moving averages that fail to anticipate demand surges or intermittent purchasing patterns.
3. **High Creative Agency Costs:** Generating tailored advertising imagery and ad copy for new products is expensive and slow.
4. **Disjointed Systems:** Point-of-sale, social listening, inventory ledgers, and marketing tools are siloed across disconnected platforms.

**AdVantage OS** solves these problems in a single, cohesive ecosystem:
- Listens to live social signals and generates mathematical 30-day trend forecasts with 95% confidence bands.
- Ingests sales receipts via OCR and computes optimal reorder points (ROP), safety stock, and purchase orders.
- Generates publication-ready marketing visuals and ad captions using enterprise AI models (NVIDIA, Groq LLaMA-3.3, Pollinations).
- Surfaces all insights in a sleek, responsive dashboard.

---

## 🏗️ System Architecture

```mermaid
graph TD
    subgraph Frontend ["Frontend (React 19 + TypeScript + Vite)"]
        UI[Executive Dashboard & Pages]
        Charts[TrendChart / BarChart / DonutChart]
        API_Client[Axios / Fetch Client]
    end

    subgraph Backend ["Backend (FastAPI Gateway)"]
        Router[Master API Router / Lifespan]
        TrendMod[Trend Predictor Module]
        InvMod[Inventory & Demand Subsystem]
        CreatMod[Creative Studio]
        OCRMod[Receipt OCR Pipeline]
    end

    subgraph Intelligence ["AI & Machine Learning Engine"]
        TrendML["TrendForecastingModel (Delta Ensemble MAE: 0.2456)"]
        DemandML["Blended HistGBR (Poisson) + TSB Intermittent Model"]
        NVIDIA["NVIDIA API Catalog / SDXL / Flux"]
        Groq["Groq LLaMA-3.3-70b Copywriter"]
        Cohere["Cohere Command-R+ Topic Filtering"]
        DistilBERT["DistilBERT + VADER Sentiment Ensemble"]
    end

    subgraph Data ["Data & Storage Layer"]
        NeonDB[("Neon PostgreSQL Serverless")]
        MLModelsTable[("ml_models (BLOB storage & sync)")]
        LocalCache[("SQLite / Local Fallback")]
    end

    UI --> API_Client
    API_Client --> Router
    Router --> TrendMod
    Router --> InvMod
    Router --> CreatMod
    Router --> OCRMod

    TrendMod --> DistilBERT
    TrendMod --> TrendML
    TrendMod --> Cohere
    InvMod --> DemandML
    CreatMod --> NVIDIA
    CreatMod --> Groq

    TrendMod --> NeonDB
    TrendML --> MLModelsTable
    InvMod --> NeonDB
    OCRMod --> NeonDB
```

---

## ✨ Key Features & Modules

### 1. Market Intelligence & Trend Predictor
* **Live Multi-Source Scraping:** Concurrent real-time listeners for Twitter/X and Reddit with fallback synthesis for unseen or niche search terms.
* **Dual Ensemble Sentiment:** 70% DistilBERT transformer polarity combined with 30% VADER lexicon heuristics.
* **`TrendForecastingModel_v2_DeltaEnsemble`:**
  * Trained on daily rate-of-change velocity ($\Delta y_t$) instead of non-stationary raw levels, eliminating autoregressive flatline attractors.
  * **Test MAE 0.2456** and **R² 0.9971** (+26.15% improvement over naive persistence).
  * Momentum damping ($\phi = 0.97^{t-1}$) simulating consumer adoption S-curves.
  * Expanding 95% confidence intervals ($\pm 1.96 \cdot \sigma \cdot \sqrt{t/3}$).
* **Zero-Disk Database Sync:** Model binaries (894 KB) are stored directly inside the Neon PostgreSQL `ml_models` table, supporting automatic hydration and cross-instance synchronization.

### 2. Autonomous Demand & Inventory Engine
* **Blended Demand Forecaster:** 50/50 blend of **HistGradientBoostingRegressor** with Poisson loss and **TSB (Teunter-Syntetos-Babai)** intermittent demand models.
* **Dynamic Stock Optimization:** Calculates safety stocks, reorder points (ROP), stockout risks, and recommended order quantities based on lead times and service level targets.
* **OCR Receipt Ingestion:** Ingests printed receipts and paper invoices via Tesseract OCR / EasyOCR and auto-populates transaction ledgers.
* **Bulk Data Onboarding:** Endpoints for bulk CSV ingestion of product catalogs and sales histories.

### 3. Generative Creative Studio
* **Multi-Provider Fallback:**
  * **NVIDIA API Catalog** (SDXL & Flux models for high-fidelity product imagery).
  * **Groq LLaMA-3.3-70b-versatile** (Sub-second marketing ad copy and social captions).
  * **Pollinations AI** (Zero-configuration fallback generation).
* **Dual-Mode Endpoints:** Exposes `/image_generation/*` and `/generate/*` with text-to-image, image-to-image, and automated hashtag generation.

### 4. Influencer Discovery & Matchmaking
* Evaluates micro-influencer profiles by category niche, audience demographic clusters, engagement velocity, and estimated campaign ROI.

### 5. Executive React 19 Dashboard
* **Clean Design System:** Premium dark-themed user interface styled with custom CSS variables and glassmorphic panels.
* **Interactive Data Visualizations:** Native canvas-backed trend trajectories with shaded 95% confidence ribbons, sales distribution donut charts, and stock level bar graphs.
* **Real-Time Model Health Badges:** Displays active ML model version, database hydration source, and accuracy metrics.

---

## 📁 Project Directory Structure

```
advantage/
├── README.md                      # Complete Project Documentation (this file)
├── .gitignore                     # Git exclusion rules (secrets, dependencies, build files)
├── metrics_fixed.json             # Evaluated ML model performance metrics
├── model_test_analysis.pkl        # Serialized ML model fallback binary
│
├── backend/                       # Python FastAPI Backend Subsystem
│   ├── main.py                    # Master API Gateway & Lifespan Orchestrator
│   ├── requirements.txt           # Python production dependencies
│   ├── .env.example               # Environment variables template
│   │
│   ├── creative_studio/           # Generative Marketing Module
│   │   ├── __init__.py
│   │   ├── creative.py            # Image generation & copy router (/image_generation)
│   │   └── nvidia_client.py       # NVIDIA + Groq + Pollinations AI API client
│   │
│   ├── trend_predictor/           # Market Intelligence & Forecasting Module
│   │   ├── __init__.py
│   │   ├── model_class.py         # TrendForecastingModel architecture definition
│   │   ├── train_model.py         # ML training, validation, & database sync script
│   │   ├── predict.py             # Trend prediction router & DB persistence (/predict)
│   │   ├── model_test_analysis.pkl# Local cached model weights
│   │   ├── reddit_scrap.py        # Reddit social scraper
│   │   ├── x_scrapper.py          # Twitter / X keyword scraper
│   │   ├── trend_engine.py        # SME Trend clustering router (/api/trends)
│   │   ├── trend_analyzer_cohere.py# Cohere LLM topic noise filter
│   │   ├── trend_db.py            # Trend database tables & queries
│   │   └── keywords.py            # Cohere semantic keyword expansion router
│   │
│   └── inventory/                 # Inventory, Retail & Demand Subsystem
│       ├── __init__.py
│       ├── demand.py              # Blended Poisson/TSB demand planner (/api/demand)
│       ├── config.py              # Application configuration & env parser
│       ├── db.py                  # SQLAlchemy engine, Neon Postgres models & migrations
│       ├── api/
│       │   ├── api_router.py      # Inventory router aggregator
│       │   └── endpoints/
│       │       ├── inventory.py   # Catalog CRUD & CSV upload endpoints
│       │       ├── dashboard.py   # Executive KPI summary endpoints
│       │       ├── influencers.py # Influencer matchmaking endpoints
│       │       └── ocr.py         # Receipt OCR scanning endpoint
│       ├── models/
│       │   └── schemas.py         # Pydantic v2 data models
│       └── services/
│           ├── inventory_service.py # Inventory analytics & business logic
│           ├── alert_service.py   # Stockout & replenishment alerts
│           └── ocr_service.py     # OCR text extraction engine
│
└── frontend/                      # React 19 + TypeScript + Vite Subsystem
    ├── package.json               # Node.js dependencies & scripts
    ├── vite.config.ts             # Vite configuration with API proxy
    ├── tsconfig.json              # TypeScript root configuration
    ├── index.html                 # Single-page application entry point
    │
    ├── src/
    │   ├── main.tsx               # React application DOM root
    │   ├── App.tsx                # Page router & layout orchestration
    │   ├── index.css              # Global styling & CSS token design system
    │   ├── App.css                # Layout styles
    │   │
    │   ├── pages/                 # Main Dashboard Views
    │   │   ├── HomePage.tsx       # Executive overview & quick actions
    │   │   ├── TrendPredictorPage.tsx # Live keyword trends & ML forecast chart
    │   │   ├── CreativeStudioPage.tsx # Generative AI image studio & copywriter
    │   │   ├── InventoryPage.tsx  # Product catalog, stock planning, & OCR upload
    │   │   └── InfluencerPage.tsx # Micro-influencer matchmaking
    │   │
    │   ├── components/
    │   │   ├── charts/            # Canvas & SVG Data Charts
    │   │   │   ├── TrendChart.tsx # 30-day forecast chart with 95% confidence bands
    │   │   │   ├── BarChart.tsx   # Inventory & sales categorical bar chart
    │   │   │   └── DonutChart.tsx # Category distribution breakdown
    │   │   └── common/            # Reusable UI Primitives
    │   │       ├── Navbar.tsx
    │   │       ├── Footer.tsx
    │   │       ├── Badge.tsx
    │   │       ├── Modal.tsx
    │   │       ├── EmptyState.tsx
    │   │       ├── LoadingSkeleton.tsx
    │   │       └── ErrorState.tsx
    │   │
    │   ├── services/
    │   │   └── api.ts             # Typed REST API service connecting to backend
    │   └── types/
    │       └── api.ts             # TypeScript interfaces for API contracts
```

---

## 💻 Prerequisites

Ensure you have the following installed on your machine:

1. **Python 3.10, 3.11, or 3.12**
   - Check with: `python --version`
2. **Node.js 18+ (Node 20+ Recommended)**
   - Check with: `node --version`
   - Includes `npm` (or you can use `pnpm` / `yarn`)
3. **Git**
   - Check with: `git --version`
4. **Neon PostgreSQL Database** (or local PostgreSQL / SQLite)
   - Free cloud PostgreSQL database from [Neon](https://neon.tech/)
5. *(Optional)* **Tesseract OCR** (for receipt scanning feature)
   - [Windows Tesseract Installer](https://github.com/UB-Mannheim/tesseract/wiki) or `brew install tesseract` on macOS / `sudo apt install tesseract-ocr` on Ubuntu.

---

## 🛠️ Installation & Setup

### 1. Clone / Import Repository

Open your terminal and clone the repository:

```bash
git clone https://github.com/jhenkargr/Advantage-OS.git
cd advantage
```

---

### 2. Backend Setup (FastAPI & ML)

Navigate to the `backend` directory, create a Python virtual environment, and install dependencies:

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment:
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Windows (CMD):
.\venv\Scripts\activate.bat
# On macOS / Linux:
source venv/bin/activate

# Upgrade pip and install production requirements
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 3. Environment Variables Configuration

In `backend/`, copy `.env.example` to create your active `.env` file:

```bash
# On Windows (PowerShell or CMD):
copy .env.example .env

# On macOS / Linux:
cp .env.example .env
```

Open `.env` in your text editor and fill in your credentials:

```dotenv
# ── Application Settings ──
APP_ENV=development
API_PREFIX=/api
CORS_ORIGINS=["http://localhost:5173","http://localhost:3000","http://127.0.0.1:5173"]

# ── Database (Neon PostgreSQL) ──
# Get your connection string from your Neon Console (https://console.neon.tech)
DATABASE_URL=postgresql://user:password@ep-cool-endpoint-pooler.us-east-2.aws.neon.tech/advantage_db?sslmode=require
DATABASE_URL_UNPOOLED=postgresql://user:password@ep-cool-endpoint.us-east-2.aws.neon.tech/advantage_db?sslmode=require

# ── AI Services & LLMs ──
COHERE_API_KEY=your_cohere_api_key_here
GROQ_API_KEY=gsk_your_groq_api_key_here
NVIDIA_API_KEY=nvapi_your_nvidia_api_key_here

# Optional: Social Scraping & Fallbacks
SCRAPEBADGER_API_KEY=your_key_here
TWITTER_API_KEY=your_twitter_key_here
```

> [!NOTE]
> If you do not provide external API keys immediately, the platform gracefully degrades to built-in fallbacks (e.g., deterministic keyword synthesis for social trends, Pollinations AI for images, and local SQLite for development if Neon is not set).

---

### 4. Database Initialization (Neon PostgreSQL)

AdVantage OS automatically connects to Neon PostgreSQL on server launch, creates necessary tables (`products`, `sales_history`, `stock_plans`, `ml_models`, and `{sme}_trends`), and auto-seeds the trained ML model.

To manually train and upload the latest trend forecast model binary to your Neon database:

```bash
# Make sure your virtual environment is active in backend/
python trend_predictor/train_model.py
```

This will:
1. Train the velocity-based `TrendForecastingModel` on 13,825 samples.
2. Evaluate test holdout metrics (**MAE 0.2456**, **R² 0.9971**).
3. Automatically serialize and save the model into the Neon PostgreSQL `ml_models` table.

---

### 5. Frontend Setup (React + Vite)

In a new terminal window, navigate to the `frontend/` directory and install Node.js dependencies:

```bash
cd frontend

# Install dependencies using npm
npm install
```

---

## 🏃 Running the Application

### 1. Start the Backend API

From the `backend/` directory with your virtual environment activated:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

* **API Gateway:** `http://localhost:8000`
* **Swagger UI (Interactive Docs):** `http://localhost:8000/docs`
* **Health Check:** `http://localhost:8000/health`
* **ML Model Status:** `http://localhost:8000/predict/model-status`

### 2. Start the Frontend Dashboard

From the `frontend/` directory:

```bash
npm run dev
```

* **Web Application:** `http://localhost:5173`

The frontend automatically proxies API calls to `http://localhost:8000` via Vite.

---

## 📡 API Reference & Documentation

Interactive OpenAPI / Swagger documentation is available at `http://localhost:8000/docs`.

### Summary of Key Endpoints:

| Category | Method | Endpoint | Description |
|---|---|---|---|
| **System** | `GET` | `/` | API module index and active features |
| **System** | `GET` | `/health` | Deep health probe (Neon DB, ML Model, OCR) |
| **Trend Forecast** | `POST` | `/predict` | Scrapes social feeds & computes 30-day trajectory with 95% CI |
| **Trend Forecast** | `GET` | `/predict/model-status` | Inspects database model storage, cache, and size |
| **Trend Forecast** | `POST` | `/predict/upload-model` | Uploads/syncs `.pkl` weights into Neon PostgreSQL `ml_models` |
| **SME Trends** | `GET` | `/api/trends/analyze/{sme}` | Discovers latent consumer trends filtered with Cohere |
| **Keywords** | `GET` | `/api/keywords/related` | Expands target keywords with semantic intent |
| **Demand Planning** | `GET` | `/api/demand/stock-plan` | Calculates blended Poisson/TSB demand & generates reorder plan |
| **Inventory** | `GET` | `/api/inventory/catalog` | Retrieves product catalog with stock status and margins |
| **Inventory** | `GET` | `/api/inventory/catalog/{id}/analytics` | Detailed sales velocity and performance history |
| **Inventory** | `POST` | `/api/products/upload-csv` | Bulk imports product catalog from CSV |
| **Inventory** | `POST` | `/api/sales-history/upload-csv` | Bulk imports transaction logs from CSV |
| **Receipt OCR** | `POST` | `/api/ocr/scan-receipt` | Extracts itemized costs & quantities from receipt image |
| **Creative Studio** | `POST` | `/image_generation/text2img` | Generates marketing product visuals via NVIDIA / SDXL |
| **Creative Studio** | `POST` | `/image_generation/caption` | Synthesizes high-converting ad copy via Groq LLaMA-3.3 |
| **Dashboard** | `GET` | `/api/dashboard/stats` | Executive KPI cards (revenue, stockout risk, top sellers) |

---

## ⚙️ Environment Variables Reference

| Variable | Required | Default | Purpose |
|---|:---:|---|---|
| `DATABASE_URL` | **Yes** | Local SQLite fallback | Neon PostgreSQL connection string (pooled) |
| `DATABASE_URL_UNPOOLED` | No | None | Direct unpooled Neon connection string for large batches |
| `COHERE_API_KEY` | No | Mock fallback | Cohere API key for trend extraction & keyword expansion |
| `GROQ_API_KEY` | No | Template copy | Groq Cloud key for ultra-fast LLaMA-3.3 ad copy generation |
| `NVIDIA_API_KEY` | No | Pollinations fallback | NVIDIA API Catalog key for Flux/SDXL image synthesis |
| `TWITTER_API_KEY` | No | Fallback synthesis | Twitter/X API key for social search |
| `CORS_ORIGINS` | No | `["*"]` | Allowed CORS origins for frontend-backend communication |
| `TESSERACT_CMD` | No | System path | Path to Tesseract executable for local OCR |

---

## 🧪 Testing & Model Verification

### 1. Test FastAPI Endpoints
Run a quick smoke test against your running backend:

```bash
# In backend/
python -c "
from fastapi.testclient import TestClient
from main import app
client = TestClient(app)
print('Health:', client.get('/health').json())
print('Model Status:', client.get('/predict/model-status').json())
"
```

### 2. Verify Frontend Build
To ensure TypeScript and Vite compile cleanly without type or bundling errors:

```bash
# In frontend/
npm run build
```

---

## 📄 Contributing & License

Contributions, bug reports, and feature requests are welcome!

1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

Distributed under the **MIT License**. See `LICENSE` for more information.
