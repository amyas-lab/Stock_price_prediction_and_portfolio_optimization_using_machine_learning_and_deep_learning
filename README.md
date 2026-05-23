---
title: InvestNature API
emoji: 🌿
colorFrom: yellow
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# CS313 Final Project — Time-Series Deep Learning for Stock Markets

## Problem Statement

Stock markets are inherently volatile and driven by dozens of overlapping signals — price momentum, trading volume, macroeconomic sentiment, industry fundamentals, and more. Traditional rule-based approaches struggle to synthesise all of these dimensions simultaneously. This project applies machine learning and deep learning models to two real markets — the **Nasdaq** (US) and the **Vietnam Stock Exchange** — to facilitate quantitative trading decisions: forecasting future prices, identifying optimal entry/exit points, and constructing risk-adjusted portfolios.

---

## Tasks Overview

| Task | Subtask | Description 
|------|---------|-------------
| **Task 1** — Nasdaq price prediction | 1.1 | Multi-feature extension (OHLCV)
| | 1.2 | k-th day forecast 
| | 1.3 | k consecutive days forecast 
| **Task 2** — Vietnam price prediction | 2.1 | Multi-feature extension 
| | 2.2 | k-th day forecast 
| | 2.3 | k consecutive days forecast 
| **Task 3** — Vietnam trading signals | 3.1 | Buying signal identification
| | 3.2 | Selling signal identification
| **Task 4** — Vietnam portfolio management | 4.1 | Portfolio composition
| | 4.2 | Risk management
| | 4.3 | Portfolio optimisation 
| **Task 5** — Deployment (extra credit) | 5.1 | Model deployment (FastAPI) 
| | 5.2 | Model as SaaS (Streamlit) 
| | 5.3 | AI engineering workflow (Airflow + MongoDB) 
| **Task 6** — Report & repository | 6.1–6.3 | Report, GitHub repo, README 

---

## How to Set Up

```bash
git clone <repo-url>
cd DL4AI-240166-project-1

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

To run the notebooks:

```bash
jupyter notebook
```

---

## Project Structure

```
DL4AI-240166-project-1/
├── notebooks/              # All Jupyter notebooks (see walkthrough below)
│   └── data/               # Prepared datasets (Nasdaq, Vietnam, sentiment)
├── src/
│   ├── api/                # FastAPI REST API (Task 5.1)
│   ├── dashboard/          # Streamlit SaaS app (Task 5.2)
│   ├── airflow/            # Airflow DAGs & pipeline (Task 5.3)
│   ├── mongodb/            # MongoDB schema definitions
│   ├── data/               # Shared data loaders, preprocessors, splitters
│   └── evaluation/         # Regression metric utilities
├── models/                 # Saved model artefacts (.keras, .pkl)
├── reports/                # Deployment write-ups for Task 5
├── run_api.py              # Entry point to launch the FastAPI server
├── requirements.txt        # Core dependencies
└── requirements_api.txt    # API-specific dependencies
```

---

## Notebook Walkthrough

> **Note:** Notebook filenames have not been renamed to avoid breaking cross-notebook dependencies. Follow the order below rather than relying on the filename prefix alone.

### Phase 1 — Data & EDA (Nasdaq)

These notebooks cover the US market pipeline and form the basis for Task 1.

| Notebook | Purpose |
|----------|---------|
| `1-exploratory-data-analysis.ipynb` | EDA for the Nasdaq dataset |
| `2-data-preprocessing.ipynb` | Feature engineering and preprocessing for Nasdaq |
| `3-nasdaq-stock-price-prediction.ipynb` | LSTM/GRU models — Tasks 1.1, 1.2, 1.3 |

### Phase 2 — Data & EDA (Vietnam)

These notebooks build the Vietnam market data pipeline before the task notebooks are run.

| Notebook | Purpose |
|----------|---------|
| `4-vnstock.ipynb` | Load and inspect Vietnam stock market data |
| `5-sentiment-analysis.ipynb` | Crawl and explore news headlines for Vietnam market |
| `6-industry-analysis.ipynb` | Crawl fundamental/financial metrics by industry |
| `7-EDA-vietnam-stock-market.ipynb` | Exploratory data analysis for Vietnam market |
| `8-data-preprocessing-vnstock.ipynb` | Preprocessing and feature engineering for Vietnam data |
| `10-sentiment_analysis_final_project-vnstock.ipynb` | Sentiment scoring via a hybrid model; outputs scores in **[−1, 1]** (−1 = negative, 0 = neutral, +1 = positive) |

### Phase 3 — Task Notebooks (Vietnam)

Run these after Phase 2 is complete.

| Notebook | Task |
|----------|------|
| `2-vietnam_stock_prediction.ipynb` | Tasks 2.1, 2.2, 2.3 — Vietnam price forecasting |
| `3-vietnam_trading_signal_identification.ipynb` | Tasks 3.1, 3.2 — buy/sell signal models |
| `4-vietnam_portfolio_management.ipynb` | Tasks 4.1, 4.2, 4.3 — portfolio composition, risk, and optimisation |

---

## Results Summary

### Task 1 — Nasdaq Stock Price Prediction
LSTM and GRU architectures were trained on 10 Nasdaq-listed tech stocks (AAPL, AMZN, GOOGL, META, MSFT, MU, NFLX, NVDA, QCOM, and a pooled model). Multi-feature inputs (Open, High, Low, Close, Volume) consistently outperformed close-only baselines. Both single-step (k-th day) and multi-step (k-day horizon) forecasting were implemented, with the best models achieving competitive MAE and RMSE on the held-out test split.

### Task 2 — Vietnam Stock Price Prediction
The same LSTM/GRU family was applied to Vietnam stocks using the `vnstock` data source. Augmenting price features with sentiment scores and industry fundamentals (from notebooks 5, 6, and 10) provided measurable improvements on multi-horizon forecasts. Seq2Seq architectures with attention were also explored for the k-consecutive-day subtask.

### Task 3 — Trading Signal Identification (Vietnam)
A multi-task learning (MTL) GRU model and an XGBoost classifier were trained on 10 Vietnam tickers (FPT, VCB, VHM, VNM, HPG, VIC, TCB, MSN, MWG, VND). Technical indicators (SMA, EMA, MACD, RSI, Bollinger Bands) were engineered as additional features. The models output buy, sell, and hold signals and were evaluated using classification metrics including F1-score and confusion matrices.

### Task 4 — Portfolio Management (Vietnam)
An XGBoost-based profitability ranking and a risk scoring model were trained across 27 Vietnam tickers. Portfolio optimisation combined the return and risk scores under Markowitz-style allocation. Three investor profiles (aggressive, balanced, conservative) were back-tested, with results visualised in `models/task4_*.png`.

---

## Task 5 — Deployment (Extra Credit)

For detailed technical write-ups, see the `reports/` folder:

- [`reports/task5-1_API_services.md`](reports/task5-1_API_services.md) — FastAPI REST API
- [`reports/task5-2_Model_SaaS.md`](reports/task5-2_Model_SaaS.md) — Streamlit web dashboard
- [`reports/task5-3_automation_pipeline.md`](reports/task5-3_automation_pipeline.md) — Airflow + MongoDB pipeline

### Quick Start — Task 5.1: FastAPI

```bash
source venv/bin/activate
pip install -r requirements_api.txt

python run_api.py
```

The server starts at `http://0.0.0.0:8000`. Interactive docs are available at:

- `http://localhost:8000/docs` (Swagger UI — recommended)
- `http://localhost:8000/redoc` (ReDoc)

Available endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | System status and loaded models |
| POST | `/predict/price` | 5-day price trajectory forecast |
| POST | `/predict/signal` | BUY / SELL / HOLD trading signal |
| GET | `/portfolio/{profile}` | Portfolio composition by risk profile |
| GET | `/portfolio/scores/profitability` | Task 4.1 profitability scores |
| GET | `/portfolio/scores/risk` | Task 4.2 risk scores |

### Quick Start — Task 5.2: Streamlit Dashboard

```bash
streamlit run src/dashboard/app.py
```

### Quick Start — Task 5.3: Airflow Pipeline

```bash
export AIRFLOW_HOME=$(pwd)/src/airflow
airflow db migrate
airflow users create --username admin --password admin \
  --firstname Admin --lastname User --role Admin --email admin@example.com
airflow webserver &
airflow scheduler &
```

The DAG `vnalpha_pipeline` in `src/airflow/dags/vnalpha_pipeline.py` handles data ingestion, transformation, model inference, and result storage to MongoDB on a scheduled basis.

---

## Data Sources

- **Nasdaq**: `yfinance` — historical OHLCV data for major tech stocks
- **Vietnam**: `vnstock` — historical OHLCV, dividend history, financial ratios, news headlines

---

## Disclaimer

This repository is for academic research and learning purposes only. It is **not** financial advice.
