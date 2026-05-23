---
title: InvestNature API
emoji: 🌿
colorFrom: yellow
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# CS313 — Stock Price Prediction & Portfolio Optimization

End-to-end ML/DL system for Vietnamese and US stock markets: price forecasting, trading signal generation, and portfolio optimization — served via a FastAPI backend and two frontend interfaces.

**Live demo:** [InvestNature on Vercel](https://frontend-nine-azure-26.vercel.app) · **API:** [HF Spaces](https://huggingface.co/spaces/amyas0107/investnature-api)

---

## Project Structure

```
notebooks/
├── task1_nasdaq/          # Task 1 — Nasdaq price prediction (LSTM, GRU, Seq2Seq)
├── task2_vietnam_price/   # Task 2 — Vietnam 5-day MTL price forecasting
├── task3_trading_signals/ # Task 3 — BUY/SELL/HOLD signal identification (XGBoost)
├── task4_portfolio/       # Task 4 — Portfolio optimization (profitability + risk scoring)
└── extras/                # Supplementary: EDA, sentiment, vnstock exploration

models/                    # Trained artifacts (see models/README.md)
src/
├── api/                   # FastAPI backend (deployed on HF Spaces)
├── dashboard/             # Streamlit frontend (Task 5.2)
├── airflow/               # Airflow pipeline — automated daily workflow (Task 5.3)
└── mongodb/               # MongoDB schema for pipeline persistence

frontend/                  # React/Vite frontend (deployed on Vercel) — InvestNature UI
```

---

## Tasks

| Task | Description | Notebook |
|------|-------------|----------|
| **Task 1** | Nasdaq multi-stock price prediction | `notebooks/task1_nasdaq/` |
| **Task 2** | Vietnam 5-day MTL price forecasting | `notebooks/task2_vietnam_price/` |
| **Task 3** | Trading signal identification (XGBoost) | `notebooks/task3_trading_signals/` |
| **Task 4** | Portfolio composition & risk management | `notebooks/task4_portfolio/` |
| **Task 5.1** | FastAPI deployment | `src/api/` |
| **Task 5.2** | Streamlit SaaS dashboard | `src/dashboard/` |
| **Task 5.3** | Airflow + MongoDB automation pipeline | `src/airflow/` |

---

## Architecture

```
[React Frontend — Vercel]
        │  HTTPS
        ▼
[FastAPI Backend — HF Spaces]
   ├── Branch 1: pre-computed features (27 known tickers, <1s)
   │       └── task4_master_features.csv + mtl_t4_final.keras
   └── Branch 2: live yfinance fetch (any HOSE ticker, ~3-5s)
           └── feature_pipeline.py + task4_feature_scaler.pkl
```

---

## Models

The production model is **MTL T4** (`models/mtl_t4_final.keras`): a multi-task LSTM trained on 27 Vietnamese stocks with 25 technical features over a 20-day window. It jointly predicts:
- **Regression head** — 5-day log-return trajectory
- **Classification head** — direction probability (P_BUY / P_HOLD / P_SELL)

Directional accuracy: **63.64%** on specialized tickers.

See [`models/README.md`](models/README.md) for the full list of production vs archived models.

---

## Running Locally

**Backend:**
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn src.api.main:app --reload --port 8000
```

**Frontend (React):**
```bash
cd frontend
npm install
npm run dev          # proxies /api → localhost:8000
```

**Streamlit:**
```bash
streamlit run src/dashboard/app.py
```

**Airflow pipeline:**
```bash
cd src/airflow
airflow standalone
```

---

## Disclaimer

Academic project (CS313). Not financial advice.
