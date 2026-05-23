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
└── extras/                # Supplementary: EDA, sentiment scraping, vnstock exploration

data/                      # Shared datasets (see data/README.md)
├── vietnam/               # HOSE raw OHLCV + production feature/signal/portfolio CSVs
├── nasdaq/                # US Nasdaq raw + engineered CSVs
└── sentiment/             # Scraped financial news (26 tickers)

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

## Deployment

The project has **two independent deployment targets** managed via separate git branches.

### Frontend → Vercel (auto-deploy)

Vercel watches the `main` branch. Every push automatically triggers a rebuild.

```bash
# 1. Commit your changes
git add frontend/src/
git commit -m "your message"

# 2. Push — Vercel deploys within ~2 min
git push origin main
```

Dashboard: [vercel.com/dashboard](https://vercel.com/dashboard) → check build logs if the deploy fails.

---

### Backend → HF Spaces (manual, `hf-deploy` branch)

The `hf-deploy` branch is a self-contained slice of the repo — it holds only the API source, models, and production data files. It is **not** kept in sync with `main` automatically; you cherry-pick changes into it manually.

```bash
# 1. Switch to the deploy branch
git checkout hf-deploy

# 2. Pull in only the API files you changed on main
git checkout main -- src/api/main.py
# If config.py changed, bring it too — but note: hf-deploy uses
# notebooks/data/ as DATA_DIR (the container was built that way).
# Only update config.py here if you also moved the data files in this branch.

# 3. Commit
git add src/api/
git commit -m "sync API changes from main"

# 4. Push to HF Spaces remote (remote is named 'hf')
git push hf hf-deploy:main

# 5. Return to main
git checkout main
```

HF Spaces rebuilds the Docker container on every push. Build logs: [huggingface.co/spaces/amyas0107/investnature-api](https://huggingface.co/spaces/amyas0107/investnature-api) → Logs tab.

> **Why two branches?** The full project repo (~500 MB with models + LFS data) exceeds HF Spaces' push limits. `hf-deploy` contains only the ~180 MB needed to run the API.

---

### Deployment summary

| Target | Trigger | Branch | Remote |
|---|---|---|---|
| Vercel (frontend) | Auto on push | `main` | `origin` |
| HF Spaces (API) | Manual push | `hf-deploy` | `hf` |

---

## Disclaimer

Academic project (CS313). Not financial advice.
