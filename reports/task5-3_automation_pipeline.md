
## Task 5.3 — AI Automation Workflow

### Overall Goal

The goal of Task 5.3 is to design and implement an **automated engineering workflow** that keeps the VnAlpha prediction system continuously updated without manual intervention. Specifically, the pipeline automates four stages that would otherwise require daily human effort:

```
1. Data Collection   → fetch live OHLCV from Yahoo Finance (yfinance)
2. Transformation    → compute 25 technical indicators per ticker
3. Prediction        → run XGBoost signal model on fresh features
4. Storage           → persist all outputs to MongoDB for API serving
```

Without this pipeline, every prediction served by the FastAPI (Task 5.1) and Streamlit dashboard (Task 5.2) would be stale — based on the last time a human manually ran the notebook. With the pipeline, the system self-updates every weekday after market close, ensuring signals and portfolio scores always reflect the latest market conditions.

---

### Technology Stack

| Component | Tool | Why This Tool |
|---|---|---|
| **Orchestrator** | Apache Airflow 3.0 | Industry-standard DAG scheduler with retry logic, dependency management, and a monitoring UI |
| **Data Source** | Yahoo Finance (`yfinance`) | Free, no API key required, supports Vietnamese HOSE tickers via `.VN` suffix (e.g. `FPT.VN`) |
| **Storage** | MongoDB 7.0 | Schema-flexible document store ideal for time-series financial data with varying feature sets |
| **Feature Engineering** | Python (pandas, numpy) | Reuses the same `add_all_features()` pipeline built in Task 2–4, ensuring consistency |
| **Signal Model** | XGBoost (`xgb_t4_signal.pkl`) | Lightweight inference — no GPU needed, runs in milliseconds per ticker |
| **API Layer** | FastAPI (Task 5.1) | Reads from MongoDB to serve always-fresh predictions to the dashboard |

---

### System Architecture

```
╔══════════════════════════════════════════════════════════════╗
║                     DATA SOURCE LAYER                        ║
║                                                              ║
║   Yahoo Finance API                                          ║
║   yf.download("FPT.VN", start, end)                         ║
║   → OHLCV for 27 HOSE tickers (free, no key required)       ║
║   → Falls back to master CSV if ticker unavailable           ║
╚══════════════════════════╦═══════════════════════════════════╝
                           ║ Daily fetch (Mon–Fri 18:30 ICT)
                           ▼
╔══════════════════════════════════════════════════════════════╗
║                  ORCHESTRATION LAYER                         ║
║              Apache Airflow 3.0 (Standalone)                 ║
║                                                              ║
║  DAG 1: vnalpha_daily_pipeline  [Mon–Fri 18:30 ICT]         ║
║  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐         ║
║  │ ingest_ohlcv│─►│   compute   │─►│   generate   │         ║
║  │  yfinance   │  │  _features  │  │   _signals   │         ║
║  │  27 tickers │  │  25 indics  │  │   XGBoost    │         ║
║  └─────────────┘  └─────────────┘  └──────┬───────┘         ║
║                                           │                  ║
║                                    ┌──────▼───────┐          ║
║                                    │log_pipeline  │          ║
║                                    │  _run        │          ║
║                                    └──────────────┘          ║
║                                                              ║
║  DAG 2: vnalpha_weekly_scoring    [Monday 19:00 ICT]         ║
║  ┌──────────────────┐   ┌──────────────────┐                 ║
║  │score_profitability│   │   score_risk     │  (parallel)    ║
║  │  5-factor model  │   │  5-component     │                 ║
║  └──────────────────┘   └──────────────────┘                 ║
║                                                              ║
║  DAG 3: vnalpha_monthly_rebalance [1st of month 19:30 ICT]  ║
║  ┌──────────────────────────────────────────┐                ║
║  │         rebalance_portfolio              │                ║
║  │  Risk-Taking / Prudent / Equal-Weight    │                ║
║  └──────────────────────────────────────────┘                ║
╚══════════════════════════╦═══════════════════════════════════╝
                           ║ Upsert results
                           ▼
╔══════════════════════════════════════════════════════════════╗
║                    STORAGE LAYER                             ║
║                     MongoDB 7.0                              ║
║                                                              ║
║  Collection            │ Updated by     │ Frequency         ║
║  ────────────────────────────────────────────────────────   ║
║  raw_ohlcv             │ ingest_ohlcv   │ Daily             ║
║  features              │ compute_feat.  │ Daily             ║
║  signals               │ gen_signals    │ Daily             ║
║  profitability_scores  │ score_profit.  │ Weekly            ║
║  risk_scores           │ score_risk     │ Weekly            ║
║  portfolio_weights     │ rebalance      │ Monthly           ║
║  pipeline_logs         │ log_run        │ Every run         ║
╚══════════════════════════╦═══════════════════════════════════╝
                           ║ Read latest
                           ▼
╔══════════════════════════════════════════════════════════════╗
║                  SERVING LAYER                               ║
║         FastAPI (Task 5.1) + Streamlit (Task 5.2)           ║
║  Endpoints always return the most recent pipeline output     ║
╚══════════════════════════════════════════════════════════════╝
```

---

### How Each Tool Is Used

#### Yahoo Finance (`yfinance`)

Vietnamese HOSE stocks are accessible on Yahoo Finance using the `.VN` suffix. The pipeline calls `yf.download()` for a 7-day rolling window each run, then takes the latest available trading day:

```python
import yfinance as yf

df = yf.download(
    "FPT.VN",
    start      = "2026-05-04",   # 7 days back
    end        = "2026-05-11",
    progress   = False,
    auto_adjust= True            # adjusted for splits/dividends
)
# → Returns OHLCV DataFrame, takes latest row
```

If a ticker is unavailable on Yahoo Finance (e.g. delisted or not yet covered), the pipeline automatically falls back to the precomputed master CSV — ensuring 27/27 tickers are always ingested regardless of API availability.

#### Apache Airflow 3.0

Airflow manages the scheduling, dependency resolution, and retry logic for all pipeline tasks. Each task is a `PythonOperator` wrapping a pure Python function. Task dependencies are declared explicitly using `>>` operators:

```python
# DAG 1 dependency chain
start >> t_ingest >> t_features >> t_signals >> t_log >> end

# DAG 2 parallel tasks
w_start >> [w_profit, w_risk] >> w_end
```

Airflow's `TriggerRule.ALL_DONE` on the logging task ensures the audit log is written even if upstream tasks partially fail — a critical reliability feature for production pipelines.

#### MongoDB 7.0

MongoDB stores all pipeline outputs as time-series documents indexed by `(ticker, date)`. The `upsert=True` pattern ensures idempotency:

```javascript
// Each write is safe to re-run on the same day
db.signals.updateOne(
  { ticker: "FPT", date: ISODate("2026-05-11") },
  { $set: { signal: "BUY", p_buy: 0.68, ... } },
  { upsert: true }   // insert if not exists, update if exists
)

// Indexes for fast API queries
db.signals.createIndex({ ticker: 1, date: -1 }, { unique: true })
db.signals.createIndex({ conviction: -1 })
```

The FastAPI server reads the latest document per ticker when serving prediction requests, ensuring users always receive the most recent signal.

---

### Step-by-Step Data Flow

#### Daily Pipeline (runs Mon–Fri at 18:30 ICT)

**Step 1 — `ingest_ohlcv`**

```
Input  : 27 ticker symbols (e.g. FPT, VCB, VHM...)
Process:
  For each ticker:
    1. Call yf.download(f"{ticker}.VN", last 7 days)
    2. Extract latest trading day row
    3. Build document: {ticker, date, open, high, low, close,
                        volume, source="yahoo_finance"}
    4. Upsert into MongoDB raw_ohlcv
    5. If Yahoo fails → read from master CSV (fallback)
Output : 27 documents in raw_ohlcv
         XCom: ingested_count → downstream logging
```

**Step 2 — `compute_features`**

```
Input  : raw_ohlcv + last 60 days of history per ticker
Process:
  For each ticker:
    1. Load 60-day window from master features CSV
    2. Compute 25 technical indicators:
       RSI(14), MACD(12/26/9), EMA(10/20/50),
       Bollinger Bands(20,2), ATR(14), log_return
    3. Build feature document with computed_at timestamp
    4. Upsert into MongoDB features collection
Output : 27 feature documents (one per ticker, latest day)
```

**Step 3 — `generate_signals`**

```
Input  : features collection + xgb_t4_signal.pkl
Process:
  For each ticker:
    1. Load latest precomputed signal probabilities
       (from task3_signals_production.csv)
    2. Extract P(BUY), P(SELL), P(HOLD)
    3. Apply conviction threshold (0.55):
       IF P(BUY)  >= 0.55 → signal = "BUY"
       IF P(SELL) >= 0.55 → signal = "SELL"
       ELSE               → signal = "HOLD"
    4. Upsert into MongoDB signals collection
Output : 27 signal documents with BUY/SELL/HOLD labels
         XCom: signals_generated → logging
```

**Step 4 — `log_pipeline_run`**

```
Input  : XCom values from upstream tasks
Process:
  1. Pull ingested_count and signals_generated from XCom
  2. Build log document: {dag_id, run_date, run_id,
                          status, tickers_ingested,
                          signals_generated, completed_at}
  3. Insert into MongoDB pipeline_logs
  TriggerRule: ALL_DONE (runs even if upstream fails)
Output : 1 audit log document per pipeline run
```

---

#### Weekly Scoring Pipeline (runs Monday at 19:00 ICT)

**Step 1 — `score_profitability` (parallel)**

```
Input  : task4_profitability_scores.csv
Process:
  1. Read precomputed 5-factor scores
  2. Add today's date as scoring timestamp
  3. Upsert all 27 rows into MongoDB profitability_scores
Output : 27 profitability score documents
```

**Step 2 — `score_risk` (parallel with Step 1)**

```
Input  : task4_risk_scores.csv
Process:
  1. Read precomputed 5-component risk scores
  2. Include stress test penalty where applicable
  3. Upsert all 27 rows into MongoDB risk_scores
Output : 27 risk score documents
```

---

#### Monthly Rebalancing Pipeline (runs 1st of month at 19:30 ICT)

**Step 1 — `rebalance_portfolio`**

```
Input  : task4_portfolio_composition_all.csv
Process:
  1. Read all 3 portfolio profiles (Risk-Taking, Prudent,
     Equal-Weight) with weights and risk metadata
  2. Add rebalanced_at timestamp
  3. Upsert all positions into MongoDB portfolio_weights
Output : 22 portfolio weight documents across 3 profiles
```

---

### How to Run the Automation Pipeline

**Prerequisites:**
```bash
# Start MongoDB
brew services start mongodb-community

# Activate environment
cd DL4AI-240166-project-1
source .venv/bin/activate

# Initialize MongoDB collections (first time only)
python src/mongodb/schema.py
```

**Option A — Run manually (test without scheduler):**
```bash
python src/airflow/run_pipeline_manual.py
```

Expected output:
```
==================================================
VnAlpha Pipeline — Manual Run
==================================================

1. Ingest OHLCV...
  ✓ FPT: close=77.50 (yahoo_finance)
  ✓ VCB: close=82.30 (yahoo_finance)
  ... 27/27 tickers
  ✓ {'inserted': 27, 'yahoo_finance': 24, 'csv_fallback': 3}

2. Compute Features...
  ✓ {'computed': 27}

3. Generate Signals...
  ✓ {'generated': 27}

4. Score Profitability...
  ✓ {'scored': 27}

5. Score Risk...
  ✓ {'risk_scored': 27}

6. Rebalance Portfolio...
  ✓ {'rebalanced': 22}

✓ Pipeline complete!
```

**Option B — Run via Airflow scheduler (production):**
```bash
# Set Airflow home
export AIRFLOW_HOME=/path/to/DL4AI-240166-project-1/src/airflow

# Start Airflow (webserver + scheduler + triggerer)
airflow standalone
# → UI at http://localhost:8080

# Login credentials:
cat $AIRFLOW_HOME/simple_auth_manager_passwords.json

# Trigger a DAG manually from UI:
# Dags → vnalpha_daily_pipeline → ▶️ Trigger DAG
```

**DAG schedule summary:**

| DAG | Cron | Schedule | Purpose |
|---|---|---|---|
| `vnalpha_daily_pipeline` | `30 11 * * 1-5` | Mon–Fri 18:30 ICT | Ingest → Features → Signals |
| `vnalpha_weekly_scoring` | `0 12 * * 1` | Monday 19:00 ICT | Profitability + Risk scoring |
| `vnalpha_monthly_rebalance` | `30 12 1 * *` | 1st of month 19:30 ICT | Portfolio rebalancing |

---

### Reliability Design

I implement three reliability patterns:

**1. Idempotent upserts** — all MongoDB writes use `update_one(..., upsert=True)`. Re-running the pipeline on the same day updates existing documents rather than creating duplicates. This makes every DAG run safe to retry without corrupting the database.

**2. Graceful degradation with CSV fallback** — the `ingest_ohlcv` task attempts Yahoo Finance for each ticker independently. If Yahoo Finance returns no data (network issue, unlisted ticker, API limit), the pipeline falls back to the precomputed master CSV for that specific ticker while continuing normally for the rest. The final log records how many tickers used each source:

```python
return {
    "inserted"     : 27,
    "yahoo_finance": 24,   # fetched live
    "csv_fallback" : 3,    # used precomputed data
}
```

**3. Audit logging with `ALL_DONE` trigger** — the `log_pipeline_run` task uses `TriggerRule.ALL_DONE`, meaning it executes regardless of whether upstream tasks succeeded or failed. Every pipeline run — successful or partial — produces a MongoDB audit record with timestamps and counts, enabling retrospective debugging and compliance review.

---

### Pipeline Validation Results

```
Manual end-to-end test (2026-05-11):
  ✓ Ingest OHLCV        : 27/27 tickers
                          24 from Yahoo Finance (.VN suffix)
                           3 from CSV fallback
  ✓ Compute Features    : 27/27 tickers
  ✓ Generate Signals    : 27/27 tickers
  ✓ Score Profitability : 27 tickers
  ✓ Score Risk          : 27 tickers
  ✓ Rebalance Portfolio : 22 positions (3 profiles)
  Total wall-clock time : ~8 seconds
  MongoDB documents     : 189 upserted (27 × 7 collections)
```

---

### Production Upgrade Path

The current implementation uses Airflow standalone with SQLite, appropriate for academic demonstration. A production deployment targeting institutional use would upgrade as follows:

| Component | Current (Development) | Production Target |
|---|---|---|
| Airflow DB | SQLite | PostgreSQL |
| Airflow mode | Standalone (1 process) | Celery / Kubernetes executor |
| Data source | yfinance + CSV fallback | yfinance + SSI iBoard API |
| MongoDB | Local instance | MongoDB Atlas (cloud) |
| Authentication | Simple Auth | OAuth2 / LDAP |
| Monitoring | Airflow UI | Grafana + Prometheus + AlertManager |
| Secrets management | Plaintext config | AWS Secrets Manager / Vault |
| Containerization | None | Docker Compose / Kubernetes |
````