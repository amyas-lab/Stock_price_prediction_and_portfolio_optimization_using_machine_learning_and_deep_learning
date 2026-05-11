
## Task 5.1 — Model Deployment: FastAPI REST API

### Why FastAPI?

I chose FastAPI as the deployment framework for three reasons. First, it generates interactive API documentation automatically via Swagger UI at `/docs` — allowing anyone to test every endpoint directly in the browser without writing a single line of client code. Second, FastAPI's Pydantic-based request/response validation catches malformed inputs before they reach the model, making the API robust against bad data. Third, FastAPI is built on ASGI (via uvicorn), making it significantly faster than Flask for ML inference workloads where multiple requests may arrive concurrently.

```
Framework comparison:
  Flask    → synchronous, manual docs, no built-in validation
  FastAPI  → async-ready, auto Swagger docs, Pydantic validation ✓
  Django   → too heavyweight for a pure API service
```

---

### How to Start the API Server

**Step 1 — Activate environment and install dependencies:**
```bash
cd DL4AI-240166-project-1
source .venv/bin/activate
pip install fastapi uvicorn pydantic tensorflow xgboost joblib pandas
```

**Step 2 — Start the server:**
```bash
python run_api.py
```

**Expected startup output:**
```
🚀 Loading models...
  ✓ MTL T4 model loaded
  ✓ MTL T3 model loaded
  ✓ xgb_signal_t4 loaded
  ✓ xgb_signal_t3 loaded
  ✓ feature_scaler loaded
  ✓ task4_scaler loaded
  ✓ ticker_encoder loaded
  ✓ profitability data loaded (27 rows)
  ✓ risk_scores data loaded (27 rows)
  ✓ portfolio data loaded (22 rows)
  ✓ signals data loaded
✓ API ready!
INFO: Uvicorn running on http://0.0.0.0:8000
```

**Step 3 — Open interactive documentation:**
```
http://localhost:8000/docs    ← Swagger UI (click any endpoint → Try it out)
http://localhost:8000/redoc   ← ReDoc UI (read-only documentation)
```

---

### Endpoint 1 — Health Check

**Request:**
```bash
curl -X GET http://localhost:8000/health
```

**Sample output:**
```json
{
  "status": "healthy",
  "models_loaded": {
    "mtl_t4": true,
    "mtl_t3": true,
    "xgb_signal_t4": true,
    "xgb_signal_t3": true,
    "feature_scaler": true,
    "task4_scaler": true,
    "ticker_encoder": true
  },
  "supported_tickers_t3": [
    "FPT", "VCB", "VHM", "VNM", "HPG",
    "VIC", "TCB", "MSN", "MWG", "VND"
  ],
  "supported_tickers_t4": [
    "FPT", "VCB", "VHM", "VNM", "HPG", "VIC", "TCB",
    "MSN", "MWG", "VND", "BID", "CTG", "MBB", "ACB",
    "HDB", "TPB", "SHB", "PDR", "KDH", "DXG", "GAS",
    "HSG", "PNJ", "SAB", "CMG", "ELC", "SGT"
  ],
  "version": "1.0.0"
}
```

---

### Endpoint 2 — Price Prediction

**Request:**
```bash
curl -X POST http://localhost:8000/predict/price \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "FPT",
    "n_days_back": 20
  }'
```

**Sample input:**
```json
{
  "ticker": "FPT",
  "n_days_back": 20
}
```

| Field | Type | Description | Default |
|---|---|---|---|
| `ticker` | string | Stock ticker (must be in supported list) | required |
| `n_days_back` | int | Historical lookback window in days | 20 |

**Sample output:**
```json
{
  "ticker": "FPT",
  "prediction_date": "2026-05-11",
  "current_price": 77.50,
  "predicted_returns": [0.2423],
  "predicted_prices": [97.99],
  "direction": "UP",
  "confidence": 0.6821,
  "model_used": "MTL_Seq2Seq_GRU_Attention"
}
```

| Field | Description |
|---|---|
| `prediction_date` | Date of latest available market data |
| `current_price` | Last closing price (VND thousands) |
| `predicted_returns` | Predicted log return(s) for forecast horizon |
| `predicted_prices` | Predicted closing price(s) in VND thousands |
| `direction` | `UP` / `DOWN` / `NEUTRAL` — dominant forecast direction |
| `confidence` | Model conviction score in [0, 1] |
| `model_used` | Model architecture identifier |

---

### Endpoint 3 — Trading Signal

**Request:**
```bash
curl -X POST http://localhost:8000/predict/signal \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "VHM",
    "threshold": 0.55
  }'
```

**Sample input:**
```json
{
  "ticker": "VHM",
  "threshold": 0.55
}
```

| Field | Type | Description | Default |
|---|---|---|---|
| `ticker` | string | Stock ticker | required |
| `threshold` | float | Conviction threshold for signal gate [0.40–0.80] | 0.55 |

**Sample output — SELL signal:**
```json
{
  "ticker": "VHM",
  "signal_date": "2026-05-11",
  "signal": "SELL",
  "p_buy": 0.1203,
  "p_sell": 0.7841,
  "p_hold": 0.0956,
  "conviction": 0.7841,
  "threshold_used": 0.55,
  "recommendation": "Strong SELL signal. P(SELL)=78.41% ≥ threshold=55.00%. Consider reducing or exiting position."
}
```

**Sample output — BUY signal:**
```json
{
  "ticker": "HDB",
  "signal_date": "2026-05-11",
  "signal": "BUY",
  "p_buy": 0.6234,
  "p_sell": 0.1102,
  "p_hold": 0.2664,
  "conviction": 0.6234,
  "threshold_used": 0.55,
  "recommendation": "Strong BUY signal. P(BUY)=62.34% ≥ threshold=55.00%. Consider entering a long position."
}
```

**Sample output — HOLD:**
```json
{
  "ticker": "MSN",
  "signal_date": "2026-05-11",
  "signal": "HOLD",
  "p_buy": 0.3812,
  "p_sell": 0.2944,
  "p_hold": 0.3244,
  "conviction": 0.3812,
  "threshold_used": 0.55,
  "recommendation": "No high-conviction signal. Max conviction=38.12% < threshold=55.00%. Stay flat or maintain current position."
}
```

**Threshold guidance:**
```
threshold=0.40 → More signals, lower precision (~91% of days flagged)
threshold=0.55 → Balanced precision/coverage (~64%) ← recommended
threshold=0.65 → Fewer signals, higher precision (~50% of days)
```

---

### Endpoint 4 — Portfolio Composition

**Request:**
```bash
# Available profiles: risk_taking | prudent | equal_weight
curl -X GET http://localhost:8000/portfolio/equal_weight
curl -X GET http://localhost:8000/portfolio/risk_taking
curl -X GET http://localhost:8000/portfolio/prudent
```

**Sample input:** URL parameter `profile` in path.

**Sample output — equal_weight:**
```json
{
  "profile": "equal_weight",
  "stocks": [
    {
      "ticker": "GAS",
      "sector": "Industrial",
      "weight": 0.10,
      "risk_score": 4.02,
      "risk_flag": "LOW"
    },
    {
      "ticker": "TCB",
      "sector": "Banking",
      "weight": 0.10,
      "risk_score": 6.04,
      "risk_flag": "MEDIUM"
    },
    {
      "ticker": "HDB",
      "sector": "Banking",
      "weight": 0.10,
      "risk_score": 4.44,
      "risk_flag": "LOW"
    },
    {
      "ticker": "VIC",
      "sector": "RealEstate",
      "weight": 0.10,
      "risk_score": 6.55,
      "risk_flag": "MEDIUM"
    },
    {
      "ticker": "VND",
      "sector": "Technology",
      "weight": 0.10,
      "risk_score": 6.18,
      "risk_flag": "MEDIUM"
    }
  ],
  "expected_return": 0.5393,
  "expected_vol": 0.2477,
  "sharpe_ratio": 1.3969,
  "total_stocks": 10
}
```

**Sample output — prudent:**
```json
{
  "profile": "prudent",
  "stocks": [
    {"ticker": "GAS", "sector": "Industrial", "weight": 0.3333, "risk_score": 4.02, "risk_flag": "LOW"},
    {"ticker": "HDB", "sector": "Banking",    "weight": 0.3333, "risk_score": 4.44, "risk_flag": "LOW"},
    {"ticker": "SAB", "sector": "Consumer",   "weight": 0.3333, "risk_score": 3.53, "risk_flag": "LOW"}
  ],
  "expected_return": 0.1644,
  "expected_vol": 0.2480,
  "sharpe_ratio": 0.4550,
  "total_stocks": 3
}
```

---

### Endpoint 5 — Profitability Scores

**Request:**
```bash
curl -X GET http://localhost:8000/portfolio/scores/profitability
```

**Sample input:** No input required.

**Sample output (truncated to top 3):**
```json
{
  "scores": [
    {
      "rank": 1,
      "ticker": "GAS",
      "sector": "Industrial",
      "mtl_score": 0.9600,
      "tech_score": 0.2600,
      "signal_score": 0.0300,
      "sharpe_score": 0.5900,
      "trend_score": 1.0000,
      "composite_score": 0.7430
    },
    {
      "rank": 2,
      "ticker": "TCB",
      "sector": "Banking",
      "mtl_score": 0.8500,
      "tech_score": 1.0000,
      "signal_score": 0.4400,
      "sharpe_score": 0.5600,
      "trend_score": 0.7000,
      "composite_score": 0.7290
    },
    {
      "rank": 3,
      "ticker": "HDB",
      "sector": "Banking",
      "mtl_score": 0.7400,
      "tech_score": 0.7400,
      "signal_score": 0.5200,
      "sharpe_score": 0.3300,
      "trend_score": 0.7400,
      "composite_score": 0.7130
    }
  ],
  "evaluation_date": "2026-05-11",
  "total_tickers": 27
}
```

---

### Endpoint 6 — Risk Scores

**Request:**
```bash
curl -X GET http://localhost:8000/portfolio/scores/risk
```

**Sample input:** No input required.

**Sample output (truncated to top 3 highest risk):**
```json
{
  "scores": [
    {
      "ticker": "PDR",
      "sector": "RealEstate",
      "volatility_risk": 8.90,
      "sell_risk": 7.40,
      "drawdown_risk": 9.60,
      "correlation_risk": 6.20,
      "reversal_risk": 6.30,
      "composite_risk": 8.83,
      "risk_flag": "EXCLUDED"
    },
    {
      "ticker": "DXG",
      "sector": "RealEstate",
      "volatility_risk": 8.50,
      "sell_risk": 8.90,
      "drawdown_risk": 8.80,
      "correlation_risk": 7.30,
      "reversal_risk": 7.00,
      "composite_risk": 7.20,
      "risk_flag": "HIGH"
    },
    {
      "ticker": "SAB",
      "sector": "Consumer",
      "volatility_risk": 2.70,
      "sell_risk": 1.10,
      "drawdown_risk": 0.70,
      "correlation_risk": 0.32,
      "reversal_risk": 0.56,
      "composite_risk": 3.53,
      "risk_flag": "LOW"
    }
  ],
  "total_tickers": 27
}
```

---

### Python Client — All Endpoints

```python
import requests

BASE = "http://localhost:8000"

# 1. Health check
print(requests.get(f"{BASE}/health").json()['status'])

# 2. Price prediction
pred = requests.post(f"{BASE}/predict/price",
    json={"ticker": "FPT", "n_days_back": 20}).json()
print(f"FPT → {pred['direction']} ({pred['confidence']:.1%})")

# 3. Trading signal
sig = requests.post(f"{BASE}/predict/signal",
    json={"ticker": "VHM", "threshold": 0.55}).json()
print(f"VHM → {sig['signal']} (P={sig['conviction']:.1%})")

# 4. Portfolio
port = requests.get(f"{BASE}/portfolio/equal_weight").json()
print(f"Equal-Weight → Sharpe {port['sharpe_ratio']:.4f}")

# 5. Profitability scores
prof = requests.get(f"{BASE}/portfolio/scores/profitability").json()
top3 = [(s['ticker'], s['composite_score'])
        for s in prof['scores'][:3]]
print(f"Top 3: {top3}")

# 6. Risk scores
risk = requests.get(f"{BASE}/portfolio/scores/risk").json()
excluded = [s['ticker'] for s in risk['scores']
            if s['risk_flag'] == 'EXCLUDED']
print(f"Excluded: {excluded}")
```

---

### Error Responses

```json
// 400 Bad Request — unsupported ticker
{
  "detail": "Ticker AAPL not supported."
}

// 503 Service Unavailable — model not loaded
{
  "detail": "Prediction model not available"
}

// 404 Not Found — invalid portfolio profile
{
  "detail": "Invalid profile. Choose: ['risk_taking', 'prudent', 'equal_weight']"
}
```
````