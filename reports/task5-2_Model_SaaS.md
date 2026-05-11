## Task 5.2 — Model as SaaS: Streamlit Dashboard (VnAlpha)

### Why Streamlit?

I chose Streamlit as the SaaS interface for three reasons. First, it allows pure Python — no HTML, CSS, or JavaScript knowledge required — making it ideal for deploying ML models quickly without a separate frontend team. Second, Streamlit's reactive model means the UI automatically re-renders whenever the user changes an input, creating a responsive experience with minimal code. Third, Streamlit integrates naturally with Plotly for interactive charts, making it well-suited for financial dashboards that require dynamic visualization.

```
SaaS tool comparison:
  Tableau / PowerBI → powerful BI, but no live ML model integration
  TensorFlow.js     → runs model in browser, but complex to deploy
  Streamlit         → pure Python, calls FastAPI, fast to build ✓
```

---

### How to Start the Web Interface

**Prerequisites — API server must be running first:**
```bash
# Terminal 1: Start FastAPI
cd DL4AI-240166-project-1
source .venv/bin/activate
python run_api.py
# → API running at http://localhost:8000
```

**Start the Streamlit dashboard:**
```bash
# Terminal 2: Start Streamlit
cd DL4AI-240166-project-1
source .venv/bin/activate
pip install streamlit streamlit-option-menu plotly requests
streamlit run src/dashboard/app.py --server.port 8501
```

**Expected output:**
```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Open `http://localhost:8501` in any browser. The sidebar will show
**"API Connected ✓"** in green if the FastAPI server is running,
or **"API Offline ✗"** in red if it is not.

---

### How the Interface Calls the API

The dashboard communicates with the FastAPI backend exclusively
through HTTP requests using the `requests` library. Every user
interaction that requires model inference triggers a call to the
corresponding endpoint:

```
User Action                  → API Call
─────────────────────────────────────────────────────────
Click "Run Prediction"       → POST /predict/price
Click "Scan All Tickers"     → POST /predict/signal (×27)
Click "Scan" (single)        → POST /predict/signal (×1)
Select portfolio profile     → GET  /portfolio/{profile}
Load Risk Analysis page      → GET  /portfolio/scores/risk
Load Overview page           → GET  /health
```

The API call pattern used throughout the dashboard:

```python
def call_api(method, endpoint, payload=None):
    url = f"http://localhost:8000{endpoint}"
    r   = requests.get(url, timeout=30) if method == 'get' \
          else requests.post(url, json=payload, timeout=30)
    return (r.json(), None) if r.status_code == 200 \
           else (None, f"API Error {r.status_code}")
```

This design means the dashboard is entirely **stateless** —
it holds no model weights itself, only displays results returned
by the API. Swapping the backend model requires no changes to
the dashboard code.

---

### Dashboard Pages

#### Page 1 — Overview

Displays the system summary: universe size, selection alpha,
best Sharpe ratio, and portfolio performance comparison.

**How to use:** Navigate to Overview from the left sidebar.
No input required — all metrics load automatically on page open.

```
Displays:
  ✓ 4 summary metric cards (universe, alpha, Sharpe, models)
  ✓ Bar chart: total return by portfolio profile vs VNI
  ✓ Model validation table (Spearman ρ, overlap, precision)
  ✓ Pipeline architecture summary
```

---

#### Page 2 — Price Prediction

Generates a 5-day price forecast for a selected ticker using
the MTL Seq2Seq GRU + Attention model.

**How to input values:**

| Input | Type | Options | Default |
|---|---|---|---|
| Ticker | Dropdown | 17 HOSE tickers | FPT |
| Lookback window | Slider | 20–60 days | 20 |

**Steps:**
1. Select a ticker from the dropdown (e.g. `FPT`)
2. Adjust the lookback slider if needed (default 20 is recommended)
3. Click **"🔮 Run Prediction"**
4. Wait ~2 seconds for the model to run

**How to read the output:**

```
Current Price   : Latest closing price in VND thousands
Direction badge : ⬆️ UP / ⬇️ DOWN / ➡️ NEUTRAL
Confidence      : Model conviction score (>55% = high conviction)
5-Day Target    : Projected price after forecast horizon
Price chart     : Visual trajectory from today to predicted target
                  Green line = UP prediction
                  Red line   = DOWN prediction
                  Orange     = NEUTRAL
Conviction gauge: Speedometer showing model confidence
                  Green zone (>55%) = reliable signal
                  Yellow zone       = borderline
                  Red zone (<40%)   = low confidence
```

**Example input/output:**

```
Input:
  Ticker       : FPT
  Lookback     : 20 days

Output:
  Current Price : 77,500 VND
  Direction     : ⬆️ UP
  Confidence    : 68.2%
  5-Day Target  : 97,990 VND (+26.4%)
  Log Return    : 0.2423
```

---

#### Page 3 — Signal Scanner

Scans one or all 18 tickers for BUY/SELL/HOLD signals using
the XGBoost classifier from Task 3.

**How to input values:**

| Input | Type | Description |
|---|---|---|
| Conviction Threshold | Slider (0.40–0.80) | Minimum confidence to generate a signal |
| Scan All / Single | Button | Scan all 18 tickers or one selected ticker |

**Steps:**
1. Adjust the conviction threshold slider (default 0.55)
2. Click **"🔍 Scan All"** to scan all tickers
   *or* select a single ticker and click **"🎯 Scan"**
3. Wait for the progress bar to complete

**How to read the output:**

```
Summary metrics : BUY count / SELL count / HOLD count
Signal chart    : Bar chart showing signal strength per ticker
                  Green bars (above threshold) = BUY signals
                  Red bars   (below threshold) = SELL signals
                  Orange bars                  = HOLD
Signal table    : Full details per ticker including
                  P(BUY), P(SELL), conviction score, signal date
```

**Example input/output:**

```
Input:
  Threshold : 0.55
  Action    : Scan All Tickers (18 tickers)

Output:
  BUY Signals  : 0   (no tickers above BUY threshold)
  SELL Signals : 9   (9 tickers with P(SELL) ≥ 0.55)
  HOLD         : 6   (remaining tickers below threshold)

  Highest conviction SELL: VHM (78.41%)
  Highest conviction BUY : none above threshold
```

---

#### Page 4 — Portfolio

Displays portfolio composition for the selected investor profile.

**How to input values:**

| Input | Type | Options |
|---|---|---|
| Profile | Radio button | 🚀 Risk-Taking / 🛡️ Prudent / ⚖️ Equal-Weight |

**Steps:**
1. Select a profile using the radio buttons at the top
2. Results load automatically — no button click needed

**How to read the output:**

```
Metric cards   : Expected annual return, volatility, Sharpe ratio,
                 number of holdings
Pie chart      : Portfolio allocation by ticker (weight %)
Bar chart      : Sector allocation breakdown
Holdings table : Full position list with weight, risk score,
                 risk flag (color-coded: green=LOW, orange=MEDIUM,
                 red=HIGH)
```

**Example input/output:**

```
Input:
  Profile: ⚖️ Equal-Weight (10 stocks)

Output:
  Expected Return : 53.93%
  Volatility      : 24.77%
  Sharpe Ratio    : 1.3969
  Holdings        : GAS(10%), TCB(10%), HDB(10%), VIC(10%),
                    VND(10%), MSN(10%), HPG(10%), VHM(10%)*,
                    MWG(10%), SAB(10%)
  (* VHM excluded from Prudent but included in Equal-Weight)
```

---

#### Page 5 — Risk Analysis

Visualizes the Task 4.2 risk scores for all 27 tickers.

**How to use:** Navigate to Risk Analysis. No input required —
all data loads automatically from the API.

**How to read the output:**

```
Summary metrics  : Count of LOW / MEDIUM / HIGH / EXCLUDED tickers
Horizontal bars  : Risk score per ticker (0–10)
                   Green  = LOW risk (score < 5)
                   Yellow = MEDIUM risk (5–7)
                   Red    = HIGH risk (7–8.5)
                   Purple = EXCLUDED (> 8.5)
  Dashed lines at 5.0 (Prudent threshold)
               and 7.0 (Risk-Taking threshold)
Heatmap          : Five risk components per ticker
                   Red cells = high risk in that component
                   Green cells = low risk
Full table       : Raw scores for all 27 tickers
```

**Example output:**
```
EXCLUDED : PDR  (composite = 8.83, stress test penalty +1.5)
HIGH     : DXG, HSG, VHM
LOW      : GAS (4.02), SAB (3.53), HDB (4.44)
```

---

### End-to-End Workflow Example

The following demonstrates a complete user workflow from
opening the dashboard to making an investment decision:

```
Step 1: Open http://localhost:8501
        → Overview page loads, API Connected ✓

Step 2: Navigate to Signal Scanner
        → Set threshold = 0.55
        → Click "Scan All"
        → Identify tickers with BUY signals

Step 3: Navigate to Price Prediction
        → Select a BUY-signaled ticker (e.g. HDB)
        → Click "Run Prediction"
        → Confirm direction = UP, confidence > 55%

Step 4: Navigate to Risk Analysis
        → Verify HDB risk_flag = LOW (score 4.44)
        → Confirms low downside risk

Step 5: Navigate to Portfolio
        → Select Equal-Weight profile
        → Verify HDB is included at 10% weight
        → Sharpe = 1.3969 ✓

Decision: HDB shows BUY signal + UP prediction
          + LOW risk + included in best-performing portfolio
          → High-confidence investment candidate
```

---

### Technical Notes

```
Framework  : Streamlit 1.x
Charts     : Plotly (interactive, hover-enabled)
Navigation : streamlit-option-menu (Bootstrap icons)
Styling    : Custom CSS injected via st.markdown
             Font: Inter (Google Fonts)
             Primary color: #5ba8ff
             Accent: #fffa93
             Background: #F8F9FA
API calls  : requests library, timeout=30s
State      : Stateless — no session data stored
Port       : 8501 (configurable via --server.port)
```
