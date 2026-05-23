# Task 3 — Trading Signal Identification

BUY / SELL / HOLD signal generation using XGBoost classifier on top of the MTL model's classification head.

| Notebook | Description |
|---|---|
| `3-vietnam_trading_signal_identification.ipynb` | Feature engineering for signals, XGBoost training, backtesting |

**Production model:** `models/task3_xgb_final_signal.pkl` + `models/xgb_t4_signal.pkl`  
**Signal logic:** P(BUY) or P(SELL) ≥ 0.55 threshold → conviction signal; else HOLD.  
**Backtesting:** Evaluated on 10 core tickers (FPT, VCB, VHM, VNM, HPG, VIC, TCB, MSN, MWG, VND).
