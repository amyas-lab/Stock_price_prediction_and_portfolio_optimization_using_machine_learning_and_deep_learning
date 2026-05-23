# Task 2 — Vietnam Stock Price Prediction

Multi-task learning (MTL) model for 5-day price trajectory forecasting on 27 HOSE stocks.

| Notebook | Description |
|---|---|
| `2-vietnam_stock_prediction.ipynb` | MTL architecture: shared encoder + regression head + classification head |

**Production model:** `models/mtl_t4_final.keras` — trained on 27 tickers, 25 technical features, 20-day window.  
**Features:** OHLCV + RSI, MACD, ATR, EMA-10/20/50, Bollinger Bands, VN-Index market features.  
**Accuracy:** Directional accuracy 63.64% on specialized tickers.
