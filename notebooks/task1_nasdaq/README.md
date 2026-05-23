# Task 1 — Nasdaq Price Prediction

Multi-stock price forecasting on US tech stocks using LSTM, GRU, and Seq2Seq models.

| Notebook | Description |
|---|---|
| `1-exploratory-data-analysis.ipynb` | EDA — price distributions, correlation, stationarity tests |
| `2-data-preprocessing.ipynb` | Feature engineering, normalization, train/val/test split |
| `3-nasdaq-stock-price-prediction.ipynb` | Model training: LSTM baseline → GRU with attention → Seq2Seq |

**Key results:** GRU with attention achieved best RMSE across AAPL, MSFT, NVDA, GOOGL, AMZN, META, NFLX, MU, QCOM.
