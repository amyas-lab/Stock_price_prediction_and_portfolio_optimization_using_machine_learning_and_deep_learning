---
title: InvestNature API
emoji: 🌿
colorFrom: yellow
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# InvestNature API

FastAPI backend for Vietnamese stock price prediction and portfolio optimization.

## Endpoints

- `GET /health` — health check
- `GET /tickers` — list supported tickers
- `POST /predict/price` — predict stock price
- `POST /predict/signal` — predict trading signal
- `GET /portfolio/{profile}` — get optimized portfolio (prudent / equal_weight / risk_taking)
- `GET /profitability-scores` — profitability scores per ticker
- `GET /risk-scores` — risk scores per ticker
