# Data

Shared datasets used across notebooks and the API. All large files (>1 MB) are tracked via Git LFS.

## Structure

```
data/
├── nasdaq/
│   └── csv/
│       ├── tech_nasdaq_stock_data.csv           # Raw OHLCV — 9 US tech tickers (2006–2026)
│       └── tech_nasdaq_stock_data_features.csv  # Engineered features (produced by notebooks/task1_nasdaq/2-data-preprocessing.ipynb)
│
├── vietnam/
│   ├── FPT_ohlcv.csv / VCB_ohlcv.csv / … (per-ticker OHLCV from vnstock)
│   ├── vnindex_ohlcv.csv                        # VN-Index benchmark
│   │
│   ├── task4_master_features.csv                # PROD — 27-ticker feature dataset (25 features × 20-day window)
│   ├── task3_signals_production.csv             # PROD — XGBoost signal outputs (BUY/SELL/HOLD + conviction)
│   ├── task4_portfolio_composition_all.csv      # PROD — 3 portfolio profiles (Risk-Taking, Prudent, Equal-Weight)
│   ├── task4_profitability_scores.csv           # PROD — 5-factor composite profitability ranking
│   ├── task4_risk_scores.csv                    # PROD — 5-component risk scores + exclusion tiers
│   ├── task4_backtest_summary.csv               # Results — walk-forward backtest performance table
│   ├── task4_portfolio_composition.csv          # Results — individual portfolio weight tables
│   │
│   ├── news_all.csv                             # Scraped — merged news for all 10 tickers
│   ├── news_raw.csv                             # Scraped — raw crawl output (pre-filter)
│   ├── news_FPT.csv / news_HPG.csv / ...        # Scraped — per-ticker filtered news (10 files)
│   ├── final_sentiment_features_v1.csv          # Sentiment features (PhoBERT + lexicon hybrid)
│   │
│   ├── finance_indicators.csv                   # Fundamental data (P/E, ROE, EPS) from vnstock
│   ├── dividend_history.csv                     # Dividend payment history
│   ├── market_cap_history.csv                   # Market cap time series
│   └── owner_capital_history.csv                # Owner equity history
│
└── sentiment/
    └── raw/                                     # Raw scraped sentiment per ticker (26 tickers, ACB_raw.csv … VRE_raw.csv)
```

## PROD files (used by the API)

These five files are loaded by `src/api/config.py` at startup:

| File | API endpoint |
|---|---|
| `task4_master_features.csv` | `/predict/price` — Branch 1 pre-computed features |
| `task3_signals_production.csv` | `/portfolio` — signal overlay |
| `task4_portfolio_composition_all.csv` | `/portfolio` — weight allocation |
| `task4_profitability_scores.csv` | `/portfolio` — profitability ranking |
| `task4_risk_scores.csv` | `/portfolio` — risk classification |

## Regenerating intermediate data

Pipeline intermediate files (feature matrices, XGBoost training sets) are not committed — they are
produced by running the notebooks in order:

1. `notebooks/task2_vietnam_price/` → `task4_master_features.csv`
2. `notebooks/task3_trading_signals/` → `task3_signals_production.csv`
3. `notebooks/task4_portfolio/` → portfolio + scoring CSVs
