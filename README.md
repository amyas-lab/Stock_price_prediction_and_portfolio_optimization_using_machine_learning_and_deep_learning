# CS313 Final Project - Time-Series Deep Learning for Stock Markets

This repository is my **in-progress** final project for **CS313 Deep Learning for Artificial Intelligence (Spring 2026)**.

The project focuses on building deep learning systems for:
- stock price forecasting,
- trading signal identification,
- portfolio and risk management,
- and (optional) industry-style deployment workflows.

## Project Goal

Social and economic conditions change continuously.  
This project applies time-series forecasting and predictive analytics to help answer practical investment questions:
- What prices are likely to happen next?
- When are potential buy/sell opportunities?
- Which stocks should be selected or excluded in a portfolio?
- How should allocation change for different risk profiles?

## Scope and Required Tasks

The course project defines 6 tasks (Tasks 1-4 core, Task 5 extra credit, Task 6 report/repository quality).  
My implementation plan follows that structure.

### Task 1 (15%) - Nasdaq Stock Price Prediction

- Task 1.1: Multivariate input (Open, High, Low, Close, Adj Close, Volume)
- Task 1.2: Predict the **k-th future day** (for example day 3, day 7)
- Task 1.3: Predict **k consecutive future days** (multi-step horizon)

### Task 2 (15%) - Vietnam Stock Price Prediction

- Task 2.1: Multivariate input (Open, High, Low, Close, Volume)
- Task 2.2: Predict the **k-th future day**
- Task 2.3: Predict **k consecutive future days**
- Evaluate whether extra Vietnam data (dividend history, industry analysis, financial ratios) improves performance

### Task 3 (20%) - Trading Signal Identification (Vietnam Market)

- Task 3.1: Buying signal model (score/probability)
- Task 3.2: Selling signal model (score/probability)
- Consider feature engineering (SMA, MACD, RSI, and related indicators)

### Task 4 (30%) - Portfolio Construction and Risk Management (Vietnam Market)

- Task 4.1: Select profitable companies and estimate return potential
- Task 4.2: Build risk scoring model to exclude risky companies
- Task 4.3: Combine return and risk outputs into portfolio allocation strategy
- Compare allocations for risk-taking vs prudent investor profiles

### Task 5 (15%, Extra Credit) - Industry Deployment

- Task 5.1: Deploy model as API service (REST/gRPC or TensorFlow Serving)
- Task 5.2: Build simple SaaS/web UI for predictions
- Task 5.3: Design AI engineering workflow (ingestion, transformation, training/inference, storage)

### Task 6 (20%) - Report and Repository Quality

- 2000+ word report describing experiments, findings, and conclusions
- Complete, reproducible GitHub repository
- Clear README with setup and running instructions

## Current Progress (Work in Progress)

- [x] Initial exploratory data analysis
- [x] Data loading/preprocessing/splitting modules in `src/data`
- [x] Baseline and advanced deep learning experiments (LSTM/GRU variants) for stock prediction
- [x] Initial Nasdaq and Vietnam notebooks
- [ ] Full task-by-task ablation and justification across all required subtasks
- [ ] Complete trading signal pipeline (Task 3) with robust evaluation
- [ ] Complete portfolio/risk pipeline (Task 4) and strategy comparison
- [ ] Optional deployment workflow (Task 5)
- [ ] Final report and polished reproducibility checklist (Task 6)

## Repository Structure

- `src/data/` - data loading, preprocessing, and time-series splitting utilities
- `src/models/`, `src/training/`, `src/evaluation/` - model and evaluation code
- `notebooks/` - EDA, preprocessing, and training experiments
- `notebooks/data/` - prepared datasets and intermediate files
- `models/` - saved model artifacts
- `report/` - report materials
- `requirements.txt` - Python dependencies

## Data Sources

- **Nasdaq data**: historical OHLCV-style market data (including adjusted close where available)
- **Vietnam data**: historical OHLCV data and optional supporting datasets (dividends, financial ratios, industry analysis)
- Main collection libraries currently used in this repository:
  - `yfinance`
  - `vnstock`

## Methodology Highlights

- Deep learning is the core modeling approach (LSTM/GRU-based variants)
- Strict chronological split for train/validation/test (no random shuffle)
- Time-series-aware validation strategy (rolling/expanding window style when applicable)
- Multi-horizon forecasting setup (single future day and multi-day horizons)
- Performance tracked with regression metrics (for example MAE, MSE, RMSE), with extensions for signal and portfolio tasks

## How to Run

### 1) Setup environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run notebooks (current main workflow)

```bash
jupyter notebook
```

Recommended order in `notebooks/`:
1. `1-exploratory-data-analysis.ipynb`
2. `2-data-preprocessing.ipynb`
3. `3-nasdaq-stock-price-prediction.ipynb`
4. Other task-specific notebooks as they are completed

## Evaluation Notes

Evaluation follows course requirements:
- appropriate time-series split and validation,
- justified architecture and hyperparameter choices,
- clear interpretation of results,
- and reproducibility through code and documentation.

## Planned Deliverables

- Completed code and notebooks for Tasks 1-4
- Optional deployment artifacts for Task 5 (extra credit)
- Final report (Task 6.1)
- Clean repository structure and README (Task 6.2, Task 6.3)

## Disclaimer

This repository is for academic research and learning purposes only.  
It is **not** financial advice.