# VnAlpha: A Multi-Stage Engineering Framework for Vietnamese Equity Alpha Discovery

**Course:** CS313 — Deep Learning
**Student:** Nguyễn Mai Anh
**Professor:** Huỳnh Thế Đăng
**Date:** May 11, 2026

---

## Abstract

Predicting price trajectories in the Vietnamese stock market (HOSE) presents unique challenges due to lower liquidity and asymmetric price movements. This report details the comprehensive development of VnAlpha, a quantitative pipeline that evolved from NASDAQ baselines (Task 1) to a specialized two-ticker pilot phase (Task 2) where a Multi-Task Seq2Seq architecture was perfected. This foundation enabled a multi-layer signal identification system (Task 3), a 27-ticker generalist portfolio (Task 4), and an automated engineering workflow (Task 5). Key findings highlight that trajectory-aware models significantly outperform isolated point-predictors, and that stock selection quality (Alpha) is more resilient than complex optimization in volatile emerging markets.

---

## 1. Task 1: Foundation and the "Independence Fallacy" (NASDAQ)

The initial phase utilized high-liquidity NASDAQ equities (e.g., AAPL) to establish the preprocessing pipeline. I implemented multi-feature extraction, expanding the basic OHLCV set into an 18-dimensional feature space.

**Architectural Insight:** My experiments with stacked GRUs and Direct MIMO (Multiple Input Multiple Output) models revealed a structural bottleneck I termed the "Independence Fallacy." By mapping a compressed 60-day context to 7 independent output neurons simultaneously, the model treated each day as mathematically independent of the previous one. This resulted in "mean-reverting straight lines" that failed to model day-to-day volatility. This necessitated the transition to a Seq2Seq (Encoder-Decoder) framework for Task 2, where recurrent decoding ensures each prediction is conditioned on the state of the previous day.

---

## 2. Task 2: The Two-Ticker Pilot Phase (Vietnam Market)

### 2.1 Specialized Training Scope

In Task 2, I narrowed the focus to a pilot set of 2 Vietnamese tickers (primarily FPT). The goal was not yet universal generalization, but rather to "crack the code" of HOSE's high-noise environment. I discovered that while yfinance provides adjusted prices for NASDAQ, local sources like vnstock require manual handling of non-stationary data. Consequently, I shifted to Log Returns and a RobustScaler to handle the frequent outliers characteristic of Vietnamese trading.

### 2.2 Architecture: Multi-Task Learning (MTL) vs. The Mean-Hugging Trap

Using these two tickers, I observed that standard regression models suffered from "Mean-Hugging"—predicting values near 0.0 to minimize MAE, which yielded zero utility for a trader.

I resolved this by developing the **Multi-Task Seq2Seq-Attention** model. This architecture shared a GRU encoder and Multi-Head Attention layer between two heads:

- **Regression Head:** For 5-day trajectory forecasting.
- **Classification Head:** For Directional Conviction (BUY/HOLD/SELL).

I implemented a Sign-Weighted Huber Loss (2.5× penalty for wrong directions) and a 5:1 loss ratio favoring classification. This forced the model to prioritize "Intent over Precision"—identifying the curvature of a move even if the exact VND magnitude had a vertical gap.

### 2.3 Finding: Trajectory Slicing

A major conclusion from the FPT pilot was that isolated n-th day prediction (Task 2.2) is a sub-problem of trajectory forecasting (Task 2.3). By slicing the t+3 value from a fully contextualized 5-day path, I achieved an **Active Directional Accuracy of 58.57%**, far superior to isolated LSTM/XGBoost baselines (~45%).

---

## 3. Task 3: Signal Identification and Conviction Gating

Task 3 bridged Task 2's deep learning prior into a concrete decision engine. I engineered a 43-feature matrix for an XGBoost Classifier (XGB_T3).

**Multi-Layer Signal Layers:**

- **Deep Learning Prior:** Injected P(BUY) and P(SELL) from the Task 2 MTL model.
- **Market Structure:** S/R zones via dynamic K-Means clustering.
- **Momentum:** EMA Golden/Death crosses.

**Key Result:** `mtl_p_up` and `mtl_p_down` ranked as the top two most important features in XGBoost, validating that the sequence-aware context from Task 2 was the primary driver of signal conviction. At a threshold of 0.55, the model achieved a **BUY Precision of 58.24%** (a +22% lift over random).

---

## 4. Task 4: Scaling to the 27-Ticker Generalist Portfolio

### 4.1 Five-Factor Profitability Scoring

With the architecture proven on the pilot tickers, I expanded the dataset to 27 tickers across all HOSE sectors. I designed a weighted profitability score combining:

| Factor | Weight |
|---|---|
| MTL Trajectory | 30% |
| Signal Conviction | 25% |
| Technical Momentum | 20% |
| Rolling Sharpe | 15% |
| ADX Trend Strength | 10% |

### 4.2 The "VIC" Anomaly and Optimization Failure

I implemented a Weighted Risk Score and attempted Mean-Variance Optimization (MVO). However, backtesting revealed that the **Equal-Weight Top 10 Portfolio (+53.9%)** crushed the **MVO Risk-Taking Portfolio (+35.7%)**. This was due to the VIC outlier (+678% rally); MVO's risk-aversion assigned it only a 2% weight, while Equal-Weighting captured its alpha more effectively. This proved that in HOSE, **Stock Selection (Alpha)** is vastly more important than complex weighting (Beta).

---

## 5. Task 5: Automation and Engineering Workflow

The final phase realized the project as a production-grade service:

- **FastAPI Backend:** A robust API serving real-time predictions and signal scanning.
- **Streamlit UI:** A high-end FinTech dashboard using custom CSS and `#fffa93` / `#5ba8ff` branding.
- **Orchestration:** Apache Airflow 3.0 was utilized to schedule daily market data ingestion into a MongoDB document store, ensuring the "Generalist" model is updated with every closing bell.

---

## Conclusion and Reflection

This report represents a deep dive into the "sufferance" of quantitative engineering. By starting with the FPT pilot in Task 2, I was able to solve the fundamental problem of directional conviction before scaling to a generalist system. While I spent the final 96 hours "cooking" these tasks with an aching back, the effort yielded a system with a statistically significant **Information Coefficient of 0.56**. VnAlpha is not just a model; it is an integrated engineering framework capable of extracting structural alpha from the HOSE market.

---

## Final Performance Dashboard

| Metric | Task 2 Pilot (FPT) | Task 4 Generalist (27 Tickers) |
|---|---|---|
| Directional Accuracy | 58.57% | 63.64% |
| Selection Alpha | N/A | +98.90% over excluded |
| Best Performer Hit | Correct (FPT) | Correct (VIC +678% at Rank #4) |

---

*VnAlpha System | 2026 — Validated on HOSE. Orchestrated by Airflow. Developed with intensity.*