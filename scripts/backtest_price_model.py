"""
Walk-forward backtest of MTL T4 price-prediction model.

For every ticker × date in task4_master_features.csv:
  - Feed the previous 20 trading days into the model
  - Record 5-day predicted log-returns + classification probabilities
  - Compare against actual forward_return_5d (ground truth)

Outputs
-------
data/vietnam/task4_price_backtest.csv   — per-ticker per-date predictions
data/vietnam/task4_equity_curve.csv     — daily strategy vs VNI benchmark
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# ── Paths ─────────────────────────────────────────────────────
MODEL_PATH   = ROOT / "models" / "mtl_t4_final.keras"
SCALER_PATH  = ROOT / "models" / "task4_feature_scaler.pkl"
DATA_PATH    = ROOT / "data" / "vietnam" / "task4_master_features.csv"
OUT_TRADES   = ROOT / "data" / "vietnam" / "task4_price_backtest.csv"
OUT_EQUITY   = ROOT / "data" / "vietnam" / "task4_equity_curve.csv"

FEATURE_COLS = [
    "volume", "log_return", "rsi",
    "macd", "macd_signal", "macd_hist",
    "atr", "ema_10", "ema_20", "ema_50",
    "bb_upper", "bb_middle", "bb_lower",
    "vni_log_return", "vni_ema_10", "vni_ema_20", "vni_ema_50",
    "vni_rsi", "vni_macd", "vni_macd_signal", "vni_macd_hist",
    "vni_atr", "vni_bb_middle", "vni_bb_upper", "vni_bb_lower",
]

N_WINDOW  = 20
THRESHOLD = 0.55

# ── Load model & scaler ───────────────────────────────────────
print("Loading model …")
model  = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
scaler = joblib.load(SCALER_PATH)
print(f"  input shape : {model.input_shape}")

# ── Load data ─────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
print(f"  {len(df):,} rows · {df['ticker'].nunique()} tickers · "
      f"{df['date'].min().date()} → {df['date'].max().date()}")

# ── Walk-forward per ticker ───────────────────────────────────
records = []

for ticker in sorted(df["ticker"].unique()):
    df_t = df[df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    n    = len(df_t)

    if n < N_WINDOW + 5:
        print(f"  ✗ {ticker}: only {n} rows, skipping")
        continue

    # Build all input windows at once → single model.predict() call
    raw_vals = df_t[FEATURE_COLS].values          # (n, 25)

    windows, valid_idx = [], []
    for i in range(N_WINDOW, n):
        chunk = raw_vals[i - N_WINDOW : i]
        if np.isnan(chunk).any():
            continue
        windows.append(scaler.transform(chunk))
        valid_idx.append(i)

    if not windows:
        continue

    X_batch  = np.stack(windows)                  # (m, 20, 25)
    reg_all, cls_all = model.predict(X_batch, verbose=0)

    for k, i in enumerate(valid_idx):
        pred_returns = reg_all[k]                  # shape (5,)
        p_sell = float(cls_all[k, 0])
        p_hold = float(cls_all[k, 1])
        p_buy  = float(cls_all[k, 2])

        signal = (
            "BUY"  if p_buy  >= THRESHOLD else
            "SELL" if p_sell >= THRESHOLD else
            "HOLD"
        )

        # Ground truth on the same row index i
        actual_1d = float(df_t["log_return"].iat[i])
        actual_5d = float(df_t["forward_return_5d"].iat[i]) \
                    if "forward_return_5d" in df_t.columns else np.nan

        pred_1d = float(pred_returns[0])
        pred_5d = float(np.sum(pred_returns))

        dir_1d = int((pred_1d > 0) == (actual_1d > 0))
        dir_5d = int((pred_5d > 0) == (actual_5d > 0)) \
                 if not np.isnan(actual_5d) else None

        records.append({
            "ticker"        : ticker,
            "date"          : str(df_t["date"].iat[i].date()),
            "close"         : round(float(df_t["close"].iat[i]), 2),
            "signal"        : signal,
            "p_buy"         : round(p_buy,  4),
            "p_hold"        : round(p_hold, 4),
            "p_sell"        : round(p_sell, 4),
            "pred_return_1d": round(pred_1d, 6),
            "pred_return_5d": round(pred_5d, 6),
            "actual_return_1d": round(actual_1d, 6),
            "actual_return_5d": round(actual_5d, 6) if not np.isnan(actual_5d) else None,
            "dir_correct_1d": dir_1d,
            "dir_correct_5d": dir_5d,
            "mae_1d"        : round(abs(pred_1d - actual_1d), 6),
            "mae_5d"        : round(abs(pred_5d - actual_5d), 6) if not np.isnan(actual_5d) else None,
        })

    print(f"  ✓ {ticker}: {len(valid_idx)} predictions")

bt = pd.DataFrame(records)
bt["date"] = pd.to_datetime(bt["date"])

# ── Summary metrics ───────────────────────────────────────────
print("\n── Backtest Summary ──────────────────────────────────────")
print(f"  Total predictions : {len(bt):,}")
print(f"  DA-1d (overall)   : {bt['dir_correct_1d'].mean():.3f}")
print(f"  DA-5d (overall)   : {bt['dir_correct_5d'].mean():.3f}")
print(f"  MAE-1d            : {bt['mae_1d'].mean():.6f}")
print(f"  MAE-5d            : {bt['mae_5d'].mean():.6f}")
print(f"  BUY signals       : {(bt['signal']=='BUY').sum():,}  "
      f"({(bt['signal']=='BUY').mean():.1%})")

# ── Equity curve ──────────────────────────────────────────────
# Strategy: on each date, equal-weight ALL tickers with BUY signal.
# P&L = average actual next-day log_return of those tickers.
# Benchmark: equal-weight all tickers (buy-and-hold equivalent).

bt_buy = bt[bt["signal"] == "BUY"].copy()

strat_daily = (
    bt_buy.groupby("date")["actual_return_1d"]
    .mean()
    .rename("strat_return")
)
bench_daily = (
    bt.groupby("date")["actual_return_1d"]
    .mean()
    .rename("bench_return")
)

equity = pd.concat([strat_daily, bench_daily], axis=1).fillna(0).sort_index()
equity["strat_cum"]  = (1 + equity["strat_return"]).cumprod() - 1
equity["bench_cum"]  = (1 + equity["bench_return"]).cumprod() - 1
equity = equity.reset_index()
equity["date"] = equity["date"].dt.strftime("%Y-%m-%d")

# Sharpe (annualised, 252 trading days)
def sharpe(returns, ann=252):
    r = returns.dropna()
    return (r.mean() / r.std() * np.sqrt(ann)).round(3) if r.std() > 0 else 0.0

strat_sharpe = sharpe(equity["strat_return"])
bench_sharpe = sharpe(equity["bench_return"])
print(f"\n  Sharpe (strategy) : {strat_sharpe}")
print(f"  Sharpe (bench)    : {bench_sharpe}")
print(f"  Cum return strat  : {equity['strat_cum'].iloc[-1]:.3f}")
print(f"  Cum return bench  : {equity['bench_cum'].iloc[-1]:.3f}")

# ── Save outputs ──────────────────────────────────────────────
bt["date"] = bt["date"].dt.strftime("%Y-%m-%d")
bt.to_csv(OUT_TRADES, index=False)
equity.to_csv(OUT_EQUITY, index=False)

print(f"\nSaved:")
print(f"  {OUT_TRADES.name}  ({len(bt):,} rows)")
print(f"  {OUT_EQUITY.name}  ({len(equity):,} rows)")
