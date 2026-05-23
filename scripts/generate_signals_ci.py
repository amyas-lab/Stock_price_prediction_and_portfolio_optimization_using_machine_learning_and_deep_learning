"""
Standalone signal generation for GitHub Actions CI.
Runs MTL T4 classification head via live yfinance data,
updates data/vietnam/task3_signals_production.csv.
No MongoDB dependency.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from src.api.feature_pipeline import build_live_features

TICKERS = [
    'FPT', 'VCB', 'VHM', 'VNM', 'HPG', 'VIC', 'TCB',
    'MSN', 'MWG', 'VND', 'BID', 'CTG', 'MBB', 'ACB',
    'HDB', 'TPB', 'SHB', 'PDR', 'KDH', 'DXG', 'GAS',
    'HSG', 'PNJ', 'SAB', 'CMG', 'ELC', 'SGT',
]

MODEL_PATH   = ROOT / 'models' / 'mtl_t4_final.keras'
SCALER_PATH  = ROOT / 'models' / 'task4_feature_scaler.pkl'
SIGNALS_PATH = ROOT / 'data' / 'vietnam' / 'task3_signals_production.csv'
THRESHOLD    = 0.55

print("Loading model and scaler...")
model  = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
scaler = joblib.load(SCALER_PATH)
print(f"Model input shape: {model.input_shape}")

rows      = []
succeeded = 0
failed    = []

for ticker in TICKERS:
    try:
        X, _, sig_date = build_live_features(ticker, scaler)
        _, cls_pred = model.predict(X, verbose=0)
        p_sell = float(cls_pred[0, 0])
        p_hold = float(cls_pred[0, 1])
        p_buy  = float(cls_pred[0, 2])

        signal = (
            'BUY'  if p_buy  >= THRESHOLD else
            'SELL' if p_sell >= THRESHOLD else
            'HOLD'
        )
        rows.append({
            "ticker": ticker,
            "date"  : sig_date,
            "p_buy" : round(p_buy,  4),
            "p_sell": round(p_sell, 4),
            "p_hold": round(p_hold, 4),
        })
        succeeded += 1
        print(f"  ✓ {ticker}: {signal} (p_buy={p_buy:.3f}, p_sell={p_sell:.3f})")

    except Exception as e:
        failed.append(ticker)
        print(f"  ✗ {ticker}: {e}")

print(f"\nGenerated: {succeeded}/{len(TICKERS)} tickers")

if rows:
    df_new = pd.DataFrame(rows)
    today  = df_new['date'].iloc[0][:10]

    if SIGNALS_PATH.exists():
        df_old = pd.read_csv(SIGNALS_PATH)
        df_old = df_old[df_old['date'].astype(str).str[:10] != today]
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new

    df_out.to_csv(SIGNALS_PATH, index=False)
    print(f"Updated {SIGNALS_PATH.name}  ({len(df_out)} rows total)")

if failed:
    print(f"Failed tickers: {failed}")
    sys.exit(1)
