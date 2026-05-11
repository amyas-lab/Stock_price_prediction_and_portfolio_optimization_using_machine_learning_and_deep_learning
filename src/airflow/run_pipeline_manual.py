#!/usr/bin/env python3
"""
Manual pipeline runner — test without Airflow scheduler.
Usage: python src/airflow/run_pipeline_manual.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.airflow.dags.vnalpha_pipeline import (
    ingest_ohlcv,
    compute_features,
    generate_signals,
    score_profitability,
    score_risk,
    rebalance_portfolio
)
from datetime import datetime

# Mock Airflow context
ctx = {
    'ds'    : datetime.today().strftime('%Y-%m-%d'),
    'run_id': 'manual_test',
    'dag'   : type('DAG', (), {'dag_id': 'manual'})(),
    'ti'    : type('TI', (), {
        'xcom_push': lambda *a, **k: None,
        'xcom_pull': lambda *a, **k: 0
    })()
}

print("=" * 50)
print("VnAlpha Pipeline — Manual Run")
print("=" * 50)

steps = [
    ("1. Ingest OHLCV",         ingest_ohlcv),
    ("2. Compute Features",     compute_features),
    ("3. Generate Signals",     generate_signals),
    ("4. Score Profitability",  score_profitability),
    ("5. Score Risk",           score_risk),
    ("6. Rebalance Portfolio",  rebalance_portfolio),
]

for name, fn in steps:
    print(f"\n{name}...")
    try:
        result = fn(**ctx)
        print(f"  ✓ {result}")
    except Exception as e:
        print(f"  ✗ {e}")

print("\n✓ Pipeline complete!")
