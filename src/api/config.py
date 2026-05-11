# ── src/api/config.py ────────────────────────────────────────
"""
Central configuration for the API.
All paths relative to project root.
"""

from pathlib import Path
import os

# Project root
ROOT = Path(__file__).parent.parent.parent

# Model paths
MODELS_DIR = ROOT / 'models'

MODEL_PATHS = {
    'mtl_production' : MODELS_DIR / 'production_mtl_v1.keras',
    'mtl_t4'         : MODELS_DIR / 'mtl_t4_final.keras',
    'xgb_signal_t3'  : MODELS_DIR / 'task3_xgb_final_signal.pkl',
    'xgb_signal_t4'  : MODELS_DIR / 'xgb_t4_signal.pkl',
    'feature_scaler' : MODELS_DIR / 'generalist_feature_scaler.pkl',
    'target_scaler'  : MODELS_DIR / 'generalist_target_multi_scaler.pkl',
    'task4_scaler'   : MODELS_DIR / 'task4_feature_scaler.pkl',
    'ticker_encoder' : MODELS_DIR / 'ticker_label_encoder.pkl',
    'sr_scaler'      : MODELS_DIR / 'task3_new_features_scaler.pkl',
}

# Data paths
DATA_DIR = ROOT / 'notebooks' / 'data' / 'vietnam'

DATA_PATHS = {
    'profitability'  : DATA_DIR / 'task4_profitability_scores.csv',
    'risk_scores'    : DATA_DIR / 'task4_risk_scores.csv',
    'signals'        : DATA_DIR / 'task3_signals_production.csv',
    'portfolio'      : DATA_DIR / 'task4_portfolio_composition_all.csv',
    'master_features': DATA_DIR / 'task4_master_features.csv',
}

# API config
API_CONFIG = {
    'host'           : '0.0.0.0',
    'port'           : 8000,
    'title'          : 'Vietnam Stock Prediction API',
    'description'    : (
        'REST API for Vietnamese stock market prediction, '
        'trading signal identification, and portfolio management.'
    ),
    'version'        : '1.0.0',
    'window_size'    : 20,
    'k_days'         : 5,
    'buy_threshold'  : 0.55,
    'sell_threshold' : 0.55,
    'rf_annual'      : 0.045,
}

# Supported tickers
SUPPORTED_TICKERS_T3 = [
    'FPT', 'VCB', 'VHM', 'VNM', 'HPG',
    'VIC', 'TCB', 'MSN', 'MWG', 'VND'
]

SUPPORTED_TICKERS_T4 = [
    'FPT', 'VCB', 'VHM', 'VNM', 'HPG', 'VIC', 'TCB',
    'MSN', 'MWG', 'VND', 'BID', 'CTG', 'MBB', 'ACB',
    'HDB', 'TPB', 'SHB', 'PDR', 'KDH', 'DXG', 'GAS',
    'HSG', 'PNJ', 'SAB', 'CMG', 'ELC', 'SGT'
]