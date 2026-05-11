# ── src/api/main.py ───────────────────────────────────────────
"""
FastAPI application — Vietnam Stock Prediction API
Endpoints:
  GET  /health                        → API health check
  POST /predict/price                 → MTL price prediction
  POST /predict/signal                → XGBoost trading signal
  GET  /portfolio/{profile}           → Portfolio composition
  GET  /portfolio/scores/profitability→ Task 4.1 scores
  GET  /portfolio/scores/risk         → Task 4.2 scores
"""

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib         import asynccontextmanager
from datetime           import datetime
from pathlib            import Path
import warnings
warnings.filterwarnings('ignore')

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.config import (
    MODEL_PATHS, DATA_PATHS, API_CONFIG,
    SUPPORTED_TICKERS_T3, SUPPORTED_TICKERS_T4
)
from src.api.models import (
    PredictionRequest, SignalRequest, PortfolioRequest,
    PricePredictionResponse, SignalResponse,
    PortfolioResponse, PortfolioStock,
    ProfitabilityResponse, ProfitabilityScore,
    RiskResponse, RiskScore, HealthResponse
)

# ── Global model store ────────────────────────────────────────
MODELS = {}
DATA   = {}


# ── Startup / Shutdown ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    print("🚀 Loading models...")

    # Load Keras models
    try:
        MODELS['mtl_t3'] = tf.keras.models.load_model(
            str(MODEL_PATHS['mtl_production']),
            compile=False
        )
        print(f"  ✓ MTL T3 model loaded")
    except Exception as e:
        print(f"  ✗ MTL T3 failed: {e}")
        MODELS['mtl_t3'] = None

    try:
        MODELS['mtl_t4'] = tf.keras.models.load_model(
            str(MODEL_PATHS['mtl_t4']),
            compile=False
        )
        print(f"  ✓ MTL T4 model loaded")
    except Exception as e:
        print(f"  ✗ MTL T4 failed: {e}")
        MODELS['mtl_t4'] = None

    # Load sklearn/xgboost models
    for key in ['xgb_signal_t3', 'xgb_signal_t4',
                'feature_scaler', 'target_scaler',
                'task4_scaler', 'ticker_encoder',
                'sr_scaler']:
        try:
            MODELS[key] = joblib.load(MODEL_PATHS[key])
            print(f"  ✓ {key} loaded")
        except Exception as e:
            print(f"  ✗ {key} failed: {e}")
            MODELS[key] = None

    # Load precomputed data
    for key, path in DATA_PATHS.items():
        try:
            DATA[key] = pd.read_csv(path)
            if 'date' in DATA[key].columns:
                DATA[key]['date'] = pd.to_datetime(
                    DATA[key]['date']
                )
            print(f"  ✓ {key} data loaded "
                  f"({len(DATA[key]):,} rows)")
        except Exception as e:
            print(f"  ✗ {key} data failed: {e}")
            DATA[key] = None

    print("✓ API ready!")
    yield

    # Cleanup
    MODELS.clear()
    DATA.clear()
    print("API shutdown complete.")


# ── App initialization ────────────────────────────────────────
app = FastAPI(
    title       = API_CONFIG['title'],
    description = API_CONFIG['description'],
    version     = API_CONFIG['version'],
    lifespan    = lifespan
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Helper Functions ──────────────────────────────────────────
def get_latest_features(ticker: str,
                          n_days: int = 20) -> np.ndarray:
    """
    Extract latest n_days of features for a ticker
    from precomputed master features.
    Returns shape (1, n_days, n_features).
    """
    if DATA['master_features'] is None:
        raise HTTPException(
            status_code=503,
            detail="Master features data not available"
        )

    df = DATA['master_features']
    ticker_df = df[
        df['ticker'] == ticker
    ].sort_values('date').tail(n_days + 10)

    if len(ticker_df) < n_days:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient data for {ticker}. "
                   f"Need {n_days} days, got {len(ticker_df)}"
        )

    # Feature columns (exclude non-feature cols)
    exclude_cols = [
        'date', 'ticker', 'open', 'high', 'low', 'close',
        'forward_return_5d', 'log_return_1d',
        'target_class_t4', 'sector'
    ]
    feature_cols = [
        c for c in ticker_df.columns
        if c not in exclude_cols
    ]

    feat_arr = ticker_df[feature_cols].values[-n_days:]
    return feat_arr.reshape(1, n_days, len(feature_cols))


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health",
         response_model=HealthResponse,
         tags=["System"])
async def health_check():
    """API health check — shows loaded models."""
    return HealthResponse(
        status  = "healthy",
        models_loaded = {
            k: v is not None for k, v in MODELS.items()
        },
        supported_tickers_t3 = SUPPORTED_TICKERS_T3,
        supported_tickers_t4 = SUPPORTED_TICKERS_T4,
        version = API_CONFIG['version']
    )


@app.post("/predict/price",
          response_model=PricePredictionResponse,
          tags=["Prediction"])
async def predict_price(request: PredictionRequest):
    """
    Predict 5-day price trajectory for a ticker.
    Uses MTL Seq2Seq model from Task 2/4.
    """
    ticker = request.ticker.upper()

    # Validate ticker
    if ticker not in SUPPORTED_TICKERS_T4:
        raise HTTPException(
            status_code=400,
            detail=f"Ticker {ticker} not supported. "
                   f"Supported: {SUPPORTED_TICKERS_T4}"
        )

    # Select model
    model = MODELS.get('mtl_t4') or MODELS.get('mtl_t3')
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model not available"
        )

    # Get features
    X = get_latest_features(ticker, request.n_days_back)

    # Predict
    reg_pred, cls_pred = model.predict(X, verbose=0)

    # Get current price
    df       = DATA['master_features']
    last_row = df[df['ticker'] == ticker].sort_values(
        'date'
    ).iloc[-1]
    current_price = float(last_row['close']) 
    if current_price < 1000:
        current_price = current_price * 1000
    last_date     = str(last_row['date'].date())

    # Decode predictions
    pred_returns = reg_pred[0].tolist() \
                   if len(reg_pred.shape) > 1 \
                   else [float(reg_pred[0])]

    # Predicted prices from log returns
    pred_prices = []
    price = current_price
    for ret in pred_returns:
        price = price * np.exp(ret)
        pred_prices.append(round(float(price), 2))

    # Direction from classification head
    p_buy  = float(cls_pred[0, 2])
    p_sell = float(cls_pred[0, 0])
    p_hold = float(cls_pred[0, 1])

    if p_buy > p_sell and p_buy > p_hold:
        direction  = "UP"
        confidence = p_buy
    elif p_sell > p_buy and p_sell > p_hold:
        direction  = "DOWN"
        confidence = p_sell
    else:
        direction  = "NEUTRAL"
        confidence = p_hold

    return PricePredictionResponse(
        ticker           = ticker,
        prediction_date  = last_date,
        current_price    = current_price,
        predicted_returns= [round(r, 6) for r in pred_returns],
        predicted_prices = pred_prices,
        direction        = direction,
        confidence       = round(confidence, 4),
        model_used       = "MTL_Seq2Seq_GRU_Attention"
    )


@app.post("/predict/signal",
          response_model=SignalResponse,
          tags=["Signal"])
async def predict_signal(request: SignalRequest):
    """
    Generate BUY/SELL/HOLD trading signal for a ticker.
    Uses XGBoost classifier from Task 3/4.
    """
    ticker    = request.ticker.upper()
    threshold = request.threshold

    # Select model and feature set
    if ticker in SUPPORTED_TICKERS_T3:
        xgb_model = MODELS.get('xgb_signal_t3')
    else:
        xgb_model = MODELS.get('xgb_signal_t4')

    if xgb_model is None:
        raise HTTPException(
            status_code=503,
            detail="Signal model not available"
        )

    # Get precomputed signals if available
    if DATA.get('signals') is not None:
        sig_df = DATA['signals']
        ticker_sig = sig_df[
            sig_df['ticker'] == ticker
        ].sort_values('date')

        if len(ticker_sig) > 0:
            last_sig   = ticker_sig.iloc[-1]
            p_buy      = float(last_sig.get('p_buy', 0.33))
            p_sell     = float(last_sig.get('p_sell', 0.33))
            p_hold     = 1 - p_buy - p_sell
            conviction = max(p_buy, p_sell)
            sig_date   = str(last_sig['date'])[:10]
        else:
            p_buy = p_sell = p_hold = 0.333
            conviction = 0.333
            sig_date = str(datetime.today().date())
    else:
        p_buy = p_sell = p_hold = 0.333
        conviction = 0.333
        sig_date = str(datetime.today().date())

    # Apply threshold
    if p_buy >= threshold:
        signal = "BUY"
        recommendation = (
            f"Strong BUY signal detected. "
            f"P(BUY)={p_buy:.2%} ≥ threshold={threshold:.2%}. "
            f"Consider entering a long position."
        )
    elif p_sell >= threshold:
        signal = "SELL"
        recommendation = (
            f"Strong SELL signal detected. "
            f"P(SELL)={p_sell:.2%} ≥ threshold={threshold:.2%}. "
            f"Consider reducing or exiting position."
        )
    else:
        signal = "HOLD"
        recommendation = (
            f"No high-conviction signal. "
            f"Max conviction={conviction:.2%} < threshold={threshold:.2%}. "
            f"Stay flat or maintain current position."
        )

    return SignalResponse(
        ticker         = ticker,
        signal_date    = sig_date,
        signal         = signal,
        p_buy          = round(p_buy,  4),
        p_sell         = round(p_sell, 4),
        p_hold         = round(p_hold, 4),
        conviction     = round(conviction, 4),
        threshold_used = threshold,
        recommendation = recommendation
    )


@app.get("/portfolio/{profile}",
         response_model=PortfolioResponse,
         tags=["Portfolio"])
async def get_portfolio(profile: str):
    """
    Get portfolio composition for a given profile.
    Profiles: risk_taking, prudent, equal_weight
    """
    valid_profiles = ['risk_taking', 'prudent', 'equal_weight']
    if profile not in valid_profiles:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid profile. Choose: {valid_profiles}"
        )

    if DATA.get('portfolio') is None:
        raise HTTPException(
            status_code=503,
            detail="Portfolio data not available"
        )

    # Map profile names
    profile_map = {
        'risk_taking' : 'Risk-Taking',
        'prudent'     : 'Prudent',
        'equal_weight': 'Equal-Weight'
    }

    port_df = DATA['portfolio']
    filtered = port_df[
        port_df['profile'] == profile_map[profile]
    ]

    if len(filtered) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for profile: {profile}"
        )

    stocks = [
        PortfolioStock(
            ticker     = row['ticker'],
            sector     = row.get('sector', 'Unknown'),
            weight     = round(float(row['weight']), 4),
            risk_score = round(float(row.get(
                'risk_score', 0
            )), 2),
            risk_flag  = str(row.get('risk_flag', 'N/A'))
        )
        for _, row in filtered.iterrows()
    ]

    # Approximate performance metrics
    perf_map = {
        'Risk-Taking' : (0.3573, 0.2952, 0.8596),
        'Prudent'     : (0.1644, 0.2480, 0.4550),
        'Equal-Weight': (0.5393, 0.2477, 1.3969),
    }
    ret, vol, sr = perf_map.get(
        profile_map[profile], (0, 0, 0)
    )

    return PortfolioResponse(
        profile         = profile,
        stocks          = stocks,
        expected_return = round(ret, 4),
        expected_vol    = round(vol, 4),
        sharpe_ratio    = round(sr,  4),
        total_stocks    = len(stocks)
    )


@app.get("/portfolio/scores/profitability",
         response_model=ProfitabilityResponse,
         tags=["Portfolio"])
async def get_profitability_scores():
    """Get Task 4.1 profitability scores for all tickers."""
    if DATA.get('profitability') is None:
        raise HTTPException(
            status_code=503,
            detail="Profitability data not available"
        )

    df = DATA['profitability']
    scores = []
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        scores.append(ProfitabilityScore(
            rank            = rank,
            ticker          = str(row.get('ticker', '')),
            sector          = str(row.get('sector', 'Unknown')),
            mtl_score       = round(float(row.get('f1_mtl',    0)), 4),
            tech_score      = round(float(row.get('f2_tech',   0)), 4),
            signal_score    = round(float(row.get('f3_signal', 0)), 4),
            sharpe_score    = round(float(row.get('f4_sharpe', 0)), 4),
            trend_score     = round(float(row.get('f5_trend',  0)), 4),
            composite_score = round(float(row.get('composite', 0)), 4),
        ))

    return ProfitabilityResponse(
        scores          = scores,
        evaluation_date = str(datetime.today().date()),
        total_tickers   = len(scores)
    )


@app.get("/portfolio/scores/risk",
         response_model=RiskResponse,
         tags=["Portfolio"])
async def get_risk_scores():
    """Get Task 4.2 risk scores for all tickers."""
    if DATA.get('risk_scores') is None:
        raise HTTPException(
            status_code=503,
            detail="Risk scores data not available"
        )

    df = DATA['risk_scores']
    scores = [
        RiskScore(
            ticker           = str(row.get('ticker', '')),
            sector           = str(row.get('sector', 'Unknown')),
            volatility_risk  = round(float(row.get('r1_vol',      0)), 2),
            sell_risk        = round(float(row.get('r2_sell',     0)), 2),
            drawdown_risk    = round(float(row.get('r3_drawdown', 0)), 2),
            correlation_risk = round(float(row.get('r4_corr',     0)), 2),
            reversal_risk    = round(float(row.get('r5_reversal', 0)), 2),
            composite_risk   = round(float(row.get(
                'final_risk_score', 0
            )), 2),
            risk_flag        = str(row.get('risk_flag', 'N/A'))
        )
        for _, row in df.iterrows()
    ]

    return RiskResponse(
        scores        = scores,
        total_tickers = len(scores)
    )