# ── src/api/main.py ───────────────────────────────────────────
"""
FastAPI application — Vietnam Stock Prediction API

Endpoints:
  GET  /health                         → API health check
  GET  /tickers                        → List supported tickers
  POST /predict/price                  → MTL price prediction
                                         Branch 1: pre-computed (27 known, <1s)
                                         Branch 2: live yfinance (any ticker, ~3-5s)
  POST /predict/signal                 → Trading signal
                                         Branch 1: pre-computed CSV
                                         Branch 2: MTL classification head
  GET  /portfolio/{profile}            → Portfolio composition
  GET  /portfolio/scores/profitability → Task 4.1 scores
  GET  /portfolio/scores/risk          → Task 4.2 scores
"""

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime   import datetime
from pathlib    import Path
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.config import (
    MODEL_PATHS, DATA_PATHS, API_CONFIG,
    SUPPORTED_TICKERS_T3, SUPPORTED_TICKERS_T4, KNOWN_TICKERS,
)
from src.api.models import (
    PredictionRequest, SignalRequest, PortfolioRequest,
    PricePredictionResponse, SignalResponse,
    PortfolioResponse, PortfolioStock,
    ProfitabilityResponse, ProfitabilityScore,
    RiskResponse, RiskScore, HealthResponse,
)
from src.api.feature_pipeline import build_live_features

# ── Global stores ─────────────────────────────────────────────
MODELS: dict = {}
DATA:   dict = {}


# ── Startup / Shutdown ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Loading models and data...")

    # Keras models
    for key, path in [
        ("mtl_t3", MODEL_PATHS["mtl_production"]),
        ("mtl_t4", MODEL_PATHS["mtl_t4"]),
    ]:
        try:
            MODELS[key] = tf.keras.models.load_model(
                str(path), compile=False
            )
            print(f"  ✓ {key} loaded  {MODELS[key].input_shape}")
        except Exception as e:
            print(f"  ✗ {key} failed: {e}")
            MODELS[key] = None

    # sklearn / XGBoost artifacts
    for key in [
        "xgb_signal_t3", "xgb_signal_t4",
        "feature_scaler", "target_scaler",
        "task4_scaler", "ticker_encoder", "sr_scaler",
    ]:
        try:
            MODELS[key] = joblib.load(MODEL_PATHS[key])
            print(f"  ✓ {key} loaded")
        except Exception as e:
            print(f"  ✗ {key} failed: {e}")
            MODELS[key] = None

    # Pre-computed CSVs
    for key, path in DATA_PATHS.items():
        try:
            DATA[key] = pd.read_csv(path)
            if "date" in DATA[key].columns:
                DATA[key]["date"] = pd.to_datetime(DATA[key]["date"])
            print(f"  ✓ {key}  ({len(DATA[key]):,} rows)")
        except Exception as e:
            print(f"  ✗ {key} failed: {e}")
            DATA[key] = None

    print("✓ API ready!")
    yield

    MODELS.clear()
    DATA.clear()
    print("API shutdown complete.")


# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title       = API_CONFIG["title"],
    description = API_CONFIG["description"],
    version     = API_CONFIG["version"],
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Shared helpers ────────────────────────────────────────────

def _get_precomputed_features(ticker: str,
                               n_days: int = 20) -> np.ndarray:
    """Return feature array (1, n_days, 25) from pre-computed master CSV."""
    if DATA["master_features"] is None:
        raise HTTPException(503, "Master features data not available")

    df   = DATA["master_features"]
    rows = df[df["ticker"] == ticker].sort_values("date")

    if len(rows) < n_days:
        raise HTTPException(
            400,
            f"Insufficient pre-computed data for {ticker}: "
            f"need {n_days} rows, have {len(rows)}",
        )

    exclude   = {
        "date", "ticker", "open", "high", "low", "close",
        "forward_return_5d", "log_return_1d",
        "target_class_t4", "sector",
    }
    feat_cols = [c for c in rows.columns if c not in exclude]
    X         = rows[feat_cols].values[-n_days:]
    return X.reshape(1, n_days, len(feat_cols))


def _decode_prediction(reg_pred, cls_pred, current_price: float, target_scaler=None):
    """Unpack MTL model output into returns, prices, direction, confidence."""
    # Inverse-scale regression output if model was trained on scaled targets
    if target_scaler is not None:
        try:
            reg_pred = target_scaler.inverse_transform(reg_pred)
        except Exception as e:
            print(f"  ⚠ target_scaler.inverse_transform failed ({e}), using raw output")

    raw          = reg_pred[0]
    pred_returns = raw.tolist() if hasattr(raw, "tolist") else [float(raw)]

    pred_prices: list = []
    price = current_price
    for r in pred_returns:
        price = price * np.exp(float(r))
        pred_prices.append(round(price, 2))

    # class order: [0]=SELL  [1]=HOLD  [2]=BUY
    p_sell = float(cls_pred[0, 0])
    p_hold = float(cls_pred[0, 1])
    p_buy  = float(cls_pred[0, 2])

    if p_buy > p_sell and p_buy > p_hold:
        direction, confidence = "UP",      p_buy
    elif p_sell > p_buy and p_sell > p_hold:
        direction, confidence = "DOWN",    p_sell
    else:
        direction, confidence = "NEUTRAL", p_hold

    return (
        [round(r, 6) for r in pred_returns],
        pred_prices,
        direction,
        round(confidence, 4),
        round(p_buy,  4),
        round(p_sell, 4),
        round(p_hold, 4),
    )


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(
        status               = "healthy",
        models_loaded        = {k: v is not None for k, v in MODELS.items()},
        supported_tickers_t3 = SUPPORTED_TICKERS_T3,
        supported_tickers_t4 = SUPPORTED_TICKERS_T4,
        version              = API_CONFIG["version"],
    )


@app.get("/tickers", tags=["System"])
async def list_tickers():
    """Known tickers (fast path) + hint about free-text input."""
    return {
        "known_tickers": SUPPORTED_TICKERS_T4,
        "note": (
            "Any HOSE ticker (e.g. SSI, VPB) is accepted. "
            "Known tickers use pre-computed features (<1 s); "
            "unknown tickers fetch live data (~3-5 s)."
        ),
    }


@app.post("/predict/price",
          response_model=PricePredictionResponse,
          tags=["Prediction"])
async def predict_price(request: PredictionRequest):
    """
    Predict next-step price trajectory.

    Branch 1 (27 known tickers): reads pre-computed master_features → <1 s.
    Branch 2 (any other ticker): fetches live OHLCV from yfinance,
      computes 25 indicators on-the-fly, scales → ~3-5 s.
    """
    ticker   = request.ticker.upper()
    n_days   = request.n_days_back
    is_known = ticker in KNOWN_TICKERS

    model = MODELS.get("mtl_t4") or MODELS.get("mtl_t3")
    if model is None:
        raise HTTPException(503, "Prediction model not available")

    # ── Branch 1: pre-computed ────────────────────────────────
    if is_known:
        X = _get_precomputed_features(ticker, n_days)

        df       = DATA["master_features"]
        last_row = df[df["ticker"] == ticker].sort_values("date").iloc[-1]
        csv_price = float(last_row["close"])
        if csv_price < 1000:
            csv_price *= 1000
        last_date   = str(last_row["date"].date())
        data_source = "pre-computed"
        model_label = "MTL_T4_Specialized"

        # Fetch live current price so chart reflects today's market
        try:
            live_df = yf.download(
                f"{ticker}.VN", period="5d",
                auto_adjust=True, progress=False, timeout=10,
            )
            if not live_df.empty:
                if isinstance(live_df.columns, pd.MultiIndex):
                    live_df.columns = live_df.columns.get_level_values(0)
                live_close = float(live_df["Close"].iloc[-1])
                current_price = live_close if live_close >= 1000 else live_close * 1000
                last_date = str(live_df.index[-1].date())
            else:
                current_price = csv_price
        except Exception:
            current_price = csv_price

    # ── Branch 2: live fetch ──────────────────────────────────
    else:
        scaler = MODELS.get("task4_scaler")
        if scaler is None:
            raise HTTPException(503, "Feature scaler not available")
        try:
            X, raw_price, last_date = build_live_features(
                ticker, scaler, n_days
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except Exception as exc:
            raise HTTPException(
                503, f"Live data fetch failed for '{ticker}': {exc}"
            )
        current_price = raw_price if raw_price >= 1000 else raw_price * 1000
        data_source   = "live-yfinance"
        model_label   = "MTL_T4_Live"

    reg_pred, cls_pred = model.predict(X, verbose=0)
    (pred_returns, pred_prices,
     direction, confidence,
     p_buy, p_sell, p_hold) = _decode_prediction(
         reg_pred, cls_pred, current_price,
         target_scaler=MODELS.get("target_scaler"),
     )

    return PricePredictionResponse(
        ticker            = ticker,
        prediction_date   = last_date,
        current_price     = current_price,
        predicted_returns = pred_returns,
        predicted_prices  = pred_prices,
        direction         = direction,
        confidence        = confidence,
        model_used        = model_label,
        is_known_ticker   = is_known,
        data_source       = data_source,
    )


@app.post("/predict/signal",
          response_model=SignalResponse,
          tags=["Signal"])
async def predict_signal(request: SignalRequest):
    """
    BUY / SELL / HOLD trading signal.

    Branch 1 (27 known tickers): reads pre-computed signal CSV → instant.
    Branch 2 (any other ticker): runs MTL classification head on live features.
    """
    ticker    = request.ticker.upper()
    threshold = request.threshold
    is_known  = ticker in KNOWN_TICKERS

    p_buy = p_sell = p_hold = 1 / 3
    sig_date    = str(datetime.today().date())
    data_source = "pre-computed"

    # ── Branch 1 ──────────────────────────────────────────────
    if is_known and DATA.get("signals") is not None:
        sig_df     = DATA["signals"]
        ticker_sig = sig_df[sig_df["ticker"] == ticker].sort_values("date")
        if len(ticker_sig) > 0:
            last     = ticker_sig.iloc[-1]
            p_buy    = float(last.get("p_buy",  1 / 3))
            p_sell   = float(last.get("p_sell", 1 / 3))
            p_hold   = max(0.0, 1 - p_buy - p_sell)
            sig_date = str(last["date"])[:10]
        else:
            is_known = False         # no CSV row → fall through to live

    # ── Branch 2 ──────────────────────────────────────────────
    if not is_known:
        model  = MODELS.get("mtl_t4")
        scaler = MODELS.get("task4_scaler")
        if model is None or scaler is None:
            raise HTTPException(503, "Model or scaler not available")
        try:
            X, _, sig_date = build_live_features(ticker, scaler)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except Exception as exc:
            raise HTTPException(
                503, f"Live data fetch failed for '{ticker}': {exc}"
            )
        _, cls_pred = model.predict(X, verbose=0)
        p_sell = float(cls_pred[0, 0])
        p_hold = float(cls_pred[0, 1])
        p_buy  = float(cls_pred[0, 2])
        data_source = "live-yfinance"

    conviction = max(p_buy, p_sell)

    if p_buy >= threshold:
        signal = "BUY"
        recommendation = (
            f"Strong BUY signal. "
            f"P(BUY)={p_buy:.2%} ≥ threshold={threshold:.2%}. "
            "Consider entering a long position."
        )
    elif p_sell >= threshold:
        signal = "SELL"
        recommendation = (
            f"Strong SELL signal. "
            f"P(SELL)={p_sell:.2%} ≥ threshold={threshold:.2%}. "
            "Consider reducing or exiting position."
        )
    else:
        signal = "HOLD"
        recommendation = (
            f"No high-conviction signal. "
            f"Max conviction={conviction:.2%} < threshold={threshold:.2%}. "
            "Stay flat or maintain current position."
        )

    return SignalResponse(
        ticker          = ticker,
        signal_date     = sig_date,
        signal          = signal,
        p_buy           = round(p_buy,       4),
        p_sell          = round(p_sell,      4),
        p_hold          = round(p_hold,      4),
        conviction      = round(conviction,  4),
        threshold_used  = threshold,
        recommendation  = recommendation,
        is_known_ticker = ticker in KNOWN_TICKERS,
        data_source     = data_source,
    )


@app.get("/portfolio/{profile}",
         response_model=PortfolioResponse,
         tags=["Portfolio"])
async def get_portfolio(profile: str):
    """Portfolio composition for risk_taking | prudent | equal_weight."""
    valid = ["risk_taking", "prudent", "equal_weight"]
    if profile not in valid:
        raise HTTPException(400, f"Invalid profile. Choose: {valid}")

    if DATA.get("portfolio") is None:
        raise HTTPException(503, "Portfolio data not available")

    profile_map = {
        "risk_taking" : "Risk-Taking",
        "prudent"     : "Prudent",
        "equal_weight": "Equal-Weight",
    }
    filtered = DATA["portfolio"][
        DATA["portfolio"]["profile"] == profile_map[profile]
    ]
    if filtered.empty:
        raise HTTPException(404, f"No data for profile: {profile}")

    stocks = [
        PortfolioStock(
            ticker     = row["ticker"],
            sector     = row.get("sector", "Unknown"),
            weight     = round(float(row["weight"]), 4),
            risk_score = round(float(row.get("risk_score", 0)), 2),
            risk_flag  = str(row.get("risk_flag", "N/A")),
        )
        for _, row in filtered.iterrows()
    ]

    perf = {
        "Risk-Taking" : (0.3573, 0.2952, 0.8596),
        "Prudent"     : (0.1644, 0.2480, 0.4550),
        "Equal-Weight": (0.5393, 0.2477, 1.3969),
    }
    ret, vol, sr = perf[profile_map[profile]]

    return PortfolioResponse(
        profile         = profile,
        stocks          = stocks,
        expected_return = round(ret, 4),
        expected_vol    = round(vol, 4),
        sharpe_ratio    = round(sr,  4),
        total_stocks    = len(stocks),
    )


@app.get("/portfolio/scores/profitability",
         response_model=ProfitabilityResponse,
         tags=["Portfolio"])
async def get_profitability_scores():
    if DATA.get("profitability") is None:
        raise HTTPException(503, "Profitability data not available")

    scores = [
        ProfitabilityScore(
            rank            = rank,
            ticker          = str(row.get("ticker", "")),
            sector          = str(row.get("sector", "Unknown")),
            mtl_score       = round(float(row.get("f1_mtl",    0)), 4),
            tech_score      = round(float(row.get("f2_tech",   0)), 4),
            signal_score    = round(float(row.get("f3_signal", 0)), 4),
            sharpe_score    = round(float(row.get("f4_sharpe", 0)), 4),
            trend_score     = round(float(row.get("f5_trend",  0)), 4),
            composite_score = round(float(row.get("composite", 0)), 4),
        )
        for rank, (_, row) in enumerate(
            DATA["profitability"].iterrows(), start=1
        )
    ]
    return ProfitabilityResponse(
        scores          = scores,
        evaluation_date = str(datetime.today().date()),
        total_tickers   = len(scores),
    )


@app.get("/portfolio/scores/risk",
         response_model=RiskResponse,
         tags=["Portfolio"])
async def get_risk_scores():
    if DATA.get("risk_scores") is None:
        raise HTTPException(503, "Risk scores data not available")

    scores = [
        RiskScore(
            ticker           = str(row.get("ticker", "")),
            sector           = str(row.get("sector", "Unknown")),
            volatility_risk  = round(float(row.get("r1_vol",           0)), 2),
            sell_risk        = round(float(row.get("r2_sell",          0)), 2),
            drawdown_risk    = round(float(row.get("r3_drawdown",      0)), 2),
            correlation_risk = round(float(row.get("r4_corr",         0)), 2),
            reversal_risk    = round(float(row.get("r5_reversal",      0)), 2),
            composite_risk   = round(float(row.get("final_risk_score", 0)), 2),
            risk_flag        = str(row.get("risk_flag", "N/A")),
        )
        for _, row in DATA["risk_scores"].iterrows()
    ]
    return RiskResponse(scores=scores, total_tickers=len(scores))
