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
    DATA_DIR,
)
from src.api.models import (
    PredictionRequest, SignalRequest, PortfolioRequest,
    PricePredictionResponse, SignalResponse,
    PortfolioResponse, PortfolioStock,
    ProfitabilityResponse, ProfitabilityScore,
    RiskResponse, RiskScore, HealthResponse,
    BacktestSummary, EquityPoint, BacktestEquityResponse,
    FeatureContext, ShapContribution, ShapExplainResponse,
)

try:
    import shap as _shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    print("  ⚠ shap not installed — /predict/signal/explain will be unavailable")
from src.api.feature_pipeline import build_live_features, build_xgb_signal_features

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
        "task4_ticker_encoder", "xgb_t4_signal_scaler",
    ]:
        try:
            MODELS[key] = joblib.load(MODEL_PATHS[key])
            print(f"  ✓ {key} loaded")
        except Exception as e:
            print(f"  ✗ {key} failed: {e}")
            MODELS[key] = None

    # SHAP explainer for XGBoost T4 signal
    if _SHAP_AVAILABLE and MODELS.get("xgb_signal_t4") is not None:
        try:
            MODELS["shap_explainer_t4"] = _shap.TreeExplainer(MODELS["xgb_signal_t4"])
            print("  ✓ shap_explainer_t4 loaded")
        except Exception as e:
            print(f"  ✗ shap_explainer_t4 failed: {e}")
            MODELS["shap_explainer_t4"] = None
    else:
        MODELS["shap_explainer_t4"] = None

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


def _decode_prediction(reg_pred, cls_pred, current_price: float):
    """Unpack MTL model output into returns, prices, direction, confidence."""
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
    Predict 5-day price trajectory using live market features.

    Primary: fetches live OHLCV via yfinance, computes 25 indicators → ~3-5 s.
    Fallback (27 known tickers): pre-computed master_features if live fetch fails.
    """
    ticker   = request.ticker.upper()
    n_days   = request.n_days_back
    is_known = ticker in KNOWN_TICKERS

    model = MODELS.get("mtl_t4") or MODELS.get("mtl_t3")
    if model is None:
        raise HTTPException(503, "Prediction model not available")

    scaler  = MODELS.get("task4_scaler")
    live_ok = False

    # ── Always try live features first (freshest market data) ────
    if scaler is not None:
        try:
            X, raw_price, last_date = build_live_features(ticker, scaler, n_days)
            current_price = raw_price if raw_price >= 1000 else raw_price * 1000
            data_source   = "live-yfinance"
            model_label   = "MTL_T4_Live"
            live_ok       = True
        except Exception as e:
            print(f"  ⚠ live features failed for {ticker}: {e}")

    # ── Pre-computed fallback (known tickers only) ────────────────
    if not live_ok:
        if not is_known:
            raise HTTPException(
                503,
                f"Live data unavailable for '{ticker}' and no pre-computed data exists.",
            )
        X = _get_precomputed_features(ticker, n_days)
        df_t      = DATA["master_features"]
        df_t      = df_t[df_t["ticker"] == ticker].sort_values("date")
        last_row  = df_t.iloc[-1]
        csv_price = float(last_row["close"])
        if csv_price < 1000:
            csv_price *= 1000
        last_date     = str(last_row["date"].date())
        current_price = csv_price
        data_source   = "pre-computed"
        model_label   = "MTL_T4_Specialized"

    # ── Historical prices from per-ticker OHLCV CSV ───────────────
    try:
        ohlcv = pd.read_csv(DATA_DIR / f"{ticker}_ohlcv.csv")
        ohlcv.columns = [c.lower().strip() for c in ohlcv.columns]
        if "date" not in ohlcv.columns:
            ohlcv = ohlcv.reset_index()
            ohlcv.columns = [c.lower().strip() for c in ohlcv.columns]
        ohlcv["date"] = pd.to_datetime(ohlcv["date"])
        ohlcv = ohlcv.sort_values("date").tail(30)
        historical_prices = [
            {"date": str(row["date"].date()), "price": round(float(row["close"]))}
            for _, row in ohlcv.iterrows()
        ]
    except Exception:
        historical_prices = []

    reg_pred, cls_pred = model.predict(X, verbose=0)
    (pred_returns, pred_prices,
     direction, confidence,
     p_buy, p_sell, p_hold) = _decode_prediction(
         reg_pred, cls_pred, current_price,
     )

    return PricePredictionResponse(
        ticker             = ticker,
        prediction_date    = last_date,
        current_price      = current_price,
        predicted_returns  = pred_returns,
        predicted_prices   = pred_prices,
        direction          = direction,
        confidence         = confidence,
        model_used         = model_label,
        is_known_ticker    = is_known,
        data_source        = data_source,
        historical_prices  = historical_prices,
    )


@app.post("/predict/signal",
          response_model=SignalResponse,
          tags=["Signal"])
async def predict_signal(request: SignalRequest):
    """
    BUY / SELL / HOLD trading signal (5-day horizon).

    Branch 1 (27 T4 tickers): XGBoost T4 live pipeline
      — fetches today's OHLCV, computes 41 features (tech + S/R + MA + MTL),
        runs xgb_t4_signal.pkl → freshest signal every request.
    Fallback A: pre-computed signal CSV (known tickers, if live fetch fails).
    Fallback B: MTL classification head (any ticker, last resort).
    """
    ticker    = request.ticker.upper()
    threshold = request.threshold
    is_known  = ticker in KNOWN_TICKERS

    p_buy = p_sell = p_hold = 1 / 3
    sig_date      = str(datetime.today().date())
    data_source   = "fallback-hold"
    feat_vec_out  = None
    feat_ctx_out  = None

    # ── Branch 1: XGBoost T4 live pipeline ───────────────────
    xgb_t4      = MODELS.get("xgb_signal_t4")
    scaler_tech = MODELS.get("task4_scaler")
    scaler_new  = MODELS.get("xgb_t4_signal_scaler")
    model_mtl   = MODELS.get("mtl_t4")
    ticker_enc  = MODELS.get("task4_ticker_encoder")

    if is_known and all(
        m is not None for m in [xgb_t4, scaler_tech, scaler_new, model_mtl, ticker_enc]
    ):
        try:
            feat_vec_out, sig_date, feat_ctx_out = build_xgb_signal_features(
                ticker, scaler_tech, scaler_new, model_mtl, ticker_enc
            )
            proba  = xgb_t4.predict_proba(feat_vec_out)[0]   # [SELL, HOLD, BUY]
            p_sell = float(proba[0])
            p_hold = float(proba[1])
            p_buy  = float(proba[2])
            data_source = "live-xgb-t4"
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except Exception as e:
            print(f"  ⚠ XGBoost T4 live pipeline failed for {ticker}: {e}")

    # ── Fallback A: pre-computed signal CSV ───────────────────
    if data_source == "fallback-hold" and is_known and DATA.get("signals") is not None:
        sig_df     = DATA["signals"]
        ticker_sig = sig_df[sig_df["ticker"] == ticker].sort_values("date")
        if len(ticker_sig) > 0:
            last        = ticker_sig.iloc[-1]
            p_buy       = float(last.get("p_buy",  1 / 3))
            p_sell      = float(last.get("p_sell", 1 / 3))
            p_hold      = max(0.0, 1 - p_buy - p_sell)
            sig_date    = str(last["date"])[:10]
            data_source = "pre-computed"

    # ── Fallback B: MTL classification head ───────────────────
    if data_source == "fallback-hold":
        scaler = MODELS.get("task4_scaler")
        mtl    = MODELS.get("mtl_t4")
        if mtl is None or scaler is None:
            raise HTTPException(503, "No signal model available")
        try:
            X, _, sig_date = build_live_features(ticker, scaler)
            _, cls_pred = mtl.predict(X, verbose=0)
            p_sell = float(cls_pred[0, 0])
            p_hold = float(cls_pred[0, 1])
            p_buy  = float(cls_pred[0, 2])
            data_source = "live-mtl-fallback"
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except Exception:
            p_buy = p_sell = p_hold = 1 / 3
            data_source = "fallback-hold"

    conviction = max(p_buy, p_sell)

    if p_buy >= threshold:
        signal = "BUY"
        recommendation = (
            f"Tín hiệu MUA (XGBoost T4, horizon 5 ngày). "
            f"P(MUA)={p_buy:.2%} ≥ ngưỡng={threshold:.2%}. "
            "XGBoost phát hiện đà tăng qua S/R zone, MA crossover và xác nhận MTL."
        )
    elif p_sell >= threshold:
        signal = "SELL"
        recommendation = (
            f"Tín hiệu BÁN (XGBoost T4, horizon 5 ngày). "
            f"P(BÁN)={p_sell:.2%} ≥ ngưỡng={threshold:.2%}. "
            "XGBoost phát hiện áp lực bán qua S/R breakdown và tín hiệu EMA."
        )
    else:
        signal = "HOLD"
        recommendation = (
            f"Chưa có tín hiệu rõ ràng (horizon 5 ngày). "
            f"Conviction cao nhất={conviction:.2%} < ngưỡng={threshold:.2%}. "
            "Giữ nguyên vị thế, chờ tín hiệu xác nhận thêm."
        )

    feature_context = FeatureContext(**feat_ctx_out) if feat_ctx_out else None

    return SignalResponse(
        ticker          = ticker,
        signal_date     = sig_date,
        signal          = signal,
        p_buy           = round(p_buy,      4),
        p_sell          = round(p_sell,     4),
        p_hold          = round(p_hold,     4),
        conviction      = round(conviction, 4),
        threshold_used  = threshold,
        recommendation  = recommendation,
        is_known_ticker = ticker in KNOWN_TICKERS,
        data_source     = data_source,
        feature_context = feature_context,
    )


@app.post("/predict/signal/explain",
          response_model=ShapExplainResponse,
          tags=["Signal"])
async def explain_signal(request: SignalRequest):
    """
    SHAP feature contributions for the XGBoost T4 signal prediction.
    Uses cached features (call /predict/signal first to warm the cache).
    Returns top-20 feature contributions ranked by |SHAP value|.
    """
    from src.api.feature_pipeline import XGB_T4_FEATURES

    if not _SHAP_AVAILABLE:
        raise HTTPException(503, "SHAP not available on this server")

    ticker    = request.ticker.upper()
    explainer = MODELS.get("shap_explainer_t4")
    xgb_t4   = MODELS.get("xgb_signal_t4")
    if explainer is None or xgb_t4 is None:
        raise HTTPException(503, "SHAP explainer not available")

    scaler_tech = MODELS.get("task4_scaler")
    scaler_new  = MODELS.get("xgb_t4_signal_scaler")
    model_mtl   = MODELS.get("mtl_t4")
    ticker_enc  = MODELS.get("task4_ticker_encoder")

    if any(m is None for m in [scaler_tech, scaler_new, model_mtl, ticker_enc]):
        raise HTTPException(503, "Signal models not fully loaded")

    try:
        feat_vec, _, _ = build_xgb_signal_features(
            ticker, scaler_tech, scaler_new, model_mtl, ticker_enc
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as e:
        raise HTTPException(500, f"Feature computation failed: {e}")

    proba  = xgb_t4.predict_proba(feat_vec)[0]
    pred_class = int(np.argmax(proba))   # 0=SELL 1=HOLD 2=BUY
    label_map  = {0: "SELL", 1: "HOLD", 2: "BUY"}

    # shap_values: list of arrays [class_0, class_1, class_2], each shape (1, 41)
    shap_vals  = explainer.shap_values(feat_vec)
    sv         = shap_vals[pred_class][0]   # shape (41,) for predicted class
    base_val   = float(explainer.expected_value[pred_class])

    # Rank by absolute contribution, take top 20
    ranked_idx = np.argsort(np.abs(sv))[::-1][:20]
    contributions = [
        ShapContribution(
            feature    = XGB_T4_FEATURES[i],
            shap_value = round(float(sv[i]), 6),
            direction  = "positive" if sv[i] >= 0 else "negative",
        )
        for i in ranked_idx
    ]

    return ShapExplainResponse(
        ticker          = ticker,
        signal          = label_map[pred_class],
        predicted_class = pred_class,
        contributions   = contributions,
        base_value      = round(base_val, 6),
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


@app.get("/backtest/price",
         response_model=BacktestEquityResponse,
         tags=["Backtest"])
async def get_price_backtest():
    """
    Walk-forward backtest results for the MTL T4 price prediction model.
    Returns equity curve (strategy vs VN benchmark) and summary metrics.
    """
    bt  = DATA.get("price_backtest")
    eq  = DATA.get("equity_curve")
    if bt is None or eq is None:
        raise HTTPException(503, "Backtest data not available")

    buy_mask = bt["signal"] == "BUY"

    summary = BacktestSummary(
        total_predictions = len(bt),
        da_1d             = round(float(bt["dir_correct_1d"].mean()), 4),
        da_5d             = round(float(bt["dir_correct_5d"].dropna().mean()), 4),
        mae_1d            = round(float(bt["mae_1d"].mean()), 6),
        mae_5d            = round(float(bt["mae_5d"].dropna().mean()), 6),
        buy_signal_pct    = round(float(buy_mask.mean()), 4),
        sharpe_strategy   = round(float(
            eq["strat_return"].mean() / eq["strat_return"].std() * (252 ** 0.5)
        ), 3) if eq["strat_return"].std() > 0 else 0.0,
        sharpe_benchmark  = round(float(
            eq["bench_return"].mean() / eq["bench_return"].std() * (252 ** 0.5)
        ), 3) if eq["bench_return"].std() > 0 else 0.0,
        cum_return_strat  = round(float(eq["strat_cum"].iloc[-1]), 4),
        cum_return_bench  = round(float(eq["bench_cum"].iloc[-1]), 4),
        date_range_start  = str(eq["date"].iloc[0]),
        date_range_end    = str(eq["date"].iloc[-1]),
    )

    equity_curve = [
        EquityPoint(
            date         = str(row["date"]),
            strat_return = round(float(row["strat_return"]), 6),
            bench_return = round(float(row["bench_return"]), 6),
            strat_cum    = round(float(row["strat_cum"]),    4),
            bench_cum    = round(float(row["bench_cum"]),    4),
        )
        for _, row in eq.iterrows()
    ]

    return BacktestEquityResponse(summary=summary, equity_curve=equity_curve)
