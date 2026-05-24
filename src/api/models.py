# ── src/api/models.py ─────────────────────────────────────────
"""
Pydantic request/response schemas.
"""

from pydantic import BaseModel, Field
from typing   import List, Optional, Dict
from datetime import date


# ── Request Schemas ───────────────────────────────────────────
class PredictionRequest(BaseModel):
    ticker      : str = Field(..., example="FPT")
    n_days_back : int = Field(
        default=20,
        description="Number of historical days to use as input",
        ge=20, le=60
    )

class SignalRequest(BaseModel):
    ticker    : str   = Field(..., example="VHM")
    threshold : float = Field(
        default=0.55,
        description="Conviction threshold for signal",
        ge=0.40, le=0.80
    )

class PortfolioRequest(BaseModel):
    profile   : str = Field(
        default="equal_weight",
        description="Portfolio profile: risk_taking, prudent, equal_weight"
    )

class ComputeRequest(BaseModel):
    tickers : List[str]
    weights : Dict[str, float]

class ComputeResponse(BaseModel):
    expected_return : float
    expected_vol    : float
    sharpe_ratio    : float


# ── Response Schemas ──────────────────────────────────────────
class PricePredictionResponse(BaseModel):
    ticker            : str
    prediction_date   : str
    current_price     : float
    predicted_returns : List[float] = Field(
        description="Predicted log returns for next K days"
    )
    predicted_prices  : List[float] = Field(
        description="Predicted absolute prices for next K days"
    )
    direction         : str   = Field(description="UP / DOWN / NEUTRAL")
    confidence        : float = Field(description="P(BUY) or P(SELL)")
    model_used        : str
    is_known_ticker   : bool  = Field(
        default=True,
        description="True = pre-computed (fast); False = live fetch"
    )
    data_source       : str   = Field(
        default="pre-computed",
        description="pre-computed | live-yfinance"
    )
    historical_prices : List[Dict] = Field(
        default=[],
        description="Last 30 actual close prices [{date, price}] for chart history"
    )


class FeatureContext(BaseModel):
    """Intermediate feature values used by XGBoost T4 — shown as rationale on UI."""
    # Technical
    rsi             : float
    macd_hist       : float
    log_return      : float
    # MA crossover
    ema_10          : float
    ema_20          : float
    ema_50          : float
    ma_alignment    : float  = Field(description="+1=bullish, -1=bearish, 0=mixed")
    ma_short_gap_pct: float  = Field(description="(EMA10-EMA20)/EMA20 *100")
    ma_long_gap_pct : float  = Field(description="(EMA20-EMA50)/EMA50 *100")
    ma_golden_cross_short: float
    ma_death_cross_short : float
    ma_golden_cross_long : float
    ma_death_cross_long  : float
    # S/R zone
    sr_distance_pct      : float
    sr_breakout_up       : float
    sr_breakout_down     : float
    sr_near_resistance   : float
    sr_near_support      : float
    # MTL model output
    mtl_p_up_t4          : float
    mtl_p_down_t4        : float
    mtl_conviction_t4    : float


class ShapContribution(BaseModel):
    feature     : str
    shap_value  : float
    direction   : str   = Field(description="positive | negative")


class ShapExplainResponse(BaseModel):
    ticker          : str
    signal          : str
    predicted_class : int
    contributions   : List[ShapContribution]
    base_value      : float


class SignalResponse(BaseModel):
    ticker          : str
    signal_date     : str
    signal          : str   = Field(description="BUY / SELL / HOLD")
    p_buy           : float
    p_sell          : float
    p_hold          : float
    conviction      : float
    threshold_used  : float
    recommendation  : str
    is_known_ticker : bool  = Field(default=True)
    data_source     : str   = Field(default="pre-computed")
    feature_context : Optional[FeatureContext] = Field(
        default=None,
        description="Raw indicator values used as model input (shown as rationale)"
    )


class PortfolioStock(BaseModel):
    ticker      : str
    sector      : str
    weight      : float
    risk_score  : float
    risk_flag   : str


class PortfolioResponse(BaseModel):
    profile         : str
    stocks          : List[PortfolioStock]
    expected_return : float
    expected_vol    : float
    sharpe_ratio    : float
    total_stocks    : int


class ProfitabilityScore(BaseModel):
    rank            : int
    ticker          : str
    sector          : str
    mtl_score       : float
    tech_score      : float
    signal_score    : float
    sharpe_score    : float
    trend_score     : float
    composite_score : float


class ProfitabilityResponse(BaseModel):
    scores          : List[ProfitabilityScore]
    evaluation_date : str
    total_tickers   : int


class RiskScore(BaseModel):
    ticker          : str
    sector          : str
    volatility_risk : float
    sell_risk       : float
    drawdown_risk   : float
    correlation_risk: float
    reversal_risk   : float
    composite_risk  : float
    risk_flag       : str


class RiskResponse(BaseModel):
    scores          : List[RiskScore]
    total_tickers   : int


class HealthResponse(BaseModel):
    status          : str
    models_loaded   : Dict[str, bool]
    supported_tickers_t3: List[str]
    supported_tickers_t4: List[str]
    version         : str


# ── Backtest Schemas ──────────────────────────────────────────
class BacktestSummary(BaseModel):
    total_predictions : int
    da_1d             : float = Field(description="Directional accuracy 1-day")
    da_5d             : float = Field(description="Directional accuracy 5-day")
    mae_1d            : float
    mae_5d            : float
    buy_signal_pct    : float
    sharpe_strategy   : float
    sharpe_benchmark  : float
    cum_return_strat  : float
    cum_return_bench  : float
    date_range_start  : str
    date_range_end    : str


class EquityPoint(BaseModel):
    date            : str
    strat_return    : float
    bench_return    : float
    strat_cum       : float
    bench_cum       : float


class BacktestEquityResponse(BaseModel):
    summary         : BacktestSummary
    equity_curve    : List[EquityPoint]