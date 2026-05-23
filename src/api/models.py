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