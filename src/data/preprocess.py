"""
Feature engineering utilities for stock OHLCV time-series.

This module is intended to be imported from notebooks/scripts and provides
reusable feature functions plus one end-to-end pipeline (`add_all_features`).
"""
import pandas as pd
import numpy as np
import ta  # Technical Analysis Library in Python

# ------------------------------------------------------------- #

REQUIRED_OHLCV = ["open", "high", "low", "close", "volume"]


def _validate_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def compute_log_returns(df: pd.DataFrame) -> pd.Series:
    """
    Compute 1-day log return from close price.

    Raw closing prices are non-stationary — their mean and variance drift over
    time, violating a core assumption of most sequence models.  Log returns
    log(P_t / P_{t-1}) transform the price series into a stationary,
    scale-free representation of day-over-day percentage change, making values
    comparable across different stocks and across different decades of data.

    We retain raw OHLC alongside log_return because the model's prediction
    target is the actual price; log_return is supplementary momentum context,
    not a replacement for the price signal.
    """
    # log(close_t / close_{t-1}): positive = price rose, negative = price fell
    return np.log(df["close"] / df["close"].shift(1))


def compute_rsi(df: pd.DataFrame) -> pd.Series:
    """
    Compute RSI(14) from close price.

    The Relative Strength Index is a momentum oscillator bounded in [0, 100].
    It measures the speed and magnitude of recent price changes:
      - RSI > 70: asset may be overbought — momentum is exhausting to the upside
      - RSI < 30: asset may be oversold  — momentum is exhausting to the downside

    This gives the model a "market state" signal that raw price cannot convey.
    Two stocks at the same absolute price level can have RSI values of 20 and
    80, implying completely opposite momentum contexts.

    The 14-day window is the standard introduced by J. Welles Wilder and is the
    most widely used period across institutional and retail trading systems.
    """
    # window=14: 14-day lookback is the Wilder standard for RSI computation
    return ta.momentum.rsi(df["close"], window=14)


def compute_macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MACD(12,26,9) and return line/signal/hist.

    MACD (Moving Average Convergence/Divergence) captures the relationship
    between short-term and long-term momentum using three derived series:

      macd       = EMA(12) - EMA(26)
                   Positive → short-term momentum is stronger than long-term
                   (bullish); negative → bearish pressure. Encodes trend direction.

      macd_signal = EMA(9) of macd
                   A smoothed version of the MACD line.  MACD-Signal crossovers
                   are among the most widely cited entry/exit signals in technical
                   analysis; the model can learn these crossover patterns implicitly
                   from the raw numeric values without explicit rule encoding.

      macd_hist  = macd - macd_signal
                   Divergence between MACD and its signal, highlighting momentum
                   acceleration (hist growing) vs deceleration (hist shrinking).

    Note on redundancy: all three columns are mathematically related
    (hist = macd − signal).  Including all three is partially redundant, but each
    representation emphasises a different aspect of the same underlying signal.
    In practice the overhead is minimal and the model selects what is useful.

    Parameters (12, 26, 9) are the industry-standard defaults originating from
    Gerald Appel's original formulation of MACD.
    """
    macd_obj = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    return pd.DataFrame({
        "macd":        macd_obj.macd(),         # EMA(12) - EMA(26): trend direction
        "macd_signal": macd_obj.macd_signal(),  # EMA(9) of MACD: smoothed trend
        "macd_hist":   macd_obj.macd_diff(),    # MACD - Signal: momentum divergence rate
    }, index=df.index)


def compute_atr(df: pd.DataFrame) -> pd.Series:
    """
    Compute ATR(14) from high/low/close.

    Average True Range measures the average daily price range over the past
    14 sessions, using the True Range at each step:
      TR = max(high - low,  |high - prev_close|,  |low - prev_close|)

    ATR quantifies market volatility in price terms:
      - High ATR: large daily swings, elevated uncertainty
      - Low ATR:  tight consolidation, lower short-term risk

    Unlike standard deviation of returns, ATR is expressed in the same units
    as price and is insensitive to the direction of movement — it measures
    *how much* the market moves, not *which way*.
    """
    # window=14: 14-day rolling true range average (Wilder standard)
    return ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)


def compute_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Bollinger Bands(20,2) and return upper/middle/lower.

    Bollinger Bands define a statistical price envelope around a 20-day
    simple moving average:
      bb_middle = SMA(20)             — dynamic mean / equilibrium price
      bb_upper  = SMA(20) + 2σ       — upper boundary (~95% of price action
                                        historically falls inside the bands)
      bb_lower  = SMA(20) - 2σ       — lower boundary

    The bands encode both trend (via bb_middle) and volatility (via band width):
      - Price near bb_upper: potential overextension upward
      - Price near bb_lower: potential oversold extension downward
      - Narrow bands (squeeze): low volatility, often preceding a breakout

    Including all three band levels gives the model the absolute price position
    *and* the current width of the bands (volatility regime) simultaneously.
    """
    # window=20: 20-day SMA baseline; window_dev=2: ±2 standard deviations
    upper  = ta.volatility.bollinger_hband(df["close"], window=20, window_dev=2)
    middle = ta.volatility.bollinger_mavg(df["close"],  window=20)
    lower  = ta.volatility.bollinger_lband(df["close"], window=20, window_dev=2)
    return pd.DataFrame({"bb_upper": upper, "bb_middle": middle, "bb_lower": lower}, index=df.index)


def compute_target_price(df: pd.DataFrame) -> pd.Series:
    """
    Return target price series for forecasting.

    - Nasdaq: use adjusted close when available.
    - Vietnam (or other sources without adjusted close): use close.

    Adjusted close accounts for stock splits and dividend distributions,
    making historical prices directly comparable across a long time window.
    For example, a 4-for-1 stock split without adjustment would introduce an
    artificial 75% price drop overnight, which the model would otherwise
    interpret as a catastrophic market event.  The vnstock API does not
    provide an adjusted series, so raw close is used for Vietnam data.
    """
    # Nasdaq: adj_close corrects for splits and dividends — preferred target
    if "adj_close" in df.columns:
        return df["adj_close"]
    # Vietnam: no adjusted price available; raw close is the only option
    return df["close"]


def compute_ema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute EMA(10/20/50) from close.

    Exponential Moving Averages assign exponentially decreasing weights to
    older observations, making them more responsive to recent price changes
    than a simple moving average of the same length.

    Three horizons are included to capture trend at different timescales:
      ema_10 (~2 trading weeks):  short-term; reacts quickly to price moves
      ema_20 (~1 trading month):  medium-term; balances responsiveness and noise
      ema_50 (~1 trading quarter): long-term; reflects the broader trend

    Cross-relationships between these levels carry predictive information:
    when ema_10 crosses above ema_50 ("golden cross"), it has historically
    been a bullish signal; the reverse ("death cross") bearish. Providing
    all three raw values lets the model learn such cross-relationships without
    requiring explicit indicator engineering.
    """
    return pd.DataFrame({
        "ema_10": ta.trend.ema_indicator(df["close"], window=10),  # ~2-week trend
        "ema_20": ta.trend.ema_indicator(df["close"], window=20),  # ~1-month trend
        "ema_50": ta.trend.ema_indicator(df["close"], window=50),  # ~1-quarter trend
    }, index=df.index)

# ------------------------------------------------------------- #

def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full feature engineering pipeline.
    Works for both Nasdaq (adj_close available) and Vietnam (no adj_close).

    Each feature group addresses a distinct limitation of using raw price alone:
      - log_return   : stationarity — removes price-level scale differences
      - rsi          : momentum state — overbought / oversold context
      - macd group   : trend direction and momentum acceleration
      - atr          : volatility regime — how much the market is moving
      - ema group    : multi-horizon trend levels and cross-signals
      - bb group     : statistical price envelope and volatility bands

    The longest indicator warmup period is EMA(50) = 50 days, so the first
    ~50 rows per ticker will contain NaN values and are dropped at the end.

    Returns:
        DataFrame with original OHLCV columns + engineered features.
        Initial rows with NaN (indicator warmup periods) are dropped.
    """
    df = df.copy()
    _validate_required_columns(df)

    # Price dynamics: stationary log-scale return, comparable across stocks/periods
    df["log_return"] = compute_log_returns(df)

    # Momentum: RSI(14) oscillator encoding overbought/oversold market state
    df["rsi"] = compute_rsi(df)

    # Momentum: MACD line, smoothed signal, and histogram of their divergence
    df[["macd", "macd_signal", "macd_hist"]] = compute_macd(df)

    # Volatility: ATR(14) measures average daily price range (direction-agnostic)
    df["atr"] = compute_atr(df)

    # Trend: EMA at 10 / 20 / 50-day horizons for short, medium, and long-term trend
    df[["ema_10", "ema_20", "ema_50"]] = compute_ema(df)

    # Volatility + trend: Bollinger Bands (SMA20 ± 2σ) encode price envelope width
    bb = compute_bollinger_bands(df)
    df[["bb_upper", "bb_middle", "bb_lower"]] = bb

    # Drop the warmup rows where indicators are NaN (longest warmup: EMA50 = 50 rows)
    return df.dropna()

if __name__ == "__main__":
    from src.data.loaders import fetch_nasdaq
    df_raw = fetch_nasdaq("NVDA", "2020-01-01", "2024-12-31")
    df_features = add_all_features(df_raw)
    print(df_features.columns.tolist())
    print(df_features.shape)
    print(df_features.head(2))