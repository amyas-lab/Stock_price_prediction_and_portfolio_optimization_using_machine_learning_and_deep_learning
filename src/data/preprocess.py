"""
This module contains the functions for preprocessing the data.
From research, model trained on multi features perform better than model trained on single feature.
Therefore, we will use multi features to train the model.
------------------------------------------------------------------
We will use the following features:
Basic features:
- Open: The price of the stock at the beginning of the trading day.
- High: The highest price of the stock during the trading day.
- Low: The lowest price of the stock during the trading day.
- Close: The price of the stock at the end of the trading day.
- Volume: The volume of the stock traded during the trading day.
------------------------------------------------------------------
Technical indicators:
- EMA (10, 20, 50)
- RSI (14): 
How to use it: This is a momentum indicator that shows the relationship between the price of the stock and the volume of the stock.
Why RSI: It is a momentum indicator that shows the relationship between the price of the stock and the volume of the stock.
- Log returns: 
How to use it: This is the natural logarithm of the price of the stock.
Why Log returns: It is more stable than the price of the stock.
- Forward adjusted close price: 
How to use it: This is the price of the stock after adjusting for splits and dividends.
- Bollinger Bands: A technical indicator that shows the relationship between the price and the volatility of the stock.
How to use it: If the price is above the upper band, it is considered to be overbought.
If the price is below the lower band, it is considered to be oversold.
- MACD: Moving Average Convergence Divergence. This is a momentum indicator that shows the relationship between two moving averages of a stock's price.
- ATR: Average True Range. This is a measure of the volatility of the stock.
"""
import pandas as pd
import numpy as np
import ta  # Technical Analysis Library in Python

# ------------------------------------------------------------- #

# log returns: 
def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the log returns of the stock.
    """
    return np.log(df['close'] / df['close'].shift(1))

# RSI (14): 
def compute_rsi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the RSI of the stock.
    """
    return ta.momentum.rsi(df['close'], window=14)

# MACD: Moving Average Convergence Divergence.
def compute_macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the MACD of the stock.
    Returns a DataFrame with macd line, signal line, and histogram.
    """
    return pd.DataFrame({
        "macd":        ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9).macd(),
        "macd_signal": ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9).macd_signal(),
        "macd_hist":   ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9).macd_diff(),
    }, index=df.index)

# ATR: Average True Range. 
def compute_atr(df: pd.DataFrame) -> pd.Series:
    """
    Compute the ATR of the stock.
    """
    return ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

# Bollinger Bands: 
def compute_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Bollinger Bands of the stock.
    Returns upper band, lower band, and middle band (SMA).
    """
    upper  = ta.volatility.bollinger_hband(df['close'], window=20, window_dev=2)
    middle = ta.volatility.bollinger_mavg(df['close'], window=20)
    lower  = ta.volatility.bollinger_lband(df['close'], window=20, window_dev=2)
    return pd.DataFrame({"bb_upper": upper, "bb_middle": middle, "bb_lower": lower}, index=df.index)

# forward adjusted close price (only valid for Nasdaq — Vietnam has no adj_close):
def compute_forward_adjusted_close_price(df: pd.DataFrame) -> pd.Series:
    """
    Compute the forward adjusted close price of the stock.
    Only applicable when adj_close column is present (Nasdaq).
    """
    if 'adj_close' not in df.columns:
        return df['close']
    return df['adj_close']

# EMA (10, 20, 50):
def compute_ema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute EMA with windows 10, 20, 50.
    Returns a DataFrame with three columns.
    """
    return pd.DataFrame({
        "ema_10": ta.trend.ema_indicator(df['close'], window=10),
        "ema_20": ta.trend.ema_indicator(df['close'], window=20),
        "ema_50": ta.trend.ema_indicator(df['close'], window=50),
    }, index=df.index)

# ------------------------------------------------------------- #

def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full feature engineering pipeline to an OHLCV DataFrame.
    Works for both Nasdaq (has adj_close) and Vietnam (no adj_close).

    Returns:
        DataFrame with original OHLCV columns + all computed features.
        Rows with NaN (from indicator warmup periods) are dropped.
    """
    df = df.copy()
    df["log_return"]  = compute_log_returns(df)
    df["adj_close_fwd"] = compute_forward_adjusted_close_price(df)
    df["rsi"]         = compute_rsi(df)
    df[["macd", "macd_signal", "macd_hist"]] = compute_macd(df)
    df["atr"]         = compute_atr(df)
    df[["ema_10", "ema_20", "ema_50"]] = compute_ema(df)
    bb = compute_bollinger_bands(df)
    df[["bb_upper", "bb_middle", "bb_lower"]] = bb
    return df.dropna()

if __name__ == "__main__":
    from src.data.loaders import fetch_nasdaq
    df_raw = fetch_nasdaq("NVDA", "2020-01-01", "2024-12-31")
    df_features = add_all_features(df_raw)
    print(df_features.columns.tolist())
    print(df_features.shape)
    print(df_features.head(2))