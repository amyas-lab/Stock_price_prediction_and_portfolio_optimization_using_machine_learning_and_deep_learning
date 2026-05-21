# ── src/api/feature_pipeline.py ──────────────────────────────
"""
Branch-2 on-the-fly pipeline for arbitrary VN stocks.
Fetches OHLCV from yfinance, computes 25 technical features,
scales with task4_feature_scaler, and caches results 1 h per ticker.

Feature column order must match task4_feature_scaler exactly:
  ['volume', 'log_return', 'rsi', 'macd', 'macd_signal', 'macd_hist',
   'atr', 'ema_10', 'ema_20', 'ema_50', 'bb_upper', 'bb_middle', 'bb_lower',
   'vni_log_return', 'vni_ema_10', 'vni_ema_20', 'vni_ema_50',
   'vni_rsi', 'vni_macd', 'vni_macd_signal', 'vni_macd_hist',
   'vni_atr', 'vni_bb_middle', 'vni_bb_upper', 'vni_bb_lower']
"""

import warnings
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
import ta
import yfinance as yf
from cachetools import TTLCache

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────
FEATURE_COLS = [
    "volume", "log_return", "rsi",
    "macd", "macd_signal", "macd_hist",
    "atr", "ema_10", "ema_20", "ema_50",
    "bb_upper", "bb_middle", "bb_lower",
    "vni_log_return", "vni_ema_10", "vni_ema_20", "vni_ema_50",
    "vni_rsi", "vni_macd", "vni_macd_signal", "vni_macd_hist",
    "vni_atr", "vni_bb_middle", "vni_bb_upper", "vni_bb_lower",
]

_FETCH_EXTRA_DAYS = 110
_VNI_CSV_PATH = (
    Path(__file__).parent.parent.parent
    / "notebooks" / "data" / "vietnam" / "csv" / "vnindex_ohlcv.csv"
)

# ── In-memory cache: max 128 tickers, 1-hour TTL ─────────────
_cache    = TTLCache(maxsize=128, ttl=3600)
_lock     = Lock()
_vni_df: pd.DataFrame | None = None
_vni_lock = Lock()


# ── VNI data: CSV primary + yfinance supplement ───────────────

def _load_vni_csv() -> pd.DataFrame:
    df = pd.read_csv(_VNI_CSV_PATH, parse_dates=["date"])
    df.columns = [c.lower() for c in df.columns]
    df = df.set_index("date")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df["volume"] = df["volume"].fillna(0)
    return (
        df[["open", "high", "low", "close", "volume"]]
        .dropna(subset=["open", "high", "low", "close"])
        .sort_index()
    )


def _get_vni_ohlcv(start: str, end: str) -> pd.DataFrame:
    """
    Return VNI OHLCV for [start, end].
    Loads CSV once; if CSV is stale > 3 days, supplements with yfinance ^VNINDEX.
    """
    global _vni_df
    with _vni_lock:
        if _vni_df is None:
            _vni_df = _load_vni_csv()
            last_date = _vni_df.index[-1]
            today     = pd.Timestamp.today().normalize()
            if (today - last_date).days > 3:
                try:
                    fetch_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    fetch_end   = today.strftime("%Y-%m-%d")
                    df_new = _fetch_ohlcv("^VNINDEX", fetch_start, fetch_end)
                    if not df_new.empty:
                        combined = pd.concat([_vni_df, df_new])
                        combined = combined[~combined.index.duplicated(keep="last")]
                        _vni_df  = combined.sort_index()
                except Exception:
                    pass  # CSV-only fallback

    df = _vni_df.loc[start:end].copy()
    if df.empty:
        raise ValueError(
            f"No VNI data for {start} → {end}. "
            f"CSV covers up to {_vni_df.index[-1].date()}."
        )
    return df


# ── Private helpers ───────────────────────────────────────────

def _fetch_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV from yfinance, normalize column names."""
    df = yf.download(symbol, start=start, end=end,
                     auto_adjust=True, progress=False, timeout=15)
    if df.empty:
        raise ValueError(
            f"yfinance returned no data for '{symbol}'. "
            "Check the ticker symbol or try again later."
        )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df.index   = pd.to_datetime(df.index).tz_localize(None)
    required   = ["open", "high", "low", "close", "volume"]
    return df[required].dropna()


def _compute_indicators(ohlcv: pd.DataFrame,
                         prefix: str = "") -> pd.DataFrame:
    """
    Compute 12 technical indicators from OHLCV.
    prefix=''     → stock features (also includes raw 'volume')
    prefix='vni_' → VN-Index market features
    """
    h, l, c = ohlcv["high"], ohlcv["low"], ohlcv["close"]
    out = pd.DataFrame(index=ohlcv.index)

    if not prefix:                          # stock only: keep raw volume
        out["volume"] = ohlcv["volume"]

    out[f"{prefix}log_return"]  = np.log(c / c.shift(1))
    out[f"{prefix}rsi"]         = ta.momentum.rsi(c, window=14)

    _macd = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
    out[f"{prefix}macd"]        = _macd.macd()
    out[f"{prefix}macd_signal"] = _macd.macd_signal()
    out[f"{prefix}macd_hist"]   = _macd.macd_diff()

    out[f"{prefix}atr"]         = ta.volatility.average_true_range(
                                      h, l, c, window=14)
    out[f"{prefix}ema_10"]      = ta.trend.ema_indicator(c, window=10)
    out[f"{prefix}ema_20"]      = ta.trend.ema_indicator(c, window=20)
    out[f"{prefix}ema_50"]      = ta.trend.ema_indicator(c, window=50)

    out[f"{prefix}bb_upper"]    = ta.volatility.bollinger_hband(
                                      c, window=20, window_dev=2)
    out[f"{prefix}bb_middle"]   = ta.volatility.bollinger_mavg(
                                      c, window=20)
    out[f"{prefix}bb_lower"]    = ta.volatility.bollinger_lband(
                                      c, window=20, window_dev=2)
    return out


# ── Public API ────────────────────────────────────────────────

def build_live_features(
    ticker: str,
    scaler,
    n_days: int = 20,
) -> tuple:
    """
    Fetch live data for *ticker*, compute 25 features, scale.

    Returns
    -------
    X_scaled  : np.ndarray, shape (1, n_days, 25)
    last_close: float  — raw close from yfinance (may be in thousands VND)
    last_date : str    — ISO date string of the last trading day
    """
    cache_key = f"{ticker}:{n_days}"
    with _lock:
        if cache_key in _cache:
            return _cache[cache_key]

    end_dt   = datetime.today()
    start_dt = end_dt - timedelta(days=n_days + _FETCH_EXTRA_DAYS)
    start    = start_dt.strftime("%Y-%m-%d")
    end      = end_dt.strftime("%Y-%m-%d")

    # Fetch stock (HOSE format: TICKER.VN) and VNI from CSV + yfinance supplement
    df_stock = _fetch_ohlcv(f"{ticker}.VN", start, end)
    df_vni   = _get_vni_ohlcv(start, end)

    # Compute indicators
    stock_feats = _compute_indicators(df_stock)
    vni_feats   = _compute_indicators(df_vni, prefix="vni_")

    # Align on shared trading dates and drop NaN warmup rows
    combined = stock_feats.join(vni_feats, how="inner").dropna()

    if len(combined) < n_days:
        raise ValueError(
            f"Only {len(combined)} valid trading days for '{ticker}' "
            f"(need {n_days} after indicator warmup). "
            "Try a more liquid stock or check the ticker symbol."
        )

    # Take last n_days rows in the exact scaler column order
    X_raw    = combined[FEATURE_COLS].values[-n_days:]
    X_scaled = scaler.transform(X_raw).reshape(1, n_days, len(FEATURE_COLS))

    last_close = float(df_stock["close"].iloc[-1])
    last_date  = str(df_stock.index[-1].date())

    result = (X_scaled, last_close, last_date)
    with _lock:
        _cache[cache_key] = result
    return result
