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

import time
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
    / "data" / "vietnam" / "vnindex_ohlcv.csv"
)

# ── In-memory cache: max 128 tickers, 1-hour TTL ─────────────
_cache    = TTLCache(maxsize=128, ttl=3600)
_lock     = Lock()
_vni_df: pd.DataFrame | None = None
_vni_lock = Lock()


# ── VNI data: CSV history + vnstock live + yfinance fallback ──

def _load_vni_csv() -> pd.DataFrame:
    # index_col=0: works whether the date column is named "date", "﻿date",
    # or is an unnamed index — covers both BOM-prefixed and index-saved CSV formats.
    df = pd.read_csv(_VNI_CSV_PATH, encoding="utf-8-sig", index_col=0)
    df.columns = [
        c.encode("ascii", "ignore").decode("ascii").lower().strip()
        for c in df.columns
    ]
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df.index = df.index.tz_localize(None) if df.index.tz is None else df.index.tz_convert(None)
    df["volume"] = df["volume"].fillna(0)
    return (
        df[["open", "high", "low", "close", "volume"]]
        .dropna(subset=["open", "high", "low", "close"])
        .sort_index()
    )


def _fetch_vni_vnstock(start: str, end: str) -> pd.DataFrame:
    """Fetch VNIndex OHLCV from vnstock v4 (VCI). Returns empty DataFrame on failure."""
    from vnstock.api.quote import Quote
    q = Quote(symbol="VNINDEX", source="VCI")
    df = q.history(start=start, end=end)
    if df.empty:
        return pd.DataFrame()
    df.columns = [c.lower() for c in df.columns]
    if "time" in df.columns:
        df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    idx = df.index
    df.index = idx.tz_localize(None) if idx.tz is None else idx.tz_convert(None)
    if "volume" not in df.columns:
        df["volume"] = 0
    needed = ["open", "high", "low", "close", "volume"]
    return df[[c for c in needed if c in df.columns]].dropna(
        subset=["open", "high", "low", "close"]
    )


def _fetch_vni_gap(fetch_start: str, fetch_end: str) -> pd.DataFrame:
    """vnstock (primary) then yfinance ^VNINDEX (fallback) for VNI gap fill."""
    try:
        df = _fetch_vni_vnstock(fetch_start, fetch_end)
        if not df.empty:
            return df
    except Exception as e:
        print(f"  ⚠ vnstock VNI failed: {e}")
    try:
        return _fetch_ohlcv("^VNINDEX", fetch_start, fetch_end)
    except Exception as e:
        print(f"  ⚠ yfinance ^VNINDEX failed: {e}")
        return pd.DataFrame()


def _get_vni_ohlcv(start: str, end: str) -> pd.DataFrame:
    """
    Return VNI OHLCV for [start, end].
    Strategy: CSV history (base) -> vnstock live gap -> yfinance ^VNINDEX fallback.
    If CSV is unavailable (e.g. Git LFS pointer on HF Spaces), bootstraps entirely
    from live sources.
    """
    global _vni_df
    with _vni_lock:
        if _vni_df is None:
            # 1. Try historical CSV (fails gracefully if HF Spaces has LFS pointer)
            df_hist = pd.DataFrame()
            try:
                df_hist = _load_vni_csv()
            except Exception as e:
                print(f"  ⚠ VNI CSV unavailable ({e}) — bootstrapping from live sources")

            today = pd.Timestamp.today().normalize()
            fetch_start = (
                (df_hist.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                if not df_hist.empty else "2020-01-01"
            )
            fetch_end = today.strftime("%Y-%m-%d")

            # 2. Fill gap: vnstock -> yfinance
            if fetch_start <= fetch_end:
                df_live = _fetch_vni_gap(fetch_start, fetch_end)
                if not df_live.empty:
                    frames  = [df_hist, df_live] if not df_hist.empty else [df_live]
                    merged  = pd.concat(frames)
                    merged  = merged[~merged.index.duplicated(keep="last")]
                    df_hist = merged.sort_index()

            if df_hist.empty:
                raise ValueError(
                    "Cannot load VNI data from CSV, vnstock, or yfinance."
                )
            _vni_df = df_hist

    df = _vni_df.loc[start:end].copy()
    if df.empty:
        raise ValueError(
            f"No VNI data for {start} -> {end}. "
            f"Data covers up to {_vni_df.index[-1].date()}."
        )
    return df


# ── Private helpers ───────────────────────────────────────────

def _fetch_ohlcv(symbol: str, start: str, end: str, max_retries: int = 3) -> pd.DataFrame:
    """Download OHLCV from yfinance with retry on rate-limit."""
    df = pd.DataFrame()
    for attempt in range(max_retries):
        df = yf.download(
            symbol, start=start, end=end,
            auto_adjust=True, progress=False, timeout=15,
        )
        if not df.empty:
            break
        if attempt < max_retries - 1:
            time.sleep(2 * (attempt + 1))

    if df.empty:
        raise ValueError(
            f"yfinance returned no data for '{symbol}'. "
            "Check the ticker symbol or try again later."
        )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    idx = pd.to_datetime(df.index)
    df.index = idx.tz_localize(None) if idx.tz is None else idx.tz_convert(None)
    required = ["open", "high", "low", "close", "volume"]
    return df[required].dropna()


def _compute_indicators(ohlcv: pd.DataFrame,
                         prefix: str = "") -> pd.DataFrame:
    """
    Compute 12 technical indicators from OHLCV.
    prefix=''     -> stock features (also includes raw 'volume')
    prefix='vni_' -> VN-Index market features
    """
    h, l, c = ohlcv["high"], ohlcv["low"], ohlcv["close"]
    out = pd.DataFrame(index=ohlcv.index)

    if not prefix:
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
    last_close: float  -- raw close from yfinance (may be in thousands VND)
    last_date : str    -- ISO date string of the last trading day
    """
    cache_key = f"{ticker}:{n_days}"
    with _lock:
        if cache_key in _cache:
            return _cache[cache_key]

    end_dt   = datetime.today()
    start_dt = end_dt - timedelta(days=n_days + _FETCH_EXTRA_DAYS)
    start    = start_dt.strftime("%Y-%m-%d")
    end      = end_dt.strftime("%Y-%m-%d")

    df_stock = _fetch_ohlcv(f"{ticker}.VN", start, end)
    df_vni   = _get_vni_ohlcv(start, end)

    stock_feats = _compute_indicators(df_stock)
    vni_feats   = _compute_indicators(df_vni, prefix="vni_")

    combined = stock_feats.join(vni_feats, how="inner").dropna()

    if len(combined) < n_days:
        raise ValueError(
            f"Only {len(combined)} valid trading days for '{ticker}' "
            f"(need {n_days} after indicator warmup). "
            "Try a more liquid stock or check the ticker symbol."
        )

    X_raw    = combined[FEATURE_COLS].values[-n_days:]
    X_scaled = scaler.transform(X_raw).reshape(1, n_days, len(FEATURE_COLS))

    last_close = float(df_stock["close"].iloc[-1])
    last_date  = str(df_stock.index[-1].date())

    result = (X_scaled, last_close, last_date)
    with _lock:
        _cache[cache_key] = result
    return result
