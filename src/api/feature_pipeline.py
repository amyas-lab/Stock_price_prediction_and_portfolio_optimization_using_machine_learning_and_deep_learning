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
from sklearn.cluster import KMeans

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

_FETCH_EXTRA_DAYS     = 110
_XGB_SIGNAL_EXTRA     = 230   # SR lookback (63) + EMA-50 warmup (49) + VNI join loss + buffer

# ── XGBoost T4 feature column names ──────────────────────────
SR_FEATURE_COLS = [
    "sr_distance_pct", "sr_breakout_up", "sr_breakout_down",
    "sr_near_resistance", "sr_near_support",
]
MA_FEATURE_COLS = [
    "ma_golden_cross_short", "ma_death_cross_short",
    "ma_golden_cross_long",  "ma_death_cross_long",
    "ma_short_gap_pct",      "ma_long_gap_pct",
    "ma_alignment",
]
MTL_FEATURE_COLS = ["mtl_p_up_t4", "mtl_p_down_t4", "mtl_conviction_t4"]

# Scaler for SR + MA + MTL (task4_xgb_signal_scaler.pkl was fit on these 15)
XGB_T4_NEW_COLS  = SR_FEATURE_COLS + MA_FEATURE_COLS + MTL_FEATURE_COLS
# Full 41-feature order expected by xgb_t4_signal.pkl
XGB_T4_FEATURES  = FEATURE_COLS + XGB_T4_NEW_COLS + ["ticker_encoded"]

_SR_LOOKBACK   = 63
_SR_ZONE_WIDTH = 0.005
_SR_K_RANGE    = range(2, 12)

_VNI_CSV_PATH = (
    Path(__file__).parent.parent.parent
    / "data" / "vietnam" / "vnindex_ohlcv.csv"
)

# ── In-memory cache: max 128 tickers, 1-hour TTL ─────────────
_cache      = TTLCache(maxsize=128, ttl=3600)
_lock       = Lock()
_xgb_cache  = TTLCache(maxsize=128, ttl=3600)
_xgb_lock   = Lock()
_vni_df: pd.DataFrame | None = None
_vni_lock   = Lock()


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


# ── XGBoost T4 helpers ────────────────────────────────────────

def _find_optimal_k(prices: np.ndarray) -> int:
    inertias = []
    for k in _SR_K_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=100)
        km.fit(prices.reshape(-1, 1))
        inertias.append(km.inertia_)
    inertias      = np.array(inertias)
    deltas        = np.diff(inertias)
    second_deltas = np.diff(deltas)
    optimal_idx   = int(np.argmax(second_deltas)) + 2
    return list(_SR_K_RANGE)[optimal_idx]


def _compute_sr_features_live(closes: np.ndarray) -> dict:
    """Compute S/R zone features for the current (last) trading day."""
    zero = {c: 0.0 for c in SR_FEATURE_COLS}
    if len(closes) < _SR_LOOKBACK + 1:
        return zero

    current_price = float(closes[-1])
    prev_price    = float(closes[-2])
    window_prices = closes[-_SR_LOOKBACK - 1 : -1]   # 63 prices ending at yesterday

    optimal_k  = _find_optimal_k(window_prices)
    km = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=100)
    km.fit(window_prices.reshape(-1, 1))
    zone_centers = np.sort(km.cluster_centers_.flatten())
    zone_width   = current_price * _SR_ZONE_WIDTH

    distances      = np.abs(zone_centers - current_price)
    nearest_center = zone_centers[int(np.argmin(distances))]
    sr_distance    = (current_price - nearest_center) / nearest_center * 100

    sr_break_up = sr_break_down = 0.0
    for center in zone_centers:
        zone_low  = center - zone_width
        zone_high = center + zone_width
        if prev_price < zone_low and current_price > zone_high:
            sr_break_up = 1.0
        elif prev_price > zone_high and current_price < zone_low:
            sr_break_down = 1.0

    sr_near_res = sr_near_sup = 0.0
    resistance  = zone_centers[zone_centers > current_price]
    support     = zone_centers[zone_centers < current_price]
    if len(resistance) > 0 and abs(current_price - resistance[0]) / current_price < _SR_ZONE_WIDTH:
        sr_near_res = 1.0
    if len(support) > 0 and abs(current_price - support[-1]) / current_price < _SR_ZONE_WIDTH:
        sr_near_sup = 1.0

    return {
        "sr_distance_pct"   : sr_distance,
        "sr_breakout_up"    : sr_break_up,
        "sr_breakout_down"  : sr_break_down,
        "sr_near_resistance": sr_near_res,
        "sr_near_support"   : sr_near_sup,
    }


def _compute_ma_features_live(ema10: np.ndarray,
                               ema20: np.ndarray,
                               ema50: np.ndarray) -> dict:
    """Compute MA crossover features for the current (last) trading day."""
    zero = {c: 0.0 for c in MA_FEATURE_COLS}
    if len(ema10) < 2:
        return zero

    e10, e20, e50   = float(ema10[-1]), float(ema20[-1]), float(ema50[-1])
    pe10, pe20, pe50 = float(ema10[-2]), float(ema20[-2]), float(ema50[-2])

    golden_short = float(pe10 <= pe20 and e10 > e20)
    death_short  = float(pe10 >= pe20 and e10 < e20)
    golden_long  = float(pe20 <= pe50 and e20 > e50)
    death_long   = float(pe20 >= pe50 and e20 < e50)
    short_gap    = (e10 - e20) / e20 * 100 if e20 != 0 else 0.0
    long_gap     = (e20 - e50) / e50 * 100 if e50 != 0 else 0.0
    bull         = e10 > e20 and e20 > e50
    bear         = e10 < e20 and e20 < e50
    alignment    = float(bull) - float(bear)

    return {
        "ma_golden_cross_short": golden_short,
        "ma_death_cross_short" : death_short,
        "ma_golden_cross_long" : golden_long,
        "ma_death_cross_long"  : death_long,
        "ma_short_gap_pct"     : short_gap,
        "ma_long_gap_pct"      : long_gap,
        "ma_alignment"         : alignment,
    }


def build_xgb_signal_features(
    ticker: str,
    scaler_tech,     # task4_feature_scaler.pkl  — for the 25 tech features
    scaler_new,      # task4_xgb_signal_scaler.pkl — for SR + MA + MTL (15 features)
    model_mtl,       # mtl_t4 keras model
    ticker_encoder,  # task4_ticker_encoder.pkl
    n_days: int = 20,
) -> tuple:
    """
    Build the full 41-feature vector for XGBoost T4 signal prediction.

    Feature order (matches xgb_t4_signal.pkl training):
      [25 tech scaled] + [5 SR scaled] + [7 MA scaled] + [3 MTL scaled] + [1 ticker int]

    Returns
    -------
    feat_vec        : np.ndarray, shape (1, 41)
    last_date       : str  — ISO date of the last trading day used
    feature_context : dict — raw (unscaled) intermediate values for UI rationale
    """
    cache_key = f"xgb_signal:{ticker}"
    with _xgb_lock:
        if cache_key in _xgb_cache:
            return _xgb_cache[cache_key]

    end_dt   = datetime.today()
    start_dt = end_dt - timedelta(days=n_days + _XGB_SIGNAL_EXTRA)
    start    = start_dt.strftime("%Y-%m-%d")
    end      = end_dt.strftime("%Y-%m-%d")

    df_stock = _fetch_ohlcv(f"{ticker}.VN", start, end)
    df_vni   = _get_vni_ohlcv(start, end)

    stock_feats = _compute_indicators(df_stock)
    vni_feats   = _compute_indicators(df_vni, prefix="vni_")
    combined    = stock_feats.join(vni_feats, how="inner").dropna()

    # MTL sequence (n_days) is a subset of the SR window (last 63 rows) — they overlap.
    # Actual minimum: 63 for SR window + current row + previous row for MA crossover.
    min_rows = _SR_LOOKBACK + 2
    if len(combined) < min_rows:
        raise ValueError(
            f"Only {len(combined)} valid rows for '{ticker}' "
            f"(need {min_rows} after indicator warmup for XGBoost T4 pipeline)."
        )

    # ── 25 tech features (scaled) ────────────────────────────
    X_raw    = combined[FEATURE_COLS].values          # shape (T, 25)
    X_scaled = scaler_tech.transform(X_raw)           # shape (T, 25)

    # ── MTL T4 features (run on last n_days scaled rows) ────
    X_seq = X_scaled[-n_days:].reshape(1, n_days, len(FEATURE_COLS))
    _, cls_pred  = model_mtl.predict(X_seq, verbose=0)
    mtl_p_up     = float(cls_pred[0, 2])
    mtl_p_down   = float(cls_pred[0, 0])
    mtl_conv     = max(mtl_p_up, mtl_p_down)

    # ── SR zone features (raw closes aligned to combined index) ──
    closes   = df_stock.loc[combined.index, "close"].values
    sr_feats = _compute_sr_features_live(closes)

    # ── MA crossover features (raw EMA values) ───────────────
    ma_feats = _compute_ma_features_live(
        combined["ema_10"].values,
        combined["ema_20"].values,
        combined["ema_50"].values,
    )

    # ── Ticker encoding ──────────────────────────────────────
    try:
        ticker_code = float(ticker_encoder.transform([ticker])[0])
    except Exception:
        ticker_code = -1.0   # unseen ticker — XGBoost trees degrade gracefully

    # ── Scale SR + MA + MTL together (15 features) ──────────
    new_raw = np.array([
        sr_feats["sr_distance_pct"],
        sr_feats["sr_breakout_up"],
        sr_feats["sr_breakout_down"],
        sr_feats["sr_near_resistance"],
        sr_feats["sr_near_support"],
        ma_feats["ma_golden_cross_short"],
        ma_feats["ma_death_cross_short"],
        ma_feats["ma_golden_cross_long"],
        ma_feats["ma_death_cross_long"],
        ma_feats["ma_short_gap_pct"],
        ma_feats["ma_long_gap_pct"],
        ma_feats["ma_alignment"],
        mtl_p_up,
        mtl_p_down,
        mtl_conv,
    ]).reshape(1, -1)
    new_scaled = scaler_new.transform(new_raw)[0]   # shape (15,)

    # ── Assemble 41-feature vector ───────────────────────────
    feat_vec  = np.concatenate([X_scaled[-1], new_scaled, [ticker_code]]).reshape(1, -1)
    last_date = str(combined.index[-1].date())

    # ── Feature context (raw values for UI rationale) ────────
    last_raw = combined[FEATURE_COLS].iloc[-1]
    feature_context = {
        "rsi"              : round(float(last_raw["rsi"]),         2),
        "macd_hist"        : round(float(last_raw["macd_hist"]),   6),
        "log_return"       : round(float(last_raw["log_return"]),  6),
        "ema_10"           : round(float(last_raw["ema_10"]),      2),
        "ema_20"           : round(float(last_raw["ema_20"]),      2),
        "ema_50"           : round(float(last_raw["ema_50"]),      2),
        "ma_alignment"     : ma_feats["ma_alignment"],
        "ma_short_gap_pct" : round(ma_feats["ma_short_gap_pct"],  4),
        "ma_long_gap_pct"  : round(ma_feats["ma_long_gap_pct"],   4),
        "ma_golden_cross_short": ma_feats["ma_golden_cross_short"],
        "ma_death_cross_short" : ma_feats["ma_death_cross_short"],
        "ma_golden_cross_long" : ma_feats["ma_golden_cross_long"],
        "ma_death_cross_long"  : ma_feats["ma_death_cross_long"],
        "sr_distance_pct"  : round(sr_feats["sr_distance_pct"],   4),
        "sr_breakout_up"   : sr_feats["sr_breakout_up"],
        "sr_breakout_down" : sr_feats["sr_breakout_down"],
        "sr_near_resistance": sr_feats["sr_near_resistance"],
        "sr_near_support"  : sr_feats["sr_near_support"],
        "mtl_p_up_t4"      : round(mtl_p_up,   4),
        "mtl_p_down_t4"    : round(mtl_p_down, 4),
        "mtl_conviction_t4": round(mtl_conv,   4),
    }

    result = (feat_vec, last_date, feature_context)
    with _xgb_lock:
        _xgb_cache[cache_key] = result
    return result
