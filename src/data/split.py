"""
split.py
========
Time-series aware data splitting utilities.

Design principles:
- No shuffling at any stage (chronological order must be preserved)
- Scaler is ALWAYS fitted on train split only, then applied to val/test
- Sequence building happens AFTER splitting and scaling to prevent leakage
- Rolling CV uses sklearn.TimeSeriesSplit (industry standard)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple, List


# ------------------------------------------------------------------ #
# 1. CHRONOLOGICAL TRAIN / VAL / TEST SPLIT
# ------------------------------------------------------------------ #

def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time-series DataFrame chronologically. No shuffling.
    test_ratio is implicitly 1 - train_ratio - val_ratio = 0.15.
    """
    assert train_ratio + val_ratio < 1.0, "train + val ratio must be < 1"
    n         = len(df)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


# ------------------------------------------------------------------ #
# 2. ROLLING WINDOW CROSS-VALIDATION
# ------------------------------------------------------------------ #

def rolling_window_cv(
    df: pd.DataFrame,
    n_splits: int = 5,
    test_size: int = None,
):
    """
    Yield (train_df, val_df) folds using TimeSeriesSplit.
    Each fold: train on past, validate on immediate future.
    Used for hyperparameter tuning — never touch test set during CV.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    for train_index, val_index in tscv.split(df):
        yield df.iloc[train_index], df.iloc[val_index]


# ------------------------------------------------------------------ #
# 3. SCALING  (fit on train only — no leakage)
# ------------------------------------------------------------------ #

def scale_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Fit MinMaxScaler on train only, then transform all three splits.

    Returns:
        train_scaled, val_scaled, test_scaled : float32 numpy arrays
        scaler : fitted scaler (keep it — needed for inverse_transform later)
    """
    scaler       = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train[feature_cols]).astype(np.float32)
    val_scaled   = scaler.transform(val[feature_cols]).astype(np.float32)
    test_scaled  = scaler.transform(test[feature_cols]).astype(np.float32)
    return train_scaled, val_scaled, test_scaled, scaler


# ------------------------------------------------------------------ #
# 4. SEQUENCE BUILDER
# ------------------------------------------------------------------ #

def make_sequences(
    data: np.ndarray,
    lookback: int,
    horizon: int,
    target_col_idx: int = 3,
    multi_step: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert 2D scaled array → (X, y) pairs for LSTM.

    Args:
        data           : scaled array of shape (n_rows, n_features)
        lookback       : number of past days used as input window
        horizon        : how many days ahead to predict
        target_col_idx : column index of target variable (default 3 = close)
        multi_step     : False → Task x.2 (predict only the k-th day)
                         True  → Task x.3 (predict all k days ahead)

    Returns:
        X : (n_samples, lookback, n_features)
        y : (n_samples,)          if multi_step=False
            (n_samples, horizon)  if multi_step=True
    """
    X, y = [], []
    for i in range(len(data) - lookback - horizon):
        X.append(data[i : i + lookback])
        if multi_step:
            y.append(data[i + lookback : i + lookback + horizon, target_col_idx])
        else:
            y.append(data[i + lookback + horizon - 1, target_col_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ------------------------------------------------------------------ #
# SANITY CHECK
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    from src.data.loaders import fetch_nasdaq
    from src.data.preprocess import add_all_features

    df = fetch_nasdaq("NVDA", "2020-01-01", "2024-12-31")
    df = add_all_features(df)

    feature_cols = ["open", "high", "low", "close", "volume",
                    "log_return", "rsi", "macd", "atr",
                    "ema_10", "ema_20", "ema_50",
                    "bb_upper", "bb_middle", "bb_lower"]

    train, val, test = train_val_test_split(df)
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)} rows")

    train_s, val_s, test_s, scaler = scale_splits(train, val, test, feature_cols)
    print(f"Scaled — Train: {train_s.shape} | Val: {val_s.shape} | Test: {test_s.shape}")

    X_tr, y_tr = make_sequences(train_s, lookback=30, horizon=1)
    print(f"\nTask x.1 (next-day):   X={X_tr.shape}, y={y_tr.shape}")

    X_tr7, y_tr7 = make_sequences(train_s, lookback=30, horizon=7)
    print(f"Task x.2 (7th day):    X={X_tr7.shape}, y={y_tr7.shape}")

    X_trm, y_trm = make_sequences(train_s, lookback=30, horizon=7, multi_step=True)
    print(f"Task x.3 (7-day seq):  X={X_trm.shape}, y={y_trm.shape}")

    print(f"\nRolling CV folds:")
    for i, (tr, vl) in enumerate(rolling_window_cv(train, n_splits=5)):
        print(f"  Fold {i+1}: train={len(tr)} | val={len(vl)}")
