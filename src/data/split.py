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

def train_val_test_split(df, train_ratio = 0.7, val_ratio = 0.15):
    n = len(df)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    return train, val, test 

def scale_splits(train, val, test, feature_cols):
    scaler = MinMaxScaler()
    train_s = scaler.fit_transform(train[feature_cols])
    val_s   = scaler.transform(val[feature_cols])
    test_s  = scaler.transform(test[feature_cols])
    return train_s, val_s, test_s, scaler  # scaler returned for inverse_transform later

# Make sequences for LSTM model
# For forcasting k-days ahead 
def make_sequences(data, lookback=30, horizon=7, target_col_idx=3):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:i+lookback, :])  # ndarray slicing
        y.append(data[i+lookback:i+lookback+horizon, target_col_idx])  # k-day target
    return np.array(X), np.array(y)

# Rolling CV
def rolling_window_cv(df, n_splits = 5, test_size = 30): # 30 days is a test size
    # TimeSeries is an object creating folders for train and validation data
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size) 
    # 
    for train_index, test_index in tscv.split(df):
        yield df.iloc[train_index], df.iloc[test_index] 

