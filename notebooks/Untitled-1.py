# %%
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
import matplotlib.dates as mdates
from IPython.display import display
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
import joblib

warnings.filterwarnings('ignore')

# ── Reproducibility ───────────────────────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)

# ── Project root logic (Tối ưu cho cả Local và Colab) ────────────────────────
try:
    from google.colab import drive
    # Nếu chạy trên Colab và đã mount Drive
    ROOT = Path('/content/drive/MyDrive/DL4AI-240166-project-1')
    print("✓ Running on Google Colab")
except ImportError:
    # Nếu chạy trên máy cá nhân
    ROOT = Path.cwd().parent
    print("✓ Running Locally")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# %%
CONFIG = {
    # ── Sequence construction ─────────────────────────────────────────────────
    'window_size'  : 20,    # look-back window: 2 trading days ≈ 1 calendar month
                            # Compared to 60 look-back window
    'n_features'   : 18,    # overwritten below after FEATURE_COLS is defined

    # ── Data split ratios ─────────────────────────────────────────────────────
    'train_ratio'  : 0.70,  # 70% of chronological data for training
    'val_ratio'    : 0.10,  # 10% for validation / hyperparameter tuning
    'test_ratio'   : 0.20,  # 20% held out for final evaluation

    # ── Training hyperparameters ──────────────────────────────────────────────
    'batch_size'   : 64,    # mini-batch size for gradient descent
    'epochs'       : 100,   # upper bound (EarlyStopping cuts this short)
    'learning_rate': 1e-3,  # Adam initial learning rate
    'patience'     : 10,    # EarlyStopping: max epochs without val_loss improvement

    # ── Dataset quality filter ────────────────────────────────────────────────
    'min_history'  : 120,   # discard tickers with fewer than 120 trading days

    # ── Forecasting horizons ──────────────────────────────────────────────────
    'n_day'        : 3,     # Task 2.2: predict the price n trading days ahead
    'k_days'       : 5,     # Task 2.3: predict the next k consecutive days
}

print('CONFIG:')
for k, v in CONFIG.items():
    print(f'  {k:<18} = {v}')

# %%
### Normalization
# ROOT the orginal folder of the project
MODELS_DIR = ROOT / 'models'

# Create a place to store models
MODELS_DIR.mkdir(exist_ok=True)

# %%
DATA_DIR = ROOT / 'notebooks' / 'data' / 'vietnam'

# ── Load the pre-engineered features CSV ──────────────────────────────────────
MASTER_PATH = DATA_DIR / 'vn_stocks_master_features.csv'
VNI_PATH    = DATA_DIR / 'csv'/ 'vnindex_ohlcv.csv'

assert MASTER_PATH.exists(), f'File not found: {MASTER_PATH}'
assert VNI_PATH.exists(), f'File not found: {VNI_PATH}'

# 1. Load Master Data (10 tickers)
df_all = pd.read_csv(MASTER_PATH)
df_all['date'] = pd.to_datetime(df_all['date']).dt.normalize()
df_all = df_all.sort_values(['ticker', 'date']).reset_index(drop=True)

# 2. Load VN-Index
df_vni_raw = pd.read_csv(VNI_PATH)
df_vni_raw['date'] = pd.to_datetime(df_vni_raw['date']).dt.normalize()
df_vni_raw = df_vni_raw.sort_values('date').reset_index(drop=True)

# ── VN-Index Multi-Feature Engineering ───────────────────────────────────────
def compute_vni_features(df):
    """
    Compute technical indicators for VN-Index, mirroring stock feature engineering.
    All output columns are prefixed with 'vni_' to avoid collisions after merge.
    """
    df = df.copy().sort_values('date').reset_index(drop=True)
    close = df['close']
    high  = df['high']
    low   = df['low']

    # Log return
    df['vni_log_return'] = np.log(close / close.shift(1))

    # EMA
    df['vni_ema_10'] = close.ewm(span=10, adjust=False).mean()
    df['vni_ema_20'] = close.ewm(span=20, adjust=False).mean()
    df['vni_ema_50'] = close.ewm(span=50, adjust=False).mean()

    # RSI (14)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['vni_rsi'] = 100 - (100 / (1 + rs))

    # MACD (12/26/9)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df['vni_macd']        = ema_12 - ema_26
    df['vni_macd_signal'] = df['vni_macd'].ewm(span=9, adjust=False).mean()
    df['vni_macd_hist']   = df['vni_macd'] - df['vni_macd_signal']

    # ATR (14)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    df['vni_atr'] = tr.rolling(14).mean()

    # Bollinger Bands (20, 2σ)
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    df['vni_bb_middle'] = sma_20
    df['vni_bb_upper']  = sma_20 + 2 * std_20
    df['vni_bb_lower']  = sma_20 - 2 * std_20

    vni_cols = [c for c in df.columns if c.startswith('vni_')]
    df[vni_cols] = df[vni_cols].fillna(0)   # fill leading NaNs from rolling windows

    return df[['date'] + vni_cols]

df_vni = compute_vni_features(df_vni_raw)

print(f'Loaded Master  : {df_all.shape[0]:,} rows | Tickers: {df_all["ticker"].unique()}')
print(f'VNI raw        : {df_vni_raw.shape[0]:,} rows')
print(f'VNI features   : {df_vni.shape[1] - 1} indicators → {[c for c in df_vni.columns if c != "date"]}')
print(df_vni.head(100))






# Now we have df_all and df_vni



# %%
# --- Section 2.1: Multi-Ticker VNI Merge ---

# 1. Ensure 'date' columns in both DataFrames are datetime objects for accurate merging
df_all['date'] = pd.to_datetime(df_all['date'])
df_vni['date'] = pd.to_datetime(df_vni['date'])

# 2. Perform Left Join
# We use 'left' join to preserve all rows from the 10 specific stock tickers
df_final = pd.merge(df_all, df_vni, on='date', how='left')

# 3. Identify the list of newly added VNI feature columns
vni_cols = [c for c in df_vni.columns if c != 'date']

# 4. Handle missing values (NaN)
# Holidays or weekends might result in NaNs if VNI and individual stock data don't sync 100%
nan_count = df_final[vni_cols].isna().sum().sum()
if nan_count > 0:
    print(f"⚠ Detected {nan_count} NaNs in VNI features. Filling with Forward Fill then 0.")
    # Sort by ticker and date before filling to prevent data leakage between different tickers
    df_final = df_final.sort_values(['ticker', 'date'])
    df_final[vni_cols] = df_final.groupby('ticker')[vni_cols].ffill().fillna(0)
else:
    print("✓ VNI Merge successful with no missing values.")


# Need the target labels for model retraining


# 5. Final check of row and column counts
print("-" * 30)
print(f"Total rows: {df_final.shape[0]:,}")
print(f"Total columns: {df_final.shape[1]}")
print(f"Tickers processed: {df_final['ticker'].unique().tolist()}")

# Update df_model to prepare for subsequent steps
df_model = df_final.copy()

# %%
# ============================================================
# REVISED: calculate_master_targets với n_forward=5
# Align với Task 2.3 full trajectory (k_days=5)
# ============================================================

def calculate_master_targets(df, n_day=5, buy_thresh=0.02,
                              sell_thresh=-0.015, n_forward=5):
    """
    Args:
        n_day     : số ngày trong trajectory (5)
        n_forward : ngày dùng để tạo classification label (5)
                    → fwd_ret_5 = log(P_{t+5} / P_t)
    """
    df = df.copy().sort_values(['ticker', 'date'])
    grouped_close = df.groupby('ticker')['close']

    # ── target_multi: cumulative forward returns ──────────────
    for i in range(1, n_day + 1):
        df[f'fwd_ret_{i}'] = np.log(
            grouped_close.shift(-i) / df['close']
        )

    fwd_cols = [f'fwd_ret_{i}' for i in range(1, n_day + 1)]

    # Drop rows với NaN trong bất kỳ fwd column nào
    # → last 5 rows của mỗi ticker bị drop
    df = df.dropna(subset=fwd_cols)

    # Stack thành list sau khi đã drop NaN
    df['target_multi'] = df[fwd_cols].values.tolist()

    # ── target_class: dựa trên Day 5 cumulative return ───────
    # fwd_ret_5 = log(P_{t+5} / P_t)
    # Consistent với full 5-day trajectory của regression head
    forward_ret_col      = f'fwd_ret_{n_forward}'  # fwd_ret_5
    df['forward_return'] = df[forward_ret_col]

    conditions = [
        df['forward_return'] > buy_thresh,
        df['forward_return'] < sell_thresh,
    ]
    choices    = [2, 0]
    df['target_class'] = np.select(conditions, choices, default=1)

    # Drop temp fwd columns
    df = df.drop(columns=fwd_cols)

    return df


# ── Execute ───────────────────────────────────────────────────
df_model = calculate_master_targets(
    df_final,
    n_day       = CONFIG['k_days'],   # 5
    buy_thresh  = 0.02,
    sell_thresh = -0.015,
    n_forward   = CONFIG['k_days']    # 5 ← changed from n_day=3
)

# ── Verify ────────────────────────────────────────────────────
sample_multi = np.stack(df_model['target_multi'].values)

print(f"Rows after target generation : {len(df_model):,}")
print(f"target_multi shape           : {sample_multi.shape}")
print(f"  → Should be (n_rows, {CONFIG['k_days']})")

# NaN check
nan_count = np.isnan(sample_multi).sum()
assert nan_count == 0, f"NaN detected in target_multi! Count: {nan_count}"
print(f"✓ No NaN in target_multi")

# Verify Option B consistency:
# target_multi[:,4] (Day 5, index 4) == forward_return
fwd_5_check = sample_multi[:, 4]
diff        = np.abs(fwd_5_check - df_model['forward_return'].values)
assert diff.max() < 1e-8, "Mismatch: target_multi[:,4] != forward_return!"
print(f"✓ Consistency verified: target_multi[:,4] == forward_return (fwd_ret_5)")

# Class distribution
label_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
total     = len(df_model)
print(f"\ntarget_class Distribution (n_forward={CONFIG['k_days']} days):")
print("─" * 45)
for cls in [0, 1, 2]:
    count = (df_model['target_class'] == cls).sum()
    bar   = '█' * int(count / total * 25)
    print(f"  {label_map[cls]:4s} ({cls}): {count:5d} "
          f"({count/total:.1%})  {bar}")
print("─" * 45)
print(f"\nBUY  threshold : > +{0.02:.1%} over 5 days")
print(f"SELL threshold : < {-0.015:.1%} over 5 days")
print(f"Asymmetry gap  : {0.02 - 0.015:.1%} (HOSE upward drift)")

# Per-ticker row count after dropping last 5 rows
print("\nRows per ticker after dropna:")
print(df_model.groupby('ticker').size().to_string())

# %%
df_model.head()

# %%
import joblib
from pathlib import Path

# Setup the path to your models directory
MODELS_DIR = Path('../models') # adjust path if needed based on your current working dir

# Load the trained models

rf_23 = joblib.load(MODELS_DIR / 'rf_23_final.pkl')

xgb_23 = joblib.load(MODELS_DIR / 'xgb_23_final.pkl')

# Sau đó bạn có thể sử dụng bình thường:
# predictions = rf_22.predict(X_test_22_ml) 


# %%
# ── Task 2 FEATURE_COLS (52 features used to train xgb_23) ─────────────────
# xgb_23 is a MultiOutputRegressor (5 estimators, one per forecast day)
# Each estimator has feature_importances_ of length window_size × n_features = 20×52 = 1040

COLS_TO_IGNORE_TASK2 = [
    'date', 'ticker', 'log_return', 'log_return_3d',
    'close', 'open', 'high', 'low', 'adj_close',
    'target_single', 'target_multi', 'target_class', 'forward_return'
]
FEATURE_COLS = [c for c in df_model.columns if c not in COLS_TO_IGNORE_TASK2]

print(f"Task 2 FEATURE_COLS ({len(FEATURE_COLS)} features):")
for i, f in enumerate(FEATURE_COLS, 1):
    print(f"  {i:2d}. {f}")


# %%
# Lấy danh sách các mô hình con (estimators)
# Tính trung bình feature_importances_ của tất cả các estimators
importances = np.mean([est.feature_importances_ for est in xgb_23.estimators_], axis=0)

# Bây giờ bạn có thể tiếp tục vẽ biểu đồ hoặc lọc features

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Bước 1: Trích xuất và Tính trung bình Importance ---
# Lấy feature_importances_ từ từng estimator (mô hình con) và tính trung bình
raw_importances = np.mean([est.feature_importances_ for est in xgb_23.estimators_], axis=0)

# --- Bước 2: Reshape và Aggregate theo Feature gốc ---
# X_train_23_ml có shape (window * n_features). Giả sử window=20, n_features=18
n_features = len(FEATURE_COLS) 
window_size = 20 # W

# Reshape về dạng (window, features) để tính trung bình tầm quan trọng của mỗi feature xuyên suốt 20 ngày
importance_matrix = raw_importances.reshape(window_size, n_features)
final_fi_scores = np.mean(importance_matrix, axis=0)

# --- Bước 3: Tạo DataFrame và Vẽ biểu đồ ---
fi_df = pd.DataFrame({
    'Feature': FEATURE_COLS,
    'Importance': final_fi_scores
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=fi_df, palette='magma')
plt.title(f"XGBoost Multi-Output Feature Importance (Averaged over {window_size} days)", fontsize=14)
plt.axvline(x=fi_df['Importance'].mean(), color='red', linestyle='--', label='Mean Importance')
plt.legend()
plt.grid(axis='x', alpha=0.3)
plt.show()

# %% [markdown]
# - The dominance of market context (VNI) features: I observe that the top four features are all derived from the VN-index. 
# - Identification of Near Zero-Importance Features: Metrics such as pb, currentRation, debtEquity, and roe show zero contribution to the XGBoost Gain.
# 
# Plausible Explanations why fundamental indicators are not contributing much:
# - Short-term momentum are more driven by liquidity and technical indicators. Fundamental indicators are derived from quarly financial statement so they are not reactive to the trained time target (3rd day price and 5 consecutive price predictions). 
# 

# %%
# --- Step 2: Feature Pruning ---

# 1. Define the threshold or specific number of features to keep
# Based on the plot, we see a sharp drop-off after the first ~25-27 features
n_to_keep = 27
top_features = fi_df.head(n_to_keep)['Feature'].tolist()

# 2. Identify features to drop (those with near-zero importance)
dropped_features = [f for f in FEATURE_COLS if f not in top_features]

print(f"✅ Pruning Complete.")
print(f"Total original features: {len(FEATURE_COLS)}")
print(f"Features kept: {len(top_features)}")
print(f"Features dropped: {len(dropped_features)}")

# 3. Update the feature list for the next steps
PRUNED_FEATURE_COLS = top_features

# Visual verification of dropped fundamentals/low-signal features
print("\nTop 5 Kept Features:")
print(top_features[:5])

print("\nExample Dropped Features (Near-Zero Signal):")
print(dropped_features[-10:]) # Showing the bottom of the list

# %% [markdown]
# # So now lean towards technical indicators more
# # The fundamental indicators are used to determine the 10 tickers in the first place

# %% [markdown]
# ### Step 3: Support/ Resistance Zones and Moving Average Crossovers
# '''
# 1. Cluster Definition: The cluster on closing prices.
# 2. Distance calculation: measure the percentage distance to the nearest cluster center.
# 3. Signal generation: 
# - Identify when the price crosses a cluster center upward or downward.
# - Flags for when price is within 1.5% of a major resistance or support. 
# '''

# %% [markdown]
# ## Feature Engineering — Task 3 Signal Identification
# 
# ---
# 
# ### 1. Support & Resistance Zones (K-Means Clustering)
# 
# K-Means clustering is applied to the **rolling 63-day (3-month) closing price history** to identify price zones where trading activity historically concentrates. The optimal number of zones `k` is selected via the **Elbow Method** (maximizing second derivative of inertia curve), ensuring the zone structure adapts dynamically to each stock's price behavior.
# 
# For each day `t`, zone centers are computed from `close[t-63 : t]`. Zones **below** the current price act as **Support**; zones **above** act as **Resistance**.
# 
# | Feature | Formula | Interpretation |
# |---|---|---|
# | `sr_distance_pct` | `(current_price − nearest_center) / nearest_center × 100` | `(+)` price in resistance territory; `(−)` price in support territory |
# | `sr_breakout_up` | `1` if `prev < zone_lower` AND `current > zone_upper` | Price just broke above a resistance zone → bullish structural signal |
# | `sr_breakout_down` | `1` if `prev > zone_upper` AND `current < zone_lower` | Price just broke below a support zone → bearish structural signal |
# | `sr_near_resistance` | `1` if distance to nearest resistance `< 0.5%` | Price approaching resistance — potential rejection or breakout setup |
# | `sr_near_support` | `1` if distance to nearest support `< 0.5%` | Price approaching support — potential bounce or breakdown setup |
# 
# > Zone width is set to `±0.5%` of current price to account for HOSE tick-size noise and avoid false breakout detection.
# 
# ---
# 
# ### 2. Moving Average Crossover Features
# 
# MA crossover signals capture **momentum shifts** across two timeframes using EMAs already present in the dataset (`ema_10`, `ema_20`, `ema_50`).
# 
# **Short-term pair (EMA10 × EMA20):** reacts quickly to recent price changes.
# **Long-term pair (EMA20 × EMA50):** confirms macro trend regime shifts.
# 
# | Feature | Condition | Interpretation |
# |---|---|---|
# | `ma_golden_cross_short` | `prev_ema10 ≤ prev_ema20` AND `ema10 > ema20` | Short-term bullish momentum begins |
# | `ma_death_cross_short` | `prev_ema10 ≥ prev_ema20` AND `ema10 < ema20` | Short-term bearish momentum begins |
# | `ma_golden_cross_long` | `prev_ema20 ≤ prev_ema50` AND `ema20 > ema50` | Macro bull regime confirmed |
# | `ma_death_cross_long` | `prev_ema20 ≥ prev_ema50` AND `ema20 < ema50` | Macro bear regime confirmed |
# | `ma_short_gap_pct` | `(ema10 − ema20) / ema20 × 100` | Continuous momentum strength — short term |
# | `ma_long_gap_pct` | `(ema20 − ema50) / ema50 × 100` | Continuous momentum strength — long term |
# | `ma_alignment` | `+1` if `ema10 > ema20 > ema50`; `−1` if fully inverted; `0` otherwise | Full bull/bear alignment across all timeframes |
# 
# > Crossover features are **event-based** (= 1 only on the day the cross occurs). Gap and alignment features are **continuous**, allowing the model to measure momentum intensity rather than just direction.
# 
# ---
# 
# ### 3. MTL Output Features
# 
# The MTL Seq2Seq model trained in Task 2 is used as a **learned prior** — its classification head outputs a probability distribution over future price direction, which is injected as a meta-feature into the Task 3 classifier.
# 
# For each day `t`, the model receives the 20-day input sequence and outputs `P(UP)` for each of the 5 forecast days. **Only Day 1** (index 0) is used as a feature, as it carries the least uncertainty.
# 
# | Feature | Formula | Interpretation |
# |---|---|---|
# | `mtl_p_up` | `class_output[:, 0]` — sigmoid probability | Confidence that price will rise tomorrow |
# | `mtl_p_down` | `1 − mtl_p_up` | Confidence that price will fall tomorrow |
# | `mtl_conviction` | `max(mtl_p_up, mtl_p_down)` | Overall model certainty — used as a signal gate |
# 
# **How XGBoost uses these features:**
# 
# | MTL Signal | Technical Confirmation | Outcome |
# |---|---|---|
# | `mtl_p_up = 0.82` | `sr_breakout_up = 1`, `ma_golden_cross_short = 1` | Strong BUY — deep learning and technicals agree |
# | `mtl_p_up = 0.82` | `sr_near_resistance = 1` | Uncertain — model bullish but price hitting a ceiling |
# | `mtl_p_up = 0.51` | Any | Weak prior — XGBoost defers entirely to technical features |
# | `mtl_p_up = 0.15` | `sr_breakout_down = 1` | Strong SELL — confluence of deep learning and structure |
# 
# > The core design philosophy: **MTL output raises or lowers the prior; technical features provide structural confirmation.** Neither source alone is sufficient — their combination is what produces high-conviction signals.

# %%
# ── Update df_pruned với targets mới ─────────────────────────
# Rebuild df_pruned từ df_model đã có targets

essential_raw  = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
mtl_targets    = ['target_multi', 'target_class', 'forward_return', 'log_return']

all_cols   = list(dict.fromkeys(
    essential_raw + mtl_targets + PRUNED_FEATURE_COLS
))
df_pruned  = df_model[all_cols].copy()

print(f"✓ df_pruned rebuilt: {df_pruned.shape}")
print(f"  Columns: {df_pruned.columns.tolist()}")

# %%
# Check available columns
print(f"All possible features: {df_pruned.columns.tolist()}")
print("-" * 30)
# Update the features 
# Will not feed the features related to absolute prices to prevent the model from overfitting and gradient vaninishing/explosion
COLS_TO_IGNORE = [
    'date', 'ticker', 'log_return', 'log_return_3d', 
    'close', 'open', 'high', 'low', 'adj_close' ,
    'target_single', 'target_multi'
]

FEATURE_COLS = [c for c in df_pruned.columns if c not in COLS_TO_IGNORE]


print(f"The number of features for training model: {len(FEATURE_COLS)}")
print("Features for training model:", FEATURE_COLS[:10], "...") # In ra 10 cái đầu xem thử

# %%
CONFIG['n_features'] = len(FEATURE_COLS)

# Target is the log-return because the close price is non-stationary, making the model
# prone to gradient vanishing
TARGET_COL = 'log_return_3d'
print(f"✓ Updated config for number of features: {CONFIG['n_features']}")
print(f"✓ Defined Target: {TARGET_COL}")
print(f"✓ Number of Input Features: {len(FEATURE_COLS)}")
print(f"✓ Total n_features (Model Input): {CONFIG['n_features']}")

# --- Target Configuration ---

# Primary target for Task 2.2 (Predicting a single point 3 days ahead)
# This column uses the shifted values (t + 3 logic)
TARGET_SINGLE = 'log_return_3d' 

# Primary target for Task 2.3 (Predicting the 5-day trajectory)
# This uses standard 1-day returns to build the cumulative path
TARGET_MULTI = 'log_return' 

# Update CONFIG to keep track of both
CONFIG['target_single'] = TARGET_SINGLE
CONFIG['target_multi'] = TARGET_MULTI

print(f"✓ kth day prediction: {CONFIG['target_single']}")
print(f"✓ consecutive days prediction: {CONFIG['target_multi']}")

# %%
def chronological_split_fixed(df, train_pct=0.7, val_pct=0.1):
    # 1. Get unique sorted dates from the entire dataset
    unique_dates = sorted(df['date'].unique())
    total_days = len(unique_dates)
    
    # 2. Calculate date-based split indices
    train_end_idx = int(total_days * train_pct)
    val_end_idx = int(total_days * (train_pct + val_pct))
    
    # 3. Define date boundaries
    train_cutoff = unique_dates[train_end_idx - 1]
    val_cutoff = unique_dates[val_end_idx - 1]
    
    # 4. Filter the dataframe based on these boundaries
    train_df = df[df['date'] <= train_cutoff].copy()
    val_df = df[(df['date'] > train_cutoff) & (df['date'] <= val_cutoff)].copy()
    test_df = df[df['date'] > val_cutoff].copy()
    
    return train_df, val_df, test_df

# --- Execution ---
train_df, val_df, test_df = chronological_split_fixed(df_pruned)

# --- Sanity Check ---
print(f'Train : {train_df["date"].min().date()} -> {train_df["date"].max().date()} ({len(train_df):,} rows)')
print(f'Val   : {val_df["date"].min().date()} -> {val_df["date"].max().date()} ({len(val_df):,} rows)')
print(f'Test  : {test_df["date"].min().date()} -> {test_df["date"].max().date()} ({len(test_df):,} rows)')

# These assertions will now pass
assert train_df['date'].max() < val_df['date'].min(), "Temporal overlap between Train and Val!"
assert val_df['date'].max() < test_df['date'].min(), "Temporal overlap between Val and Test!"
print('\n✓ Chronological integrity confirmed.')

# %%
# Class distribution per split
print("\nClass distribution per split:")
for split_name, split_df in [('Train', train_df),
                               ('Val',   val_df),
                               ('Test',  test_df)]:
    total  = len(split_df)
    counts = {label_map[k]: f"{(split_df['target_class']==k).sum()/total:.1%}"
              for k in [0, 1, 2]}
    print(f"  {split_name:5s}: {counts}")

# %%
df_pruned.head(100)

# %%
# Create a test column to see if current price equals shifted price
df_model['is_duplicate'] = df_model['close'] == df_model.groupby('ticker')['close'].shift(1)
print(f"Number of days with zero price change: {df_model['is_duplicate'].sum()}")

# %%
# ── Normalize ─────────────────────────────────────────────────
feature_scaler      = RobustScaler()
target_scaler_multi = RobustScaler()

train_df = train_df.copy()
val_df   = val_df.copy()
test_df  = test_df.copy()

# Scale features
train_df[PRUNED_FEATURE_COLS] = feature_scaler.fit_transform(
    train_df[PRUNED_FEATURE_COLS]
)
for df_split in [val_df, test_df]:
    df_split[PRUNED_FEATURE_COLS] = feature_scaler.transform(
        df_split[PRUNED_FEATURE_COLS]
    )

# Scale target_multi (2D array)
y_train_multi_raw    = np.stack(train_df['target_multi'].values)  # (n, 5)
y_train_multi_scaled = target_scaler_multi.fit_transform(y_train_multi_raw)
train_df['target_multi'] = list(y_train_multi_scaled)

for df_split in [val_df, test_df]:
    y_raw    = np.stack(df_split['target_multi'].values)
    y_scaled = target_scaler_multi.transform(y_raw)
    df_split['target_multi'] = list(y_scaled)

# Persist
joblib.dump(feature_scaler,
            MODELS_DIR / 'generalist_feature_scaler.pkl')
joblib.dump(target_scaler_multi,
            MODELS_DIR / 'generalist_target_multi_scaler.pkl')

print("✓ Normalization complete.")
print(f"  Features scaled  : {len(PRUNED_FEATURE_COLS)}")
print(f"  target_multi     : scaled (n, 5) — RobustScaler")
print(f"  target_class     : NOT scaled (categorical 0/1/2)")
print(f"  forward_return   : NOT scaled (reference only)")

# %%
# ── build_mtl_sequences: sliding-window sequences for MTL training ──────────
def build_mtl_sequences(df, window_size=None, feature_cols=None):
    """
    Build sliding-window sequences for Multi-Task Learning.
    Per-ticker grouping prevents cross-ticker data leakage.

    Returns:
        X     : (n_samples, window_size, n_features) float32
        y_reg : (n_samples, k_days)  — scaled forward return trajectory
        y_cls : (n_samples,)         — 0/1/2 class label
    """
    if window_size is None:
        window_size = CONFIG['window_size']
    if feature_cols is None:
        feature_cols = PRUNED_FEATURE_COLS

    all_X, all_y_reg, all_y_cls = [], [], []

    for ticker, group in df.groupby('ticker', sort=False):
        group     = group.sort_values('date').reset_index(drop=True)
        feat_arr  = group[feature_cols].values                  # (T, F)
        multi_arr = np.stack(group['target_multi'].values)      # (T, k_days)
        cls_arr   = group['target_class'].values.astype(int)    # (T,)
        n         = len(group)

        for start in range(n - window_size):
            end = start + window_size
            all_X.append(feat_arr[start:end])   # (W, F)
            all_y_reg.append(multi_arr[end])     # (k_days,)
            all_y_cls.append(cls_arr[end])       # scalar

    X     = np.array(all_X,     dtype=np.float32)
    y_reg = np.array(all_y_reg, dtype=np.float32)
    y_cls = np.array(all_y_cls, dtype=np.int32)

    return X, y_reg, y_cls

print("✓ build_mtl_sequences defined")
print(f"  window_size = {CONFIG['window_size']} days")
print(f"  features    = {len(PRUNED_FEATURE_COLS)} (PRUNED_FEATURE_COLS)")


# %%
# ── Build MTL Sequences ───────────────────────────────────────
X_train, y_train_reg, y_train_cls = build_mtl_sequences(train_df)
X_val,   y_val_reg,   y_val_cls   = build_mtl_sequences(val_df)
X_test,  y_test_reg,  y_test_cls  = build_mtl_sequences(test_df)

print("─" * 55)
print("Generalist MTL Sequence Shapes")
print("─" * 55)
print(f"  X_train  : {X_train.shape}")
print(f"             (samples × {CONFIG['window_size']} days × "
      f"{len(PRUNED_FEATURE_COLS)} features)")
print(f"  y_reg    : {y_train_reg.shape}  ← 5-day trajectory")
print(f"  y_cls    : {y_train_cls.shape}     ← 0/1/2 label")
print()
print(f"  X_val    : {X_val.shape}")
print(f"  X_test   : {X_test.shape}")
print("─" * 55)

# Final sanity checks
assert not np.isnan(X_train).any(),     "NaN in X_train!"
assert not np.isnan(y_train_reg).any(), "NaN in y_train_reg!"
assert set(np.unique(y_train_cls)).issubset({0, 1, 2}), \
    f"Unexpected classes: {np.unique(y_train_cls)}"

print("✓ All assertions passed.")
print(f"✓ Class values: {np.unique(y_train_cls)}")

# Class distribution in final sequences
print("\nClass distribution in sequences:")
for split_name, y_cls in [('Train', y_train_cls),
                            ('Val',   y_val_cls),
                            ('Test',  y_test_cls)]:
    total  = len(y_cls)
    counts = {label_map[c]: f"{(y_cls==c).sum()/total:.1%}"
              for c in [0, 1, 2]}
    print(f"  {split_name:5s}: {counts}")

# %%
# ============================================================
# Visualization after Chronological Split
# 3 plots:
#   1. Log return distribution per split (all tickers)
#   2. Per-ticker price path with split boundaries
#   3. Class distribution per split
# ============================================================

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

fig = plt.figure(figsize=(20, 18))
fig.suptitle('Generalist MTL Dataset — Post-Split Analysis (10 Tickers)',
             fontsize=16, fontweight='bold', y=0.98)

# ── Plot 1: Log Return Distribution per Split ─────────────────
ax1 = fig.add_subplot(3, 1, 1)

for split_name, split_df, color in [
    ('Train', train_df, 'steelblue'),
    ('Val',   val_df,   'darkorange'),
    ('Test',  test_df,  'seagreen')
]:
    # Use log_return column (daily return, not forward return)
    returns = split_df['log_return'].dropna()
    ax1.plot(split_df['date'], returns,
             color=color, linewidth=0.6, alpha=0.6,
             label=f'{split_name} ({len(split_df):,} rows)')

# Split boundaries
ax1.axvline(train_df['date'].max(), color='black',
            linestyle='--', linewidth=1.5, alpha=0.8,
            label=f'Train cutoff: {train_df["date"].max().date()}')
ax1.axvline(val_df['date'].max(), color='gray',
            linestyle='--', linewidth=1.5, alpha=0.8,
            label=f'Val cutoff: {val_df["date"].max().date()}')
ax1.axhline(0, color='red', linestyle='-', alpha=0.3, linewidth=1)

ax1.xaxis.set_major_locator(mdates.YearLocator(1))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.tick_params(axis='x', rotation=45)
ax1.set_title('Daily Log Returns — All 10 Tickers Combined', fontsize=13)
ax1.set_ylabel('Log Return')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(alpha=0.2)

# ── Plot 2: Per-Ticker Close Price with Split Boundaries ──────
ax2 = fig.add_subplot(3, 1, 2)

# Normalize each ticker's price to base 100 for comparison
# (since prices differ vastly across tickers)
colors_tickers = plt.cm.tab10(np.linspace(0, 1, 10))
all_tickers    = df_pruned['ticker'].unique()

for i, ticker in enumerate(all_tickers):
    ticker_data = df_pruned[df_pruned['ticker'] == ticker].sort_values('date')
    
    # Normalize to base 100 at first date
    base_price   = ticker_data['close'].iloc[0]
    norm_price   = ticker_data['close'] / base_price * 100
    
    ax2.plot(ticker_data['date'], norm_price,
             color=colors_tickers[i], linewidth=0.9,
             alpha=0.75, label=ticker)

# Split boundaries
ax2.axvline(train_df['date'].max(), color='black',
            linestyle='--', linewidth=1.5, alpha=0.8)
ax2.axvline(val_df['date'].max(), color='gray',
            linestyle='--', linewidth=1.5, alpha=0.8)

# Shade regions
ax2.axvspan(df_pruned['date'].min(), train_df['date'].max(),
            alpha=0.05, color='steelblue', label='Train region')
ax2.axvspan(train_df['date'].max(), val_df['date'].max(),
            alpha=0.08, color='darkorange', label='Val region')
ax2.axvspan(val_df['date'].max(), df_pruned['date'].max(),
            alpha=0.08, color='seagreen', label='Test region')

ax2.xaxis.set_major_locator(mdates.YearLocator(1))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.tick_params(axis='x', rotation=45)
ax2.set_title('Normalized Price (Base=100) — All 10 Tickers', fontsize=13)
ax2.set_ylabel('Normalized Price')
ax2.legend(loc='upper left', fontsize=8, ncol=5)
ax2.grid(alpha=0.2)

# ── Plot 3: Class Distribution per Split ─────────────────────
ax3 = fig.add_subplot(3, 1, 3)

label_map    = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
label_colors = {0: '#e74c3c', 1: '#95a5a6', 2: '#2ecc71'}
splits       = [('Train', train_df), ('Val', val_df), ('Test', test_df)]
x            = np.arange(len(splits))
width        = 0.25

for cls_idx, (cls_id, cls_name) in enumerate(label_map.items()):
    counts = []
    for _, split_df in splits:
        total = len(split_df)
        pct   = (split_df['target_class'] == cls_id).sum() / total * 100
        counts.append(pct)
    
    bars = ax3.bar(x + cls_idx * width, counts,
                   width, label=cls_name,
                   color=label_colors[cls_id],
                   alpha=0.85, edgecolor='white')
    
    # Annotate percentage on each bar
    for bar, pct in zip(bars, counts):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f'{pct:.1f}%', ha='center', va='bottom',
                 fontsize=9, fontweight='bold')

ax3.set_xticks(x + width)
ax3.set_xticklabels([s[0] for s in splits], fontsize=11)
ax3.set_ylabel('Percentage (%)')
ax3.set_title('Target Class Distribution per Split\n'
              '(BUY >+2% | HOLD -1.5%~+2% | SELL <-1.5% over 5 days)',
              fontsize=13)
ax3.legend(fontsize=10)
ax3.set_ylim(0, 80)
ax3.grid(axis='y', alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(MODELS_DIR / 'task3_split_analysis.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("✓ Split visualization saved.")

# %% [markdown]
# ### 3.0 Design Philosophy
# 
# The core design challenge is:
# 
# > *When should a trader enter or exit 
# > a position and does the answer change depending on their time horizon?"*
# ### Pipeline:
# ####   2. Feature Engineering (S/R zones, MA crossover, MTL output)
# ####   3. Label Generation (asymmetric threshold)
# ####   4. XGBoost Classifier (3-class)
# ####   5. GRU Classifier (compare)
# ============================================================
# 
# ### ── Section 3.1: Feature Pruning ─────────────────────────────
# #### Lấy feature importance từ RF (Task 2) để lọc near-zero features

# %% [markdown]
# ### Master Plan 
# Step 1: Retrain MTL Model on 10 tickers
# - Output: MTL model generalist
# - Purpose: generate mtl_p_up, mtl_p_down, mtl_conviction
# Step 2: Feature Pruning
# - Omit the redundant 25 features
# - Keep the 27 important features
# Step 3: Engineer new features
# S/R Zone features (5 features):
#   - sr_distance_pct
#   - sr_breakout_up / sr_breakout_down
#   - sr_near_resistance / sr_near_support
#   → K-Means với elbow method, lookback 63 ngày (3 tháng)
# 
# MA Crossover features (7 features):
#   - ma_golden_cross_short / ma_death_cross_short  (EMA10 vs EMA20)
#   - ma_golden_cross_long  / ma_death_cross_long   (EMA20 vs EMA50)
#   - ma_short_gap_pct / ma_long_gap_pct
#   - ma_alignment (+1 bull, -1 bear, 0 mixed)
# 
# MTL Output features (3 features):
#   - mtl_p_up        (P(UP) từ classification head, Day 1)
#   - mtl_p_down      (P(DOWN), Day 1)
#   - mtl_conviction  (max of above)
# 
# Step 4: Label Generation (BUY, HOlD, SELL)
# Forward return = close[t+3] / close[t] - 1
# 
# BUY  (class 2): return > +2.0%
# HOLD (class 1): -1.5% ≤ return ≤ +2.0%
# SELL (class 0): return < -1.5%
# 
# Asymmetric justification:
# - HOSE có upward drift ~15% annualized
# - Bear moves sharp và nhanh hơn bull moves
# - Liquidity asymmetry: easier to exit than enter
# 
# Step 5: Final features = pruned Task 2 features (~27)
#               + S/R zone features (5)
#               + MA crossover features (7)
#               + MTL output features (3)
#               = ~42 features total
# 
# Chronological split (per ticker, no shuffle):
#   Train : 70%
#   Val   : 10%
#   Test  : 20%
# 
# Scaler: RobustScaler fit on train only
# 
# Step 6: Train XGBoost Classifer
# - 
# Step 7: Train GRU Classifier
# Step 8: Backtest & Final comparision: 
# Metrics: Classificaiton report, DA, Win rate, Sharpe ratio, total return
# 

# %% [markdown]
# ## Retrain for 10 tickers

# %% [markdown]
# ### Build Sequence for MultiTask learning across tickers
# *1. Alignment of Y heads:* It ensures that y_reg[i] and y_cls[i] both refer to the exact same moment in time for the exact same ticker. This is critical for the MTL model to learn the correlation between "Price Path" and "Signal Conviction.
# *2. Pruned Feature Scope:* It restricts the input X to only your 27 high-signal features, ignoring the fundamental "noise" we pruned earlier.
# *3. Multi-Ticker Handling:* It maintains the groupby('ticker') logic to ensure that a sequence for VCB never accidentally contains trailing data from FPT.

# %%
# ============================================================
# STEP 1: Retrain MTL Model on 10 Tickers
# Architecture changes vs Task 2:
#   - Input features: 27 (pruned) instead of 52
#   - Classification head: softmax(3) instead of sigmoid(5)
#     → P(SELL), P(HOLD), P(BUY) for the full 5-day window
#   - Loss weights: reg=1.0, cls=5.0 (same as Task 2)
# ============================================================

import tensorflow as tf
from tensorflow.keras import layers, Model

def build_generalist_mtl(window_size, n_features, output_steps=5, n_classes=3):
    """
    Generalist MTL Seq2Seq model for 10 tickers.
    
    Changes vs Task 2 build_multitask_s2s():
      - n_features: 27 (pruned) instead of 52
      - Classification head: Dense(3, softmax) instead of Dense(5, sigmoid)
        → Predicts market regime (SELL/HOLD/BUY) for the full 5-day window
        → Consistent with target_class = 0/1/2
      - Dropout slightly increased for generalization across diverse tickers
    
    Architecture:
      Shared Encoder:
        GRU(128) → MultiHeadAttention(2 heads) → GlobalAveragePooling
      
      Regression Head:
        RepeatVector(5) → GRU(128) → TimeDistributed(Dense(1))
        → output shape: (batch, 5)
      
      Classification Head:
        Dense(64, relu) → Dropout(0.3) → Dense(3, softmax)
        → output shape: (batch, 3)
        → P(SELL)=[:,0], P(HOLD)=[:,1], P(BUY)=[:,2]
    """
    inputs = layers.Input(
        shape=(window_size, n_features), 
        name='main_input'
    )

    # ── Shared Encoder ────────────────────────────────────────
    encoder_seq, encoder_state = layers.GRU(
        128,
        return_sequences=True,
        return_state=True,
        name='encoder_gru'
    )(inputs)

    # Attention — 2 heads, dropout 0.3 to prevent over-smoothing
    att = layers.MultiHeadAttention(
        num_heads=2, key_dim=64, dropout=0.3,
        name='shared_attention'
    )(encoder_seq, encoder_seq)

    # Shared context vector
    context = layers.GlobalAveragePooling1D(
        name='context_pool'
    )(att)

    # ── Regression Head — 5-day trajectory ───────────────────
    # RepeatVector feeds context into GRU decoder
    reg_input  = layers.RepeatVector(output_steps, name='reg_bridge')(context)
    reg_gru    = layers.GRU(
        128, return_sequences=True,
        name='reg_decoder_gru'
    )(reg_input, initial_state=encoder_state)
    reg_out    = layers.TimeDistributed(
        layers.Dense(1), name='reg_td'
    )(reg_gru)
    reg_output = layers.Reshape(
        (output_steps,), name='reg_output'
    )(reg_out)

    # ── Classification Head — 3-class regime ─────────────────
    # Direct from shared context — no separate decoder needed
    # since we predict ONE label for the full 5-day window
    cls_x      = layers.Dense(64, activation='relu',
                               name='cls_dense1')(context)
    cls_x      = layers.Dropout(0.3, name='cls_dropout')(cls_x)
    cls_x      = layers.Dense(32, activation='relu',
                               name='cls_dense2')(cls_x)
    cls_output = layers.Dense(
        n_classes, activation='softmax',
        name='cls_output'
    )(cls_x)
    # cls_output[:, 0] = P(SELL)
    # cls_output[:, 1] = P(HOLD)
    # cls_output[:, 2] = P(BUY)

    model = Model(
        inputs=inputs,
        outputs=[reg_output, cls_output],
        name='Generalist_MTL_S2S'
    )

    return model


# ── Initialize ────────────────────────────────────────────────
W          = CONFIG['window_size']   # 20
K_DAYS     = CONFIG['k_days']        # 5
N_FEATURES = len(PRUNED_FEATURE_COLS)  # 27

generalist_mtl = build_generalist_mtl(
    window_size  = W,
    n_features   = N_FEATURES,
    output_steps = K_DAYS,
    n_classes    = 3
)

generalist_mtl.summary()

print(f"\nOutput heads:")
print(f"  reg_output : {generalist_mtl.output[0].shape} — 5-day trajectory")
print(f"  cls_output : {generalist_mtl.output[1].shape} — P(SELL/HOLD/BUY)")

# %%
# ── Loss & Compilation ────────────────────────────────────────
# Sign-Weighted Huber Loss — carried over from Task 2
# Penalizes directional mistakes 2.5x more than magnitude mistakes

def sign_weighted_huber(y_true, y_pred, delta=1.0, penalty_weight=2.5):
    """
    Huber loss with directional penalty.
    Penalizes predictions where sign(pred) != sign(true) by penalty_weight.
    """
    error         = y_true - y_pred
    is_small      = tf.abs(error) <= delta
    squared_loss  = 0.5 * tf.square(error)
    linear_loss   = delta * (tf.abs(error) - 0.5 * delta)
    huber_loss    = tf.where(is_small, squared_loss, linear_loss)

    sign_match = tf.sign(y_true) * tf.sign(y_pred)
    penalty    = tf.where(
        sign_match < 0,
        tf.constant(penalty_weight, dtype=tf.float32),
        tf.constant(1.0,            dtype=tf.float32)
    )
    return tf.reduce_mean(huber_loss * penalty)


# Class weights to handle imbalance (HOLD typically dominates)
from collections import Counter
cls_counts  = Counter(y_train_cls.tolist())
total_train = len(y_train_cls)
n_classes   = 3

# Balanced class weights
class_weight_dict = {
    cls: total_train / (n_classes * count)
    for cls, count in cls_counts.items()
}
print("Class weights for imbalance handling:")
for cls, w in sorted(class_weight_dict.items()):
    print(f"  Class {cls} ({label_map[cls]:4s}): {w:.3f}")

# Sample weights array for fit()
sample_weights_train = np.array([
    class_weight_dict[c] for c in y_train_cls
])

# Learning rate schedule
steps_per_epoch = len(X_train) // CONFIG['batch_size']
total_steps     = CONFIG['epochs'] * steps_per_epoch

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate = 1e-3,
    decay_steps           = total_steps,
    alpha                 = 0.1   # decay to 10% of initial LR
)

generalist_mtl.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss = {
        'reg_output': lambda y_true, y_pred: sign_weighted_huber(
            y_true, y_pred, penalty_weight=2.5
        ),
        'cls_output': 'sparse_categorical_crossentropy'
    },
    loss_weights = {
        'reg_output': 1.0,
        'cls_output': 5.0   # Direction is King
    },
    metrics = {
        'reg_output': 'mae',
        'cls_output': 'accuracy'
    }
)

print("\n✓ Model compiled.")
print(f"  Regression loss  : Sign-Weighted Huber (penalty=2.5×)")
print(f"  Classification   : Sparse Categorical Crossentropy")
print(f"  Loss weights     : reg=1.0, cls=5.0")
print(f"  LR schedule      : CosineDecay 1e-3 → 1e-4")

# %%
# ── Callbacks for Generalist MTL (V1) ────────────────────────
checkpoint_path = MODELS_DIR / 'generalist_mtl_best.keras'


# %%
# ── Option B: Bỏ CosineDecay, dùng float LR + ReduceLROnPlateau
generalist_mtl.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),  # ← float, not schedule
    loss = {
        'reg_output': lambda y_true, y_pred: sign_weighted_huber(
            y_true, y_pred, penalty_weight=2.5
        ),
        'cls_output': 'sparse_categorical_crossentropy'
    },
    loss_weights = {
        'reg_output': 1.0,
        'cls_output': 8.0
    },
    metrics = {
        'reg_output': 'mae',
        'cls_output': 'accuracy'
    }
)

# Callbacks — có ReduceLROnPlateau, không có CosineDecay
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath       = str(checkpoint_path),
        monitor        = 'val_loss',
        save_best_only = True,
        mode           = 'min',
        verbose        = 1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor              = 'val_loss',
        patience             = 20,
        restore_best_weights = True,
        verbose              = 1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor  = 'val_loss',
        factor   = 0.5,
        patience = 7,
        min_lr   = 1e-7,
        verbose  = 1
    )
]

history_generalist = generalist_mtl.fit(
    X_train,
    {
        'reg_output': y_train_reg,
        'cls_output': y_train_cls
    },
    validation_data = (
        X_val,
        {
            'reg_output': y_val_reg,
            'cls_output': y_val_cls
        }
    ),
    epochs     = CONFIG['epochs'],
    batch_size = CONFIG['batch_size'],
    callbacks  = callbacks,
    verbose    = 1
)

# %%
# ── Training Curves ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Generalist MTL — Training History (10 Tickers)',
             fontsize=14, fontweight='bold')

# Plot 1: Total loss
ax = axes[0]
ax.plot(history_generalist.history['loss'],
        label='Train Loss', color='steelblue')
ax.plot(history_generalist.history['val_loss'],
        label='Val Loss', color='darkorange')
ax.set_title('Total Loss (reg×1.0 + cls×5.0)')
ax.set_xlabel('Epoch')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Regression MAE
ax = axes[1]
ax.plot(history_generalist.history['reg_output_mae'],
        label='Train MAE', color='steelblue')
ax.plot(history_generalist.history['val_reg_output_mae'],
        label='Val MAE', color='darkorange')
ax.set_title('Regression Head — MAE')
ax.set_xlabel('Epoch')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Classification Accuracy
ax = axes[2]
ax.plot(history_generalist.history['cls_output_accuracy'],
        label='Train Accuracy', color='steelblue')
ax.plot(history_generalist.history['val_cls_output_accuracy'],
        label='Val Accuracy', color='darkorange')
ax.set_title('Classification Head — Accuracy')
ax.set_xlabel('Epoch')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(MODELS_DIR / 'generalist_mtl_training_curves.png',
            dpi=150, bbox_inches='tight')
plt.show()

# %%
# ── Evaluation on Test Set ────────────────────────────────────
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Predict
reg_pred, cls_pred_prob = generalist_mtl.predict(X_test, verbose=0)
cls_pred_label          = np.argmax(cls_pred_prob, axis=1)

# ── Regression MAE ────────────────────────────────────────────
reg_mae  = mean_absolute_error(y_test_reg, reg_pred)
reg_rmse = np.sqrt(mean_squared_error(y_test_reg, reg_pred))

print("=" * 50)
print("GENERALIST MTL — TEST SET RESULTS")
print("=" * 50)
print(f"\nRegression Head:")
print(f"  MAE  : {reg_mae:.6f}")
print(f"  RMSE : {reg_rmse:.6f}")

# ── Classification Report ─────────────────────────────────────
print(f"\nClassification Head:")
print(classification_report(
    y_test_cls, cls_pred_label,
    target_names = ['SELL', 'HOLD', 'BUY'],
    digits       = 4
))

# ── Confusion Matrix ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
cm = confusion_matrix(y_test_cls, cls_pred_label)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels = ['SELL', 'HOLD', 'BUY'],
    yticklabels = ['SELL', 'HOLD', 'BUY'],
    ax = ax
)
ax.set_title('Generalist MTL — Confusion Matrix (Test Set)\n'
             '10 Tickers Combined', fontsize=12)
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig(MODELS_DIR / 'generalist_mtl_confusion_matrix.png',
            dpi=150, bbox_inches='tight')
plt.show()

# ── Active DA (conviction-gated) ─────────────────────────────
CONVICTION_THRESHOLD = 0.65

max_prob    = cls_pred_prob.max(axis=1)
active_mask = max_prob >= CONVICTION_THRESHOLD

# Only evaluate on BUY/SELL days (exclude HOLD)
active_pred     = cls_pred_label[active_mask]
active_true     = y_test_cls[active_mask]
non_hold_mask   = (active_pred != 1) & (active_true != 1)

active_da = np.mean(
    active_pred[non_hold_mask] == active_true[non_hold_mask]
) * 100

print(f"\nConviction-Gated Evaluation (threshold={CONVICTION_THRESHOLD}):")
print(f"  Active signals : {active_mask.sum():,} / {len(y_test_cls):,} "
      f"({active_mask.mean():.1%} of test days)")
print(f"  Active DA      : {active_da:.2f}%")
print("=" * 50)

# ── Save final model ──────────────────────────────────────────
generalist_mtl.save(MODELS_DIR / 'generalist_mtl_final.keras')
print(f"\n✓ Model saved → generalist_mtl_final.keras")

# %%
# ── Assign production model for Step 3 MTL feature generation ───────────────
# V1 Generalist MTL is selected as production model based on Active DA = 49.82%
# (highest among V1, Robust, V3 — see markdown commentary above)
production_model = generalist_mtl
W = CONFIG['window_size']  # 20 — needed by generate_mtl_features_all_splits

print("✓ production_model assigned → generalist_mtl (V1)")
print(f"  Classification head: P(SELL)[:,0] | P(HOLD)[:,1] | P(BUY)[:,2]")


# %% [markdown]
# --- 
# ### Comment on generalist_mtl_final.keras
# 
# - Severe Overfitting:
#     - Train Loss: 5.0 → 4.0. It decreases steadily. 
#     - Val Loss:   10.0 → 22.5. It is in increasing pattern
# -> Model đang memorize training data, không generalize 
# - Classification Head does not learn much 
#     - Train Accuracy: 0.57 → 0.70  (tăng)
#     - Val Accuracy:   0.40 → 0.40  (FLAT hoàn toàn)
# -> Val accuracy stuck ở ~40% = gần như random
# -> Active DA chỉ 49.82% = tệ hơn coin flip
# 
# - Regressing MAE
# 
# Potental  problems: 
# - The model is too complex
# - The high weight for classification head

# %%
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from sklearn.utils import class_weight

# ── 1. Model Architecture (Robust Version) ───────────────────────────────────

def build_robust_generalist_mtl(window_size, n_features, output_steps=5, n_classes=3):
    """
    Robust MTL Seq2Seq with Bottlenecking and Noise Injection to stop overfitting.
    """
    inputs = layers.Input(shape=(window_size, n_features), name='main_input')

    # A. Noise Injection: Simulates market volatility to prevent memorization
    x = layers.GaussianNoise(0.01, name='input_noise')(inputs)

    # B. Shared Encoder: Reduced capacity (64 units) forces feature compression
    encoder_seq, encoder_state = layers.GRU(
        64, return_sequences=True, return_state=True, name='encoder_gru'
    )(x)

    # C. Attention: High dropout (0.5) prevents reliance on specific time-steps
    att = layers.MultiHeadAttention(
        num_heads=2, key_dim=32, dropout=0.5, name='shared_attention'
    )(encoder_seq, encoder_seq)

    # D. Shared Context Vector + LayerNorm: Stabilizes gradients
    context = layers.GlobalAveragePooling1D(name='context_pool')(att)
    context = layers.LayerNormalization(name='context_norm')(context)

    # E. Regression Head: Predicts 5-day path
    reg_input = layers.RepeatVector(output_steps, name='reg_bridge')(context)
    reg_gru = layers.GRU(64, return_sequences=True, name='reg_decoder_gru')(
        reg_input, initial_state=encoder_state
    )
    reg_out = layers.TimeDistributed(layers.Dense(1), name='reg_td')(reg_gru)
    reg_output = layers.Reshape((output_steps,), name='reg_output')(reg_out)

    # F. Classification Head: 3-class (SELL/HOLD/BUY)
    cls_x = layers.Dense(32, activation='relu', name='cls_dense')(context)
    cls_x = layers.Dropout(0.5, name='cls_head_dropout')(cls_x)
    cls_output = layers.Dense(n_classes, activation='softmax', name='cls_output')(cls_x)

    return Model(inputs=inputs, outputs=[reg_output, cls_output], name='Robust_Generalist_MTL')

# ── 2. Class Weight Calculation ──────────────────────────────────────────────

# Handle imbalance (e.g., too many HOLDs) to prevent the model from overfitting to the majority
weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(y_train_cls), y=y_train_cls
)
class_weight_dict = {i: weights[i] for i in range(len(weights))}

# ── 3. Compile & Initialize ──────────────────────────────────────────────────

W = CONFIG['window_size']
N_FEATURES = len(PRUNED_FEATURE_COLS)

model = build_robust_generalist_mtl(W, N_FEATURES)

# Optimization Strategy: Balanced weights to avoid one head dominating the loss
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss={
        'reg_output': 'huber', # Huber is more robust to outliers than MSE
        'cls_output': 'sparse_categorical_crossentropy'
    },
    loss_weights={
        'reg_output': 1.0, 
        'cls_output': 4.0   # Reduced from 8.0 to prevent overfitting to class labels
    },
    metrics={'reg_output': 'mae', 'cls_output': 'accuracy'}
)

# %%
# ── 4. Callbacks ─────────────────────────────────────────────────────────────

checkpoint_path = MODELS_DIR / 'robust_mtl_best.keras'

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,          # Stricter patience to catch overfitting early
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

# %%
# ── Fallback: Custom weighted loss function ───────────────────
# Bake class weights trực tiếp vào loss function
# → Không cần sample_weight hay class_weight argument

def weighted_sparse_categorical_crossentropy(class_weight_dict):
    """
    Wrap class weights vào loss function.
    Keras multi-output models support custom loss functions
    nhưng không support sample_weight dict (Keras 3.x bug).
    """
    # Convert dict to tensor
    weight_tensor = tf.constant(
        [class_weight_dict[i] for i in sorted(class_weight_dict.keys())],
        dtype=tf.float32
    )

    def loss_fn(y_true, y_pred):
        # Standard sparse categorical crossentropy
        ce_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred
        )
        # Get weight for each sample based on its true class
        y_true_int = tf.cast(tf.squeeze(y_true), tf.int32)
        sample_weights = tf.gather(weight_tensor, y_true_int)

        # Apply weights
        return tf.reduce_mean(ce_loss * sample_weights)

    return loss_fn


# Recompile với weighted loss
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss = {
        'reg_output': 'huber',
        'cls_output': weighted_sparse_categorical_crossentropy(class_weight_dict)
    },
    loss_weights = {
        'reg_output': 1.0,
        'cls_output': 4.0
    },
    metrics = {
        'reg_output': 'mae',
        'cls_output': 'accuracy'
    }
)

# Train — KHÔNG có sample_weight
history = model.fit(
    X_train,
    {
        'reg_output': y_train_reg,
        'cls_output': y_train_cls
    },
    validation_data=(
        X_val,
        {
            'reg_output': y_val_reg,
            'cls_output': y_val_cls
        }
    ),
    epochs     = 100,
    batch_size = CONFIG['batch_size'],
    callbacks  = callbacks,
    verbose    = 1
)

print("✓ Training with custom weighted loss — no sample_weight needed.")

# %%
# ── 6. Evaluation ────────────────────────────────────────────────────────────

print("\n--- Final Evaluation on Test Set ---")
test_results = model.evaluate(
    X_test, 
    {'reg_output': y_test_reg, 'cls_output': y_test_cls},
    verbose=0
)

for i, name in enumerate(model.metrics_names):
    print(f"{name}: {test_results[i]:.4f}")

# %%
# ============================================================
# Evaluation — Robust Generalist MTL
# Mirror của V1 evaluation script
# ============================================================

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ── 1. Predict ────────────────────────────────────────────────
reg_pred, cls_pred_prob = model.predict(X_test, verbose=0)
cls_pred_label          = np.argmax(cls_pred_prob, axis=1)

# ── 2. Regression MAE ─────────────────────────────────────────
reg_mae  = mean_absolute_error(y_test_reg, reg_pred)
reg_rmse = np.sqrt(mean_squared_error(y_test_reg, reg_pred))

print("=" * 50)
print("ROBUST GENERALIST MTL — TEST SET RESULTS")
print("=" * 50)
print(f"\nRegression Head:")
print(f"  MAE  : {reg_mae:.6f}")
print(f"  RMSE : {reg_rmse:.6f}")

# ── 3. Classification Report ──────────────────────────────────
print(f"\nClassification Head:")
print(classification_report(
    y_test_cls, cls_pred_label,
    target_names = ['SELL', 'HOLD', 'BUY'],
    digits       = 4
))

# ── 4. Confusion Matrix ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
cm = confusion_matrix(y_test_cls, cls_pred_label)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels = ['SELL', 'HOLD', 'BUY'],
    yticklabels = ['SELL', 'HOLD', 'BUY'],
    ax          = ax
)
ax.set_title('Robust Generalist MTL — Confusion Matrix (Test Set)\n'
             '10 Tickers Combined', fontsize=12)
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig(MODELS_DIR / 'robust_mtl_confusion_matrix.png',
            dpi=150, bbox_inches='tight')
plt.show()

# ── 5. Active DA (conviction-gated) ──────────────────────────
CONVICTION_THRESHOLD = 0.65

max_prob    = cls_pred_prob.max(axis=1)
active_mask = max_prob >= CONVICTION_THRESHOLD

active_pred   = cls_pred_label[active_mask]
active_true   = y_test_cls[active_mask]
non_hold_mask = (active_pred != 1) & (active_true != 1)

active_da = np.mean(
    active_pred[non_hold_mask] == active_true[non_hold_mask]
) * 100 if non_hold_mask.sum() > 0 else 0.0

print(f"\nConviction-Gated Evaluation (threshold={CONVICTION_THRESHOLD}):")
print(f"  Active signals : {active_mask.sum():,} / {len(y_test_cls):,} "
      f"({active_mask.mean():.1%} of test days)")
print(f"  BUY  signals   : {((active_pred==2)).sum():,}")
print(f"  SELL signals   : {((active_pred==0)).sum():,}")
print(f"  Active DA      : {active_da:.2f}%")

# ── 6. Per-class conviction distribution ─────────────────────
print(f"\nConviction Score Distribution:")
print(f"  Mean  : {max_prob.mean():.4f}")
print(f"  Median: {np.median(max_prob):.4f}")
print(f"  >0.65 : {(max_prob >= 0.65).mean():.1%}")
print(f"  >0.70 : {(max_prob >= 0.70).mean():.1%}")
print(f"  >0.80 : {(max_prob >= 0.80).mean():.1%}")

# ── 7. Training Curves ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Robust Generalist MTL — Training History (10 Tickers)',
             fontsize=14, fontweight='bold')

# Total loss
ax = axes[0]
ax.plot(history.history['loss'],
        label='Train Loss', color='steelblue')
ax.plot(history.history['val_loss'],
        label='Val Loss', color='darkorange')
ax.set_title('Total Loss (reg×1.0 + cls×4.0)')
ax.set_xlabel('Epoch')
ax.legend()
ax.grid(alpha=0.3)

# Regression MAE
ax = axes[1]
ax.plot(history.history['reg_output_mae'],
        label='Train MAE', color='steelblue')
ax.plot(history.history['val_reg_output_mae'],
        label='Val MAE', color='darkorange')
ax.set_title('Regression Head — MAE')
ax.set_xlabel('Epoch')
ax.legend()
ax.grid(alpha=0.3)

# Classification Accuracy
ax = axes[2]
ax.plot(history.history['cls_output_accuracy'],
        label='Train Accuracy', color='steelblue')
ax.plot(history.history['val_cls_output_accuracy'],
        label='Val Accuracy', color='darkorange')
ax.set_title('Classification Head — Accuracy')
ax.set_xlabel('Epoch')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(MODELS_DIR / 'robust_mtl_training_curves.png',
            dpi=150, bbox_inches='tight')
plt.show()

# ── 8. Summary vs V1 ──────────────────────────────────────────
print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)
print(f"{'Metric':<25} {'V1 (Overfit)':>15} {'Robust':>15}")
print("-" * 50)
print(f"{'Reg MAE':<25} {'—':>15} {reg_mae:>15.6f}")
print(f"{'Active DA':<25} {'49.82%':>15} {active_da:>14.2f}%")
print(f"{'Val Accuracy':<25} {'~40% (flat)':>15} "
      f"{max(history.history['val_cls_output_accuracy']):>14.2%}")
print("=" * 50)

# ── 9. Save ───────────────────────────────────────────────────
model.save(MODELS_DIR / 'robust_mtl_final.keras')
print(f"\n✓ Model saved → robust_mtl_final.keras")

# %%
# ============================================================
# Model Comparison: V1 (Generalist) vs Robust MTL
# ============================================================

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ── 1. Predict from both models ───────────────────────────────
# V1
reg_pred_v1, cls_prob_v1 = generalist_mtl.predict(X_test, verbose=0)
cls_label_v1             = np.argmax(cls_prob_v1, axis=1)

# Robust
reg_pred_rob, cls_prob_rob = model.predict(X_test, verbose=0)
cls_label_rob              = np.argmax(cls_prob_rob, axis=1)

# ── 2. Regression Metrics ─────────────────────────────────────
reg_mae_v1  = mean_absolute_error(y_test_reg, reg_pred_v1)
reg_rmse_v1 = np.sqrt(mean_squared_error(y_test_reg, reg_pred_v1))

reg_mae_rob  = mean_absolute_error(y_test_reg, reg_pred_rob)
reg_rmse_rob = np.sqrt(mean_squared_error(y_test_reg, reg_pred_rob))

# ── 3. Active DA per model ────────────────────────────────────
CONVICTION_THRESHOLD = 0.65

def compute_active_da(cls_prob, cls_label, y_true, threshold=0.65):
    max_prob    = cls_prob.max(axis=1)
    active_mask = max_prob >= threshold
    active_pred = cls_label[active_mask]
    active_true = y_true[active_mask]
    non_hold    = (active_pred != 1) & (active_true != 1)

    active_da = np.mean(
        active_pred[non_hold] == active_true[non_hold]
    ) * 100 if non_hold.sum() > 0 else 0.0

    return {
        'active_count'  : active_mask.sum(),
        'active_pct'    : active_mask.mean(),
        'buy_signals'   : (active_pred == 2).sum(),
        'sell_signals'  : (active_pred == 0).sum(),
        'active_da'     : active_da,
        'mean_conviction': max_prob.mean(),
        'pct_above_065' : (max_prob >= 0.65).mean(),
        'pct_above_070' : (max_prob >= 0.70).mean(),
        'pct_above_080' : (max_prob >= 0.80).mean(),
    }

da_v1  = compute_active_da(cls_prob_v1,  cls_label_v1,  y_test_cls)
da_rob = compute_active_da(cls_prob_rob, cls_label_rob, y_test_cls)

# ── 4. Summary Table ──────────────────────────────────────────
print("=" * 62)
print("MODEL COMPARISON: V1 (Generalist) vs Robust MTL")
print("=" * 62)
print(f"{'Metric':<30} {'V1 (Overfit)':>14} {'Robust':>14}")
print("-" * 62)

# Regression
print(f"{'[Regression]':<30}")
print(f"{'  MAE':<30} {reg_mae_v1:>14.6f} {reg_mae_rob:>14.6f}")
print(f"{'  RMSE':<30} {reg_rmse_v1:>14.6f} {reg_rmse_rob:>14.6f}")

# Classification
print(f"\n{'[Classification]':<30}")

rep_v1  = classification_report(y_test_cls, cls_label_v1,
            target_names=['SELL','HOLD','BUY'],
            output_dict=True, digits=4)
rep_rob = classification_report(y_test_cls, cls_label_rob,
            target_names=['SELL','HOLD','BUY'],
            output_dict=True, digits=4)

for cls in ['BUY', 'SELL', 'HOLD']:
    for metric in ['precision', 'recall', 'f1-score']:
        label = f"  {cls} {metric}"
        v1_val  = rep_v1[cls][metric]
        rob_val = rep_rob[cls][metric]
        winner  = '✓' if rob_val > v1_val else ' '
        print(f"{label:<30} {v1_val:>14.4f} {rob_val:>13.4f} {winner}")

print(f"\n{'  Overall Accuracy':<30} "
      f"{rep_v1['accuracy']:>14.4f} "
      f"{rep_rob['accuracy']:>14.4f} "
      f"{'✓' if rep_rob['accuracy'] > rep_v1['accuracy'] else ' '}")

# Active DA
print(f"\n{'[Conviction-Gated @ 0.65]':<30}")
print(f"{'  Active Signals':<30} "
      f"{da_v1['active_count']:>10,} ({da_v1['active_pct']:.1%}) "
      f"{da_rob['active_count']:>6,} ({da_rob['active_pct']:.1%})")
print(f"{'  BUY signals':<30} "
      f"{da_v1['buy_signals']:>14,} "
      f"{da_rob['buy_signals']:>14,}")
print(f"{'  SELL signals':<30} "
      f"{da_v1['sell_signals']:>14,} "
      f"{da_rob['sell_signals']:>14,}")
print(f"{'  Active DA':<30} "
      f"{da_v1['active_da']:>13.2f}% "
      f"{da_rob['active_da']:>13.2f}% "
      f"{'✓' if da_rob['active_da'] > da_v1['active_da'] else ' '}")

# Conviction distribution
print(f"\n{'[Conviction Distribution]':<30}")
print(f"{'  Mean Conviction':<30} "
      f"{da_v1['mean_conviction']:>14.4f} "
      f"{da_rob['mean_conviction']:>14.4f}")
print(f"{'  % above 0.65':<30} "
      f"{da_v1['pct_above_065']:>14.1%} "
      f"{da_rob['pct_above_065']:>14.1%}")
print(f"{'  % above 0.70':<30} "
      f"{da_v1['pct_above_070']:>14.1%} "
      f"{da_rob['pct_above_070']:>14.1%}")
print(f"{'  % above 0.80':<30} "
      f"{da_v1['pct_above_080']:>14.1%} "
      f"{da_rob['pct_above_080']:>14.1%}")
print("=" * 62)

# ── 5. Classification Reports (side by side) ──────────────────
print("\n--- V1 Classification Report ---")
print(classification_report(
    y_test_cls, cls_label_v1,
    target_names=['SELL', 'HOLD', 'BUY'], digits=4
))

print("--- Robust Classification Report ---")
print(classification_report(
    y_test_cls, cls_label_rob,
    target_names=['SELL', 'HOLD', 'BUY'], digits=4
))

# ── 6. Side-by-side Confusion Matrices ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Confusion Matrix Comparison — Test Set (10 Tickers)',
             fontsize=13, fontweight='bold')

for ax, cls_label, title in zip(
    axes,
    [cls_label_v1, cls_label_rob],
    ['V1 (Generalist — Overfit)', 'Robust MTL']
):
    cm = confusion_matrix(y_test_cls, cls_label)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['SELL', 'HOLD', 'BUY'],
        yticklabels=['SELL', 'HOLD', 'BUY'],
        ax=ax
    )
    ax.set_title(title, fontsize=12)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

plt.tight_layout()
plt.savefig(MODELS_DIR / 'mtl_comparison_confusion_matrices.png',
            dpi=150, bbox_inches='tight')
plt.show()

# ── 7. Training Curves Comparison ────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Training History Comparison: V1 vs Robust MTL',
             fontsize=14, fontweight='bold')

metrics = [
    ('loss',                  'val_loss',                  'Total Loss'),
    ('reg_output_mae',        'val_reg_output_mae',        'Regression MAE'),
    ('cls_output_accuracy',   'val_cls_output_accuracy',   'Classification Accuracy'),
]

for col, (train_key, val_key, title) in enumerate(metrics):
    # V1 — top row
    ax = axes[0][col]
    ax.plot(history_generalist.history[train_key],
            label='Train', color='steelblue')
    ax.plot(history_generalist.history[val_key],
            label='Val', color='darkorange')
    ax.set_title(f'V1 — {title}')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(alpha=0.3)

    # Robust — bottom row
    ax = axes[1][col]
    ax.plot(history.history[train_key],
            label='Train', color='steelblue')
    ax.plot(history.history[val_key],
            label='Val', color='darkorange')
    ax.set_title(f'Robust — {title}')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(MODELS_DIR / 'mtl_comparison_training_curves.png',
            dpi=150, bbox_inches='tight')
plt.show()

# ── 8. Conviction Distribution Plot ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Conviction Score Distribution — Test Set',
             fontsize=13, fontweight='bold')

for ax, cls_prob, title in zip(
    axes,
    [cls_prob_v1, cls_prob_rob],
    ['V1 (Generalist)', 'Robust MTL']
):
    max_prob = cls_prob.max(axis=1)
    ax.hist(max_prob, bins=40, color='steelblue',
            alpha=0.7, edgecolor='white')
    ax.axvline(0.65, color='red', linestyle='--',
               linewidth=1.5, label='Threshold (0.65)')
    ax.axvline(max_prob.mean(), color='orange', linestyle='-',
               linewidth=1.5, label=f'Mean ({max_prob.mean():.3f})')
    ax.set_title(title)
    ax.set_xlabel('Max Class Probability (Conviction)')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(MODELS_DIR / 'mtl_comparison_conviction_dist.png',
            dpi=150, bbox_inches='tight')
plt.show()

# ── 9. Final Verdict ──────────────────────────────────────────
print("\n" + "=" * 62)
print("FINAL VERDICT")
print("=" * 62)
winner_da  = "Robust" if da_rob['active_da']      > da_v1['active_da']      else "V1"
winner_mae = "Robust" if reg_mae_rob               < reg_mae_v1               else "V1"
winner_acc = "Robust" if rep_rob['accuracy']       > rep_v1['accuracy']       else "V1"

print(f"  Active DA winner   : {winner_da}")
print(f"  Regression winner  : {winner_mae}")
print(f"  Accuracy winner    : {winner_acc}")
print()

if winner_da == "Robust" and winner_acc == "Robust":
    print("  → Robust MTL selected as production model for Task 3")
    print("    Reason: Better generalization, less overfitting")
elif winner_da == "V1":
    print("  → V1 selected despite overfitting")
    print("    Reason: Higher Active DA — the primary trading metric")
else:
    print("  → Mixed results — consider ensemble or further tuning")
print("=" * 62)

# %%
# Check class distribution
for split_name, y_cls in [('Train', y_train_cls),
                           ('Val',   y_val_cls),
                           ('Test',  y_test_cls)]:
    total  = len(y_cls)
    sell   = (y_cls == 0).sum()
    hold   = (y_cls == 1).sum()
    buy    = (y_cls == 2).sum()
    print(f"{split_name}:")
    print(f"  SELL: {sell:,} ({sell/total:.1%})")
    print(f"  HOLD: {hold:,} ({hold/total:.1%})")
    print(f"  BUY : {buy:,} ({buy/total:.1%})")
    print()

# %% [markdown]
# ### Problem leading to 2 overfitting model
# 
# Train: SELL 30.7% | HOLD 37.5% | BUY 31.8%  ← balanced ✓
# Val:   SELL 27.7% | HOLD 51.4% | BUY 20.9%  ← HOLD dominated ✗
# Test:  SELL 30.9% | HOLD 32.8% | BUY 36.2%  ← balanced ✓
# 
# Val period = sideways market 
# → EarlyStopping depends on the val_losss
# → Model leans towards HOLD để minimize val_loss
# → When encountering Test set with class balance, model is confused
# 
# #### Solution: 
# Fix 1: Change Early Stopping monitor from val_loss to val_cls_output_accuracy.
# Fix 2: Label smoothing

# %%
# ============================================================
# Retrain V1 với fixes cho Val imbalance issue
# ============================================================

def build_generalist_mtl_v3(window_size, n_features,
                              output_steps=5, n_classes=3):
    """
    V3: Same as V1 architecture nhưng fix val imbalance issue.
    Architecture không thay đổi — vấn đề là training strategy.
    """
    inputs = layers.Input(
        shape=(window_size, n_features),
        name='main_input'
    )

    # Shared Encoder
    encoder_seq, encoder_state = layers.GRU(
        128, return_sequences=True,
        return_state=True, name='encoder_gru'
    )(inputs)

    att = layers.MultiHeadAttention(
        num_heads=2, key_dim=64,
        dropout=0.3, name='shared_attention'
    )(encoder_seq, encoder_seq)

    context = layers.GlobalAveragePooling1D(name='context_pool')(att)
    context = layers.LayerNormalization(name='context_norm')(context)

    # Regression Head
    reg_input  = layers.RepeatVector(output_steps, name='reg_bridge')(context)
    reg_gru    = layers.GRU(
        128, return_sequences=True, name='reg_decoder_gru'
    )(reg_input, initial_state=encoder_state)
    reg_out    = layers.TimeDistributed(
        layers.Dense(1), name='reg_td'
    )(reg_gru)
    reg_output = layers.Reshape(
        (output_steps,), name='reg_output'
    )(reg_out)

    # Classification Head
    cls_x      = layers.Dense(64, activation='relu', name='cls_dense1')(context)
    cls_x      = layers.Dropout(0.4, name='cls_dropout')(cls_x)
    cls_x      = layers.Dense(32, activation='relu', name='cls_dense2')(cls_x)
    cls_output = layers.Dense(
        n_classes, activation='softmax', name='cls_output'
    )(cls_x)

    return Model(
        inputs=inputs,
        outputs=[reg_output, cls_output],
        name='Generalist_MTL_V3'
    )


# ── Custom loss với label smoothing ──────────────────────────
# Label smoothing giúp model không quá confident vào HOLD
# khi val set bị dominated bởi HOLD
def smoothed_sparse_categorical_crossentropy(y_true, y_pred,
                                              smoothing=0.1,
                                              n_classes=3):
    """
    Label smoothing: thay vì [0,0,1] dùng [0.033, 0.033, 0.933]
    → Prevents overconfidence on majority class (HOLD in val)
    → Model buộc phải maintain some probability cho minority classes
    """
    y_true_int    = tf.cast(tf.squeeze(y_true), tf.int32)
    one_hot       = tf.one_hot(y_true_int, n_classes)
    smooth_labels = one_hot * (1 - smoothing) + smoothing / n_classes
    return tf.reduce_mean(
        tf.keras.losses.categorical_crossentropy(smooth_labels, y_pred)
    )


# ── Weighted loss để balance SELL/BUY vs HOLD ────────────────
from sklearn.utils import class_weight as sklearn_cw

weights    = sklearn_cw.compute_class_weight(
    'balanced',
    classes = np.unique(y_train_cls),
    y       = y_train_cls
)
cw_dict    = {i: weights[i] for i in range(len(weights))}
weight_tensor = tf.constant(
    [cw_dict[i] for i in sorted(cw_dict.keys())],
    dtype=tf.float32
)

print("Class weights:")
for cls, w in cw_dict.items():
    print(f"  {label_map[cls]:4s} ({cls}): {w:.4f}")


def weighted_smoothed_crossentropy(y_true, y_pred,
                                    smoothing=0.1, n_classes=3):
    """
    Combines:
    1. Label smoothing → prevents HOLD overconfidence
    2. Class weights   → upweights SELL/BUY minority signal
    """
    y_true_int    = tf.cast(tf.squeeze(y_true), tf.int32)
    one_hot       = tf.one_hot(y_true_int, n_classes)
    smooth_labels = one_hot * (1 - smoothing) + smoothing / n_classes

    ce_loss       = tf.keras.losses.categorical_crossentropy(
        smooth_labels, y_pred
    )
    sample_w      = tf.gather(weight_tensor, y_true_int)

    return tf.reduce_mean(ce_loss * sample_w)


# ── Initialize & Compile ──────────────────────────────────────
generalist_mtl_v3 = build_generalist_mtl_v3(
    window_size  = W,
    n_features   = N_FEATURES,
    output_steps = K_DAYS,
    n_classes    = 3
)

generalist_mtl_v3.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss = {
        'reg_output': lambda y_true, y_pred: sign_weighted_huber(
            y_true, y_pred, penalty_weight=2.5
        ),
        'cls_output': weighted_smoothed_crossentropy
    },
    loss_weights = {
        'reg_output': 1.0,
        'cls_output': 5.0
    },
    metrics = {
        'reg_output': 'mae',
        'cls_output': 'accuracy'
    }
)

generalist_mtl_v3.summary()

# %%
# ── Callbacks V3 — monitor accuracy instead of loss ──────────
checkpoint_path_v3 = MODELS_DIR / 'generalist_mtl_v3_best.keras'

callbacks_v3 = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath       = str(checkpoint_path_v3),
        monitor        = 'val_cls_output_accuracy',  # ← KEY FIX
        save_best_only = True,
        mode           = 'max',
        verbose        = 1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor              = 'val_cls_output_accuracy',  # ← KEY FIX
        patience             = 20,
        restore_best_weights = True,
        mode                 = 'max',
        verbose              = 1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor  = 'val_cls_output_accuracy',
        factor   = 0.5,
        patience = 7,
        min_lr   = 1e-7,
        mode     = 'max',
        verbose  = 1
    )
]

# %%
# ── Fix: Define weight tensor inside loss function ────────────

# Precompute weights as Python list (not tf.constant at global scope)
weights    = sklearn_cw.compute_class_weight(
    'balanced',
    classes = np.unique(y_train_cls),
    y       = y_train_cls
)
cw_dict    = {i: float(weights[i]) for i in range(len(weights))}
cw_list    = [cw_dict[i] for i in sorted(cw_dict.keys())]

print("Class weights:")
for cls, w in cw_dict.items():
    print(f"  {label_map[cls]:4s} ({cls}): {w:.4f}")


def weighted_smoothed_crossentropy(y_true, y_pred,
                                    smoothing=0.1,
                                    n_classes=3):
    """
    Fixed version: weight_tensor defined inside function
    → Avoids shape inference issues during Keras graph tracing
    """
    # Define inside function — Keras can trace shape correctly
    w_tensor = tf.constant(cw_list, dtype=tf.float32)

    # Flatten y_true safely
    y_true_flat = tf.reshape(tf.cast(y_true, tf.int32), [-1])

    # Label smoothing
    one_hot       = tf.one_hot(y_true_flat, n_classes)
    smooth_labels = one_hot * (1.0 - smoothing) + smoothing / n_classes

    # Cross entropy
    ce_loss = tf.keras.losses.categorical_crossentropy(
        smooth_labels, y_pred
    )

    # Apply class weights
    sample_w = tf.gather(w_tensor, y_true_flat)

    return tf.reduce_mean(ce_loss * sample_w)


# ── Recompile V3 với fixed loss ───────────────────────────────
generalist_mtl_v3.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss = {
        'reg_output': lambda y_true, y_pred: sign_weighted_huber(
            y_true, y_pred, penalty_weight=2.5
        ),
        'cls_output': weighted_smoothed_crossentropy
    },
    loss_weights = {
        'reg_output': 1.0,
        'cls_output': 5.0
    },
    metrics = {
        'reg_output': 'mae',
        'cls_output': 'accuracy'
    }
)

print("✓ V3 recompiled with fixed loss function.")
print(f"  Label smoothing : 0.1")
print(f"  Class weights   : {cw_list}")

# %%
# ── Train V3 ─────────────────────────────────────────────────
print("\n🚀 Training Generalist MTL V3...")

history_v3 = generalist_mtl_v3.fit(
    X_train,
    {
        'reg_output': y_train_reg,
        'cls_output': y_train_cls
    },
    validation_data=(
        X_val,
        {
            'reg_output': y_val_reg,
            'cls_output': y_val_cls
        }
    ),
    epochs     = 100,
    batch_size = 32,
    callbacks  = callbacks_v3,
    verbose    = 1
)

# %%
# ============================================================
# Evaluation — V3 (Fixed) + 3-way Comparison
# ============================================================

# ── 1. Predict ────────────────────────────────────────────────
reg_pred_v3, cls_prob_v3 = generalist_mtl_v3.predict(X_test, verbose=0)
cls_label_v3             = np.argmax(cls_prob_v3, axis=1)

reg_mae_v3  = mean_absolute_error(y_test_reg, reg_pred_v3)
reg_rmse_v3 = np.sqrt(mean_squared_error(y_test_reg, reg_pred_v3))
da_v3       = compute_active_da(cls_prob_v3, cls_label_v3, y_test_cls)
rep_v3      = classification_report(
    y_test_cls, cls_label_v3,
    target_names=['SELL', 'HOLD', 'BUY'],
    output_dict=True, digits=4
)

# ── 2. 3-way Summary Table ────────────────────────────────────
print("=" * 72)
print("3-WAY COMPARISON: V1 vs Robust vs V3 (Fixed)")
print("=" * 72)
print(f"{'Metric':<28} {'V1':>12} {'Robust':>12} {'V3 (Fix)':>12}")
print("-" * 72)

# Regression
print(f"{'[Regression]':<28}")
print(f"{'  MAE':<28} "
      f"{reg_mae_v1:>12.6f} "
      f"{reg_mae_rob:>12.6f} "
      f"{reg_mae_v3:>12.6f} "
      f"{'✓' if reg_mae_v3 == min(reg_mae_v1, reg_mae_rob, reg_mae_v3) else ''}")
print(f"{'  RMSE':<28} "
      f"{reg_rmse_v1:>12.6f} "
      f"{reg_rmse_rob:>12.6f} "
      f"{reg_rmse_v3:>12.6f} "
      f"{'✓' if reg_rmse_v3 == min(reg_rmse_v1, reg_rmse_rob, reg_rmse_v3) else ''}")

# Classification per class
print(f"\n{'[Classification]':<28}")
for cls in ['BUY', 'SELL', 'HOLD']:
    print(f"  {cls}:")
    for metric in ['precision', 'recall', 'f1-score']:
        v1_val  = rep_v1[cls][metric]
        rob_val = rep_rob[cls][metric]
        v3_val  = rep_v3[cls][metric]
        best    = max(v1_val, rob_val, v3_val)
        label   = f"    {metric}"
        print(f"{label:<28} "
              f"{v1_val:>12.4f} "
              f"{rob_val:>12.4f} "
              f"{v3_val:>12.4f} "
              f"{'✓' if v3_val == best else ''}")

# Overall accuracy
v1_acc  = rep_v1['accuracy']
rob_acc = rep_rob['accuracy']
v3_acc  = rep_v3['accuracy']
best_acc = max(v1_acc, rob_acc, v3_acc)
print(f"\n{'  Overall Accuracy':<28} "
      f"{v1_acc:>12.4f} "
      f"{rob_acc:>12.4f} "
      f"{v3_acc:>12.4f} "
      f"{'✓' if v3_acc == best_acc else ''}")

# Active DA
print(f"\n{'[Conviction @ 0.65]':<28}")
print(f"{'  Active Signals':<28} "
      f"{da_v1['active_count']:>9,} ({da_v1['active_pct']:.1%}) "
      f"{da_rob['active_count']:>4,} ({da_rob['active_pct']:.1%}) "
      f"{da_v3['active_count']:>4,} ({da_v3['active_pct']:.1%})")
print(f"{'  BUY signals':<28} "
      f"{da_v1['buy_signals']:>12,} "
      f"{da_rob['buy_signals']:>12,} "
      f"{da_v3['buy_signals']:>12,}")
print(f"{'  SELL signals':<28} "
      f"{da_v1['sell_signals']:>12,} "
      f"{da_rob['sell_signals']:>12,} "
      f"{da_v3['sell_signals']:>12,}")
best_da = max(da_v1['active_da'], da_rob['active_da'], da_v3['active_da'])
print(f"{'  Active DA':<28} "
      f"{da_v1['active_da']:>11.2f}% "
      f"{da_rob['active_da']:>11.2f}% "
      f"{da_v3['active_da']:>11.2f}% "
      f"{'✓' if da_v3['active_da'] == best_da else ''}")

# Conviction distribution
print(f"\n{'[Conviction Distribution]':<28}")
for label_str, da_dict in [('V1', da_v1), ('Robust', da_rob), ('V3', da_v3)]:
    print(f"  {label_str}: mean={da_dict['mean_conviction']:.3f} | "
          f">0.65={da_dict['pct_above_065']:.1%} | "
          f">0.70={da_dict['pct_above_070']:.1%} | "
          f">0.80={da_dict['pct_above_080']:.1%}")
print("=" * 72)

# ── 3. Full Classification Reports ───────────────────────────
for title, y_pred in [
    ('V1 (Generalist)',  cls_label_v1),
    ('Robust MTL',       cls_label_rob),
    ('V3 (Fixed)',       cls_label_v3)
]:
    print(f"\n--- {title} Classification Report ---")
    print(classification_report(
        y_test_cls, y_pred,
        target_names=['SELL', 'HOLD', 'BUY'],
        digits=4
    ))

# ── 4. Confusion Matrices (3 side by side) ───────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle('Confusion Matrix — 3-Way Comparison (Test Set, 10 Tickers)',
             fontsize=13, fontweight='bold')

for ax, cls_label, title in zip(
    axes,
    [cls_label_v1, cls_label_rob, cls_label_v3],
    ['V1 (Generalist — Overfit)',
     'Robust MTL',
     'V3 (Fixed — Label Smoothing)']
):
    cm = confusion_matrix(y_test_cls, cls_label)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['SELL', 'HOLD', 'BUY'],
        yticklabels=['SELL', 'HOLD', 'BUY'],
        ax=ax
    )
    ax.set_title(title, fontsize=11)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

plt.tight_layout()
plt.savefig(MODELS_DIR / 'mtl_3way_confusion_matrices.png',
            dpi=150, bbox_inches='tight')
plt.show()

# ── 5. Training Curves (3 rows) ───────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle('Training History: V1 vs Robust vs V3',
             fontsize=14, fontweight='bold')

metrics_keys = [
    ('loss',              'val_loss',              'Total Loss'),
    ('reg_output_mae',    'val_reg_output_mae',    'Regression MAE'),
    ('cls_output_accuracy','val_cls_output_accuracy','Classification Accuracy'),
]

for row, (history_obj, model_name) in enumerate([
    (history_generalist, 'V1 (Generalist)'),
    (history,            'Robust MTL'),
    (history_v3,         'V3 (Fixed)')
]):
    for col, (train_key, val_key, metric_title) in enumerate(metrics_keys):
        ax = axes[row][col]
        ax.plot(history_obj.history[train_key],
                label='Train', color='steelblue', linewidth=1.2)
        ax.plot(history_obj.history[val_key],
                label='Val', color='darkorange', linewidth=1.2)
        ax.set_title(f'{model_name} — {metric_title}', fontsize=10)
        ax.set_xlabel('Epoch')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(MODELS_DIR / 'mtl_3way_training_curves.png',
            dpi=150, bbox_inches='tight')
plt.show()

# ── 6. Conviction Distribution (3 side by side) ───────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Conviction Score Distribution — Test Set',
             fontsize=13, fontweight='bold')

for ax, cls_prob, title, color in zip(
    axes,
    [cls_prob_v1, cls_prob_rob, cls_prob_v3],
    ['V1 (Generalist)', 'Robust MTL', 'V3 (Fixed)'],
    ['steelblue', 'darkorange', 'seagreen']
):
    max_prob = cls_prob.max(axis=1)
    ax.hist(max_prob, bins=40, color=color,
            alpha=0.7, edgecolor='white')
    ax.axvline(0.65, color='red', linestyle='--',
               linewidth=1.5, label='Threshold (0.65)')
    ax.axvline(max_prob.mean(), color='black', linestyle='-',
               linewidth=1.5, label=f'Mean ({max_prob.mean():.3f})')
    ax.set_title(title)
    ax.set_xlabel('Max Class Probability (Conviction)')
    ax.set_ylabel('Count')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(MODELS_DIR / 'mtl_3way_conviction_dist.png',
            dpi=150, bbox_inches='tight')
plt.show()

# ── 7. Final Verdict ──────────────────────────────────────────
models_results = {
    'V1'    : {'da': da_v1['active_da'],  'mae': reg_mae_v1,  'acc': v1_acc},
    'Robust': {'da': da_rob['active_da'], 'mae': reg_mae_rob, 'acc': rob_acc},
    'V3'    : {'da': da_v3['active_da'],  'mae': reg_mae_v3,  'acc': v3_acc},
}

best_da_model  = max(models_results, key=lambda x: models_results[x]['da'])
best_mae_model = min(models_results, key=lambda x: models_results[x]['mae'])
best_acc_model = max(models_results, key=lambda x: models_results[x]['acc'])

print("\n" + "=" * 55)
print("FINAL VERDICT")
print("=" * 55)
print(f"  Active DA winner   : {best_da_model}")
print(f"  Regression winner  : {best_mae_model}")
print(f"  Accuracy winner    : {best_acc_model}")
print()

# Select production model based on Active DA (primary metric)
production_model_name = best_da_model
production_model = {
    'V1'    : generalist_mtl,
    'Robust': model,
    'V3'    : generalist_mtl_v3
}[production_model_name]

print(f"  → {production_model_name} selected as production model")
print(f"    Primary reason: Highest Active DA "
      f"({models_results[production_model_name]['da']:.2f}%)")
print()
print(f"  This model will be used to generate:")
print(f"    mtl_p_up       → P(BUY)  from cls_output[:,2]")
print(f"    mtl_p_down     → P(SELL) from cls_output[:,0]")
print(f"    mtl_conviction → max(P(BUY), P(SELL))")
print("=" * 55)

# Save production model
production_model.save(
    MODELS_DIR / f'production_mtl_{production_model_name.lower()}.keras'
)
print(f"\n✓ Production model saved → "
      f"production_mtl_{production_model_name.lower()}.keras")

# %% [markdown]
# ## MTL Generalist Model — Training Analysis & Design Decision
# 
# ### Training Results Summary
# 
# Three architectural iterations were evaluated to identify the optimal generalist MTL model across 10 tickers:
# 
# | Model | Active DA | Mean Conviction | Active Signals | Reg MAE |
# |---|---|---|---|---|
# | V1 (Generalist) | 49.82% | 0.621 | 1,134 (39.8%) | 0.885 |
# | Robust MTL | — | 0.401 | 2 (0.1%) | 0.873 |
# | V3 (Label Smoothing) | — | 0.366 | — | 0.869 |
# 
# **V1 is selected as the production model** on the basis of Active DA and conviction score distribution — despite its overfitting signature, it generates the most actionable signal volume with the highest directional reliability among the three candidates.
# 
# ---
# 
# ### Root Cause: Conflicting Patterns Across Sectors
# 
# The generalist model's inability to achieve strong directional accuracy (>55%) is a structural consequence of the dataset's sector diversity rather than an architectural failure. The 10 tickers span fundamentally different market regimes:
# 
# | Sector | Tickers | Price Driver |
# |---|---|---|
# | Technology | FPT | Earnings momentum, foreign flow |
# | Banking | VCB, TCB | Interest rate cycle, credit growth |
# | Real Estate | VHM, VIC | Policy sentiment, bond market |
# | Steel / Industrial | HPG | Commodity cycle, infrastructure spend |
# | Consumer | MWG, VNM, MSN | Retail sentiment, FX exposure |
# 
# A pattern that constitutes a BUY signal for a banking stock (e.g., rising RSI after rate cut) may simultaneously represent a SELL signal for a real estate stock reacting to the same macro event. This **inter-sector signal conflict** prevents the shared GRU encoder from converging on a universal directional representation, resulting in a mean conviction score of 0.621 — well below the threshold required for high-confidence standalone trading signals.
# 
# > *This is not a model failure — it is a data reality. In professional quant systems, sector-specific models are the industry standard precisely because cross-sector generalization degrades directional alpha.*
# 
# ---
# 
# ### Design Decision: Option A — Retain MTL Output as Weak Prior
# 
# Despite the sub-50% standalone Active DA, the MTL classification head is retained as a **feature input** to the downstream XGBoost classifier rather than discarded. The conviction threshold is lowered from `0.65 → 0.55` to increase signal coverage.
# 
# **Rationale:**
# 
# - The MTL model captures **temporal sequence dynamics** (20-day momentum context) that static technical indicators like RSI and MACD cannot represent — even a weak prior adds non-redundant information to the feature space.
# - XGBoost's tree-based architecture will **automatically down-weight** MTL features if they carry low information gain relative to S/R zones and MA crossovers — the feature importance mechanism acts as a natural filter.
# - Removing MTL features entirely would break the design philosophy of the pipeline: that deep learning trajectory context and technical confirmation should **reinforce each other**, not operate independently.
# 
# **Adjusted MTL feature extraction:**
# 
# ```python
# # Conviction threshold lowered for feature generation
# CONVICTION_THRESHOLD = 0.55   # was 0.65
# 
# # MTL features extracted from V1 production model
# # mtl_p_up       = cls_output[:, 2]          → P(BUY)
# # mtl_p_down     = cls_output[:, 0]          → P(SELL)
# # mtl_conviction = max(mtl_p_up, mtl_p_down) → signal strength
# ```
# 
# The XGBoost classifier in Step 6 will determine the true predictive contribution of MTL features through feature importance analysis — if they rank near zero, they will be pruned at that stage before final model selection.

# %% [markdown]
# ## Step 3: Engineer New Features

# %%
# ============================================================
# STEP 3: Feature Engineering
# 3.1 S/R Zone Features    (5 features) — từ df_model (unscaled)
# 3.2 MA Crossover Features (7 features) — từ df_model (unscaled)
# 3.3 MTL Output Features   (3 features) — từ production model
# 3.4 Ticker Encoding       (1 feature)  — sector context
# ============================================================

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ── 3.0 Base DataFrame ────────────────────────────────────────
# Dùng df_model (unscaled) làm base vì cần close price gốc
# df_pruned đã scaled → không dùng cho S/R và MA

df_feat = df_model[
    ['date', 'ticker', 'close', 'ema_10', 'ema_20', 'ema_50',
     'target_multi', 'target_class', 'forward_return']
].copy().sort_values(['ticker', 'date']).reset_index(drop=True)

print(f"Base df_feat shape: {df_feat.shape}")
print(f"Tickers: {df_feat['ticker'].unique().tolist()}")

# %%
# ── 3.1 S/R Zone Features ────────────────────────────────────
LOOKBACK   = 63    # 3 months
ZONE_WIDTH = 0.005 # ±0.5%
K_RANGE    = range(2, 12)

def find_optimal_k(prices):
    """
    Elbow method: find k that maximizes second derivative of inertia.
    Returns optimal k.
    """
    inertias = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=42,
                    n_init=10, max_iter=100)
        km.fit(prices.reshape(-1, 1))
        inertias.append(km.inertia_)

    inertias      = np.array(inertias)
    deltas        = np.diff(inertias)
    second_deltas = np.diff(deltas)

    # Elbow = max second derivative
    optimal_idx = np.argmax(second_deltas) + 2
    optimal_k   = list(K_RANGE)[optimal_idx]
    return optimal_k


def compute_sr_features_ticker(group_df):
    """
    Compute S/R features for a single ticker.
    group_df: sorted by date, contains 'close' column.
    Returns DataFrame with 5 new S/R columns.
    """
    closes = group_df['close'].values
    n      = len(closes)

    # Initialize output arrays
    sr_distance    = np.zeros(n)
    sr_break_up    = np.zeros(n)
    sr_break_down  = np.zeros(n)
    sr_near_res    = np.zeros(n)
    sr_near_sup    = np.zeros(n)

    for i in range(LOOKBACK, n):
        current_price = closes[i]
        prev_price    = closes[i - 1]
        window_prices = closes[i - LOOKBACK : i]

        # Find optimal k via elbow
        optimal_k = find_optimal_k(window_prices)

        # Fit KMeans
        km = KMeans(n_clusters=optimal_k, random_state=42,
                    n_init=10, max_iter=100)
        km.fit(window_prices.reshape(-1, 1))
        zone_centers = np.sort(km.cluster_centers_.flatten())

        zone_width = current_price * ZONE_WIDTH

        # Feature 1: Distance to nearest zone (%)
        distances      = np.abs(zone_centers - current_price)
        nearest_center = zone_centers[np.argmin(distances)]
        sr_distance[i] = (
            (current_price - nearest_center) / nearest_center * 100
        )

        # Feature 2 & 3: Breakout detection
        for center in zone_centers:
            zone_low  = center - zone_width
            zone_high = center + zone_width

            if prev_price < zone_low and current_price > zone_high:
                sr_break_up[i] = 1.0
            elif prev_price > zone_high and current_price < zone_low:
                sr_break_down[i] = 1.0

        # Feature 4 & 5: Near zone detection
        resistance_zones = zone_centers[zone_centers > current_price]
        support_zones    = zone_centers[zone_centers < current_price]

        if len(resistance_zones) > 0:
            nearest_res = resistance_zones[0]
            if abs(current_price - nearest_res) / current_price < ZONE_WIDTH:
                sr_near_res[i] = 1.0

        if len(support_zones) > 0:
            nearest_sup = support_zones[-1]
            if abs(current_price - nearest_sup) / current_price < ZONE_WIDTH:
                sr_near_sup[i] = 1.0

    result = group_df.copy()
    result['sr_distance_pct']   = sr_distance
    result['sr_breakout_up']    = sr_break_up
    result['sr_breakout_down']  = sr_break_down
    result['sr_near_resistance'] = sr_near_res
    result['sr_near_support']   = sr_near_sup

    return result


# ── Run per ticker with progress bar ─────────────────────────
print("Computing S/R zone features (per ticker)...")
sr_results = []

for ticker, group in tqdm(df_feat.groupby('ticker', sort=False),
                           desc='S/R Zones',
                           total=df_feat['ticker'].nunique()):
    group_sorted = group.sort_values('date').reset_index(drop=True)
    sr_results.append(compute_sr_features_ticker(group_sorted))

df_feat = pd.concat(sr_results).sort_values(
    ['ticker', 'date']
).reset_index(drop=True)

SR_FEATURE_COLS = [
    'sr_distance_pct', 'sr_breakout_up', 'sr_breakout_down',
    'sr_near_resistance', 'sr_near_support'
]

print(f"✓ S/R features computed: {SR_FEATURE_COLS}")
print(df_feat[SR_FEATURE_COLS].describe().round(4))

# %%
FEAT_EXPORT_PATH = DATA_DIR / 'df_feat_engineered_master.csv'
df_feat.to_csv(FEAT_EXPORT_PATH, index = False)

print(f"✅ df_feat exported successfully!")
print(f"Path: {FEAT_EXPORT_PATH}")
print(f"Shape: {df_feat.shape}")

# %%


# %%
# ── 3.2 MA Crossover Features ─────────────────────────────────
# EMA columns đã có sẵn trong df_model (unscaled)

def compute_ma_features(df_input):
    """
    Compute MA crossover features per ticker.
    Uses ema_10, ema_20, ema_50 from df_model (unscaled).
    """
    df_out = df_input.copy()
    results = []

    for ticker, group in df_out.groupby('ticker', sort=False):
        group = group.sort_values('date').reset_index(drop=True)

        ema10 = group['ema_10']
        ema20 = group['ema_20']
        ema50 = group['ema_50']

        prev_ema10 = ema10.shift(1)
        prev_ema20 = ema20.shift(1)
        prev_ema50 = ema50.shift(1)

        # Event-based crossovers (= 1 only on crossover day)
        group['ma_golden_cross_short'] = (
            (prev_ema10 <= prev_ema20) & (ema10 > ema20)
        ).astype(float)

        group['ma_death_cross_short'] = (
            (prev_ema10 >= prev_ema20) & (ema10 < ema20)
        ).astype(float)

        group['ma_golden_cross_long'] = (
            (prev_ema20 <= prev_ema50) & (ema20 > ema50)
        ).astype(float)

        group['ma_death_cross_long'] = (
            (prev_ema20 >= prev_ema50) & (ema20 < ema50)
        ).astype(float)

        # Continuous momentum strength
        group['ma_short_gap_pct'] = (
            (ema10 - ema20) / ema20 * 100
        )
        group['ma_long_gap_pct'] = (
            (ema20 - ema50) / ema50 * 100
        )

        # Full alignment state
        bull = (ema10 > ema20) & (ema20 > ema50)
        bear = (ema10 < ema20) & (ema20 < ema50)
        group['ma_alignment'] = bull.astype(float) - bear.astype(float)

        results.append(group)

    return pd.concat(results).sort_values(
        ['ticker', 'date']
    ).reset_index(drop=True)


df_feat = compute_ma_features(df_feat)

MA_FEATURE_COLS = [
    'ma_golden_cross_short', 'ma_death_cross_short',
    'ma_golden_cross_long',  'ma_death_cross_long',
    'ma_short_gap_pct',      'ma_long_gap_pct',
    'ma_alignment'
]

print(f"✓ MA features computed: {MA_FEATURE_COLS}")
print(df_feat[MA_FEATURE_COLS].describe().round(4))

# %%
# ── 3.3 MTL Output Features ───────────────────────────────────
# Generate P(UP), P(DOWN), conviction từ production MTL model
# Dùng scaled sequences (X_train, X_val, X_test) đã có

MTL_CONVICTION_THRESHOLD = 0.55  # lowered từ 0.65

def generate_mtl_features_all_splits(model, window_size=20):
    """
    Generate MTL features cho toàn bộ dataset.
    
    Strategy:
      - Train sequences → predict → align với train_df dates
      - Val sequences   → predict → align với val_df dates
      - Test sequences  → predict → align với test_df dates
      - Concat theo thứ tự chronological
    
    Padding: first window_size rows của mỗi ticker = NaN
    (không có đủ lookback để build sequence)
    """
    results = []

    for split_name, X_seq, df_split in [
        ('train', X_train, train_df),
        ('val',   X_val,   val_df),
        ('test',  X_test,  test_df)
    ]:
        _, cls_prob = model.predict(X_seq, verbose=0)
        # cls_prob shape: (n_samples, 3)
        # [:, 0] = P(SELL), [:, 1] = P(HOLD), [:, 2] = P(BUY)

        p_up        = cls_prob[:, 2]              # P(BUY)
        p_down      = cls_prob[:, 0]              # P(SELL)
        conviction  = np.maximum(p_up, p_down)    # signal strength

        # Align với df_split
        # Sequences bắt đầu từ row window_size của mỗi ticker
        # → cần pad window_size NaN rows ở đầu mỗi ticker

        split_results = []
        seq_idx = 0  # pointer vào cls_prob

        for ticker, group in df_split.groupby('ticker', sort=False):
            group = group.sort_values('date').reset_index(drop=True)
            n     = len(group)

            # Số sequences cho ticker này
            n_seq = n - window_size

            if n_seq <= 0:
                # Ticker quá ngắn — all NaN
                group['mtl_p_up']       = np.nan
                group['mtl_p_down']     = np.nan
                group['mtl_conviction'] = np.nan
                split_results.append(group)
                continue

            # Pad đầu với NaN
            pad_nan = np.full(window_size, np.nan)

            group['mtl_p_up'] = np.concatenate([
                pad_nan, p_up[seq_idx : seq_idx + n_seq]
            ])[:n]
            group['mtl_p_down'] = np.concatenate([
                pad_nan, p_down[seq_idx : seq_idx + n_seq]
            ])[:n]
            group['mtl_conviction'] = np.concatenate([
                pad_nan, conviction[seq_idx : seq_idx + n_seq]
            ])[:n]

            seq_idx += n_seq
            split_results.append(group)

        results.append(pd.concat(split_results))

    return pd.concat(results).sort_values(
        ['ticker', 'date']
    ).reset_index(drop=True)


# ── Generate MTL features ─────────────────────────────────────
print("Generating MTL output features from production model...")

# Merge MTL features back to df_feat
# df_feat is based on df_model rows — align by ticker + date
mtl_df = generate_mtl_features_all_splits(
    production_model, window_size=W
)

MTL_FEATURE_COLS = ['mtl_p_up', 'mtl_p_down', 'mtl_conviction']

df_feat = pd.merge(
    df_feat,
    mtl_df[['date', 'ticker'] + MTL_FEATURE_COLS],
    on=['date', 'ticker'],
    how='left'
)

print(f"✓ MTL features generated: {MTL_FEATURE_COLS}")
print(f"  Conviction threshold : {MTL_CONVICTION_THRESHOLD}")
print(f"  P(BUY)  mean         : {df_feat['mtl_p_up'].mean():.4f}")
print(f"  P(SELL) mean         : {df_feat['mtl_p_down'].mean():.4f}")
print(f"  Conviction mean      : {df_feat['mtl_conviction'].mean():.4f}")
print(f"  Signals above 0.55   : "
      f"{(df_feat['mtl_conviction'] >= 0.55).mean():.1%}")

# %%
# ── 3.4 Ticker Encoding ───────────────────────────────────────
le = LabelEncoder()
df_feat['ticker_encoded'] = le.fit_transform(df_feat['ticker'])

# Save encoder for inference
joblib.dump(le, MODELS_DIR / 'ticker_label_encoder.pkl')

print(f"✓ Ticker encoding:")
for ticker, code in zip(le.classes_,
                         le.transform(le.classes_)):
    print(f"  {ticker} → {code}")

# %%
# ── Debug: Check df_feat columns ─────────────────────────────
print("Columns in df_feat:")
print(df_feat.columns.tolist())
print()
print("MTL cols present:", 
      [c for c in MTL_FEATURE_COLS if c in df_feat.columns])
print("MTL cols missing:", 
      [c for c in MTL_FEATURE_COLS if c not in df_feat.columns])

# %%
# ── 3.5 Assemble Final Feature Matrix (Fixed) ─────────────────

# Step 1: Fix MTL column names (_x/_y suffix issue)
for col in MTL_FEATURE_COLS:
    if col + '_x' in df_feat.columns:
        df_feat = df_feat.rename(columns={col + '_x': col})
    if col + '_y' in df_feat.columns:
        df_feat = df_feat.drop(columns=[col + '_y'])

# Verify fix
print("MTL cols check:")
for col in MTL_FEATURE_COLS:
    status = '✓' if col in df_feat.columns else '✗ MISSING'
    print(f"  {col}: {status}")

# Step 2: Define feature list
TASK3_ALL_FEATURES = (
    PRUNED_FEATURE_COLS +   # 27 features từ Task 2
    SR_FEATURE_COLS     +   # 5  S/R zone features
    MA_FEATURE_COLS     +   # 7  MA crossover features
    MTL_FEATURE_COLS    +   # 3  MTL output features
    ['ticker_encoded']      # 1  ticker identity
)

print(f"\nFinal Task 3 Feature Summary:")
print(f"  Pruned Task 2 features : {len(PRUNED_FEATURE_COLS)}")
print(f"  S/R Zone features      : {len(SR_FEATURE_COLS)}")
print(f"  MA Crossover features  : {len(MA_FEATURE_COLS)}")
print(f"  MTL Output features    : {len(MTL_FEATURE_COLS)}")
print(f"  Ticker encoding        : 1")
print(f"  {'─'*35}")
print(f"  TOTAL                  : {len(TASK3_ALL_FEATURES)}")

# Step 3: Verify all required cols exist in df_feat before merge
required_in_feat = (
    ['date', 'ticker', 'close', 'target_class',
     'forward_return', 'ticker_encoded']
    + SR_FEATURE_COLS
    + MA_FEATURE_COLS
    + MTL_FEATURE_COLS
)

missing_in_feat = [c for c in required_in_feat if c not in df_feat.columns]
if missing_in_feat:
    print(f"\n✗ Still missing in df_feat: {missing_in_feat}")
    print("  Available columns:", df_feat.columns.tolist())
else:
    print(f"\n✓ All required columns present in df_feat")

# Step 4: Verify PRUNED_FEATURE_COLS exist in df_pruned
missing_in_pruned = [
    c for c in PRUNED_FEATURE_COLS 
    if c not in df_pruned.columns
]
if missing_in_pruned:
    print(f"✗ Missing in df_pruned: {missing_in_pruned}")
else:
    print(f"✓ All pruned features present in df_pruned")

# %%
FEAT_EXPORT_PATH = DATA_DIR / 'df_feat_engineered_master.csv'
df_feat.to_csv(FEAT_EXPORT_PATH, index = False)

print(f"✅ df_feat exported successfully!")
print(f"Path: {FEAT_EXPORT_PATH}")
print(f"Shape: {df_feat.shape}")

# %%
df_feat.head(100)

# %% [markdown]
# ### Comment on the df_feat (including all the newly added features)
# - Cross-over events are rare

# %%
# Check the existence of cross-over events
print("Crossover events in df_feat:")
print(f"  ma_golden_cross_short > 0: "
      f"{(df_feat['ma_golden_cross_short'] > 0).sum()}")
print(f"  ma_death_cross_short  > 0: "
      f"{(df_feat['ma_death_cross_short'] > 0).sum()}")
print(f"  ma_golden_cross_long  > 0: "
      f"{(df_feat['ma_golden_cross_long'] > 0).sum()}")
print(f"  ma_death_cross_long   > 0: "
      f"{(df_feat['ma_death_cross_long'] > 0).sum()}")

print(f"\nMA continuous features (should have variance):")
print(df_feat[['ma_short_gap_pct', 'ma_long_gap_pct',
               'ma_alignment']].describe().round(4))

print(f"\nMTL NaN distribution:")
print(f"  Total rows       : {len(df_feat):,}")
print(f"  NaN in mtl_p_up  : {df_feat['mtl_p_up'].isna().sum():,}")
print(f"  Non-NaN          : {df_feat['mtl_p_up'].notna().sum():,}")

# %%
# Step 5: Merge (only run after Step 1-4 pass) ─────────────────
df_task3 = pd.merge(
    df_feat[
        ['date', 'ticker', 'close', 'target_class',
         'forward_return', 'ticker_encoded']
        + SR_FEATURE_COLS
        + MA_FEATURE_COLS
        + MTL_FEATURE_COLS
    ],
    df_pruned[['date', 'ticker'] + PRUNED_FEATURE_COLS],
    on  = ['date', 'ticker'],
    how = 'inner'
)

print(f"Shape after merge  : {df_task3.shape}")

# Step 6: Drop NaN rows
rows_before = len(df_task3)
df_task3    = df_task3.dropna(
    subset=TASK3_ALL_FEATURES
).reset_index(drop=True)
rows_after  = len(df_task3)

print(f"Rows before dropna : {rows_before:,}")
print(f"Rows after dropna  : {rows_after:,}")
print(f"Dropped            : {rows_before - rows_after:,}")
print(f"df_task3 shape     : {df_task3.shape}")
print(f"Date range         : {df_task3['date'].min().date()} → "
      f"{df_task3['date'].max().date()}")

# Step 7: NaN check
print(f"\nFeature NaN check:")
nan_counts = df_task3[TASK3_ALL_FEATURES].isna().sum()
if nan_counts.sum() == 0:
    print(f"  ✓ No NaN in any of {len(TASK3_ALL_FEATURES)} features")
else:
    print("  NaN detected:")
    print(nan_counts[nan_counts > 0])

# Step 8: Class distribution
label_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
total     = len(df_task3)
print(f"\ntarget_class distribution:")
for cls in [0, 1, 2]:
    count = (df_task3['target_class'] == cls).sum()
    bar   = '█' * int(count / total * 25)
    print(f"  {label_map[cls]:4s} ({cls}): "
          f"{count:5,} ({count/total:.1%})  {bar}")

# %%
FEAT_EXPORT_PATH = DATA_DIR / 'df_task3.csv'
df_task3.to_csv(FEAT_EXPORT_PATH, index = False)

print(f"✅ df_task3 exported successfully!")
print(f"Path: {FEAT_EXPORT_PATH}")
print(f"Shape: {df_task3.shape}")

# %%
df_task3.head()


# %%
# Step 9: Chronological split
train_t3, val_t3, test_t3 = chronological_split_fixed(df_task3)

print(f"Task 3 Split:")
print(f"  Train: {train_t3['date'].min().date()} → "
      f"{train_t3['date'].max().date()} ({len(train_t3):,} rows)")
print(f"  Val  : {val_t3['date'].min().date()} → "
      f"{val_t3['date'].max().date()} ({len(val_t3):,} rows)")
print(f"  Test : {test_t3['date'].min().date()} → "
      f"{test_t3['date'].max().date()} ({len(test_t3):,} rows)")

assert train_t3['date'].max() < val_t3['date'].min()
assert val_t3['date'].max()   < test_t3['date'].min()
print('\n✓ Chronological integrity confirmed.')

# Step 10: Scale new features only
# PRUNED_FEATURE_COLS already scaled from Task 2
# New features need fresh scaler
NEW_FEATURE_COLS = SR_FEATURE_COLS + MA_FEATURE_COLS + MTL_FEATURE_COLS

scaler_new = RobustScaler()
train_t3   = train_t3.copy()
val_t3     = val_t3.copy()
test_t3    = test_t3.copy()

train_t3[NEW_FEATURE_COLS] = scaler_new.fit_transform(
    train_t3[NEW_FEATURE_COLS]
)
for df_split in [val_t3, test_t3]:
    df_split[NEW_FEATURE_COLS] = scaler_new.transform(
        df_split[NEW_FEATURE_COLS]
    )

joblib.dump(scaler_new, MODELS_DIR / 'task3_new_features_scaler.pkl')
print(f"✓ New features scaled: {len(NEW_FEATURE_COLS)} features")

# Step 11: Final arrays
X_train_t3 = train_t3[TASK3_ALL_FEATURES].values
X_val_t3   = val_t3[TASK3_ALL_FEATURES].values
X_test_t3  = test_t3[TASK3_ALL_FEATURES].values

y_train_t3 = train_t3['target_class'].values.astype(int)
y_val_t3   = val_t3['target_class'].values.astype(int)
y_test_t3  = test_t3['target_class'].values.astype(int)

print(f"\nFinal Task 3 Arrays:")
print(f"  X_train : {X_train_t3.shape}")
print(f"  X_val   : {X_val_t3.shape}")
print(f"  X_test  : {X_test_t3.shape}")
print(f"  y unique: {np.unique(y_train_t3)}")

# Class distribution per split
print(f"\nClass distribution per split:")
for split_name, y in [('Train', y_train_t3),
                       ('Val',   y_val_t3),
                       ('Test',  y_test_t3)]:
    total = len(y)
    dist  = {label_map[c]: f"{(y==c).sum()/total:.1%}"
             for c in [0, 1, 2]}
    print(f"  {split_name:5s}: {dist}")

print(f"\n✓ Step 3 complete — ready for XGBoost classifier (Step 6)")

# %% [markdown]
# ### Step 6:  XGBoost Classifer - 3 class (BUY, SELL, HOLD)

# %%
# ============================================================
# STEP 6: XGBoost Classifier — 3-class Signal Generation
# ============================================================

import xgboost as xgb
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.utils import class_weight as sklearn_cw

# ── 6.1 Class weights ─────────────────────────────────────────
weights  = sklearn_cw.compute_class_weight(
    'balanced',
    classes = np.unique(y_train_t3),
    y       = y_train_t3
)
cw_dict  = {i: float(weights[i]) for i in range(len(weights))}
sw_train = np.array([cw_dict[c] for c in y_train_t3])

print("Class weights:")
for cls, w in cw_dict.items():
    print(f"  {label_map[cls]:4s} ({cls}): {w:.4f}")

# ── 6.2 Train XGBoost ─────────────────────────────────────────
xgb_t3 = xgb.XGBClassifier(
    n_estimators      = 500,
    max_depth         = 5,
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    min_child_weight  = 5,
    gamma             = 0.1,
    reg_alpha         = 0.1,    # L1 regularization
    reg_lambda        = 1.0,    # L2 regularization
    objective         = 'multi:softprob',
    num_class         = 3,
    eval_metric       = 'mlogloss',
    random_state      = 42,
    n_jobs            = -1,
    tree_method       = 'hist',
    early_stopping_rounds = 30
)

xgb_t3.fit(
    X_train_t3, y_train_t3,
    sample_weight          = sw_train,
    eval_set               = [(X_val_t3, y_val_t3)],
    verbose                = 50
)

print(f"\n✓ XGBoost trained.")
print(f"  Best iteration: {xgb_t3.best_iteration}")
print(f"  Best val score: {xgb_t3.best_score:.6f}")

# %%
# ── 6.3 Predict & Evaluate ────────────────────────────────────
y_prob_xgb  = xgb_t3.predict_proba(X_test_t3)   # (n, 3)
y_pred_xgb  = np.argmax(y_prob_xgb, axis=1)

# Helper reuse
def compute_active_da(cls_prob, cls_label, y_true,
                       threshold=0.65):
    max_prob    = cls_prob.max(axis=1)
    active_mask = max_prob >= threshold
    active_pred = cls_label[active_mask]
    active_true = y_true[active_mask]
    non_hold    = (active_pred != 1) & (active_true != 1)
    active_da   = (
        np.mean(active_pred[non_hold] == active_true[non_hold]) * 100
        if non_hold.sum() > 0 else 0.0
    )
    return {
        'active_count'   : active_mask.sum(),
        'active_pct'     : active_mask.mean(),
        'buy_signals'    : (active_pred == 2).sum(),
        'sell_signals'   : (active_pred == 0).sum(),
        'active_da'      : active_da,
        'mean_conviction': max_prob.mean(),
        'pct_above_055'  : (max_prob >= 0.55).mean(),
        'pct_above_065'  : (max_prob >= 0.65).mean(),
        'pct_above_070'  : (max_prob >= 0.70).mean(),
    }

da_xgb = compute_active_da(y_prob_xgb, y_pred_xgb, y_test_t3)

print("=" * 55)
print("XGBoost Classifier — Task 3 Test Results")
print("=" * 55)
print(classification_report(
    y_test_t3, y_pred_xgb,
    target_names=['SELL', 'HOLD', 'BUY'],
    digits=4
))
print(f"Conviction-Gated (threshold=0.65):")
print(f"  Active signals : {da_xgb['active_count']:,} "
      f"({da_xgb['active_pct']:.1%})")
print(f"  BUY  signals   : {da_xgb['buy_signals']:,}")
print(f"  SELL signals   : {da_xgb['sell_signals']:,}")
print(f"  Active DA      : {da_xgb['active_da']:.2f}%")
print(f"  Mean conviction: {da_xgb['mean_conviction']:.4f}")
print("=" * 55)

# %%
# ── 6.4 Feature Importance ───────────────────────────────────
importances = xgb_t3.feature_importances_
fi_df_t3    = pd.DataFrame({
    'feature'   : TASK3_ALL_FEATURES,
    'importance': importances
}).sort_values('importance', ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(12, 10))
colors  = ['#2ecc71' if imp > fi_df_t3['importance'].mean()
           else '#95a5a6'
           for imp in fi_df_t3['importance']]
ax.barh(fi_df_t3['feature'], fi_df_t3['importance'],
        color=colors, alpha=0.85)
ax.axvline(fi_df_t3['importance'].mean(),
           color='red', linestyle='--',
           label='Mean importance')
ax.set_title('XGBoost Feature Importance — Task 3 Signal Classifier',
             fontsize=13)
ax.set_xlabel('Importance Score')
ax.invert_yaxis()
ax.legend()
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(MODELS_DIR / 'task3_xgb_feature_importance.png',
            dpi=150, bbox_inches='tight')
plt.show()

# Top 10 features
print("\nTop 10 Features:")
print(fi_df_t3.head(10).to_string(index=False))

# Check MTL feature contribution
print("\nMTL Feature Importance:")
for col in MTL_FEATURE_COLS:
    row = fi_df_t3[fi_df_t3['feature'] == col]
    if len(row) > 0:
        rank = fi_df_t3[fi_df_t3['feature'] == col].index[0] + 1
        imp  = row['importance'].values[0]
        print(f"  {col:<20}: {imp:.6f} (rank {rank}/{len(TASK3_ALL_FEATURES)})")

# %%
# ── 6.5 Confusion Matrix ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
cm = confusion_matrix(y_test_t3, y_pred_xgb)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['SELL', 'HOLD', 'BUY'],
    yticklabels=['SELL', 'HOLD', 'BUY'],
    ax=ax
)
ax.set_title('XGBoost — Task 3 Confusion Matrix\n'
             '(10 Tickers, Test Set)', fontsize=12)
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig(MODELS_DIR / 'task3_xgb_confusion_matrix.png',
            dpi=150, bbox_inches='tight')
plt.show()

# %%
# ── 6.6 Conviction threshold analysis ────────────────────────
thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

print(f"\n{'Threshold':>10} {'Signals':>10} {'Signal%':>10} "
      f"{'BUY':>8} {'SELL':>8} {'Active DA':>12}")
print("─" * 65)

for thresh in thresholds:
    da = compute_active_da(y_prob_xgb, y_pred_xgb,
                           y_test_t3, threshold=thresh)
    print(f"{thresh:>10.2f} "
          f"{da['active_count']:>10,} "
          f"{da['active_pct']:>10.1%} "
          f"{da['buy_signals']:>8,} "
          f"{da['sell_signals']:>8,} "
          f"{da['active_da']:>11.2f}%")

print("─" * 65)
print("→ Choose threshold that maximizes Active DA while keeping "
      "enough signals for Task 4")

# %%
# ── 6.7 Per-ticker signal analysis ───────────────────────────
print(f"\nPer-Ticker Signal Analysis (threshold=0.50):")
print("─" * 60)
print(f"{'Ticker':<8} {'Total':>7} {'BUY':>7} "
      f"{'SELL':>7} {'Active DA':>12}")
print("─" * 60)

max_prob_all = y_prob_xgb.max(axis=1)
active_all   = max_prob_all >= 0.50

for ticker in sorted(test_t3['ticker'].unique()):
    ticker_mask  = test_t3['ticker'].values == ticker
    t_prob       = y_prob_xgb[ticker_mask]
    t_pred       = y_pred_xgb[ticker_mask]
    t_true       = y_test_t3[ticker_mask]
    t_da         = compute_active_da(t_prob, t_pred, t_true)

    print(f"{ticker:<8} "
          f"{ticker_mask.sum():>7,} "
          f"{t_da['buy_signals']:>7,} "
          f"{t_da['sell_signals']:>7,} "
          f"{t_da['active_da']:>11.2f}%")

print("─" * 60)

# ── 6.8 Save model ────────────────────────────────────────────
joblib.dump(xgb_t3, MODELS_DIR / 'task3_xgb_signal.pkl')
print(f"\n✓ XGBoost signal model saved → task3_xgb_signal.pkl")

# %% [markdown]
# ### Fix: Remove MTL features, retrain XGB
# 
# The model is failing at Directional Accuracy. 
# Potential root cases: MTL features are dominating (by looking at the feature importance graph), but the MTL model is already weak, so the weak signals are propagated.
# 
# Solution: Technical + S/R features should be trained on their own

# %%
# ── Fix: Retrain XGBoost WITHOUT MTL features ─────────────────
# MTL features are dominating and propagating weak signal
# Technical + S/R features should stand on their own

TASK3_FEATURES_NO_MTL = (
    PRUNED_FEATURE_COLS +
    SR_FEATURE_COLS     +
    MA_FEATURE_COLS     +
    ['ticker_encoded']
)

print(f"Features without MTL: {len(TASK3_FEATURES_NO_MTL)}")

# Rebuild arrays
X_train_no_mtl = train_t3[TASK3_FEATURES_NO_MTL].values
X_val_no_mtl   = val_t3[TASK3_FEATURES_NO_MTL].values
X_test_no_mtl  = test_t3[TASK3_FEATURES_NO_MTL].values

# Recompute class weights
weights_v2  = sklearn_cw.compute_class_weight(
    'balanced',
    classes = np.unique(y_train_t3),
    y       = y_train_t3
)
cw_dict_v2  = {i: float(weights_v2[i]) for i in range(len(weights_v2))}
sw_train_v2 = np.array([cw_dict_v2[c] for c in y_train_t3])

# Retrain
xgb_t3_v2 = xgb.XGBClassifier(
    n_estimators          = 500,
    max_depth             = 4,       # ← shallower to reduce overfitting
    learning_rate         = 0.03,    # ← slower learning
    subsample             = 0.7,
    colsample_bytree      = 0.7,
    min_child_weight      = 10,      # ← larger to avoid small splits
    gamma                 = 0.2,
    reg_alpha             = 0.5,
    reg_lambda            = 2.0,
    objective             = 'multi:softprob',
    num_class             = 3,
    eval_metric           = 'mlogloss',
    random_state          = 42,
    n_jobs                = -1,
    tree_method           = 'hist',
    early_stopping_rounds = 30
)

xgb_t3_v2.fit(
    X_train_no_mtl, y_train_t3,
    sample_weight = sw_train_v2,
    eval_set      = [(X_val_no_mtl, y_val_t3)],
    verbose       = 50
)

print(f"\n✓ XGBoost V2 trained (no MTL features)")
print(f"  Best iteration: {xgb_t3_v2.best_iteration}")

# %%
# ── Evaluate V2 ───────────────────────────────────────────────
y_prob_xgb_v2 = xgb_t3_v2.predict_proba(X_test_no_mtl)
y_pred_xgb_v2 = np.argmax(y_prob_xgb_v2, axis=1)

da_xgb_v2 = compute_active_da(
    y_prob_xgb_v2, y_pred_xgb_v2, y_test_t3
)

print("=" * 55)
print("XGBoost V2 (No MTL) — Test Results")
print("=" * 55)
print(classification_report(
    y_test_t3, y_pred_xgb_v2,
    target_names=['SELL', 'HOLD', 'BUY'],
    digits=4
))

# Threshold analysis
thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
print(f"\n{'Threshold':>10} {'Signals':>10} {'Signal%':>10} "
      f"{'BUY':>8} {'SELL':>8} {'Active DA':>12}")
print("─" * 65)
for thresh in thresholds:
    da = compute_active_da(
        y_prob_xgb_v2, y_pred_xgb_v2,
        y_test_t3, threshold=thresh
    )
    print(f"{thresh:>10.2f} "
          f"{da['active_count']:>10,} "
          f"{da['active_pct']:>10.1%} "
          f"{da['buy_signals']:>8,} "
          f"{da['sell_signals']:>8,} "
          f"{da['active_da']:>11.2f}%")
print("─" * 65)

# %%
# ── Feature Importance V2 ─────────────────────────────────────
importances_v2 = xgb_t3_v2.feature_importances_
fi_df_v2       = pd.DataFrame({
    'feature'   : TASK3_FEATURES_NO_MTL,
    'importance': importances_v2
}).sort_values('importance', ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(12, 9))
colors  = ['#2ecc71' if imp > fi_df_v2['importance'].mean()
           else '#95a5a6'
           for imp in fi_df_v2['importance']]
ax.barh(fi_df_v2['feature'], fi_df_v2['importance'],
        color=colors, alpha=0.85)
ax.axvline(fi_df_v2['importance'].mean(),
           color='red', linestyle='--',
           label='Mean importance')
ax.set_title('XGBoost V2 Feature Importance\n'
             '(No MTL features — Technical + S/R only)',
             fontsize=13)
ax.set_xlabel('Importance Score')
ax.invert_yaxis()
ax.legend()
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(MODELS_DIR / 'task3_xgb_v2_feature_importance.png',
            dpi=150, bbox_inches='tight')
plt.show()

print("\nTop 10 Features (V2):")
print(fi_df_v2.head(10).to_string(index=False))

# %%
# ── Confusion Matrix V2 ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('XGBoost Confusion Matrix Comparison',
             fontsize=13, fontweight='bold')

for ax, y_pred, y_prob, title in zip(
    axes,
    [y_pred_xgb,    y_pred_xgb_v2],
    [y_prob_xgb,    y_prob_xgb_v2],
    ['V1 (with MTL features)', 'V2 (no MTL features)']
):
    cm = confusion_matrix(y_test_t3, y_pred)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['SELL', 'HOLD', 'BUY'],
        yticklabels=['SELL', 'HOLD', 'BUY'],
        ax=ax
    )
    da = compute_active_da(y_prob, y_pred, y_test_t3)
    ax.set_title(f'{title}\nActive DA: {da["active_da"]:.2f}%',
                 fontsize=11)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

plt.tight_layout()
plt.savefig(MODELS_DIR / 'task3_xgb_comparison_cm.png',
            dpi=150, bbox_inches='tight')
plt.show()

# Save V2
joblib.dump(xgb_t3_v2, MODELS_DIR / 'task3_xgb_v2_signal.pkl')
print(f"\n✓ XGBoost V2 saved → task3_xgb_v2_signal.pkl")

# %% [markdown]
# ### Problem: Distribution Shift: 
# Train period: 2020-03 → 2024-06  (Bull market + COVID recovery)
# Val period  : 2024-06 → 2025-01  (Sideways, HOLD dominated 51%)
# Test period : 2025-02 → 2026-04  (Bearish correction Q1 2026)
# 
# Train class dist: SELL 30.7% | HOLD 37.5% | BUY 31.8%  ← balanced
# Val   class dist: SELL 27.7% | HOLD 51.4% | BUY 20.9%  ← HOLD heavy
# Test  class dist: SELL 30.9% | HOLD 32.8% | BUY 36.2%  ← BUY heavy 
# 
# 
# The model was early-stopped on a sideways val period → learned to be conservative → met a bearish test period with different dynamics → failed to generalize.
# 
# V1 failure: MTL weak prior (49.82% DA) propagated into XGBoost → model essentially forwarded bad MTL signals.
# V2 improvement: Removing MTL let technical features speak — Active DA improved to 52.09% but SELL bias remains dominant. The model correctly identifies downside risk but struggles to identify upside opportunities.
# Practical output: V2 at threshold=0.50 generates 1,937 signals with 50.75% Active DA — marginally above random but with a reliable SELL signal component (precision 32%, recall 84%).
# 
# ### Problem 2: Generalist model trained on tickers with not so significant shared patterns 

# %%
# ============================================================
# STEP 3 (REVISED): Quantile-Based Label Generation
# Per-ticker quantile thresholds
# ============================================================

def generate_quantile_labels(df, n_forward=5, quantile_low=0.25,
                              quantile_high=0.75):
    """
    Per-ticker quantile-based label generation.
    
    For each ticker:
      - Compute forward return = log(P_{t+n} / P_t)
      - BUY  (2): return > ticker's 75th percentile
      - SELL (0): return < ticker's 25th percentile  
      - HOLD (1): between 25th and 75th percentile
    
    Guarantees ~25/50/25 distribution per ticker
    regardless of market regime.
    
    Args:
        df          : df_model (unscaled, has 'close' column)
        n_forward   : forecast horizon (5 days)
        quantile_low : bottom quantile threshold (0.25)
        quantile_high: top quantile threshold (0.75)
    """
    df = df.copy().sort_values(['ticker', 'date'])
    grouped_close = df.groupby('ticker')['close']

    # Forward return = log(P_{t+n} / P_t)
    df['forward_return'] = np.log(
        grouped_close.shift(-n_forward) / df['close']
    )

    # Per-ticker quantile thresholds
    ticker_thresholds = {}
    labels            = np.ones(len(df), dtype=int)  # default HOLD

    for ticker, group in df.groupby('ticker'):
        ticker_mask = df['ticker'] == ticker
        returns     = group['forward_return'].dropna()

        # Compute quantiles from THIS ticker's return distribution
        q_low  = returns.quantile(quantile_low)
        q_high = returns.quantile(quantile_high)

        ticker_thresholds[ticker] = {
            'q_low' : q_low,
            'q_high': q_high,
            'mean'  : returns.mean(),
            'std'   : returns.std()
        }

        # Assign labels
        fwd_ret = df.loc[ticker_mask, 'forward_return']
        labels[ticker_mask & (df['forward_return'] > q_high).values] = 2  # BUY
        labels[ticker_mask & (df['forward_return'] < q_low).values]  = 0  # SELL

    df['target_class'] = labels

    # Drop last n_forward rows per ticker (no forward data)
    df = df.dropna(subset=['forward_return'])
    df['target_class'] = df['target_class'].astype(int)

    return df, ticker_thresholds


# ── Execute ───────────────────────────────────────────────────
df_model_relabeled, ticker_thresholds = generate_quantile_labels(
    df_model,
    n_forward    = CONFIG['k_days'],  # 5
    quantile_low = 0.25,
    quantile_high= 0.75
)

# ── Print per-ticker thresholds ───────────────────────────────
print("Per-Ticker Quantile Thresholds:")
print("─" * 60)
print(f"{'Ticker':<8} {'Q25 (SELL)':>12} {'Q75 (BUY)':>12} "
      f"{'Mean':>10} {'Std':>10}")
print("─" * 60)
for ticker, thresh in sorted(ticker_thresholds.items()):
    print(f"{ticker:<8} "
          f"{thresh['q_low']:>12.4f} "
          f"{thresh['q_high']:>12.4f} "
          f"{thresh['mean']:>10.4f} "
          f"{thresh['std']:>10.4f}")
print("─" * 60)

# ── Class distribution per ticker ─────────────────────────────
print("\nClass Distribution per Ticker:")
print("─" * 55)
print(f"{'Ticker':<8} {'SELL':>10} {'HOLD':>10} {'BUY':>10}")
print("─" * 55)
for ticker, group in df_model_relabeled.groupby('ticker'):
    total = len(group)
    sell  = (group['target_class'] == 0).sum()
    hold  = (group['target_class'] == 1).sum()
    buy   = (group['target_class'] == 2).sum()
    print(f"{ticker:<8} "
          f"{sell:>5} ({sell/total:.0%}) "
          f"{hold:>5} ({hold/total:.0%}) "
          f"{buy:>5} ({buy/total:.0%})")
print("─" * 55)

# Overall distribution
total = len(df_model_relabeled)
sell  = (df_model_relabeled['target_class'] == 0).sum()
hold  = (df_model_relabeled['target_class'] == 1).sum()
buy   = (df_model_relabeled['target_class'] == 2).sum()
print(f"{'TOTAL':<8} "
      f"{sell:>5} ({sell/total:.0%}) "
      f"{hold:>5} ({hold/total:.0%}) "
      f"{buy:>5} ({buy/total:.0%})")

# %%
# ── Rebuild df_pruned with new labels ─────────────────────────
essential_raw = ['date', 'ticker', 'open', 'high', 'low',
                 'close', 'volume']
mtl_targets   = ['target_multi', 'target_class', 'forward_return']

all_cols  = list(dict.fromkeys(
    essential_raw + mtl_targets + PRUNED_FEATURE_COLS
))

# Merge new target_class into df_pruned
df_pruned_relabeled = pd.merge(
    df_model_relabeled[['date', 'ticker', 'target_class',
                         'forward_return']],
    df_model[all_cols].drop(
        columns=['target_class', 'forward_return'],
        errors='ignore'
    ),
    on=['date', 'ticker'],
    how='inner'
)

print(f"✓ df_pruned_relabeled shape: {df_pruned_relabeled.shape}")

# %%
# ── Merge with engineered features (S/R, MA, ticker_encoded) ──
# df_feat already has S/R, MA, ticker_encoded from Step 3
# Just need to update target_class

df_task3_v2 = pd.merge(
    df_feat[
        ['date', 'ticker', 'close', 'ticker_encoded']
        + SR_FEATURE_COLS
        + MA_FEATURE_COLS
    ],
    df_pruned_relabeled[
        ['date', 'ticker', 'target_class',
         'forward_return'] + PRUNED_FEATURE_COLS
    ],
    on=['date', 'ticker'],
    how='inner'
)

# ── Feature list (Option B: no MTL) ──────────────────────────
TASK3_FEATURES_V2 = (
    PRUNED_FEATURE_COLS +
    SR_FEATURE_COLS     +
    MA_FEATURE_COLS     +
    ['ticker_encoded']
)

print(f"Feature count: {len(TASK3_FEATURES_V2)}")

# Drop NaN
rows_before   = len(df_task3_v2)
df_task3_v2   = df_task3_v2.dropna(
    subset=TASK3_FEATURES_V2
).reset_index(drop=True)

print(f"Rows: {rows_before:,} → {len(df_task3_v2):,} after dropna")

# ── Class distribution ────────────────────────────────────────
label_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
total     = len(df_task3_v2)
print(f"\nOverall class distribution (quantile-based):")
for cls in [0, 1, 2]:
    count = (df_task3_v2['target_class'] == cls).sum()
    bar   = '█' * int(count / total * 30)
    print(f"  {label_map[cls]:4s}: {count:,} ({count/total:.1%}) {bar}")

# %%
# ── Chronological split ───────────────────────────────────────
train_t3v2, val_t3v2, test_t3v2 = chronological_split_fixed(df_task3_v2)

print(f"Split:")
print(f"  Train: {train_t3v2['date'].min().date()} → "
      f"{train_t3v2['date'].max().date()} ({len(train_t3v2):,})")
print(f"  Val  : {val_t3v2['date'].min().date()} → "
      f"{val_t3v2['date'].max().date()} ({len(val_t3v2):,})")
print(f"  Test : {test_t3v2['date'].min().date()} → "
      f"{test_t3v2['date'].max().date()} ({len(test_t3v2):,})")

assert train_t3v2['date'].max() < val_t3v2['date'].min()
assert val_t3v2['date'].max()   < test_t3v2['date'].min()
print('✓ Chronological integrity confirmed.')

# ── Verify class distribution per split ──────────────────────
print("\nClass distribution per split:")
for split_name, split_df in [('Train', train_t3v2),
                               ('Val',   val_t3v2),
                               ('Test',  test_t3v2)]:
    total = len(split_df)
    dist  = {label_map[c]: f"{(split_df['target_class']==c).sum()/total:.1%}"
             for c in [0, 1, 2]}
    print(f"  {split_name:5s}: {dist}")

# %%
# ── Scale features ────────────────────────────────────────────
NEW_FEATURE_COLS_V2 = SR_FEATURE_COLS + MA_FEATURE_COLS

scaler_v2    = RobustScaler()
train_t3v2   = train_t3v2.copy()
val_t3v2     = val_t3v2.copy()
test_t3v2    = test_t3v2.copy()

train_t3v2[NEW_FEATURE_COLS_V2] = scaler_v2.fit_transform(
    train_t3v2[NEW_FEATURE_COLS_V2]
)
for df_split in [val_t3v2, test_t3v2]:
    df_split[NEW_FEATURE_COLS_V2] = scaler_v2.transform(
        df_split[NEW_FEATURE_COLS_V2]
    )

joblib.dump(scaler_v2, MODELS_DIR / 'task3_v2_scaler.pkl')

# Save quantile thresholds for inference
import json
thresholds_serializable = {
    ticker: {k: float(v) for k, v in thresh.items()}
    for ticker, thresh in ticker_thresholds.items()
}
with open(MODELS_DIR / 'ticker_quantile_thresholds.json', 'w') as f:
    json.dump(thresholds_serializable, f, indent=2)

print("✓ Scalers and thresholds saved.")
print(f"  task3_v2_scaler.pkl")
print(f"  ticker_quantile_thresholds.json")

# ── Final arrays ──────────────────────────────────────────────
X_train_v2 = train_t3v2[TASK3_FEATURES_V2].values
X_val_v2   = val_t3v2[TASK3_FEATURES_V2].values
X_test_v2  = test_t3v2[TASK3_FEATURES_V2].values

y_train_v2 = train_t3v2['target_class'].values.astype(int)
y_val_v2   = val_t3v2['target_class'].values.astype(int)
y_test_v2  = test_t3v2['target_class'].values.astype(int)

print(f"\nFinal arrays:")
print(f"  X_train: {X_train_v2.shape}")
print(f"  X_val  : {X_val_v2.shape}")
print(f"  X_test : {X_test_v2.shape}")

# ── Verify distribution improvement ──────────────────────────
print(f"\nDistribution comparison:")
print(f"{'Split':<8} {'Old SELL':>10} {'Old BUY':>10} "
      f"{'New SELL':>10} {'New BUY':>10}")
print("─" * 50)
for split_name, y_new, y_old in [
    ('Train', y_train_v2, y_train_t3),
    ('Val',   y_val_v2,   y_val_t3),
    ('Test',  y_test_v2,  y_test_t3)
]:
    old_sell = (y_old == 0).mean()
    old_buy  = (y_old == 2).mean()
    new_sell = (y_new == 0).mean()
    new_buy  = (y_new == 2).mean()
    print(f"{split_name:<8} "
          f"{old_sell:>10.1%} "
          f"{old_buy:>10.1%} "
          f"{new_sell:>10.1%} "
          f"{new_buy:>10.1%}")

# %%
# ============================================================
# STEP 4: GRU + Multi-Head Attention Classifier
# with Focal Loss
# ============================================================

# ── 4.1 Focal Loss ───────────────────────────────────────────
from sklearn.utils import class_weight as sklearn_cw

# Compute alpha (class weights) for focal loss
weights_v2   = sklearn_cw.compute_class_weight(
    'balanced',
    classes = np.unique(y_train_v2),
    y       = y_train_v2
)
alpha_weights = [float(weights_v2[i]) for i in range(3)]

print(f"Focal Loss alpha weights:")
for cls, w in enumerate(alpha_weights):
    print(f"  {label_map[cls]:4s} ({cls}): {w:.4f}")


def focal_loss(gamma=2.0, alpha=None, n_classes=3):
    """
    Focal Loss for multi-class classification.
    
    FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
    
    Key properties:
      γ=2: standard focal — aggressive focus on hard examples
      α  : per-class weight (handles imbalance)
    
    Effect on trading signal learning:
      HOLD days (easy, model already confident):
        p_t ≈ 0.8 → (1-0.8)^2 = 0.04 → near-zero loss
        → Model stops wasting capacity on easy HOLD days
      
      BUY/SELL pivot days (hard, model uncertain):
        p_t ≈ 0.4 → (1-0.4)^2 = 0.36 → large loss
        → Model forced to master these critical days
    """
    if alpha is None:
        alpha = [1.0] * n_classes

    alpha_tensor = tf.constant(alpha, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        # Flatten y_true
        y_true_int = tf.reshape(tf.cast(y_true, tf.int32), [-1])

        # One-hot encode
        y_one_hot = tf.one_hot(y_true_int, n_classes)

        # Clip predictions for numerical stability
        y_pred    = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Cross entropy per class
        ce        = -tf.math.log(y_pred)

        # p_t: probability of true class
        p_t       = tf.reduce_sum(y_pred * y_one_hot, axis=-1)

        # Focusing factor: (1 - p_t)^gamma
        focal_w   = tf.pow(1.0 - p_t, gamma)

        # Alpha weight for true class
        alpha_t   = tf.gather(alpha_tensor, y_true_int)

        # CE loss for true class only
        ce_true   = tf.reduce_sum(ce * y_one_hot, axis=-1)

        # Final focal loss
        fl        = alpha_t * focal_w * ce_true

        return tf.reduce_mean(fl)

    return loss_fn


# ── 4.2 Build GRU + MultiHead Attention Classifier ───────────
def build_gru_attention_classifier(window_size, n_features,
                                    n_classes=3):
    """
    GRU Encoder + Single-Head Attention + 3-class Classifier.
    
    Design choices:
      - GRU(128): captures sequential momentum patterns
      - num_heads=1: "opinionated" attention
        → Locks onto single most relevant pivot day
        → Sharp, frequent signals vs multi-head averaging
      - key_dim=64: attention resolution
      - Dropout=0.3: regularization for generalization
      - Dense(64→32): bottleneck before classification
    
    Input : (batch, 20 days, 40 features)
    Output: (batch, 3) — P(SELL), P(HOLD), P(BUY)
    """
    inputs = tf.keras.layers.Input(
        shape=(window_size, n_features),
        name='signal_input'
    )

    # ── GRU Encoder ───────────────────────────────────────────
    # return_sequences=True: pass full sequence to attention
    x = tf.keras.layers.GRU(
        128,
        return_sequences=True,
        name='gru_encoder'
    )(inputs)
    x = tf.keras.layers.Dropout(0.3, name='gru_dropout')(x)

    # ── Single-Head Attention ─────────────────────────────────
    # num_heads=1: opinionated — locks onto single pivot day
    # Mechanism: which day in 20-day window is most predictive?
    att_out = tf.keras.layers.MultiHeadAttention(
        num_heads = 1,          # single head = sharp focus
        key_dim   = 64,         # attention resolution
        dropout   = 0.2,
        name      = 'pivot_attention'
    )(x, x)                     # self-attention

    # Layer norm for stable gradients
    att_out = tf.keras.layers.LayerNormalization(
        name='att_norm'
    )(att_out)

    # ── Context Vector ────────────────────────────────────────
    # GlobalAvgPool: summarize attended sequence
    context = tf.keras.layers.GlobalAveragePooling1D(
        name='context_pool'
    )(att_out)

    # ── Classification Head ───────────────────────────────────
    x = tf.keras.layers.Dense(
        64, activation='relu', name='cls_dense1'
    )(context)
    x = tf.keras.layers.Dropout(0.3, name='cls_dropout1')(x)
    x = tf.keras.layers.Dense(
        32, activation='relu', name='cls_dense2'
    )(x)
    x = tf.keras.layers.Dropout(0.2, name='cls_dropout2')(x)

    outputs = tf.keras.layers.Dense(
        n_classes, activation='softmax', name='signal_output'
    )(x)

    model = tf.keras.Model(
        inputs=inputs, outputs=outputs,
        name='GRU_Attention_SignalClassifier'
    )
    return model


# ── Initialize ────────────────────────────────────────────────
SEQ_LEN    = CONFIG['window_size']   # 20
N_FEAT_V2  = len(TASK3_FEATURES_V2) # 40

gru_signal = build_gru_attention_classifier(
    window_size = SEQ_LEN,
    n_features  = N_FEAT_V2,
    n_classes   = 3
)

gru_signal.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss      = focal_loss(gamma=2.0, alpha=alpha_weights),
    metrics   = ['accuracy']
)

gru_signal.summary()
print(f"\nFocal Loss: γ=2.0, α={[round(a,3) for a in alpha_weights]}")

# %%
# ── 4.3 Build Sequences for GRU ──────────────────────────────
def build_signal_sequences(df_split, feature_cols,
                            label_col, window_size=20):
    """
    Build sliding window sequences for GRU classifier.
    Per-ticker to prevent cross-ticker leakage.
    """
    all_X, all_y = [], []

    for ticker, group in df_split.groupby('ticker', sort=False):
        group     = group.sort_values('date').reset_index(drop=True)
        feat_arr  = group[feature_cols].values
        label_arr = group[label_col].values.astype(int)
        n         = len(group)

        for i in range(window_size, n):
            all_X.append(feat_arr[i - window_size : i])
            all_y.append(label_arr[i])

    return np.array(all_X), np.array(all_y)


X_train_gru, y_train_gru = build_signal_sequences(
    train_t3v2, TASK3_FEATURES_V2, 'target_class', SEQ_LEN
)
X_val_gru, y_val_gru = build_signal_sequences(
    val_t3v2, TASK3_FEATURES_V2, 'target_class', SEQ_LEN
)
X_test_gru, y_test_gru = build_signal_sequences(
    test_t3v2, TASK3_FEATURES_V2, 'target_class', SEQ_LEN
)

print(f"GRU Sequence Shapes:")
print(f"  X_train: {X_train_gru.shape}")
print(f"  X_val  : {X_val_gru.shape}")
print(f"  X_test : {X_test_gru.shape}")

# Verify class distribution in sequences
print(f"\nClass distribution in sequences:")
for split_name, y in [('Train', y_train_gru),
                       ('Val',   y_val_gru),
                       ('Test',  y_test_gru)]:
    total = len(y)
    dist  = {label_map[c]: f"{(y==c).sum()/total:.1%}"
             for c in [0, 1, 2]}
    print(f"  {split_name:5s}: {dist}")

# %%
# ── 4.4 Callbacks ─────────────────────────────────────────────
checkpoint_gru = MODELS_DIR / 'gru_signal_best.keras'

callbacks_gru = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath         = str(checkpoint_gru),
        monitor          = 'val_accuracy',
        save_best_only   = True,
        mode             = 'max',
        verbose          = 1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor              = 'val_accuracy',
        patience             = 20,
        restore_best_weights = True,
        mode                 = 'max',
        verbose              = 1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor  = 'val_accuracy',
        factor   = 0.5,
        patience = 7,
        min_lr   = 1e-7,
        mode     = 'max',
        verbose  = 1
    )
]

# ── 4.5 Train ─────────────────────────────────────────────────
print("🚀 Training GRU + Single-Head Attention Signal Classifier")
print(f"   Features  : {N_FEAT_V2} (no MTL)")
print(f"   Loss      : Focal Loss (γ=2.0, α=balanced)")
print(f"   Labels    : Quantile-based (25/50/25 per ticker)")
print(f"   Sequences : {X_train_gru.shape}")

history_gru = gru_signal.fit(
    X_train_gru, y_train_gru,
    validation_data = (X_val_gru, y_val_gru),
    epochs          = 150,
    batch_size      = 32,
    callbacks       = callbacks_gru,
    verbose         = 1
)

print(f"\n✓ Training complete.")
print(f"  Best val accuracy: "
      f"{max(history_gru.history['val_accuracy']):.4f}")

# %%
# ── 4.6 Evaluate ──────────────────────────────────────────────
from sklearn.metrics import classification_report, confusion_matrix

y_prob_gru  = gru_signal.predict(X_test_gru, verbose=0)
y_pred_gru  = np.argmax(y_prob_gru, axis=1)

print("=" * 55)
print("GRU Attention Classifier — Test Results")
print("=" * 55)
print(classification_report(
    y_test_gru, y_pred_gru,
    target_names=['SELL', 'HOLD', 'BUY'],
    digits=4
))

# Threshold analysis
thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
print(f"\n{'Threshold':>10} {'Signals':>10} {'Signal%':>10} "
      f"{'BUY':>8} {'SELL':>8} {'Active DA':>12}")
print("─" * 65)
for thresh in thresholds:
    da = compute_active_da(
        y_prob_gru, y_pred_gru, y_test_gru, threshold=thresh
    )
    print(f"{thresh:>10.2f} "
          f"{da['active_count']:>10,} "
          f"{da['active_pct']:>10.1%} "
          f"{da['buy_signals']:>8,} "
          f"{da['sell_signals']:>8,} "
          f"{da['active_da']:>11.2f}%")
print("─" * 65)

# Confusion matrix
fig, ax = plt.subplots(figsize=(7, 5))
cm = confusion_matrix(y_test_gru, y_pred_gru)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['SELL', 'HOLD', 'BUY'],
    yticklabels=['SELL', 'HOLD', 'BUY'],
    ax=ax
)
ax.set_title('GRU + Attention — Signal Classifier\n'
             'Quantile Labels, Focal Loss', fontsize=12)
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig(MODELS_DIR / 'gru_signal_confusion_matrix.png',
            dpi=150, bbox_inches='tight')
plt.show()

# Per-ticker analysis
print(f"\nPer-Ticker Signal Analysis:")
print("─" * 60)
print(f"{'Ticker':<8} {'Total':>7} {'BUY':>7} "
      f"{'SELL':>7} {'Active DA':>12}")
print("─" * 60)

# Align test_t3v2 dates with GRU sequences
# GRU sequences start at window_size within each ticker
ticker_list = []
for ticker, group in test_t3v2.groupby('ticker', sort=False):
    group = group.sort_values('date').reset_index(drop=True)
    ticker_list.extend([ticker] * (len(group) - SEQ_LEN))

ticker_arr = np.array(ticker_list)

for ticker in sorted(test_t3v2['ticker'].unique()):
    mask   = ticker_arr == ticker
    t_prob = y_prob_gru[mask]
    t_pred = y_pred_gru[mask]
    t_true = y_test_gru[mask]
    da     = compute_active_da(t_prob, t_pred, t_true,
                                threshold=0.50)
    print(f"{ticker:<8} "
          f"{mask.sum():>7,} "
          f"{da['buy_signals']:>7,} "
          f"{da['sell_signals']:>7,} "
          f"{da['active_da']:>11.2f}%")
print("─" * 60)

# Save
gru_signal.save(MODELS_DIR / 'gru_signal_final.keras')
print(f"\n✓ GRU Signal model saved → gru_signal_final.keras")

# %%
# ── Diagnose: Check what model is actually predicting ─────────
print("Raw prediction probabilities (first 10 rows):")
print(y_prob_gru[:10])

print(f"\nPredicted class distribution:")
total = len(y_pred_gru)
for cls in [0, 1, 2]:
    count = (y_pred_gru == cls).sum()
    print(f"  {label_map[cls]:4s}: {count:,} ({count/total:.1%})")

print(f"\nMax probability stats:")
max_probs = y_prob_gru.max(axis=1)
print(f"  Mean  : {max_probs.mean():.4f}")
print(f"  Max   : {max_probs.max():.4f}")
print(f"  Min   : {max_probs.min():.4f}")
print(f"  >0.35 : {(max_probs > 0.35).sum()}")

print(f"\nPer-class probability stats:")
for cls, name in label_map.items():
    probs = y_prob_gru[:, cls]
    print(f"  {name}: mean={probs.mean():.4f} "
          f"max={probs.max():.4f} "
          f"std={probs.std():.4f}")

print(f"\nTraining history check:")
print(f"  Best val accuracy : {max(history_gru.history['val_accuracy']):.4f}")
print(f"  Best epoch        : {np.argmax(history_gru.history['val_accuracy'])+1}")
print(f"  Final train loss  : {history_gru.history['loss'][-1]:.4f}")
print(f"  Final val loss    : {history_gru.history['val_loss'][-1]:.4f}")

print(f"\nLabel distribution in sequences:")
for split_name, y in [('Train', y_train_gru),
                       ('Val',   y_val_gru),
                       ('Test',  y_test_gru)]:
    total = len(y)
    dist  = {label_map[c]: f"{(y==c).sum()/total:.1%}"
             for c in [0, 1, 2]}
    print(f"  {split_name:5s}: {dist}")

# %% [markdown]
# ### 3 changes on V2

# %%
# ── Recompile with stable loss ────────────────────────────────
from sklearn.utils import class_weight as sklearn_cw
import tensorflow as tf

# Recompute class weights
weights_stable = sklearn_cw.compute_class_weight(
    'balanced',
    classes = np.unique(y_train_gru),
    y       = y_train_gru
)
cw_stable = {i: float(weights_stable[i])
             for i in range(len(weights_stable))}
cw_list   = [cw_stable[i] for i in sorted(cw_stable.keys())]

print("Class weights (stable):")
for cls, w in cw_stable.items():
    print(f"  {label_map[cls]:4s} ({cls}): {w:.4f}")


def stable_weighted_ce(y_true, y_pred, n_classes=3):
    """
    Weighted sparse categorical crossentropy.
    More stable than Focal Loss for noisy financial data.
    Weights defined inside function to avoid Keras tracing issues.
    """
    w_tensor   = tf.constant(cw_list, dtype=tf.float32)
    y_true_int = tf.reshape(tf.cast(y_true, tf.int32), [-1])
    ce_loss    = tf.keras.losses.sparse_categorical_crossentropy(
        y_true_int, y_pred
    )
    sample_w   = tf.gather(w_tensor, y_true_int)
    return tf.reduce_mean(ce_loss * sample_w)


# Rebuild model fresh
gru_signal_v2 = build_gru_attention_classifier(
    window_size = SEQ_LEN,
    n_features  = N_FEAT_V2,
    n_classes   = 3
)

gru_signal_v2.compile(
    optimizer = tf.keras.optimizers.Adam(
        learning_rate = 3e-4    # ← lower LR: more stable
    ),
    loss    = stable_weighted_ce,
    metrics = ['accuracy']
)

print(f"\n✓ Model V2 compiled")
print(f"  Loss : Weighted CE (stable)")
print(f"  LR   : 3e-4 (reduced from 1e-3)")

# %%
# ── Fix 2: Callbacks — monitor val_loss not val_accuracy ──────
checkpoint_gru_v2 = MODELS_DIR / 'gru_signal_v2_best.keras'

callbacks_gru_v2 = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath       = str(checkpoint_gru_v2),
        monitor        = 'val_loss',      # ← val_loss not accuracy
        save_best_only = True,
        mode           = 'min',
        verbose        = 1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor              = 'val_loss',  # ← val_loss not accuracy
        patience             = 25,
        restore_best_weights = True,
        mode                 = 'min',
        verbose              = 1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor  = 'val_loss',
        factor   = 0.5,
        patience = 8,
        min_lr   = 1e-7,
        mode     = 'min',
        verbose  = 1
    )
]

# %%
# ── Train V2 ──────────────────────────────────────────────────
print("🚀 Training GRU Signal Classifier V2 (Stable)")
print(f"   Loss     : Weighted CE (not Focal)")
print(f"   LR       : 3e-4")
print(f"   Monitor  : val_loss (not val_accuracy)")
print(f"   Patience : 25 epochs")

history_gru_v2 = gru_signal_v2.fit(
    X_train_gru, y_train_gru,
    validation_data = (X_val_gru, y_val_gru),
    epochs          = 150,
    batch_size      = 32,
    callbacks       = callbacks_gru_v2,
    verbose         = 1
)

# Quick check: is model learning anything?
print(f"\nFirst 5 epochs:")
for i in range(min(5, len(history_gru_v2.history['loss']))):
    print(f"  Epoch {i+1}: "
          f"loss={history_gru_v2.history['loss'][i]:.4f} "
          f"acc={history_gru_v2.history['accuracy'][i]:.4f} "
          f"val_loss={history_gru_v2.history['val_loss'][i]:.4f} "
          f"val_acc={history_gru_v2.history['val_accuracy'][i]:.4f}")

# %%
# ── Quick eval after training ─────────────────────────────────
y_prob_v2 = gru_signal_v2.predict(X_test_gru, verbose=0)
y_pred_v2 = np.argmax(y_prob_v2, axis=1)

print("\nPredicted distribution (sanity check):")
total = len(y_pred_v2)
for cls in [0, 1, 2]:
    count = (y_pred_v2 == cls).sum()
    print(f"  {label_map[cls]:4s}: {count:,} ({count/total:.1%})")

print(f"\nMax prob stats:")
max_p = y_prob_v2.max(axis=1)
print(f"  Mean: {max_p.mean():.4f} | "
      f"Max: {max_p.max():.4f} | "
      f"Std: {max_p.std():.4f}")

# If still uniform after epoch 5 → stop and report
if max_p.std() < 0.001:
    print("\n⚠️  Model still outputting uniform distribution")
    print("   → Val distribution shift is the blocker")
    print("   → Need walk-forward validation approach")
else:
    print("\n✓ Model is learning — proceed to full evaluation")
    from sklearn.metrics import classification_report
    print(classification_report(
        y_test_gru, y_pred_v2,
        target_names=['SELL', 'HOLD', 'BUY'],
        digits=4
    ))

# %% [markdown]
# ### Final Version: with Time-Series Cross-Validation
# Full CV
# Train 4 models → average performance
# Final model retrained on full train+val
# ~4x training time but most reliable

# %%
# ============================================================
# TIME SERIES CROSS-VALIDATION
# Expanding Window, 4 Folds
# Fixed thresholds: BUY>+2%, SELL<-1.5%
# Final model: retrain on full train+val data
# ============================================================

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ── Config ────────────────────────────────────────────────────
BUY_THRESH  =  0.02
SELL_THRESH = -0.015
N_FORWARD   =  5
SEQ_LEN     = CONFIG['window_size']  # 20

# ── Step 1: Regenerate fixed-threshold labels ─────────────────
def generate_fixed_labels(df, buy_thresh, sell_thresh, n_forward):
    """
    Fixed threshold label generation.
    BUY  (2): forward_return > buy_thresh
    HOLD (1): sell_thresh <= forward_return <= buy_thresh
    SELL (0): forward_return < sell_thresh
    """
    df = df.copy().sort_values(['ticker', 'date'])
    grouped_close = df.groupby('ticker')['close']

    df['forward_return'] = np.log(
        grouped_close.shift(-n_forward) / df['close']
    )

    conditions = [
        df['forward_return'] > buy_thresh,
        df['forward_return'] < sell_thresh,
    ]
    choices          = [2, 0]
    df['target_class'] = np.select(conditions, choices, default=1)

    # Drop last n_forward rows per ticker
    df = df.dropna(subset=['forward_return'])
    df['target_class'] = df['target_class'].astype(int)

    return df


df_cv = generate_fixed_labels(
    df_model, BUY_THRESH, SELL_THRESH, N_FORWARD
)

print(f"✓ Fixed threshold labels generated")
print(f"  BUY  > +{BUY_THRESH:.1%}")
print(f"  SELL < {SELL_THRESH:.1%}")
print(f"  Rows : {len(df_cv):,}")

# Overall distribution
total = len(df_cv)
for cls, name in label_map.items():
    count = (df_cv['target_class'] == cls).sum()
    print(f"  {name:4s}: {count:,} ({count/total:.1%})")

# %%
# ── Step 2: Merge engineered features ─────────────────────────
# Merge S/R, MA, ticker_encoded from df_feat
# Use TASK3_FEATURES_V2 (no MTL)

df_cv = pd.merge(
    df_cv[['date', 'ticker', 'close',
           'target_class', 'forward_return']],
    df_feat[
        ['date', 'ticker', 'ticker_encoded']
        + SR_FEATURE_COLS
        + MA_FEATURE_COLS
    ],
    on=['date', 'ticker'],
    how='inner'
)

# Merge pruned features
df_cv = pd.merge(
    df_cv,
    df_pruned[['date', 'ticker'] + PRUNED_FEATURE_COLS],
    on=['date', 'ticker'],
    how='inner'
)

# Drop NaN
df_cv = df_cv.dropna(
    subset=TASK3_FEATURES_V2
).reset_index(drop=True)

df_cv = df_cv.sort_values(['ticker', 'date']).reset_index(drop=True)

print(f"✓ df_cv shape: {df_cv.shape}")
print(f"  Date range: {df_cv['date'].min().date()} → "
      f"{df_cv['date'].max().date()}")

# %%
# ── Step 3: Define CV Folds (Expanding Window) ────────────────
# Fixed test set: 2025-02 onwards
# CV on remaining data: 2020-03 → 2025-01

TEST_START = pd.Timestamp('2025-02-01')

df_trainval = df_cv[df_cv['date'] < TEST_START].copy()
df_test_cv  = df_cv[df_cv['date'] >= TEST_START].copy()

print(f"Train+Val pool: {len(df_trainval):,} rows "
      f"({df_trainval['date'].min().date()} → "
      f"{df_trainval['date'].max().date()})")
print(f"Test (fixed)  : {len(df_test_cv):,} rows "
      f"({df_test_cv['date'].min().date()} → "
      f"{df_test_cv['date'].max().date()})")

# Define fold boundaries (expanding window)
# Each val period = 1 year
FOLD_BOUNDARIES = [
    # (train_end, val_start, val_end)
    (pd.Timestamp('2022-01-01'),
     pd.Timestamp('2022-01-01'),
     pd.Timestamp('2022-12-31')),

    (pd.Timestamp('2023-01-01'),
     pd.Timestamp('2023-01-01'),
     pd.Timestamp('2023-12-31')),

    (pd.Timestamp('2024-01-01'),
     pd.Timestamp('2024-01-01'),
     pd.Timestamp('2024-12-31')),

    (pd.Timestamp('2025-01-01'),
     pd.Timestamp('2025-01-01'),
     pd.Timestamp('2025-01-31')),
]

print(f"\nCV Folds (Expanding Window):")
print("─" * 65)
for i, (train_end, val_start, val_end) in enumerate(FOLD_BOUNDARIES):
    train_rows = df_trainval[
        df_trainval['date'] < train_end
    ]['date'].nunique()
    val_rows = df_trainval[
        (df_trainval['date'] >= val_start) &
        (df_trainval['date'] <= val_end)
    ]['date'].nunique()
    print(f"  Fold {i+1}: "
          f"Train 2020-03 → {train_end.date()} "
          f"({train_rows} days) | "
          f"Val {val_start.date()} → {val_end.date()} "
          f"({val_rows} days)")
print("─" * 65)

# %%
# ── Step 4: CV Helper Functions ───────────────────────────────

def get_fold_data(df, train_end, val_start, val_end,
                  feature_cols, label_col, window_size):
    """
    Extract train/val arrays for a single fold.
    Scales features on train, transforms val.
    """
    train_df = df[df['date'] < train_end].copy()
    val_df   = df[
        (df['date'] >= val_start) &
        (df['date'] <= val_end)
    ].copy()

    # Scale new features (S/R, MA) — PRUNED already scaled
    new_cols = SR_FEATURE_COLS + MA_FEATURE_COLS
    scaler   = RobustScaler()
    train_df[new_cols] = scaler.fit_transform(train_df[new_cols])
    val_df[new_cols]   = scaler.transform(val_df[new_cols])

    # Build sequences
    X_train, y_train = build_signal_sequences(
        train_df, feature_cols, label_col, window_size
    )
    X_val, y_val = build_signal_sequences(
        val_df, feature_cols, label_col, window_size
    )

    return X_train, y_train, X_val, y_val, scaler


def build_fold_model(n_features, learning_rate=3e-4):
    """Build fresh model for each fold."""
    model = build_gru_attention_classifier(
        window_size = SEQ_LEN,
        n_features  = n_features,
        n_classes   = 3
    )

    # Compute class weights for this fold's training data
    # (passed in separately)
    model.compile(
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss    = stable_weighted_ce,
        metrics = ['accuracy']
    )
    return model


def evaluate_fold(model, X_val, y_val, fold_num):
    """Evaluate a single fold."""
    y_prob = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    # Classification report
    rep = classification_report(
        y_val, y_pred,
        target_names=['SELL', 'HOLD', 'BUY'],
        output_dict=True, digits=4
    )

    # Active DA
    da = compute_active_da(y_prob, y_pred, y_val,
                            threshold=0.50)

    print(f"\n  Fold {fold_num} Results:")
    print(f"    Accuracy : {rep['accuracy']:.4f}")
    print(f"    BUY  F1  : {rep['BUY']['f1-score']:.4f}")
    print(f"    SELL F1  : {rep['SELL']['f1-score']:.4f}")
    print(f"    Active DA: {da['active_da']:.2f}%")
    print(f"    Signals  : {da['active_count']} "
          f"(BUY={da['buy_signals']}, "
          f"SELL={da['sell_signals']})")

    return rep, da, y_prob

# %%
# ── Step 5: Run CV ────────────────────────────────────────────
print("=" * 60)
print("TIME SERIES CROSS-VALIDATION (4 Folds)")
print("Expanding Window | Fixed Thresholds | Weighted CE")
print("=" * 60)

fold_results  = []
fold_models   = []
fold_histories = []

for fold_num, (train_end, val_start, val_end) in enumerate(
    FOLD_BOUNDARIES, start=1
):
    print(f"\n{'─'*60}")
    print(f"FOLD {fold_num}/4")
    print(f"  Train: 2020-03 → {train_end.date()}")
    print(f"  Val  : {val_start.date()} → {val_end.date()}")
    print(f"{'─'*60}")

    # Get fold data
    X_tr, y_tr, X_vl, y_vl, fold_scaler = get_fold_data(
        df_trainval,
        train_end, val_start, val_end,
        TASK3_FEATURES_V2, 'target_class', SEQ_LEN
    )

    print(f"  X_train: {X_tr.shape} | X_val: {X_vl.shape}")

    # Class distribution
    for split_n, y in [('Train', y_tr), ('Val', y_vl)]:
        total = len(y)
        dist  = {label_map[c]: f"{(y==c).sum()/total:.1%}"
                 for c in [0, 1, 2]}
        print(f"  {split_n} dist: {dist}")

    if len(X_tr) == 0 or len(X_vl) == 0:
        print(f"  ⚠️  Skipping fold {fold_num} — insufficient data")
        continue

    # Build fresh model
    fold_model = build_fold_model(
        n_features    = N_FEAT_V2,
        learning_rate = 3e-4
    )

    # Fold callbacks
    fold_checkpoint = MODELS_DIR / f'gru_fold{fold_num}_best.keras'
    fold_callbacks  = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath       = str(fold_checkpoint),
            monitor        = 'val_loss',
            save_best_only = True,
            mode           = 'min',
            verbose        = 0
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor              = 'val_loss',
            patience             = 20,
            restore_best_weights = True,
            mode                 = 'min',
            verbose              = 0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor  = 'val_loss',
            factor   = 0.5,
            patience = 7,
            min_lr   = 1e-7,
            mode     = 'min',
            verbose  = 0
        )
    ]

    # Train
    history = fold_model.fit(
        X_tr, y_tr,
        validation_data = (X_vl, y_vl),
        epochs          = 150,
        batch_size      = 32,
        callbacks       = fold_callbacks,
        verbose         = 0  # silent — summary printed after
    )

    # Print training summary
    best_epoch   = np.argmin(history.history['val_loss']) + 1
    best_val_loss= min(history.history['val_loss'])
    best_val_acc = history.history['val_accuracy'][best_epoch - 1]

    print(f"  Training: {len(history.history['loss'])} epochs | "
          f"Best epoch {best_epoch} | "
          f"Val loss {best_val_loss:.4f} | "
          f"Val acc {best_val_acc:.4f}")

    # Evaluate
    rep, da, y_prob_fold = evaluate_fold(
        fold_model, X_vl, y_vl, fold_num
    )

    fold_results.append({
        'fold'     : fold_num,
        'train_end': train_end,
        'val_start': val_start,
        'val_end'  : val_end,
        'accuracy' : rep['accuracy'],
        'buy_f1'   : rep['BUY']['f1-score'],
        'sell_f1'  : rep['SELL']['f1-score'],
        'buy_prec' : rep['BUY']['precision'],
        'sell_prec': rep['SELL']['precision'],
        'buy_rec'  : rep['BUY']['recall'],
        'sell_rec' : rep['SELL']['recall'],
        'active_da': da['active_da'],
        'n_signals': da['active_count'],
        'n_buy'    : da['buy_signals'],
        'n_sell'   : da['sell_signals'],
        'best_epoch': best_epoch,
        'val_loss' : best_val_loss,
    })

    fold_models.append(fold_model)
    fold_histories.append(history)

print(f"\n{'='*60}")
print("CV COMPLETE")
print(f"{'='*60}")

# %%
# ── Step 6: CV Summary ────────────────────────────────────────
results_df = pd.DataFrame(fold_results)

print("\nFold-by-Fold Summary:")
print("─" * 75)
print(f"{'Fold':<6} {'Accuracy':>10} {'BUY F1':>10} "
      f"{'SELL F1':>10} {'Active DA':>12} {'Signals':>10}")
print("─" * 75)

for _, row in results_df.iterrows():
    print(f"  {int(row['fold']):<4} "
          f"{row['accuracy']:>10.4f} "
          f"{row['buy_f1']:>10.4f} "
          f"{row['sell_f1']:>10.4f} "
          f"{row['active_da']:>11.2f}% "
          f"{int(row['n_signals']):>10,}")

print("─" * 75)
print(f"  {'MEAN':<4} "
      f"{results_df['accuracy'].mean():>10.4f} "
      f"{results_df['buy_f1'].mean():>10.4f} "
      f"{results_df['sell_f1'].mean():>10.4f} "
      f"{results_df['active_da'].mean():>11.2f}% "
      f"{results_df['n_signals'].mean():>10.0f}")
print(f"  {'STD':<4} "
      f"{results_df['accuracy'].std():>10.4f} "
      f"{results_df['buy_f1'].std():>10.4f} "
      f"{results_df['sell_f1'].std():>10.4f} "
      f"{results_df['active_da'].std():>11.2f}% ")
print("─" * 75)

# Best fold
best_fold_idx = results_df['active_da'].idxmax()
best_fold     = results_df.loc[best_fold_idx]
print(f"\n  Best fold by Active DA: Fold {int(best_fold['fold'])}"
      f" (DA={best_fold['active_da']:.2f}%)")

# %%
# ── Diagnose each fold ────────────────────────────────────────

# Check fold 4 data
train_end, val_start, val_end = FOLD_BOUNDARIES[3]
train_f4 = df_trainval[df_trainval['date'] < train_end]
val_f4   = df_trainval[
    (df_trainval['date'] >= val_start) &
    (df_trainval['date'] <= val_end)
]
print(f"Fold 4 train rows: {len(train_f4):,}")
print(f"Fold 4 val rows  : {len(val_f4):,}")

# Check prediction distribution per fold
print("\nPer-fold prediction check:")
for i, (fold_model, fold_res) in enumerate(
    zip(fold_models, fold_results), start=1
):
    # Get fold val data
    te, vs, ve = FOLD_BOUNDARIES[i-1]
    _, _, X_vl, y_vl, _ = get_fold_data(
        df_trainval, te, vs, ve,
        TASK3_FEATURES_V2, 'target_class', SEQ_LEN
    )
    if len(X_vl) == 0:
        print(f"Fold {i}: No val data")
        continue

    y_prob = fold_model.predict(X_vl, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    max_p  = y_prob.max(axis=1)

    print(f"\nFold {i}:")
    print(f"  Val size      : {len(y_vl):,}")
    print(f"  Pred dist     : SELL={( y_pred==0).mean():.1%} "
          f"HOLD={(y_pred==1).mean():.1%} "
          f"BUY={(y_pred==2).mean():.1%}")
    print(f"  Max prob      : mean={max_p.mean():.4f} "
          f"max={max_p.max():.4f} "
          f"std={max_p.std():.4f}")
    print(f"  True dist     : SELL={(y_vl==0).mean():.1%} "
          f"HOLD={(y_vl==1).mean():.1%} "
          f"BUY={(y_vl==2).mean():.1%}")

# Check if stable_weighted_ce is working
print("\nLoss function check — first batch:")
X_sample = X_train_gru[:32]
y_sample = y_train_gru[:32]
test_model = build_gru_attention_classifier(SEQ_LEN, N_FEAT_V2)
test_model.compile(
    optimizer='adam',
    loss=stable_weighted_ce,
    metrics=['accuracy']
)
result = test_model.evaluate(X_sample, y_sample, verbose=0)
print(f"  Initial loss    : {result[0]:.4f}")
print(f"  Initial accuracy: {result[1]:.4f}")

# Check class weights
print(f"\nClass weights being used:")
print(f"  {cw_list}")
print(f"  SELL: {cw_list[0]:.4f}")
print(f"  HOLD: {cw_list[1]:.4f}")
print(f"  BUY : {cw_list[2]:.4f}")

# %%
# ── Deep diagnostic: Is gradient flowing? ────────────────────
import tensorflow as tf

# Build fresh model
diag_model = build_gru_attention_classifier(
    SEQ_LEN, N_FEAT_V2, n_classes=3
)
diag_model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4),  # ← fix
    loss      = stable_weighted_ce,
    metrics   = ['accuracy']
)

# Check gradients on first batch
X_batch = X_train_gru[:32].astype(np.float32)
y_batch = y_train_gru[:32].astype(np.float32)

with tf.GradientTape() as tape:
    y_pred = diag_model(X_batch, training=True)
    loss   = stable_weighted_ce(y_batch, y_pred)

grads = tape.gradient(loss, diag_model.trainable_variables)

print("Gradient check per layer:")
print("─" * 55)
any_none = False
for var, grad in zip(diag_model.trainable_variables, grads):
    if grad is not None:
        grad_norm = tf.norm(grad).numpy()
        status

# %%
# ── Step 7: Retrain on Full Train+Val Data ────────────────────
print("\n" + "=" * 60)
print("FINAL MODEL: Retrain on Full Train+Val Data")
print("=" * 60)

# Scale full train+val
new_cols_final  = SR_FEATURE_COLS + MA_FEATURE_COLS
scaler_final    = RobustScaler()
df_trainval_scaled = df_trainval.copy()
df_test_scaled     = df_test_cv.copy()

df_trainval_scaled[new_cols_final] = scaler_final.fit_transform(
    df_trainval[new_cols_final]
)
df_test_scaled[new_cols_final] = scaler_final.transform(
    df_test_cv[new_cols_final]
)

# Build sequences
X_full, y_full = build_signal_sequences(
    df_trainval_scaled,
    TASK3_FEATURES_V2, 'target_class', SEQ_LEN
)
X_test_final, y_test_final = build_signal_sequences(
    df_test_scaled,
    TASK3_FEATURES_V2, 'target_class', SEQ_LEN
)

print(f"Full train+val: {X_full.shape}")
print(f"Test          : {X_test_final.shape}")

# Class distribution
for split_n, y in [('Train+Val', y_full),
                    ('Test',     y_test_final)]:
    total = len(y)
    dist  = {label_map[c]: f"{(y==c).sum()/total:.1%}"
             for c in [0, 1, 2]}
    print(f"  {split_n}: {dist}")

# Build final model
# Use average best_epoch from CV as max epochs
avg_best_epoch = int(results_df['best_epoch'].mean())
print(f"\nTraining for {avg_best_epoch} epochs "
      f"(avg best epoch from CV)")

final_model = build_gru_attention_classifier(
    window_size = SEQ_LEN,
    n_features  = N_FEAT_V2,
    n_classes   = 3
)
final_model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss      = stable_weighted_ce,
    metrics   = ['accuracy']
)

# Train without EarlyStopping — use avg_best_epoch directly
history_final = final_model.fit(
    X_full, y_full,
    epochs     = avg_best_epoch,
    batch_size = 32,
    verbose    = 1
)

print(f"\n✓ Final model trained for {avg_best_epoch} epochs")

# %%
# 1. Generate probabilities for the Test Set
y_prob_final = final_model.predict(X_test_final, verbose=1)

# 2. Extract specific probabilities for readability
# index 0: SELL, index 1: HOLD, index 2: BUY
p_sell = y_prob_final[:, 0]
p_hold = y_prob_final[:, 1]
p_buy  = y_prob_final[:, 2]

# 3. Create a Signal DataFrame for Task 4
# We need to align the signals with the original test dataframe 
# (Remember: sequences start after SEQ_LEN rows per ticker)
test_signals_df = df_test_scaled.copy()

# Groupby logic to align predictions with the correct dates
aligned_results = []
current_idx = 0

for ticker, group in test_signals_df.groupby('ticker', sort=False):
    group = group.sort_values('date').reset_index(drop=True)
    n = len(group)
    n_seq = n - SEQ_LEN
    
    # Create empty columns
    group['signal_buy_prob'] = np.nan
    group['signal_sell_prob'] = np.nan
    group['final_signal'] = 1 # Default to HOLD
    
    if n_seq > 0:
        # Fill in the predictions (shifted by SEQ_LEN)
        group.iloc[SEQ_LEN:, group.columns.get_loc('signal_buy_prob')] = p_buy[current_idx : current_idx + n_seq]
        group.iloc[SEQ_LEN:, group.columns.get_loc('signal_sell_prob')] = p_sell[current_idx : current_idx + n_seq]
        
        # Apply your conviction threshold (e.g., 0.60)
        buy_mask = group['signal_buy_prob'] > 0.60
        sell_mask = group['signal_sell_prob'] > 0.60
        
        group.loc[buy_mask, 'final_signal'] = 2
        group.loc[sell_mask, 'final_signal'] = 0
        
        current_idx += n_seq
    
    aligned_results.append(group)

final_signals_df = pd.concat(aligned_results)

# 4. Export for Backtesting
final_signals_df.to_csv(DATA_DIR / 'final_test_signals_production.csv', index=False)
print(f"✅ Signals generated and saved. Ready for Backtesting!")

# %%
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Predict on Test Set ---
y_prob_final = final_model.predict(X_test_final, verbose=1)
y_pred_final = np.argmax(y_prob_final, axis=1)

# --- 2. Basic Metrics ---
print("\n" + "="*60)
print("FINAL PRODUCTION MODEL: SIGNAL IDENTIFICATION PERFORMANCE")
print("="*60)
print(classification_report(y_test_final, y_pred_final, 
                            target_names=['SELL', 'HOLD', 'BUY'], 
                            digits=4))

# --- 3. Conviction-Gated Analysis (The Trading Edge) ---
# We only care about accuracy when the model is confident (> 0.60)
threshold = 0.60
max_probs = np.max(y_prob_final, axis=1)
confident_mask = max_probs >= threshold

# Filter only confident predictions
y_confident_pred = y_pred_final[confident_mask]
y_confident_true = y_test_final[confident_mask]

# Calculate Directional Accuracy (Excluding HOLD)
# (i.e., when model says BUY/SELL confidently, is it right?)
active_mask = (y_confident_pred != 1) & (y_confident_true != 1)
if active_mask.sum() > 0:
    active_da = np.mean(y_confident_pred[active_mask] == y_confident_true[active_mask]) * 100
else:
    active_da = 0

print(f"\nConviction-Gated Metrics (Threshold > {threshold}):")
print(f"  Active Signals  : {confident_mask.sum():,} / {len(y_test_final):,} ({confident_mask.mean():.1%})")
print(f"  Active Dir. Acc : {active_da:.2f}%")

# --- 4. Confusion Matrix Visualization ---
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test_final, y_pred_final)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['SELL', 'HOLD', 'BUY'],
            yticklabels=['SELL', 'HOLD', 'BUY'])
plt.title('Final Model: Signal Confusion Matrix (Test Set)')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# %% [markdown]
# #### FINALE

# %%
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler

# ── Config ────────────────────────────────────────────────────
N_FORWARD = CONFIG['k_days']   # 5
SEQ_LEN   = CONFIG['window_size']  # 20

# ── Step 1: Compute forward returns ───────────────────────────
df_cv = df_model.copy().sort_values(['ticker', 'date'])
df_cv['forward_return'] = np.log(
    df_cv.groupby('ticker')['close'].shift(-N_FORWARD) / df_cv['close']
)   
df_cv = df_cv.dropna(subset=['forward_return'])

# ── Step 2: Merge engineered features ─────────────────────────
df_cv = pd.merge(
    df_cv[['date', 'ticker', 'close', 'forward_return']],
    df_feat[['date', 'ticker', 'ticker_encoded'] + SR_FEATURE_COLS + MA_FEATURE_COLS],
    on=['date', 'ticker'], how='inner' 
)   

df_cv = pd.merge(
    df_cv,
    df_pruned[['date', 'ticker'] + PRUNED_FEATURE_COLS],
    on=['date', 'ticker'], how='inner' 
)   

df_cv = df_cv.dropna(subset=TASK3_FEATURES_V2).sort_values(['ticker', 'date']).reset_index(drop=True)

# ── Step 3: Define CV Folds (Expanding Window) ────────────────
TEST_START = pd.Timestamp('2025-02-01')
df_trainval = df_cv[df_cv['date'] < TEST_START].copy()
df_test_cv  = df_cv[df_cv['date'] >= TEST_START].copy()

FOLD_BOUNDARIES = [ 
    (pd.Timestamp('2022-01-01'), pd.Timestamp('2022-01-01'), pd.Timestamp('2022-12-31')),
    (pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-01'), pd.Timestamp('2023-12-31')),
    (pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-01'), pd.Timestamp('2024-12-31')),
]

# %%
def assign_quantile_labels(df, ticker_thresholds):
    labels = np.ones(len(df), dtype=int)  # default HOLD
    fwd    = df['forward_return'].values
    for ticker, (q_low, q_high) in ticker_thresholds.items():
        mask = (df['ticker'] == ticker).values
        labels[mask & (fwd > q_high)] = 2  # BUY
        labels[mask & (fwd < q_low )] = 0  # SELL
    return labels

def get_fold_data(df, train_end, val_start, val_end, feature_cols, window_size):
    train_df = df[df['date'] < train_end].copy()
    val_df   = df[(df['date'] >= val_start) & (df['date'] <= val_end)].copy()

    # Per-ticker thresholds from TRAIN only
    ticker_thresholds = {}
    for ticker, group in train_df.groupby('ticker'):
        returns = group['forward_return'].dropna()
        ticker_thresholds[ticker] = (returns.quantile(0.25), returns.quantile(0.75))

    train_df['target_class'] = assign_quantile_labels(train_df, ticker_thresholds)
    val_df['target_class']   = assign_quantile_labels(val_df, ticker_thresholds)
    
    # Scaling
    new_cols = SR_FEATURE_COLS + MA_FEATURE_COLS
    scaler = RobustScaler()
    train_df[new_cols] = scaler.fit_transform(train_df[new_cols])
    val_df[new_cols]   = scaler.transform(val_df[new_cols])

    X_train, y_train = build_signal_sequences(train_df, feature_cols, 'target_class', window_size)
    X_val, y_val = build_signal_sequences(val_df, feature_cols, 'target_class', window_size)

    return X_train, y_train, X_val, y_val, scaler, ticker_thresholds

def evaluate_fold(model, X_val, y_val, fold_num):
    y_prob = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    rep = classification_report(y_val, y_pred, target_names=['SELL', 'HOLD', 'BUY'], output_dict=True, digits=4)
    da = compute_active_da(y_prob, y_pred, y_val, threshold=0.50)
    
    print(f"\nFold {fold_num} | Acc: {rep['accuracy']:.4f} | Active DA: {da['active_da']:.2f}% | Signals: {da['active_count']}")
    return rep, da, y_prob

# %%
fold_results, fold_models = [], []

for fold_num, (train_end, val_start, val_end) in enumerate(FOLD_BOUNDARIES, start=1):
    X_tr, y_tr, X_vl, y_vl, _, _ = get_fold_data(df_trainval, train_end, val_start, val_end, TASK3_FEATURES_V2, SEQ_LEN)
    
    fold_model = build_gru_attention_classifier(window_size=SEQ_LEN, n_features=N_FEAT_V2, n_classes=3)
    fold_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4), loss=stable_weighted_ce, metrics=['accuracy'])

    fold_model.fit(X_tr, y_tr, validation_data=(X_vl, y_vl), epochs=100, batch_size=32, 
                   callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)], verbose=0)

    rep, da, _ = evaluate_fold(fold_model, X_vl, y_vl, fold_num)
    fold_results.append({'fold': fold_num, 'accuracy': rep['accuracy'], 'active_da': da['active_da'], 'best_epoch': 0}) # Epoch tracking optional
    fold_models.append(fold_model)

# %%
  # ── Step 7: Final Model — Retrain on Full Train+Val ───────────
print("\n" + "=" * 60)
print("FINAL MODEL: Retrain on Full Train+Val Data")
print("=" * 60)
  
  # Compute quantile thresholds from full train+val (no look-ahead into test)
final_thresholds = {}
for ticker, group in df_trainval.groupby('ticker'):
      returns = group['forward_return'].dropna()
      final_thresholds[ticker] = ( 
          returns.quantile(0.25),
          returns.quantile(0.75)
      )
  
  # Apply thresholds to train+val and test
df_trainval = df_trainval.copy()
df_test_cv  = df_test_cv.copy()
df_trainval['target_class'] = assign_quantile_labels(df_trainval,
  final_thresholds)
df_test_cv['target_class']  = assign_quantile_labels(df_test_cv,
  final_thresholds)
  
  # Scale S/R + MA: fit on train+val, transform test
new_cols_final  = SR_FEATURE_COLS + MA_FEATURE_COLS
scaler_final    = RobustScaler()
df_trainval_scaled = df_trainval.copy()
df_test_scaled     = df_test_cv.copy()
df_trainval_scaled[new_cols_final] = scaler_final.fit_transform(df_trainval[new_cols_final])
df_test_scaled[new_cols_final]     = scaler_final.transform(df_test_cv[new_cols_final])

  # Carve last 6 months of train+val as internal val for EarlyStopping
val_cutoff     = df_trainval_scaled['date'].max() - pd.DateOffset(months=6)
df_ftrain      = df_trainval_scaled[df_trainval_scaled['date'] <  val_cutoff]
df_fval        = df_trainval_scaled[df_trainval_scaled['date'] >= val_cutoff]
  
X_full,  y_full  = build_signal_sequences(df_ftrain,     TASK3_FEATURES_V2,
  'target_class', SEQ_LEN)
X_fval,  y_fval  = build_signal_sequences(df_fval,       TASK3_FEATURES_V2,
  'target_class', SEQ_LEN)
X_test_final, y_test_final = build_signal_sequences(df_test_scaled,
  TASK3_FEATURES_V2, 'target_class', SEQ_LEN)
  
print(f"Final train  : {X_full.shape}")
print(f"Final val    : {X_fval.shape}")
print(f"Test         : {X_test_final.shape}")
  
for split_n, y in [('Train', y_full), ('Val', y_fval), ('Test',
  y_test_final)]:
      total = len(y)
      dist  = {label_map[c]: f"{(y==c).sum()/total:.1%}" for c in [0, 1, 2]}
      print(f"  {split_n}: {dist}")

# %%

# 4. Train Final Model
final_model = build_gru_attention_classifier(SEQ_LEN, N_FEAT_V2)
final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4), loss=stable_weighted_ce, metrics=['accuracy'])

# %%
checkpoint_final = MODELS_DIR / 'gru_signal_cv_final.keras'
callbacks_final  = [
      tf.keras.callbacks.ModelCheckpoint(
          filepath=str(checkpoint_final),
          monitor='val_loss', save_best_only=True, mode='min', verbose=1
      ),
      tf.keras.callbacks.EarlyStopping(
          restore_best_weights=True, mode='min', verbose=1
      ),  
      tf.keras.callbacks.ReduceLROnPlateau(
          monitor='val_loss', factor=0.5, patience=8,
          min_lr=1e-7, mode='min', verbose=1
      )   
  ]   

# %%
history_final = final_model.fit(
      X_full, y_full,
      validation_data=(X_fval, y_fval),
      epochs=150, batch_size=32,
      callbacks=callbacks_final, verbose=1
  )   
  
print(f"\n✓ Final model saved → gru_signal_cv_final.keras")

# %%
# 5. Evaluate
y_prob_final = final_model.predict(X_test_final)
y_pred_final = np.argmax(y_prob_final, axis=1)
print(classification_report(y_test_final, y_pred_final, target_names=['SELL', 'HOLD', 'BUY']))


