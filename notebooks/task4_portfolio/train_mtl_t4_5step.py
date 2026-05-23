"""
Retrain MTL T4 with 5-step regression output.

Changes vs original:
  - output_steps = 5  (was 1)
  - targets: [r1d, r2d, r3d, r4d, r5d] cumulative log-returns (was just r5d)
  - No target scaling — model outputs raw log-returns directly

Run:
  cd <project_root>
  venv/bin/python notebooks/task4_portfolio/train_mtl_t4_5step.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.utils.class_weight import compute_class_weight

# ── Paths ─────────────────────────────────────────────────────
DATA_DIR   = ROOT / "data" / "vietnam"
MODELS_DIR = ROOT / "models"

# ── Config ────────────────────────────────────────────────────
WINDOW     = 20
OUTPUT_STEPS = 5        # predict 5 daily forward returns
BUY_T      =  0.02
SELL_T     = -0.015
TRAIN_END  = pd.Timestamp("2024-06-21")
VAL_END    = pd.Timestamp("2025-01-31")

FEATURE_COLS = [
    "volume", "log_return", "rsi",
    "macd", "macd_signal", "macd_hist",
    "atr", "ema_10", "ema_20", "ema_50",
    "bb_upper", "bb_middle", "bb_lower",
    "vni_log_return", "vni_ema_10", "vni_ema_20", "vni_ema_50",
    "vni_rsi", "vni_macd", "vni_macd_signal", "vni_macd_hist",
    "vni_atr", "vni_bb_middle", "vni_bb_upper", "vni_bb_lower",
]

# ── Load & prepare data ───────────────────────────────────────
print("Loading master_features.csv ...")
df = pd.read_csv(DATA_DIR / "task4_master_features.csv", parse_dates=["date"])
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

# Compute 5 cumulative forward returns per ticker
print("Computing 5-step forward returns ...")
for k in range(1, 6):
    df[f"fr_{k}d"] = (
        df.groupby("ticker")["close"]
          .transform(lambda s: np.log(s.shift(-k) / s))
    )

# Classification label  (0=SELL, 1=HOLD, 2=BUY)
df["target_cls"] = np.select(
    [df["forward_return_5d"] > BUY_T,
     df["forward_return_5d"] < SELL_T],
    [2, 0],
    default=1,
)

# Drop rows without enough forward data
df = df.dropna(subset=[f"fr_{k}d" for k in range(1, 6)])

# ── Feature scaling (reuse existing scaler — same as production) ──
print("Loading task4_feature_scaler ...")
scaler = joblib.load(MODELS_DIR / "task4_feature_scaler.pkl")

df_tr = df[df["date"] <= TRAIN_END].copy()
df_vl = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)].copy()
df_te = df[df["date"] > VAL_END].copy()

for split in [df_tr, df_vl, df_te]:
    split[FEATURE_COLS] = scaler.transform(split[FEATURE_COLS])

print(f"Split sizes — train: {len(df_tr):,}  val: {len(df_vl):,}  test: {len(df_te):,}")

# ── Sequence builder ──────────────────────────────────────────
TARGET_COLS = [f"fr_{k}d" for k in range(1, 6)]

def build_sequences(df_split):
    all_X, all_y_reg, all_y_cls = [], [], []
    for _, grp in df_split.groupby("ticker", sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        n   = len(grp)
        X_a = grp[FEATURE_COLS].values
        Y_a = grp[TARGET_COLS].values      # (n, 5)
        C_a = grp["target_cls"].values
        for i in range(WINDOW, n - 1):
            all_X.append(X_a[i - WINDOW : i])
            all_y_reg.append(Y_a[i])       # shape (5,)
            all_y_cls.append(C_a[i])
    return (
        np.array(all_X,     dtype=np.float32),
        np.array(all_y_reg, dtype=np.float32),
        np.array(all_y_cls, dtype=np.int32),
    )

print("Building sequences ...")
X_tr, y_reg_tr, y_cls_tr = build_sequences(df_tr)
X_vl, y_reg_vl, y_cls_vl = build_sequences(df_vl)
X_te, y_reg_te, y_cls_te = build_sequences(df_te)

print(f"Sequence shapes:")
print(f"  X_train : {X_tr.shape}   y_reg: {y_reg_tr.shape}   y_cls: {y_cls_tr.shape}")
print(f"  X_val   : {X_vl.shape}")

# ── Class weights ─────────────────────────────────────────────
classes = np.array([0, 1, 2])
cw      = compute_class_weight("balanced", classes=classes, y=y_cls_tr)
cw_dict = {0: cw[0], 1: cw[1], 2: cw[2]}
print(f"Class weights: {cw_dict}")

def weighted_ce(y_true, y_pred):
    weights = tf.constant([cw_dict[0], cw_dict[1], cw_dict[2]], dtype=tf.float32)
    y_true_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    sample_w = tf.gather(weights, y_true_int)
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        tf.reshape(y_true, [-1]), y_pred
    )
    return tf.reduce_mean(loss * sample_w)

# ── Model ─────────────────────────────────────────────────────
def build_mtl_t4(window_size, n_features, output_steps=5, n_classes=3):
    inputs = layers.Input(shape=(window_size, n_features), name="input")

    encoder_seq, encoder_state = layers.GRU(
        64, return_sequences=True, return_state=True, name="encoder_gru"
    )(inputs)

    att = layers.MultiHeadAttention(
        num_heads=2, key_dim=32, dropout=0.3, name="attention"
    )(encoder_seq, encoder_seq)

    context = layers.GlobalAveragePooling1D(name="context")(att)
    context = layers.LayerNormalization(name="norm")(context)

    # Regression decoder — outputs output_steps values
    reg_in  = layers.RepeatVector(output_steps)(context)
    reg_gru = layers.GRU(64, return_sequences=True)(reg_in, initial_state=encoder_state)
    reg_out = layers.TimeDistributed(layers.Dense(1))(reg_gru)
    reg_out = layers.Reshape((output_steps,), name="reg_output")(reg_out)

    # Classification head
    cls_x   = layers.Dense(32, activation="relu")(context)
    cls_x   = layers.Dropout(0.3)(cls_x)
    cls_out = layers.Dense(n_classes, activation="softmax", name="cls_output")(cls_x)

    return Model(inputs, [reg_out, cls_out], name="MTL_T4_5step")

model = build_mtl_t4(
    window_size  = WINDOW,
    n_features   = len(FEATURE_COLS),
    output_steps = OUTPUT_STEPS,
)
model.summary()

model.compile(
    optimizer    = tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss         = {"reg_output": "huber", "cls_output": weighted_ce},
    loss_weights = {"reg_output": 1.0,     "cls_output": 4.0},
    metrics      = {"reg_output": "mae",   "cls_output": "accuracy"},
)

# ── Train ─────────────────────────────────────────────────────
BEST_PATH  = MODELS_DIR / "mtl_t4_best.keras"
FINAL_PATH = MODELS_DIR / "mtl_t4_final.keras"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        str(BEST_PATH), monitor="val_loss",
        save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15,
        restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=7, min_lr=1e-7, verbose=1
    ),
]

print("\nStarting training ...")
history = model.fit(
    X_tr,
    {"reg_output": y_reg_tr, "cls_output": y_cls_tr},
    validation_data=(
        X_vl,
        {"reg_output": y_reg_vl, "cls_output": y_cls_vl}
    ),
    epochs     = 80,
    batch_size = 64,
    callbacks  = callbacks,
    verbose    = 1,
)

model.save(str(FINAL_PATH))
print(f"\n✓ Saved: {FINAL_PATH}")
print(f"✓ Best checkpoint: {BEST_PATH}")

# ── Quick eval ────────────────────────────────────────────────
y_reg_pred, y_cls_pred = model.predict(X_te, verbose=0)
mae_5d = np.mean(np.abs(y_reg_pred[:, -1] - y_reg_te[:, -1]))
acc    = np.mean(np.argmax(y_cls_pred, axis=1) == y_cls_te)
print(f"\nTest results:")
print(f"  MAE (5d return) : {mae_5d:.5f}")
print(f"  Cls accuracy    : {acc:.3f}")
print(f"  Reg output shape: {y_reg_pred.shape}  ← should be (n, 5)")
