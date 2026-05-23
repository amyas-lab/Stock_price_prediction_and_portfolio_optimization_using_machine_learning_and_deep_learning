# Models

## Production (used by the API)

These are the models loaded by `src/api/main.py` and `src/api/config.py`:

| File | Role |
|---|---|
| `mtl_t4_final.keras` | Main MTL model — 5-day price trajectory + direction (27 VN tickers) |
| `production_mtl_v1.keras` | Fallback generalist MTL (10 tickers, Task 3 era) |
| `task4_feature_scaler.pkl` | Feature scaler for MTL T4 (25 features) |
| `generalist_feature_scaler.pkl` | Feature scaler for the generalist model |
| `generalist_target_multi_scaler.pkl` | Target scaler — inverse-transforms regression output |
| `task3_new_features_scaler.pkl` | Scaler for Task 3 signal features |
| `task3_xgb_final_signal.pkl` | XGBoost signal classifier (Task 3, 10 tickers) |
| `xgb_t4_signal.pkl` | XGBoost signal classifier (Task 4, 27 tickers) |
| `ticker_label_encoder.pkl` | Label encoder for ticker embeddings |

## Archive (experiments — not in production)

**Nasdaq models** (`AAPL/AMZN/GOOGL/META/MSFT/MU/NFLX/NVDA/QCOM .lstm.keras`): Per-ticker LSTM models from Task 1.

**Vietnam experiments:**
- `fpt_*.keras` — FPT-specific Seq2Seq / Transformer experiments
- `gru_fold*.keras` — Cross-validation GRU folds
- `gru_signal*.keras` — GRU-based signal models (pre-XGBoost)
- `generalist_mtl_v*.keras` — Intermediate MTL versions before T4
- `robust_mtl_*.keras` — Robust MTL experiments
- `mtl_s2s_best_weights.keras`, `mtl_t4_best.keras` — Checkpoint saves
- `lstm_*.keras`, `pooled_lstm*.keras` — Early baseline LSTMs
- `s2s_*.keras` — Seq2Seq variants
- `rf_*.pkl`, `xgb_*.pkl` (non-final) — Random Forest / XGBoost ablations

**PNG files** — Training curves, confusion matrices, signal charts from experiments.
