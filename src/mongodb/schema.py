# ============================================================
# TASK 5.3 — MongoDB Schema Design
# Collections for VnAlpha automation pipeline
# ============================================================

from pymongo import MongoClient, ASCENDING, DESCENDING
from datetime import datetime

MONGO_URI = "mongodb://localhost:27017"
DB_NAME   = "vnalpha"

def get_db():
    client = MongoClient(MONGO_URI)
    return client[DB_NAME]

def setup_collections():
    """Create collections with indexes."""
    db = get_db()

    # ── 1. raw_ohlcv ─────────────────────────────────────────
    # Raw daily OHLCV data per ticker
    db.raw_ohlcv.create_index(
        [("ticker", ASCENDING), ("date", DESCENDING)],
        unique=True
    )
    db.raw_ohlcv.create_index([("date", DESCENDING)])

    # ── 2. features ──────────────────────────────────────────
    # Computed technical features per ticker per day
    db.features.create_index(
        [("ticker", ASCENDING), ("date", DESCENDING)],
        unique=True
    )

    # ── 3. signals ───────────────────────────────────────────
    # XGBoost BUY/SELL/HOLD signals per ticker per day
    db.signals.create_index(
        [("ticker", ASCENDING), ("date", DESCENDING)],
        unique=True
    )
    db.signals.create_index([("signal", ASCENDING)])
    db.signals.create_index([("conviction", DESCENDING)])

    # ── 4. profitability_scores ───────────────────────────────
    # Weekly 5-factor profitability scores
    db.profitability_scores.create_index(
        [("date", DESCENDING), ("ticker", ASCENDING)],
        unique=True
    )
    db.profitability_scores.create_index(
        [("composite_score", DESCENDING)]
    )

    # ── 5. risk_scores ────────────────────────────────────────
    # Weekly risk scores per ticker
    db.risk_scores.create_index(
        [("date", DESCENDING), ("ticker", ASCENDING)],
        unique=True
    )

    # ── 6. portfolio_weights ──────────────────────────────────
    # Monthly portfolio allocations
    db.portfolio_weights.create_index(
        [("date", DESCENDING), ("profile", ASCENDING)]
    )

    # ── 7. pipeline_logs ─────────────────────────────────────
    # Audit trail for each pipeline run
    db.pipeline_logs.create_index([("run_date", DESCENDING)])
    db.pipeline_logs.create_index([("dag_id", ASCENDING)])

    print("✓ MongoDB collections and indexes created")
    print(f"  Collections: {db.list_collection_names()}")
    return db

# Document schemas (for documentation)
SCHEMAS = {
    "raw_ohlcv": {
        "ticker": "str",
        "date"  : "datetime",
        "open"  : "float",
        "high"  : "float",
        "low"   : "float",
        "close" : "float",
        "volume": "int",
        "ingested_at": "datetime"
    },
    "features": {
        "ticker"      : "str",
        "date"        : "datetime",
        "rsi"         : "float",
        "macd"        : "float",
        "macd_signal" : "float",
        "macd_hist"   : "float",
        "ema_10"      : "float",
        "ema_20"      : "float",
        "ema_50"      : "float",
        "bb_upper"    : "float",
        "bb_middle"   : "float",
        "bb_lower"    : "float",
        "atr"         : "float",
        "log_return"  : "float",
        "vni_features": "dict",
        "computed_at" : "datetime"
    },
    "signals": {
        "ticker"     : "str",
        "date"       : "datetime",
        "signal"     : "str",   # BUY/SELL/HOLD
        "p_buy"      : "float",
        "p_sell"     : "float",
        "p_hold"     : "float",
        "conviction" : "float",
        "threshold"  : "float",
        "model"      : "str",
        "generated_at": "datetime"
    },
    "profitability_scores": {
        "ticker"         : "str",
        "date"           : "datetime",
        "f1_mtl"         : "float",
        "f2_tech"        : "float",
        "f3_signal"      : "float",
        "f4_sharpe"      : "float",
        "f5_trend"       : "float",
        "composite_score": "float",
        "rank"           : "int",
        "scored_at"      : "datetime"
    },
    "portfolio_weights": {
        "date"           : "datetime",
        "profile"        : "str",
        "ticker"         : "str",
        "weight"         : "float",
        "risk_score"     : "float",
        "risk_flag"      : "str",
        "rebalanced_at"  : "datetime"
    }
}

if __name__ == "__main__":
    setup_collections()
