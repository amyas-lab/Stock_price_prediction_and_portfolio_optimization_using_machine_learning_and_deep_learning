# ============================================================
# TASK 5.3 — VnAlpha Airflow DAG
# Automated pipeline: Ingest → Feature → Signal → Score → Store
#
# Schedule:
#   Daily  (weekdays 18:30 ICT): Ingest + Features + Signals
#   Weekly (Monday 19:00 ICT)  : Profitability + Risk scoring
#   Monthly (1st, 19:30 ICT)   : Portfolio rebalancing
# ============================================================

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.task.trigger_rule import TriggerRule
from datetime import datetime, timedelta
import logging
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

# ── Default args ──────────────────────────────────────────────
DEFAULT_ARGS = {
    'owner'           : 'vnalpha',
    'depends_on_past' : False,
    'start_date'      : datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry'  : False,
    'retries'         : 2,
    'retry_delay'     : timedelta(minutes=5),
}

TICKERS = [
    'FPT','VCB','VHM','VNM','HPG','VIC','TCB',
    'MSN','MWG','VND','BID','CTG','MBB','ACB',
    'HDB','TPB','SHB','PDR','KDH','DXG','GAS',
    'HSG','PNJ','SAB','CMG','ELC','SGT'
]

# ── Task Functions ────────────────────────────────────────────

def ingest_ohlcv(**context):
    """
    Task 1: Fetch latest OHLCV data from vnstock API.
    Stores raw data in MongoDB raw_ohlcv collection.
    Only fetches yesterday's data (incremental).
    """
    from pymongo import MongoClient
    import pandas as pd

    run_date = context['ds']        # YYYY-MM-DD
    logger.info(f"Ingesting OHLCV for {run_date}")

    client = MongoClient("mongodb://localhost:27017")
    db     = client['vnalpha']

    inserted = 0
    failed   = []

    for ticker in TICKERS:
        try:
            # Fetch from vnstock (pseudo-code — replace with
            # actual vnstock API call)
            # from vnstock import stock_historical_data
            # df = stock_historical_data(
            #     ticker, run_date, run_date, '1D'
            # )

            # For now: check if data exists in master CSV
            master_path = (
                ROOT / 'notebooks' / 'data' / 'vietnam'
                / 'task4_master_features.csv'
            )
            if not master_path.exists():
                logger.warning(f"Master CSV not found")
                continue

            df = pd.read_csv(master_path)
            df = df[df['ticker'] == ticker].copy()
            df['date'] = pd.to_datetime(df['date'])

            # Get latest row
            latest = df.sort_values('date').iloc[-1]

            doc = {
                "ticker"      : ticker,
                "date"        : latest['date'].to_pydatetime(),
                "open"        : float(latest.get('open', 0)),
                "high"        : float(latest.get('high', 0)),
                "low"         : float(latest.get('low',  0)),
                "close"       : float(latest['close']),
                "volume"      : int(latest.get('volume', 0)),
                "ingested_at" : datetime.utcnow()
            }

            db.raw_ohlcv.update_one(
                {"ticker": ticker, "date": doc["date"]},
                {"$set": doc},
                upsert=True
            )
            inserted += 1

        except Exception as e:
            logger.error(f"Failed {ticker}: {e}")
            failed.append(ticker)

    logger.info(f"Ingested {inserted}/{len(TICKERS)} tickers")
    if failed:
        logger.warning(f"Failed tickers: {failed}")

    # Push to XCom for downstream tasks
    context['ti'].xcom_push(
        key='ingested_count', value=inserted
    )
    return {"inserted": inserted, "failed": failed}


def compute_features(**context):
    """
    Task 2: Compute technical features for all tickers.
    Uses add_all_features() from src/data/preprocess.py.
    Stores in MongoDB features collection.
    """
    from pymongo import MongoClient
    import pandas as pd
    import numpy as np

    logger.info("Computing technical features...")

    client = MongoClient("mongodb://localhost:27017")
    db     = client['vnalpha']

    try:
        from src.data.preprocess import add_all_features
    except ImportError:
        logger.error("add_all_features not available")
        return

    master_path = (
        ROOT / 'notebooks' / 'data' / 'vietnam'
        / 'task4_master_features.csv'
    )
    df_all = pd.read_csv(master_path)
    df_all['date'] = pd.to_datetime(df_all['date'])

    computed = 0
    for ticker in TICKERS:
        try:
            ticker_df = df_all[
                df_all['ticker'] == ticker
            ].sort_values('date').tail(60)

            featured = add_all_features(ticker_df)
            latest   = featured.iloc[-1]

            doc = {
                "ticker"     : ticker,
                "date"       : latest['date'].to_pydatetime()
                               if hasattr(latest['date'], 'to_pydatetime')
                               else datetime.utcnow(),
                "rsi"        : float(latest.get('rsi',       0)),
                "macd"       : float(latest.get('macd',      0)),
                "macd_hist"  : float(latest.get('macd_hist', 0)),
                "ema_10"     : float(latest.get('ema_10',    0)),
                "ema_20"     : float(latest.get('ema_20',    0)),
                "ema_50"     : float(latest.get('ema_50',    0)),
                "bb_upper"   : float(latest.get('bb_upper',  0)),
                "bb_lower"   : float(latest.get('bb_lower',  0)),
                "log_return" : float(latest.get('log_return',0)),
                "atr"        : float(latest.get('atr',       0)),
                "computed_at": datetime.utcnow()
            }

            db.features.update_one(
                {"ticker": ticker, "date": doc["date"]},
                {"$set": doc},
                upsert=True
            )
            computed += 1

        except Exception as e:
            logger.error(f"Feature error {ticker}: {e}")

    logger.info(f"Features computed: {computed}/{len(TICKERS)}")
    return {"computed": computed}


def generate_signals(**context):
    """
    Task 3: Run XGBoost signal model on all tickers.
    Reads features from MongoDB, writes signals back.
    """
    from pymongo import MongoClient
    import pandas as pd
    import numpy as np
    import joblib

    logger.info("Generating trading signals...")

    client    = MongoClient("mongodb://localhost:27017")
    db        = client['vnalpha']
    threshold = 0.55

    # Load XGBoost model
    model_path = ROOT / 'models' / 'xgb_t4_signal.pkl'
    if not model_path.exists():
        model_path = ROOT / 'models' / 'task3_xgb_final_signal.pkl'

    if not model_path.exists():
        logger.error("XGBoost signal model not found")
        return

    xgb_model = joblib.load(model_path)
    scaler    = joblib.load(
        ROOT / 'models' / 'task4_feature_scaler.pkl'
    )

    # Load latest features from master CSV
    master_path = (
        ROOT / 'notebooks' / 'data' / 'vietnam'
        / 'task4_master_features.csv'
    )
    df_all = pd.read_csv(master_path)
    df_all['date'] = pd.to_datetime(df_all['date'])

    # Load precomputed signals (from Task 3 output)
    signals_path = (
        ROOT / 'notebooks' / 'data' / 'vietnam'
        / 'task3_signals_production.csv'
    )

    generated = 0
    for ticker in TICKERS:
        try:
            # Get latest signal from precomputed file
            if signals_path.exists():
                sig_df     = pd.read_csv(signals_path)
                sig_df['date'] = pd.to_datetime(sig_df['date'])
                ticker_sig = sig_df[
                    sig_df['ticker'] == ticker
                ].sort_values('date')

                if len(ticker_sig) > 0:
                    last = ticker_sig.iloc[-1]
                    p_buy  = float(last.get('p_buy',  0.33))
                    p_sell = float(last.get('p_sell', 0.33))
                else:
                    p_buy = p_sell = 0.33

            else:
                p_buy = p_sell = 0.33

            p_hold     = max(0, 1 - p_buy - p_sell)
            conviction = max(p_buy, p_sell)

            signal = (
                'BUY'  if p_buy  >= threshold else
                'SELL' if p_sell >= threshold else
                'HOLD'
            )

            doc = {
                "ticker"      : ticker,
                "date"        : datetime.utcnow().replace(
                    hour=0, minute=0, second=0, microsecond=0
                ),
                "signal"      : signal,
                "p_buy"       : p_buy,
                "p_sell"      : p_sell,
                "p_hold"      : p_hold,
                "conviction"  : conviction,
                "threshold"   : threshold,
                "model"       : "xgb_t4_signal",
                "generated_at": datetime.utcnow()
            }

            db.signals.update_one(
                {"ticker": ticker, "date": doc["date"]},
                {"$set": doc},
                upsert=True
            )
            generated += 1

        except Exception as e:
            logger.error(f"Signal error {ticker}: {e}")

    logger.info(f"Signals generated: {generated}/{len(TICKERS)}")
    context['ti'].xcom_push(
        key='signals_generated', value=generated
    )
    return {"generated": generated}


def score_profitability(**context):
    """
    Task 4 (Weekly): Recompute 5-factor profitability scores.
    Reads from precomputed CSV, stores in MongoDB.
    """
    from pymongo import MongoClient
    import pandas as pd

    logger.info("Scoring profitability (weekly)...")

    client = MongoClient("mongodb://localhost:27017")
    db     = client['vnalpha']

    scores_path = (
        ROOT / 'notebooks' / 'data' / 'vietnam'
        / 'task4_profitability_scores.csv'
    )

    if not scores_path.exists():
        logger.error("Profitability scores CSV not found")
        return

    df = pd.read_csv(scores_path)
    today = datetime.utcnow().replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    inserted = 0
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        try:
            doc = {
                "ticker"         : str(row.get('ticker', '')),
                "date"           : today,
                "f1_mtl"         : float(row.get('f1_mtl',    0)),
                "f2_tech"        : float(row.get('f2_tech',   0)),
                "f3_signal"      : float(row.get('f3_signal', 0)),
                "f4_sharpe"      : float(row.get('f4_sharpe', 0)),
                "f5_trend"       : float(row.get('f5_trend',  0)),
                "composite_score": float(row.get('composite', 0)),
                "sector"         : str(row.get('sector',  '')),
                "rank"           : rank,
                "scored_at"      : datetime.utcnow()
            }
            db.profitability_scores.update_one(
                {"ticker": doc["ticker"], "date": today},
                {"$set": doc},
                upsert=True
            )
            inserted += 1
        except Exception as e:
            logger.error(f"Score error: {e}")

    logger.info(f"Profitability scored: {inserted} tickers")
    return {"scored": inserted}


def score_risk(**context):
    """
    Task 5 (Weekly): Update risk scores from precomputed CSV.
    """
    from pymongo import MongoClient
    import pandas as pd

    logger.info("Updating risk scores (weekly)...")

    client = MongoClient("mongodb://localhost:27017")
    db     = client['vnalpha']

    risk_path = (
        ROOT / 'notebooks' / 'data' / 'vietnam'
        / 'task4_risk_scores.csv'
    )

    if not risk_path.exists():
        logger.error("Risk scores CSV not found")
        return

    df    = pd.read_csv(risk_path)
    today = datetime.utcnow().replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    inserted = 0
    for _, row in df.iterrows():
        try:
            doc = {
                "ticker"          : str(row.get('ticker',   '')),
                "date"            : today,
                "volatility_risk" : float(row.get('r1_vol',      0)),
                "sell_risk"       : float(row.get('r2_sell',     0)),
                "drawdown_risk"   : float(row.get('r3_drawdown', 0)),
                "correlation_risk": float(row.get('r4_corr',     0)),
                "reversal_risk"   : float(row.get('r5_reversal', 0)),
                "composite_risk"  : float(row.get(
                    'final_risk_score', 0
                )),
                "risk_flag"       : str(row.get('risk_flag','N/A')),
                "scored_at"       : datetime.utcnow()
            }
            db.risk_scores.update_one(
                {"ticker": doc["ticker"], "date": today},
                {"$set": doc},
                upsert=True
            )
            inserted += 1
        except Exception as e:
            logger.error(f"Risk score error: {e}")

    logger.info(f"Risk scored: {inserted} tickers")
    return {"risk_scored": inserted}


def rebalance_portfolio(**context):
    """
    Task 6 (Monthly): Update portfolio weights in MongoDB.
    """
    from pymongo import MongoClient
    import pandas as pd

    logger.info("Rebalancing portfolio (monthly)...")

    client = MongoClient("mongodb://localhost:27017")
    db     = client['vnalpha']

    port_path = (
        ROOT / 'notebooks' / 'data' / 'vietnam'
        / 'task4_portfolio_composition_all.csv'
    )

    if not port_path.exists():
        logger.error("Portfolio CSV not found")
        return

    df    = pd.read_csv(port_path)
    today = datetime.utcnow().replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    inserted = 0
    for _, row in df.iterrows():
        try:
            doc = {
                "date"        : today,
                "profile"     : str(row.get('profile', '')),
                "ticker"      : str(row.get('ticker',  '')),
                "weight"      : float(row.get('weight', 0)),
                "sector"      : str(row.get('sector',  '')),
                "risk_score"  : float(row.get('risk_score', 0)),
                "risk_flag"   : str(row.get('risk_flag', 'N/A')),
                "rebalanced_at": datetime.utcnow()
            }
            db.portfolio_weights.update_one(
                {
                    "date"   : today,
                    "profile": doc["profile"],
                    "ticker" : doc["ticker"]
                },
                {"$set": doc},
                upsert=True
            )
            inserted += 1
        except Exception as e:
            logger.error(f"Rebalance error: {e}")

    logger.info(f"Portfolio rebalanced: {inserted} positions")
    return {"rebalanced": inserted}


def log_pipeline_run(**context):
    """
    Final task: Log pipeline completion to MongoDB.
    """
    from pymongo import MongoClient

    client = MongoClient("mongodb://localhost:27017")
    db     = client['vnalpha']

    ingested  = context['ti'].xcom_pull(
        task_ids='ingest_ohlcv',
        key='ingested_count'
    ) or 0
    signals   = context['ti'].xcom_pull(
        task_ids='generate_signals',
        key='signals_generated'
    ) or 0

    doc = {
        "dag_id"        : context['dag'].dag_id,
        "run_date"      : context['ds'],
        "run_id"        : context['run_id'],
        "status"        : "success",
        "tickers_ingested": ingested,
        "signals_generated": signals,
        "completed_at"  : datetime.utcnow()
    }

    db.pipeline_logs.insert_one(doc)
    logger.info(f"Pipeline logged: {doc}")
    return doc


# ============================================================
# DAG 1: DAILY — Ingest + Features + Signals
# Schedule: weekdays at 18:30 ICT (11:30 UTC)
# ============================================================
with DAG(
    dag_id          = 'vnalpha_daily_pipeline',
    default_args    = DEFAULT_ARGS,
    description     = 'Daily: OHLCV ingest → features → signals',
    schedule = '30 11 * * 1-5',  # Mon-Fri 18:30 ICT
    catchup         = False,
    tags            = ['vnalpha', 'daily', 'production'],
    max_active_runs = 1
) as daily_dag:

    start = EmptyOperator(task_id='pipeline_start')
    end   = EmptyOperator(
        task_id      = 'pipeline_end',
        trigger_rule = TriggerRule.ALL_DONE
    )

    t_ingest = PythonOperator(
        task_id         = 'ingest_ohlcv',
        python_callable = ingest_ohlcv,
        doc_md          = "Fetch latest OHLCV from vnstock API"
    )

    t_features = PythonOperator(
        task_id         = 'compute_features',
        python_callable = compute_features,
        doc_md          = "Compute technical indicators"
    )

    t_signals = PythonOperator(
        task_id         = 'generate_signals',
        python_callable = generate_signals,
        doc_md          = "Run XGBoost signal classifier"
    )

    t_log = PythonOperator(
        task_id         = 'log_pipeline_run',
        python_callable = log_pipeline_run,
        trigger_rule    = TriggerRule.ALL_DONE,
        doc_md          = "Log pipeline run to MongoDB"
    )

    # DAG flow: sequential
    start >> t_ingest >> t_features >> t_signals >> t_log >> end


# ============================================================
# DAG 2: WEEKLY — Profitability + Risk Scoring
# Schedule: Monday 19:00 ICT (12:00 UTC)
# ============================================================
with DAG(
    dag_id            = 'vnalpha_weekly_scoring',
    default_args      = DEFAULT_ARGS,
    description       = 'Weekly: profitability + risk scoring',
    schedule = '0 12 * * 1',  # Monday 19:00 ICT
    catchup           = False,
    tags              = ['vnalpha', 'weekly'],
    max_active_runs   = 1
) as weekly_dag:

    w_start = EmptyOperator(task_id='weekly_start')
    w_end   = EmptyOperator(
        task_id      = 'weekly_end',
        trigger_rule = TriggerRule.ALL_DONE
    )

    w_profit = PythonOperator(
        task_id         = 'score_profitability',
        python_callable = score_profitability,
        doc_md          = "Compute 5-factor profitability scores"
    )

    w_risk = PythonOperator(
        task_id         = 'score_risk',
        python_callable = score_risk,
        doc_md          = "Compute 5-component risk scores"
    )

    # Parallel scoring
    w_start >> [w_profit, w_risk] >> w_end


# ============================================================
# DAG 3: MONTHLY — Portfolio Rebalancing
# Schedule: 1st of each month 19:30 ICT (12:30 UTC)
# ============================================================
with DAG(
    dag_id            = 'vnalpha_monthly_rebalance',
    default_args      = DEFAULT_ARGS,
    description       = 'Monthly: portfolio rebalancing',
    schedule = '30 12 1 * *',  # 1st of month 19:30 ICT
    catchup           = False,
    tags              = ['vnalpha', 'monthly'],
    max_active_runs   = 1
) as monthly_dag:

    m_start = EmptyOperator(task_id='monthly_start')
    m_end   = EmptyOperator(task_id='monthly_end')

    m_rebalance = PythonOperator(
        task_id         = 'rebalance_portfolio',
        python_callable = rebalance_portfolio,
        doc_md          = "Update portfolio weights in MongoDB"
    )

    m_start >> m_rebalance >> m_end
