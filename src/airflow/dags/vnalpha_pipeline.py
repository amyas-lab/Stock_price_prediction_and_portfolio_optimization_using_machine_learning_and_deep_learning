# ============================================================
# TASK 5.3 — VnAlpha Airflow DAG
# Automated pipeline: Ingest → Feature → Signal → Score → Store
#
# Schedule:
#   Daily  (weekdays 18:30 ICT): Ingest + Features + Signals
#   Weekly (Monday 19:00 ICT)  : Profitability + Risk scoring
#   Monthly (1st, 19:30 ICT)   : Portfolio rebalancing + Model retrain
# ============================================================

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.task.trigger_rule import TriggerRule
from datetime import datetime, timedelta
import logging
import sys
from pathlib import Path

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

DATA_DIR = ROOT / 'data' / 'vietnam'

TICKERS = [
    'FPT','VCB','VHM','VNM','HPG','VIC','TCB',
    'MSN','MWG','VND','BID','CTG','MBB','ACB',
    'HDB','TPB','SHB','PDR','KDH','DXG','GAS',
    'HSG','PNJ','SAB','CMG','ELC','SGT'
]

# ── Task Functions ────────────────────────────────────────────

def ingest_ohlcv(**context):
    """Fetch latest OHLCV from yfinance, store in MongoDB."""
    from pymongo import MongoClient
    import pandas as pd
    import yfinance as yf
    from datetime import datetime, timedelta

    run_date = context['ds']
    logger.info(f"Ingesting OHLCV via yfinance for {run_date}")

    client = MongoClient("mongodb://localhost:27017")
    db     = client['vnalpha']

    end_date   = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=7)).strftime('%Y-%m-%d')

    inserted = 0
    failed   = []

    for ticker in TICKERS:
        try:
            df = yf.download(
                f"{ticker}.VN",
                start       = start_date,
                end         = end_date,
                progress    = False,
                auto_adjust = True,
            )
            if df.empty:
                failed.append(ticker)
                continue

            df = df.reset_index()
            df.columns = [
                c[0] if isinstance(c, tuple) else c for c in df.columns
            ]
            df.columns = [c.lower() for c in df.columns]
            latest = df.sort_values('date').iloc[-1]

            doc = {
                "ticker"     : ticker,
                "date"       : pd.Timestamp(latest['date']).to_pydatetime()
                               .replace(hour=0, minute=0, second=0,
                                        microsecond=0, tzinfo=None),
                "open"       : float(latest.get('open',  0)),
                "high"       : float(latest.get('high',  0)),
                "low"        : float(latest.get('low',   0)),
                "close"      : float(latest.get('close', 0)),
                "volume"     : int(latest.get('volume',  0)),
                "source"     : "yahoo_finance",
                "ingested_at": datetime.utcnow(),
            }

            if doc['close'] <= 0:
                failed.append(ticker)
                continue

            db.raw_ohlcv.update_one(
                {"ticker": ticker, "date": doc["date"]},
                {"$set": doc}, upsert=True,
            )
            inserted += 1
            logger.info(f"  ✓ {ticker}: close={doc['close']:.2f}")

        except Exception as e:
            logger.error(f"  ✗ {ticker}: {e}")
            failed.append(ticker)

    # CSV fallback for failed tickers
    if failed:
        logger.warning(f"CSV fallback for: {failed}")
        master_path = DATA_DIR / 'task4_master_features.csv'
        if master_path.exists():
            df_m = pd.read_csv(master_path)
            df_m['date'] = pd.to_datetime(df_m['date'])
            for ticker in failed:
                try:
                    t_df = df_m[df_m['ticker'] == ticker].sort_values('date')
                    if len(t_df) == 0:
                        continue
                    latest = t_df.iloc[-1]
                    doc = {
                        "ticker"     : ticker,
                        "date"       : latest['date'].to_pydatetime(),
                        "open"       : float(latest.get('open',  0)),
                        "high"       : float(latest.get('high',  0)),
                        "low"        : float(latest.get('low',   0)),
                        "close"      : float(latest['close']),
                        "volume"     : int(latest.get('volume',  0)),
                        "source"     : "csv_fallback",
                        "ingested_at": datetime.utcnow(),
                    }
                    db.raw_ohlcv.update_one(
                        {"ticker": ticker, "date": doc["date"]},
                        {"$set": doc}, upsert=True,
                    )
                    inserted += 1
                    logger.info(f"  ✓ {ticker}: CSV fallback ok")
                except Exception as e:
                    logger.error(f"  ✗ {ticker} CSV: {e}")

    logger.info(f"Ingested {inserted}/{len(TICKERS)} tickers")
    context['ti'].xcom_push(key='ingested_count', value=inserted)
    return {"inserted": inserted, "failed": len(failed)}


def compute_features(**context):
    """Compute technical features for all tickers, store in MongoDB."""
    from pymongo import MongoClient
    import pandas as pd
    from datetime import datetime

    logger.info("Computing technical features...")

    client = MongoClient("mongodb://localhost:27017")
    db     = client['vnalpha']

    try:
        from src.data.preprocess import add_all_features
    except ImportError:
        logger.error("add_all_features not available in src.data.preprocess")
        return

    master_path = DATA_DIR / 'task4_master_features.csv'
    df_all = pd.read_csv(master_path)
    df_all['date'] = pd.to_datetime(df_all['date'])

    computed = 0
    for ticker in TICKERS:
        try:
            ticker_df = (
                df_all[df_all['ticker'] == ticker]
                .sort_values('date')
                .tail(60)
            )
            if len(ticker_df) < 10:
                logger.warning(f"{ticker}: not enough rows, skipping")
                continue

            featured = add_all_features(ticker_df)
            latest   = featured.iloc[-1]

            doc = {
                "ticker"      : ticker,
                "date"        : (
                    latest['date'].to_pydatetime()
                    if hasattr(latest['date'], 'to_pydatetime')
                    else datetime.utcnow()
                ),
                "rsi"         : float(latest.get('rsi',        0)),
                "macd"        : float(latest.get('macd',       0)),
                "macd_signal" : float(latest.get('macd_signal',0)),
                "macd_hist"   : float(latest.get('macd_hist',  0)),
                "ema_10"      : float(latest.get('ema_10',     0)),
                "ema_20"      : float(latest.get('ema_20',     0)),
                "ema_50"      : float(latest.get('ema_50',     0)),
                "bb_upper"    : float(latest.get('bb_upper',   0)),
                "bb_middle"   : float(latest.get('bb_middle',  0)),
                "bb_lower"    : float(latest.get('bb_lower',   0)),
                "log_return"  : float(latest.get('log_return', 0)),
                "atr"         : float(latest.get('atr',        0)),
                "computed_at" : datetime.utcnow(),
            }

            db.features.update_one(
                {"ticker": ticker, "date": doc["date"]},
                {"$set": doc}, upsert=True,
            )
            computed += 1

        except Exception as e:
            logger.error(f"Feature error {ticker}: {e}")

    logger.info(f"Features computed: {computed}/{len(TICKERS)}")
    return {"computed": computed}


def generate_signals(**context):
    """
    Run MTL T4 classification head on live yfinance data for all tickers.
    Writes results to MongoDB + updates task3_signals_production.csv for API.
    """
    import joblib
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from pymongo import MongoClient
    from datetime import datetime
    from src.api.feature_pipeline import build_live_features

    logger.info("Generating trading signals via MTL T4 model...")

    client    = MongoClient("mongodb://localhost:27017")
    db        = client['vnalpha']
    threshold = 0.55

    model_path   = ROOT / 'models' / 'mtl_t4_final.keras'
    scaler_path  = ROOT / 'models' / 'task4_feature_scaler.pkl'
    signals_path = DATA_DIR / 'task3_signals_production.csv'

    if not model_path.exists():
        logger.error("MTL T4 model not found")
        return
    if not scaler_path.exists():
        logger.error("task4_feature_scaler not found")
        return

    model  = tf.keras.models.load_model(str(model_path), compile=False)
    scaler = joblib.load(scaler_path)

    rows      = []
    generated = 0

    for ticker in TICKERS:
        try:
            X, _, sig_date = build_live_features(ticker, scaler)
            _, cls_pred = model.predict(X, verbose=0)
            p_sell = float(cls_pred[0, 0])
            p_hold = float(cls_pred[0, 1])
            p_buy  = float(cls_pred[0, 2])

            signal = (
                'BUY'  if p_buy  >= threshold else
                'SELL' if p_sell >= threshold else
                'HOLD'
            )

            doc = {
                "ticker"      : ticker,
                "date"        : sig_date,
                "signal"      : signal,
                "p_buy"       : p_buy,
                "p_sell"      : p_sell,
                "p_hold"      : p_hold,
                "conviction"  : max(p_buy, p_sell),
                "model"       : "mtl_t4_final",
                "generated_at": datetime.utcnow(),
            }

            db.signals.update_one(
                {"ticker": ticker, "date": sig_date},
                {"$set": doc}, upsert=True,
            )

            rows.append({
                "ticker": ticker,
                "date"  : sig_date,
                "p_buy" : round(p_buy,  4),
                "p_sell": round(p_sell, 4),
                "p_hold": round(p_hold, 4),
            })
            generated += 1
            logger.info(f"  ✓ {ticker}: {signal} "
                        f"(p_buy={p_buy:.3f}, p_sell={p_sell:.3f})")

        except Exception as e:
            logger.error(f"  ✗ {ticker}: {e}")

    # Update signals CSV so API Branch-1 serves fresh signals
    if rows:
        df_new = pd.DataFrame(rows)
        today  = df_new['date'].iloc[0][:10]
        if signals_path.exists():
            df_old = pd.read_csv(signals_path)
            df_old = df_old[
                df_old['date'].astype(str).str[:10] != today
            ]
            df_out = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_out = df_new
        df_out.to_csv(signals_path, index=False)
        logger.info(f"Updated {signals_path.name} with {len(rows)} tickers")

    logger.info(f"Signals generated: {generated}/{len(TICKERS)}")
    context['ti'].xcom_push(key='signals_generated', value=generated)
    return {"generated": generated}


def score_profitability(**context):
    """Weekly: read profitability CSV, store in MongoDB."""
    from pymongo import MongoClient
    import pandas as pd
    from datetime import datetime

    logger.info("Scoring profitability (weekly)...")

    client = MongoClient("mongodb://localhost:27017")
    db     = client['vnalpha']

    scores_path = DATA_DIR / 'task4_profitability_scores.csv'
    if not scores_path.exists():
        logger.error(f"Not found: {scores_path}")
        return

    df    = pd.read_csv(scores_path)
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    inserted = 0
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        try:
            doc = {
                "ticker"         : str(row.get('ticker',    '')),
                "date"           : today,
                "f1_mtl"         : float(row.get('f1_mtl',    0)),
                "f2_tech"        : float(row.get('f2_tech',   0)),
                "f3_signal"      : float(row.get('f3_signal', 0)),
                "f4_sharpe"      : float(row.get('f4_sharpe', 0)),
                "f5_trend"       : float(row.get('f5_trend',  0)),
                "composite_score": float(row.get('composite', 0)),
                "sector"         : str(row.get('sector',  'Unknown')),
                "rank"           : rank,
                "scored_at"      : datetime.utcnow(),
            }
            db.profitability_scores.update_one(
                {"ticker": doc["ticker"], "date": today},
                {"$set": doc}, upsert=True,
            )
            inserted += 1
        except Exception as e:
            logger.error(f"Score error: {e}")

    logger.info(f"Profitability scored: {inserted} tickers")
    return {"scored": inserted}


def score_risk(**context):
    """Weekly: read risk CSV, store in MongoDB."""
    from pymongo import MongoClient
    import pandas as pd
    from datetime import datetime

    logger.info("Updating risk scores (weekly)...")

    client = MongoClient("mongodb://localhost:27017")
    db     = client['vnalpha']

    risk_path = DATA_DIR / 'task4_risk_scores.csv'
    if not risk_path.exists():
        logger.error(f"Not found: {risk_path}")
        return

    df    = pd.read_csv(risk_path)
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    inserted = 0
    for _, row in df.iterrows():
        try:
            doc = {
                "ticker"          : str(row.get('ticker',           '')),
                "date"            : today,
                "volatility_risk" : float(row.get('r1_vol',          0)),
                "sell_risk"       : float(row.get('r2_sell',         0)),
                "drawdown_risk"   : float(row.get('r3_drawdown',     0)),
                "correlation_risk": float(row.get('r4_corr',         0)),
                "reversal_risk"   : float(row.get('r5_reversal',     0)),
                "composite_risk"  : float(row.get('final_risk_score',0)),
                "risk_flag"       : str(row.get('risk_flag',      'N/A')),
                "scored_at"       : datetime.utcnow(),
            }
            db.risk_scores.update_one(
                {"ticker": doc["ticker"], "date": today},
                {"$set": doc}, upsert=True,
            )
            inserted += 1
        except Exception as e:
            logger.error(f"Risk score error: {e}")

    logger.info(f"Risk scored: {inserted} tickers")
    return {"risk_scored": inserted}


def rebalance_portfolio(**context):
    """Monthly: read portfolio CSV, store in MongoDB."""
    from pymongo import MongoClient
    import pandas as pd
    from datetime import datetime

    logger.info("Rebalancing portfolio (monthly)...")

    client = MongoClient("mongodb://localhost:27017")
    db     = client['vnalpha']

    port_path = DATA_DIR / 'task4_portfolio_composition_all.csv'
    if not port_path.exists():
        logger.error(f"Not found: {port_path}")
        return

    df    = pd.read_csv(port_path)
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    inserted = 0
    for _, row in df.iterrows():
        try:
            doc = {
                "date"         : today,
                "profile"      : str(row.get('profile',    '')),
                "ticker"       : str(row.get('ticker',     '')),
                "weight"       : float(row.get('weight',    0)),
                "sector"       : str(row.get('sector',     '')),
                "risk_score"   : float(row.get('risk_score',0)),
                "risk_flag"    : str(row.get('risk_flag','N/A')),
                "rebalanced_at": datetime.utcnow(),
            }
            db.portfolio_weights.update_one(
                {"date": today, "profile": doc["profile"], "ticker": doc["ticker"]},
                {"$set": doc}, upsert=True,
            )
            inserted += 1
        except Exception as e:
            logger.error(f"Rebalance error: {e}")

    logger.info(f"Portfolio rebalanced: {inserted} positions")
    return {"rebalanced": inserted}


def retrain_model(**context):
    """
    Monthly: retrain MTL T4 with latest data.
    Runs train_mtl_t4_5step.py as a subprocess so it gets its own TF session.
    """
    import subprocess
    from datetime import datetime

    logger.info("Retraining MTL T4 model (monthly)...")
    script = ROOT / 'notebooks' / 'task4_portfolio' / 'train_mtl_t4_5step.py'

    if not script.exists():
        logger.error(f"Training script not found: {script}")
        return

    python = ROOT / 'venv' / 'bin' / 'python'
    if not python.exists():
        python = sys.executable  # fallback to Airflow's Python

    result = subprocess.run(
        [str(python), str(script)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    # Log last 3000 chars of output to avoid overflow
    if result.stdout:
        logger.info(f"Retrain stdout:\n{result.stdout[-3000:]}")
    if result.stderr:
        logger.warning(f"Retrain stderr:\n{result.stderr[-1000:]}")

    if result.returncode != 0:
        raise RuntimeError(
            f"Model retrain failed (exit {result.returncode}). "
            f"Check logs above."
        )

    logger.info(f"Retrain completed successfully at {datetime.utcnow()}")
    return {"retrain": "success", "completed_at": str(datetime.utcnow())}


def log_pipeline_run(**context):
    """Final task: log pipeline completion to MongoDB."""
    from pymongo import MongoClient
    from datetime import datetime

    client = MongoClient("mongodb://localhost:27017")
    db     = client['vnalpha']

    ingested = context['ti'].xcom_pull(
        task_ids='ingest_ohlcv', key='ingested_count'
    ) or 0
    signals = context['ti'].xcom_pull(
        task_ids='generate_signals', key='signals_generated'
    ) or 0

    doc = {
        "dag_id"            : context['dag'].dag_id,
        "run_date"          : context['ds'],
        "run_id"            : context['run_id'],
        "status"            : "success",
        "tickers_ingested"  : ingested,
        "signals_generated" : signals,
        "completed_at"      : datetime.utcnow(),
    }

    db.pipeline_logs.insert_one(doc)
    logger.info(f"Pipeline logged: {doc}")
    return doc


# ============================================================
# DAG 1: DAILY — Ingest + Features + Signals
# Mon-Fri 18:30 ICT (11:30 UTC)
# ============================================================
with DAG(
    dag_id          = 'vnalpha_daily_pipeline',
    default_args    = DEFAULT_ARGS,
    description     = 'Daily: OHLCV ingest → features → MTL signals',
    schedule        = '30 11 * * 1-5',
    catchup         = False,
    tags            = ['vnalpha', 'daily', 'production'],
    max_active_runs = 1,
) as daily_dag:

    start = EmptyOperator(task_id='pipeline_start')
    end   = EmptyOperator(
        task_id      = 'pipeline_end',
        trigger_rule = TriggerRule.ALL_DONE,
    )

    t_ingest = PythonOperator(
        task_id         = 'ingest_ohlcv',
        python_callable = ingest_ohlcv,
    )

    t_features = PythonOperator(
        task_id         = 'compute_features',
        python_callable = compute_features,
    )

    t_signals = PythonOperator(
        task_id         = 'generate_signals',
        python_callable = generate_signals,
    )

    t_log = PythonOperator(
        task_id         = 'log_pipeline_run',
        python_callable = log_pipeline_run,
        trigger_rule    = TriggerRule.ALL_DONE,
    )

    start >> t_ingest >> t_features >> t_signals >> t_log >> end


# ============================================================
# DAG 2: WEEKLY — Profitability + Risk Scoring
# Monday 19:00 ICT (12:00 UTC)
# ============================================================
with DAG(
    dag_id          = 'vnalpha_weekly_scoring',
    default_args    = DEFAULT_ARGS,
    description     = 'Weekly: profitability + risk scoring',
    schedule        = '0 12 * * 1',
    catchup         = False,
    tags            = ['vnalpha', 'weekly'],
    max_active_runs = 1,
) as weekly_dag:

    w_start = EmptyOperator(task_id='weekly_start')
    w_end   = EmptyOperator(
        task_id      = 'weekly_end',
        trigger_rule = TriggerRule.ALL_DONE,
    )

    w_profit = PythonOperator(
        task_id         = 'score_profitability',
        python_callable = score_profitability,
    )

    w_risk = PythonOperator(
        task_id         = 'score_risk',
        python_callable = score_risk,
    )

    # Parallel scoring
    w_start >> [w_profit, w_risk] >> w_end


# ============================================================
# DAG 3: MONTHLY — Portfolio Rebalancing + Model Retrain
# 1st of each month 19:30 ICT (12:30 UTC)
# ============================================================
with DAG(
    dag_id          = 'vnalpha_monthly_rebalance',
    default_args    = DEFAULT_ARGS,
    description     = 'Monthly: portfolio rebalancing + model retrain',
    schedule        = '30 12 1 * *',
    catchup         = False,
    tags            = ['vnalpha', 'monthly'],
    max_active_runs = 1,
) as monthly_dag:

    m_start = EmptyOperator(task_id='monthly_start')
    m_end   = EmptyOperator(task_id='monthly_end')

    m_rebalance = PythonOperator(
        task_id         = 'rebalance_portfolio',
        python_callable = rebalance_portfolio,
    )

    m_retrain = PythonOperator(
        task_id         = 'retrain_model',
        python_callable = retrain_model,
        execution_timeout = timedelta(hours=2),
    )

    # Rebalance first, then retrain with latest data
    m_start >> m_rebalance >> m_retrain >> m_end
