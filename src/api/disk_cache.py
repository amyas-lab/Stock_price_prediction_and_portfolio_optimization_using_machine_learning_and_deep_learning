# ── src/api/disk_cache.py ─────────────────────────────────────
"""
File-based hourly cache for expensive API results.

Cache key  : {prefix}_{ticker}_{YYYY-MM-DD_HH}.json
Location   : /tmp/market_cache/  (always writable, reset on container restart)
TTL        : one clock hour — the first request each hour triggers a live fetch;
             all subsequent requests within the same hour get the cached result.

Why hourly: intraday prices change with each continuous-order session.
Hourly granularity keeps data fresh while still batching requests within a
60-minute window (typical HF Free Tier cold-start: ~8 s, cached: ~0.05 s).

Why /tmp: Hugging Face Free Tier containers reset their filesystem on every
restart (typically after inactivity). /tmp is guaranteed writable; losing
stale data on restart is fine since we always want fresh market data.
"""

import json
from datetime import datetime
from pathlib import Path

CACHE_DIR = Path("/tmp/market_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _path(prefix: str, ticker: str) -> Path:
    hour_str = datetime.now().strftime("%Y-%m-%d_%H")
    return CACHE_DIR / f"{prefix}_{ticker}_{hour_str}.json"


def get(prefix: str, ticker: str) -> dict | None:
    """Return cached dict if a file exists for today, else None."""
    p = _path(prefix, ticker)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            print(f"  → [CACHE HIT]  {prefix}/{ticker}")
            return data
        except Exception as e:
            print(f"  ⚠ disk_cache read error ({p.name}): {e}")
            p.unlink(missing_ok=True)
    return None


def put(prefix: str, ticker: str, data: dict) -> None:
    """Write data as today's cache file. Silently skips on write error."""
    try:
        _path(prefix, ticker).write_text(
            json.dumps(data, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(f"  → [CACHE SET]  {prefix}/{ticker}")
    except Exception as e:
        print(f"  ⚠ disk_cache write error: {e}")


def purge_old(keep_hours: int = 24) -> int:
    """Delete cache files older than keep_hours. Returns number of files removed."""
    from datetime import timedelta
    cutoff = datetime.now() - timedelta(hours=keep_hours)
    removed = 0
    for f in CACHE_DIR.glob("*.json"):
        try:
            # filename: prefix_TICKER_YYYY-MM-DD_HH.json
            # stem ends with _YYYY-MM-DD_HH → last two underscore-parts form the timestamp
            parts = f.stem.rsplit("_", 2)  # [prefix_TICKER, YYYY-MM-DD, HH]
            date_hour_str = f"{parts[-2]}_{parts[-1]}"
            file_dt = datetime.strptime(date_hour_str, "%Y-%m-%d_%H")
            if file_dt < cutoff:
                f.unlink()
                removed += 1
        except Exception:
            pass
    return removed
