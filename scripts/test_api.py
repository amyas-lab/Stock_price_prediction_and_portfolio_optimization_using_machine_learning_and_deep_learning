#!/usr/bin/env python3
"""
Manual endpoint test against the live HF Spaces API.

Usage:
    python scripts/test_api.py
    python scripts/test_api.py --base http://localhost:8000
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error

DEFAULT_BASE = "https://amyas-lab-investnature.hf.space"

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"


def _req(base: str, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    url  = base + path
    data = json.dumps(body).encode() if body else None
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())
    except Exception as e:
        return 0, {"error": str(e)}


def check(results: list, label: str, status: int, body: dict, expect_keys: list[str]) -> dict:
    ok  = status == 200 and all(k in body for k in expect_keys)
    tag = PASS if ok else FAIL
    print(f"  {tag} {label}  [{status}]")
    if not ok:
        missing = [k for k in expect_keys if k not in body]
        if missing:
            print(f"      missing keys: {missing}")
        print(f"      body: {json.dumps(body)[:300]}")
    results.append(ok)
    return body


def sep(title: str):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


def main(base: str):
    results: list[bool] = []
    get  = lambda path: _req(base, "GET",  path)
    post = lambda path, body: _req(base, "POST", path, body)

    # ── Health ────────────────────────────────────────────────
    sep("GET /health")
    t0 = time.time()
    s, b = get("/health")
    elapsed = time.time() - t0
    check(results, f"health check  ({elapsed:.1f}s)", s, b,
          ["status", "models_loaded", "supported_tickers_t3", "supported_tickers_t4"])
    if s == 200:
        for m, ok in b.get("models_loaded", {}).items():
            print(f"      {'✓' if ok else '✗'} {m}: {ok}")

    # ── Tickers ───────────────────────────────────────────────
    sep("GET /tickers")
    s, b = get("/tickers")
    check(results, "list tickers", s, b, ["known_tickers"])

    # ── Price prediction ──────────────────────────────────────
    sep("POST /predict/price")
    for ticker in ["FPT", "VCB", "HDB"]:
        t0 = time.time()
        s, b = post("/predict/price", {"ticker": ticker, "n_days_back": 20})
        elapsed = time.time() - t0
        check(results, f"{ticker}  ({elapsed:.1f}s)", s, b,
              ["ticker", "current_price", "predicted_prices", "direction",
               "confidence", "data_source", "historical_prices"])
        if s == 200:
            print(f"      price={b['current_price']:,.0f}  "
                  f"dir={b['direction']}  conf={b['confidence']:.2%}  "
                  f"src={b['data_source']}  hist={len(b['historical_prices'])} pts")

    # second FPT call — should be cache hit
    t0 = time.time()
    s, b = post("/predict/price", {"ticker": "FPT", "n_days_back": 20})
    elapsed = time.time() - t0
    tag = PASS if elapsed < 2 else WARN
    print(f"  {tag} FPT cache hit  ({elapsed:.1f}s)")
    results.append(True)

    # ── Signal ────────────────────────────────────────────────
    sep("POST /predict/signal")
    for ticker in ["FPT", "VHM", "TCB"]:
        t0 = time.time()
        s, b = post("/predict/signal", {"ticker": ticker, "threshold": 0.55})
        elapsed = time.time() - t0
        check(results, f"{ticker}  ({elapsed:.1f}s)", s, b,
              ["ticker", "signal", "p_buy", "p_sell", "p_hold",
               "conviction", "data_source", "recommendation"])
        if s == 200:
            fc = b.get("feature_context") or {}
            print(f"      signal={b['signal']}  p_buy={b['p_buy']:.2%}  "
                  f"p_sell={b['p_sell']:.2%}  src={b['data_source']}")
            if fc:
                print(f"      rsi={fc.get('rsi', 0):.1f}  "
                      f"sr_dist={fc.get('sr_distance_pct', 0):.2%}  "
                      f"mtl_p_up={fc.get('mtl_p_up_t4', 0):.2%}")

    # cache hit
    t0 = time.time()
    s, _ = post("/predict/signal", {"ticker": "FPT", "threshold": 0.55})
    elapsed = time.time() - t0
    tag = PASS if elapsed < 2 else WARN
    print(f"  {tag} FPT signal cache hit  ({elapsed:.1f}s)")
    results.append(True)

    # different threshold on cached probs
    t0 = time.time()
    s, b = post("/predict/signal", {"ticker": "FPT", "threshold": 0.40})
    elapsed = time.time() - t0
    check(results, f"FPT threshold=0.40 (cached probs, {elapsed:.1f}s)", s, b,
          ["signal", "p_buy"])

    # ── SHAP explain ──────────────────────────────────────────
    sep("POST /predict/signal/explain")
    t0 = time.time()
    s, b = post("/predict/signal/explain", {"ticker": "FPT", "threshold": 0.55})
    elapsed = time.time() - t0
    if s == 503 and "SHAP" in str(b):
        print(f"  {WARN} SHAP not installed on server  [{s}]")
        results.append(True)
    else:
        check(results, f"FPT SHAP explain  ({elapsed:.1f}s)", s, b,
              ["ticker", "signal", "contributions", "base_value"])
        if s == 200:
            for c in b["contributions"][:3]:
                print(f"      {c['feature']}: {c['shap_value']:+.4f} ({c['direction']})")

    # ── Portfolio ─────────────────────────────────────────────
    sep("GET /portfolio/{profile}")
    for profile in ["risk_taking", "prudent", "equal_weight"]:
        s, b = get(f"/portfolio/{profile}")
        check(results, profile, s, b,
              ["profile", "stocks", "expected_return", "expected_vol", "sharpe_ratio"])
        if s == 200:
            print(f"      stocks={b['total_stocks']}  "
                  f"ret={b['expected_return']:.2%}  "
                  f"vol={b['expected_vol']:.2%}  "
                  f"sharpe={b['sharpe_ratio']:.2f}")

    s, b = get("/portfolio/invalid_profile")
    ok = (s == 400)
    print(f"  {PASS if ok else FAIL} invalid profile → 400  [{s}]")
    results.append(ok)

    # ── Profitability & Risk ──────────────────────────────────
    sep("GET /portfolio/scores/*")
    s, b = get("/portfolio/scores/profitability")
    check(results, "profitability scores", s, b, ["scores", "total_tickers"])
    if s == 200:
        print(f"      {b['total_tickers']} tickers scored")

    s, b = get("/portfolio/scores/risk")
    check(results, "risk scores", s, b, ["scores", "total_tickers"])
    if s == 200:
        print(f"      {b['total_tickers']} tickers assessed")

    # ── Backtest ──────────────────────────────────────────────
    sep("GET /backtest/price")
    s, b = get("/backtest/price")
    check(results, "price backtest", s, b, ["summary", "equity_curve"])
    if s == 200:
        sm = b["summary"]
        print(f"      da_1d={sm['da_1d']:.2%}  da_5d={sm['da_5d']:.2%}  "
              f"sharpe_strat={sm['sharpe_strategy']:.2f}  "
              f"cum_strat={sm['cum_return_strat']:.2%}")

    # ── Summary ───────────────────────────────────────────────
    passed = sum(results)
    total  = len(results)
    print(f"\n{'═'*55}")
    print(f"  Results: {passed}/{total} passed")
    if passed == total:
        print(f"  {PASS} All tests passed")
    else:
        print(f"  {FAIL} {total - passed} test(s) failed")
    print(f"{'═'*55}\n")
    return passed == total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InvestNature API test suite")
    parser.add_argument(
        "--base",
        default=DEFAULT_BASE,
        help=f"API base URL (default: {DEFAULT_BASE})",
    )
    args = parser.parse_args()
    success = main(args.base.rstrip("/"))
    sys.exit(0 if success else 1)
