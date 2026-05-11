#!/usr/bin/env python3
"""
Quick API test script.
Run after: python run_api.py
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_endpoint(name, method, url, payload=None):
    print(f"\nTesting: {name}")
    print(f"  {method.upper()} {url}")
    try:
        if method == 'get':
            r = requests.get(url, timeout=30)
        else:
            r = requests.post(url, json=payload, timeout=30)
        print(f"  Status : {r.status_code}")
        data = r.json()
        print(f"  Response: {str(data)[:200]}...")
        return data
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

test_endpoint("Health Check",         "get",  f"{BASE_URL}/health")
test_endpoint("Price Prediction FPT", "post", f"{BASE_URL}/predict/price",   {"ticker": "FPT", "n_days_back": 20})
test_endpoint("Trading Signal VHM",   "post", f"{BASE_URL}/predict/signal",  {"ticker": "VHM", "threshold": 0.55})
test_endpoint("Portfolio Equal-Wt",   "get",  f"{BASE_URL}/portfolio/equal_weight")
test_endpoint("Profitability Scores", "get",  f"{BASE_URL}/portfolio/scores/profitability")
test_endpoint("Risk Scores",          "get",  f"{BASE_URL}/portfolio/scores/risk")

print("\n✓ All tests complete!")
