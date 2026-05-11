#!/usr/bin/env python3
"""
Run the FastAPI server.
Usage: python run_api.py
"""
import uvicorn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = True,
        workers = 1
    )
