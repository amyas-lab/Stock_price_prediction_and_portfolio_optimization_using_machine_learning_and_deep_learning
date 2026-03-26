import yfinance as yf
from vnstock import Vnstock 
import pandas as pd
import os 

# Check the format of the data
# df = yf.download("NVDA", start = "2020-01-01", end = "2020-12-31", auto_adjust = False)
# print(df.columns.tolist())
# print(df.head(2))

# Fetch the data from Yahoo Finance

def fetch_nasdaq(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df.index.name = "date"
    df = df[["open", "high", "low", "close", "adj_close", "volume"]]
    return _normalize(df)

# Test the function
# df = fetch_nasdaq("NVDA", "2025-01-01", "2026-03-25")
# print(df.columns.tolist())
# print(df.head(2))

# Normalize the data
def _normalize (df: pd.DataFrame) -> pd.DataFrame:
    # Convert the index to proper datetime type
    df.index = pd.to_datetime(df.index)
    df = df.sort_index() # Sort the index to ensure proper datetime order
    # Remove duplicates
    df = df[~df.index.duplicated()]
    return df.dropna() # Drop missing data

# Fetch the data from Vietnam Stock Exchange
# vnstock 3.x returns: time, open, high, low, close, volume (no adj_close)
# 'time' is a regular column, not the index — we promote it to index

def fetch_vnstock(ticker: str, start: str, end: str, source: str = "VCI") -> pd.DataFrame:
    stock = Vnstock().stock(symbol=ticker, source=source)
    df = stock.quote.history(start=start, end=end)
    df = df.rename(columns={"time": "date"})
    df = df.set_index("date") # Set index 
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df[["open", "high", "low", "close", "volume"]]
    return _normalize(df)

# Test the function
df = fetch_vnstock("VNM", "2020-01-01", "2026-03-25")
print(df.columns.tolist())
print(df.head(2))