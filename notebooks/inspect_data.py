import pandas as pd
from pathlib import Path

DATA_DIR = Path('/Users/cps/DL4AI-240166-project-1/notebooks/data/vietnam')
MASTER_PATH = DATA_DIR / 'vn_stocks_master_features.csv'
VNI_PATH    = DATA_DIR / 'csv' / 'vnindex_ohlcv.csv'

df_all = pd.read_csv(MASTER_PATH)
df_all['date'] = pd.to_datetime(df_all['date'])
df_all = df_all.sort_values(['ticker', 'date']).reset_index(drop=True)

df_vni_raw = pd.read_csv(VNI_PATH)
df_vni_raw['date'] = pd.to_datetime(df_vni_raw['date'])
df_vni_raw = df_vni_raw.sort_values('date').reset_index(drop=True)

print(f"Master data rows: {len(df_all)}")
print(f"VNI raw rows: {len(df_vni_raw)}")
print(f"Master date range: {df_all['date'].min()} to {df_all['date'].max()}")
print(f"VNI date range: {df_vni_raw['date'].min()} to {df_vni_raw['date'].max()}")

df_fpt = df_all[df_all['ticker'] == 'FPT'].copy().reset_index(drop=True)
print(f"FPT rows: {len(df_fpt)}")

intersection = set(df_fpt['date']).intersection(set(df_vni_raw['date']))
print(f"Number of intersecting dates between FPT and VNI: {len(intersection)}")

missing_in_vni = set(df_fpt['date']) - set(df_vni_raw['date'])
print(f"Number of dates in FPT but missing in VNI: {len(missing_in_vni)}")

if len(missing_in_vni) > 0:
    print("Some missing dates examples:", list(missing_in_vni)[:5])

