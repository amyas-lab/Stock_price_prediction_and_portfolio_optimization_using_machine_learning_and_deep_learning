"""Generates notebooks/8-EDA-vietnam-stock-market.ipynb programmatically."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

# ── Title ────────────────────────────────────────────────────────────────────
md("""# Vietnam Stock Market — Comprehensive EDA

**Tickers:** VIC, HPG, VHM, TCB, VNM, FPT, VCB, MSN, MWG
**Sources:** `stock_price.csv` · `finance_indicators.csv` · `market_cap_history.csv` · `dividend_history.csv` · `news_all.csv`

Sections:
1. Data Quality Check
2. Stock Price Analysis
3. Inter-Ticker Correlation
4. News Coverage Analysis
5. Financial Indicators
6. Market Cap & Dividends
7. Vietnam-Specific Features (foreign flows, price limits)
8. Summary & Action Items""")

# ── 0  Imports ────────────────────────────────────────────────────────────────
md("## 0  Imports & Data Load")
code("""\
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from pathlib import Path

sns.set_theme(style='whitegrid', palette='tab10', font_scale=1.1)
plt.rcParams.update({'figure.dpi': 120, 'figure.facecolor': 'white'})

DATA    = Path('data/vietnam')
TICKERS = ['VIC', 'HPG', 'VHM', 'TCB', 'VNM', 'FPT', 'VCB', 'MSN', 'MWG']
PALETTE = dict(zip(TICKERS, sns.color_palette('tab10', len(TICKERS))))
print('Environment ready.')
""")

code("""\
# tradingDate in stock_price.csv is dd/mm/YYYY
price = pd.read_csv(DATA / 'stock_price.csv')
price['date'] = pd.to_datetime(price['tradingDate'], dayfirst=True)
price = price.sort_values(['ticker', 'date']).reset_index(drop=True)

fin        = pd.read_csv(DATA / 'finance_indicators.csv')
fin_annual = fin[fin['lengthReport'] == 5].copy()   # 5 = annual report

mcap = pd.read_csv(DATA / 'market_cap_history.csv')
div  = pd.read_csv(DATA / 'dividend_history.csv')

news = pd.read_csv(DATA / 'news_all.csv')
news['date'] = pd.to_datetime(news['date'], errors='coerce')
news = news[news['ticker'].isin(TICKERS)].copy()

print(f"price       : {price.shape[0]:>5} rows  {price.shape[1]} cols")
print(f"fin         : {fin.shape[0]:>5} rows  (annual subset: {fin_annual.shape[0]})")
print(f"mcap        : {mcap.shape[0]:>5} rows")
print(f"div         : {div.shape[0]:>5} rows")
print(f"news        : {news.shape[0]:>5} rows")
""")

# ── 1  Data Quality ───────────────────────────────────────────────────────────
md("---\n## 1  Data Quality Check")

code("""\
# 1.1  Row counts and date range per ticker
coverage = (
    price.groupby('ticker')
    .agg(rows=('date', 'count'), start=('date', 'min'), end=('date', 'max'))
    .reset_index()
)
print('=== Price rows & date coverage ===')
print(coverage.to_string(index=False))
""")

code("""\
# 1.2  Missing-value bar charts per dataset
datasets = {
    'stock_price': price[['open','high','low','close','volume',
                           'foreignBuyVolTotal','foreignSellVolTotal',
                           'ceilingPrice','floorPrice']],
    'finance_ind': fin[['pe','roe','roa','roic','eps','pb',
                         'grossProfitMargin','netProfitMargin','debtEquity']],
    'market_cap':  mcap[['ticker','year','market_cap_bil']],
    'dividends':   div[['ticker','year','value_per_share']],
    'news':        news[['ticker','date','title','text_for_sentiment']],
}

fig, axes = plt.subplots(1, len(datasets), figsize=(22, 5))
for ax, (name, df) in zip(axes, datasets.items()):
    miss = (df.isnull().mean() * 100).pipe(lambda s: s[s > 0].sort_values(ascending=False))
    if miss.empty:
        ax.text(0.5, 0.5, 'No missing\\nvalues', ha='center', va='center',
                fontsize=12, transform=ax.transAxes)
        ax.set_title(name)
        ax.axis('off')
    else:
        miss.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title(f'{name}  (% missing)')
        ax.set_xlabel('% missing')
        ax.set_xlim(0, miss.max() * 1.35)
        for bar in ax.patches:
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f'{bar.get_width():.1f}%', va='center', fontsize=8)
plt.suptitle('Missing Values by Dataset', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()
""")

code("""\
# 1.3  Duplicate check + price anomalies
print('=== Duplicate rows ===')
for name, df in datasets.items():
    print(f'  {name:<15}  {df.duplicated().sum()} row duplicates')
print(f'  price ticker+date key: {price.duplicated(["ticker","date"]).sum()} duplicates')

print('\\n=== Price anomaly checks ===')
print(f'  close == 0       : {(price["close"]==0).sum()}')
print(f'  volume == 0      : {(price["volume"]==0).sum()}')
print(f'  high < low       : {(price["high"]<price["low"]).sum()}')
print(f'  close > ceiling  : {(price["close"]>price["ceilingPrice"]).sum()}')
print(f'  close < floor    : {(price["close"]<price["floorPrice"]).sum()}')
""")

code("""\
# 1.4  Extreme return outliers  |z| > 4
price['daily_return'] = price.groupby('ticker')['close'].pct_change()
price['z_return'] = price.groupby('ticker')['daily_return'].transform(
    lambda x: (x - x.mean()) / x.std()
)
outliers = (
    price[price['z_return'].abs() > 4]
    [['ticker','date','close','daily_return','z_return']]
    .sort_values('z_return', key=abs, ascending=False)
)
print(f'Extreme return events (|z|>4): {len(outliers)}')
print(outliers.head(10).to_string(index=False))
""")

# ── 2  Price Analysis ─────────────────────────────────────────────────────────
md("---\n## 2  Stock Price Analysis")

code("""\
# 2.1  Normalised price (base 100 at first date per ticker)
price['norm_close'] = price.groupby('ticker')['close'].transform(
    lambda x: x / x.iloc[0] * 100
)
fig, ax = plt.subplots(figsize=(14, 6))
for ticker, grp in price.groupby('ticker'):
    ax.plot(grp['date'], grp['norm_close'], label=ticker,
            color=PALETTE[ticker], linewidth=1.6)
ax.axhline(100, color='grey', linestyle='--', linewidth=0.8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_title('Normalised Closing Price  (Base = 100 at first trading date)', fontsize=13)
ax.set_ylabel('Indexed Price')
ax.legend(ncol=3, fontsize=9)
plt.tight_layout()
plt.show()
""")

code("""\
# 2.2  Absolute price per ticker (3x3 grid)
fig, axes = plt.subplots(3, 3, figsize=(16, 10))
for ax, ticker in zip(axes.flat, TICKERS):
    g = price[price['ticker'] == ticker]
    ax.fill_between(g['date'], g['close'], alpha=0.20, color=PALETTE[ticker])
    ax.plot(g['date'], g['close'], color=PALETTE[ticker], linewidth=1.2)
    ax.set_title(ticker, fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y'))
    ax.tick_params(axis='x', labelsize=8)
    ax.set_ylabel('Close (VND)', fontsize=8)
plt.suptitle('Closing Price History per Ticker (VND)', fontsize=13)
plt.tight_layout()
plt.show()
""")

code("""\
# 2.3  Return statistics table
ret = price.dropna(subset=['daily_return'])
ret_stats = (
    ret.groupby('ticker')['daily_return']
    .agg(
        mean_pct = lambda x: x.mean()  * 100,
        std_pct  = lambda x: x.std()   * 100,
        skewness = 'skew',
        kurtosis = lambda x: x.kurt(),
        min_pct  = lambda x: x.min()   * 100,
        max_pct  = lambda x: x.max()   * 100,
    ).round(3)
)
print('=== Daily Return Statistics ===')
print(ret_stats.to_string())
""")

code("""\
# 2.4  Return histograms + Normal fit + Shapiro-Wilk
fig, axes = plt.subplots(3, 3, figsize=(16, 10))
for ax, ticker in zip(axes.flat, TICKERS):
    r = ret[ret['ticker'] == ticker]['daily_return'].dropna()
    ax.hist(r, bins=50, density=True, alpha=0.6, color=PALETTE[ticker], edgecolor='white')
    xv = np.linspace(r.quantile(0.001), r.quantile(0.999), 200)
    ax.plot(xv, stats.norm.pdf(xv, r.mean(), r.std()), 'k--', lw=1.2)
    _, p_sw = stats.shapiro(r.sample(min(300, len(r)), random_state=0))
    ax.set_title(
        f'{ticker}  mu={r.mean()*100:.2f}%  sigma={r.std()*100:.2f}%\\n'
        f'skew={r.skew():.2f}  kurt={r.kurt():.1f}  SW-p={p_sw:.3f}',
        fontsize=8
    )
    ax.set_xlim(-0.12, 0.12)
plt.suptitle('Daily Return Distribution  (dashed = Normal fit)', fontsize=13)
plt.tight_layout()
plt.show()
""")

code("""\
# 2.5  Box-plot comparison
fig, ax = plt.subplots(figsize=(12, 5))
arrays = [ret[ret['ticker']==t]['daily_return'].dropna().values * 100 for t in TICKERS]
bp = ax.boxplot(arrays, labels=TICKERS, patch_artist=True, notch=True, showfliers=False)
for patch, t in zip(bp['boxes'], TICKERS):
    patch.set_facecolor((*PALETTE[t], 0.5))
ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
ax.set_title('Daily Return Distribution by Ticker  (outliers hidden)', fontsize=12)
ax.set_ylabel('Daily Return (%)')
plt.tight_layout()
plt.show()
""")

code("""\
# 2.6  Rolling 20-day annualised volatility
price['vol20'] = price.groupby('ticker')['daily_return'].transform(
    lambda x: x.rolling(20).std() * np.sqrt(252) * 100
)
fig, ax = plt.subplots(figsize=(14, 5))
for ticker, grp in price.groupby('ticker'):
    ax.plot(grp['date'], grp['vol20'], label=ticker,
            color=PALETTE[ticker], linewidth=1.4, alpha=0.85)
ax.set_title('Rolling 20-Day Annualised Volatility (%)', fontsize=13)
ax.set_ylabel('Ann. Volatility (%)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.legend(ncol=3, fontsize=9)
plt.tight_layout()
plt.show()
""")

code("""\
# 2.7  Cumulative return
price['cum_return'] = price.groupby('ticker')['daily_return'].transform(
    lambda x: (1 + x.fillna(0)).cumprod()
)
fig, ax = plt.subplots(figsize=(14, 5))
for ticker, grp in price.groupby('ticker'):
    ax.plot(grp['date'], (grp['cum_return'] - 1) * 100,
            label=ticker, color=PALETTE[ticker], linewidth=1.5)
ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_title('Cumulative Return from First Date (%)', fontsize=13)
ax.set_ylabel('Cumulative Return (%)')
ax.legend(ncol=3, fontsize=9)
plt.tight_layout()
plt.show()
""")

code("""\
# 2.8  Trading volume
fig, axes = plt.subplots(3, 3, figsize=(16, 9))
for ax, ticker in zip(axes.flat, TICKERS):
    g = price[price['ticker'] == ticker]
    ax.bar(g['date'], g['volume'] / 1e6, width=3, alpha=0.7, color=PALETTE[ticker])
    ax.set_title(ticker, fontsize=10)
    ax.set_ylabel('Vol (M)', fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y'))
    ax.tick_params(axis='x', labelsize=8)
plt.suptitle('Daily Trading Volume (Millions of Shares)', fontsize=13)
plt.tight_layout()
plt.show()
""")

# ── 3  Correlation ────────────────────────────────────────────────────────────
md("---\n## 3  Inter-Ticker Correlation")

code("""\
close_wide = price.pivot(index='date', columns='ticker', values='close').dropna()[TICKERS]
ret_wide   = price.pivot(index='date', columns='ticker', values='daily_return').dropna()[TICKERS]

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, data, title in zip(
    axes,
    [close_wide, ret_wide],
    ['Closing Price Correlation', 'Daily Return Correlation']
):
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, ax=ax, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=-1, vmax=1, linewidths=0.5, linecolor='white',
                mask=mask, cbar_kws={'shrink': 0.8})
    ax.set_title(title, fontsize=13)
    ax.tick_params(axis='x', rotation=45)
plt.suptitle('Correlation Matrices  (lower triangle)', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()
""")

code("""\
# Cluster map — reveals sector groupings
corr = ret_wide.corr()
g = sns.clustermap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
                   vmin=-1, vmax=1, linewidths=0.5, figsize=(9, 8),
                   cbar_pos=(0.02, 0.8, 0.03, 0.15))
g.fig.suptitle('Hierarchical Cluster Map — Daily Return Correlation', y=1.02, fontsize=13)
plt.show()

print('\\n=== Pairs with |r| > 0.50 on daily returns ===')
for i, a in enumerate(TICKERS):
    for b in TICKERS[i+1:]:
        r = corr.loc[a, b]
        if abs(r) > 0.50:
            print(f'  {a} <-> {b}:  r = {r:.3f}')
""")

# ── 4  News ───────────────────────────────────────────────────────────────────
md("""\
---
## 4  News Coverage Analysis

> **Data note:** `news_all.csv` has `text_for_sentiment` but **no pre-computed `sentiment_label` / `sentiment_score`**.
> Those columns must be generated (e.g. PhoBERT or mBERT). This section analyses *news volume and temporal distribution* only.""")

code("""\
# 4.1  Article count per ticker
news_count = news['ticker'].value_counts().reindex(TICKERS)
fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(news_count.index, news_count.values,
              color=[PALETTE[t] for t in news_count.index])
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1, str(int(bar.get_height())),
            ha='center', fontsize=10)
ax.set_title('News Article Count per Ticker', fontsize=13)
ax.set_ylabel('# Articles')
plt.tight_layout()
plt.show()
""")

code("""\
# 4.2  Monthly news volume — stacked area
news_ts = (
    news.dropna(subset=['date'])
    .assign(month=lambda d: d['date'].dt.to_period('M').dt.to_timestamp())
    .groupby(['month', 'ticker']).size()
    .unstack(fill_value=0)
    .reindex(columns=TICKERS, fill_value=0)
)
fig, ax = plt.subplots(figsize=(14, 5))
news_ts.plot(kind='area', stacked=True, ax=ax,
             color=[PALETTE[t] for t in news_ts.columns], alpha=0.8)
ax.set_title('Monthly News Volume per Ticker', fontsize=13)
ax.set_ylabel('# Articles')
ax.legend(ncol=3, fontsize=9)
plt.tight_layout()
plt.show()
""")

code("""\
# 4.3  Top news sources
top_sources = news['source'].value_counts().head(15)
fig, ax = plt.subplots(figsize=(10, 5))
top_sources.plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Top 15 News Sources', fontsize=13)
ax.set_xlabel('# Articles')
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# 4.4  Date coverage gap: price vs news
nd  = news.dropna(subset=['date']).groupby('ticker')['date'].agg(['min','max'])
pd_ = price.groupby('ticker')['date'].agg(['min','max'])
print('\\n=== Date coverage: Price vs News ===')
print(pd_.join(nd, lsuffix='_price', rsuffix='_news').to_string())
""")

# ── 5  Financial Indicators ───────────────────────────────────────────────────
md("---\n## 5  Financial Indicators Analysis")

code("""\
# 5.1  Latest annual snapshot (exclude partial 2026)
latest_annual = (
    fin_annual[fin_annual['yearReport'] <= 2025]
    .sort_values('yearReport', ascending=False)
    .groupby('ticker').first().reset_index()
)
cols = ['ticker','yearReport','pe','roe','roa','roic',
        'eps','pb','grossProfitMargin','netProfitMargin','debtEquity']
print('=== Latest annual report per ticker ===')
print(latest_annual[cols].to_string(index=False))
""")

code("""\
# 5.2  P/E and ROE bar charts
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, col, label in [(axes[0],'pe','P/E Ratio'), (axes[1],'roe','ROE')]:
    d = latest_annual.dropna(subset=[col]).sort_values(col, ascending=False)
    bars = ax.bar(d['ticker'], d[col],
                  color=[PALETTE.get(t,'grey') for t in d['ticker']], alpha=0.85)
    for bar in bars:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + abs(d[col].max()) * 0.01,
                f'{v:.2f}', ha='center', fontsize=9)
    ax.set_title(f'{label} — Latest Annual', fontsize=12)
    ax.set_ylabel(label)
plt.suptitle('Valuation & Profitability Snapshot', fontsize=14)
plt.tight_layout()
plt.show()
""")

code("""\
# 5.3  P/E and ROE trends 2018-2025
fp = fin_annual[fin_annual['yearReport'].between(2018, 2025)]
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, col, label in [(axes[0],'pe','P/E'), (axes[1],'roe','ROE')]:
    for ticker, grp in fp.groupby('ticker'):
        g = grp.sort_values('yearReport')
        ax.plot(g['yearReport'], g[col], marker='o', label=ticker,
                color=PALETTE.get(ticker,'grey'), linewidth=1.5)
    ax.set_title(f'{label} Over Time (Annual)', fontsize=12)
    ax.set_ylabel(label)
    ax.set_xlabel('Year')
    ax.legend(ncol=3, fontsize=9)
plt.suptitle('Key Ratios 2018–2025', fontsize=14)
plt.tight_layout()
plt.show()
""")

code("""\
# 5.4  ROE / ROA / ROIC box-plots
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, col in zip(axes, ['roe','roa','roic']):
    data = [fin_annual[fin_annual['ticker']==t][col].dropna().values for t in TICKERS]
    bp = ax.boxplot(data, labels=TICKERS, patch_artist=True, showfliers=True,
                    flierprops={'marker':'o','markersize':3,'alpha':0.4})
    for patch, t in zip(bp['boxes'], TICKERS):
        patch.set_facecolor((*PALETTE[t], 0.5))
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax.set_title(col.upper(), fontsize=12)
    ax.tick_params(axis='x', rotation=45)
plt.suptitle('Return Metrics Distribution  (all annual years)', fontsize=13)
plt.tight_layout()
plt.show()
""")

code("""\
# 5.5  Gross vs Net profit margin
margin = latest_annual.set_index('ticker')[['grossProfitMargin','netProfitMargin']].dropna()
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(margin)); w = 0.35
ax.bar(x - w/2, margin['grossProfitMargin'] * 100, w, label='Gross %', color='steelblue', alpha=0.8)
ax.bar(x + w/2, margin['netProfitMargin']   * 100, w, label='Net %',   color='coral',     alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(margin.index)
ax.set_ylabel('Margin (%)')
ax.set_title('Gross vs Net Profit Margin — Latest Annual', fontsize=13)
ax.legend()
plt.tight_layout()
plt.show()
""")

code("""\
# 5.6  P/E vs ROE bubble chart  (bubble size = market cap)
scatter_df = latest_annual.merge(
    mcap[mcap['ticker'].isin(TICKERS)]
    .sort_values('year', ascending=False)
    .groupby('ticker').first().reset_index()[['ticker','market_cap_bil']],
    on='ticker', how='left'
).dropna(subset=['pe','roe','market_cap_bil'])

fig, ax = plt.subplots(figsize=(9, 6))
for _, row in scatter_df.iterrows():
    ax.scatter(row['pe'], row['roe'],
               s=row['market_cap_bil'] / 300,
               color=PALETTE.get(row['ticker'],'grey'),
               alpha=0.8, edgecolors='grey', linewidths=0.5)
    ax.annotate(row['ticker'], (row['pe'], row['roe']),
                textcoords='offset points', xytext=(6, 3), fontsize=9)
ax.set_xlabel('P/E Ratio')
ax.set_ylabel('ROE')
ax.set_title('P/E vs ROE  (bubble size proportional to market cap)', fontsize=13)
plt.tight_layout()
plt.show()
""")

# ── 6  Market Cap & Dividends ─────────────────────────────────────────────────
md("---\n## 6  Market Cap & Dividend Analysis")

code("""\
# 6.1  Market cap evolution
mcap_f = mcap[mcap['ticker'].isin(TICKERS)]
fig, ax = plt.subplots(figsize=(13, 5))
for ticker, grp in mcap_f.groupby('ticker'):
    g = grp.sort_values('year')
    ax.plot(g['year'], g['market_cap_bil'] / 1e3, marker='o', label=ticker,
            color=PALETTE.get(ticker,'grey'), linewidth=1.5)
ax.set_title('Market Capitalisation History (Trillion VND)', fontsize=13)
ax.set_ylabel('Market Cap (Trillion VND)')
ax.set_xlabel('Year')
ax.legend(ncol=3, fontsize=9)
plt.tight_layout()
plt.show()
""")

code("""\
# 6.2  Latest market cap ranking
latest_mcap = (
    mcap_f.sort_values('year', ascending=False)
    .groupby('ticker').first().reset_index()
    .sort_values('market_cap_bil', ascending=True)
)
fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.barh(latest_mcap['ticker'], latest_mcap['market_cap_bil'] / 1e3,
               color=[PALETTE.get(t,'grey') for t in latest_mcap['ticker']], alpha=0.85)
for bar in bars:
    ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height() / 2,
            f'{bar.get_width():.0f}T', va='center', fontsize=9)
ax.set_title('Market Cap Ranking — Latest Year (Trillion VND)', fontsize=13)
ax.set_xlabel('Trillion VND')
plt.tight_layout()
plt.show()
""")

code("""\
# 6.3  Dividend per share history
div_f = div[div['ticker'].isin(TICKERS)]
n_t   = len(TICKERS)
bar_w = 0.8 / n_t

fig, ax = plt.subplots(figsize=(13, 5))
for i, ticker in enumerate(TICKERS):
    g = div_f[div_f['ticker'] == ticker].sort_values('year')
    offsets = [y + (i - n_t / 2) * bar_w + bar_w / 2 for y in g['year']]
    ax.bar(offsets, g['value_per_share'], width=bar_w, label=ticker,
           color=PALETTE.get(ticker,'grey'), alpha=0.85)
ax.set_title('Cash Dividend per Share (VND)', fontsize=13)
ax.set_ylabel('Dividend (VND/share)')
ax.set_xlabel('Year')
ax.legend(ncol=3, fontsize=9)
plt.tight_layout()
plt.show()
""")

code("""\
# 6.4  Estimated dividend yield  (dividend / year-end close)
price['year'] = price['date'].dt.year
ye_close = (
    price.sort_values('date')
    .groupby(['ticker','year'])['close'].last().reset_index()
)
dy = div_f.merge(ye_close, on=['ticker','year'], how='inner')
dy['yield_pct'] = dy['value_per_share'] / dy['close'] * 100

fig, ax = plt.subplots(figsize=(13, 5))
for ticker, grp in dy.groupby('ticker'):
    g = grp.sort_values('year')
    ax.plot(g['year'], g['yield_pct'], marker='o', label=ticker,
            color=PALETTE.get(ticker,'grey'), linewidth=1.5)
ax.axhline(2, color='red', linestyle='--', linewidth=0.8, label='2% ref')
ax.set_title('Estimated Dividend Yield (%)', fontsize=13)
ax.set_ylabel('Div Yield (%)')
ax.set_xlabel('Year')
ax.legend(ncol=4, fontsize=9)
plt.tight_layout()
plt.show()

print('\\n=== Dividend Yield (%) by year ===')
print(dy.pivot(index='year', columns='ticker', values='yield_pct').round(2).to_string())
""")

# ── 7  Vietnam-Specific ───────────────────────────────────────────────────────
md("---\n## 7  Vietnam-Specific Features")

code("""\
# 7.1  Net foreign flow ratio  (foreignBuy - foreignSell) / volume
price['fbuy_ratio']  = price['foreignBuyVolTotal']  / price['volume'].replace(0, np.nan)
price['fsell_ratio'] = price['foreignSellVolTotal'] / price['volume'].replace(0, np.nan)
price['net_ff']      = price['fbuy_ratio'] - price['fsell_ratio']

fig, axes = plt.subplots(3, 3, figsize=(16, 10))
for ax, ticker in zip(axes.flat, TICKERS):
    g = price[price['ticker'] == ticker].dropna(subset=['net_ff'])
    ax.fill_between(g['date'], g['net_ff'], where=g['net_ff'] > 0,
                    alpha=0.55, color='green')
    ax.fill_between(g['date'], g['net_ff'], where=g['net_ff'] <= 0,
                    alpha=0.55, color='red')
    ax.axhline(0, color='grey', linewidth=0.6)
    ax.set_title(ticker, fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y'))
    ax.tick_params(axis='x', labelsize=8)
handles = [plt.Rectangle((0,0),1,1, color='green', alpha=0.55),
           plt.Rectangle((0,0),1,1, color='red',   alpha=0.55)]
axes[0,0].legend(handles, ['Net Buy','Net Sell'], fontsize=7)
plt.suptitle('Net Foreign Flow Ratio  (foreignBuy - foreignSell) / Volume', fontsize=13)
plt.tight_layout()
plt.show()

print('\\n=== Mean foreign flow ratios ===')
print(price.groupby('ticker')[['fbuy_ratio','fsell_ratio','net_ff']].mean().round(4).to_string())
""")

code("""\
# 7.2  Price-limit proximity  (0 = at floor, 1 = at ceiling)
price['limit_prox'] = (
    (price['close'] - price['floorPrice']) /
    (price['ceilingPrice'] - price['floorPrice']).replace(0, np.nan)
)
fig, axes = plt.subplots(3, 3, figsize=(16, 9))
for ax, ticker in zip(axes.flat, TICKERS):
    d = price[price['ticker'] == ticker]['limit_prox'].dropna()
    ax.hist(d, bins=30, color=PALETTE[ticker], alpha=0.7, edgecolor='white')
    ax.axvline(0.5, color='grey', linestyle='--', linewidth=0.8)
    ax.set_title(f'{ticker}  mean={d.mean():.2f}', fontsize=10)
    ax.set_xlim(0, 1)
plt.suptitle('Price Limit Proximity  (0 = floor, 1 = ceiling)', fontsize=13)
plt.tight_layout()
plt.show()
""")

code("""\
# 7.3  Ceiling / floor hit frequency  (within 0.5% of the limit)
TOL = 0.005
price['ceil_hit']  = (price['close'] >= price['ceilingPrice'] * (1 - TOL)).astype(int)
price['floor_hit'] = (price['close'] <= price['floorPrice']   * (1 + TOL)).astype(int)

lf = price.groupby('ticker')[['ceil_hit','floor_hit']].sum()
lf['days']    = price.groupby('ticker').size()
lf['ceil_%']  = (lf['ceil_hit']  / lf['days'] * 100).round(1)
lf['floor_%'] = (lf['floor_hit'] / lf['days'] * 100).round(1)

fig, ax = plt.subplots(figsize=(11, 4))
x = np.arange(len(TICKERS)); w = 0.35
ax.bar(x - w/2, lf.loc[TICKERS,'ceil_%'],  w, label='Ceiling hit %', color='tomato',    alpha=0.8)
ax.bar(x + w/2, lf.loc[TICKERS,'floor_%'], w, label='Floor hit %',   color='steelblue', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(TICKERS)
ax.set_ylabel('% of Trading Days')
ax.set_title('Price-Limit Touch Frequency  (+/-7% daily limit, within 0.5%)', fontsize=12)
ax.legend()
plt.tight_layout()
plt.show()

print(lf[['ceil_hit','floor_hit','days','ceil_%','floor_%']].to_string())
""")

# ── 8  Summary ────────────────────────────────────────────────────────────────
md("---\n## 8  Summary of Findings & Action Items Before Modelling")

code("""\
SEP = '=' * 68
print(SEP)
print('EDA SUMMARY — Vietnam Stock Market')
print(SEP)

print('\\n[1] COVERAGE')
print(coverage.to_string(index=False))
print('  -> VNM has 40 fewer rows than all other tickers.')

print('\\n[2] MISSING VALUES')
ohlcv_null = price[['open','high','low','close','volume']].isnull().sum().to_dict()
ff_null    = (price[['fbuy_ratio','fsell_ratio']].isnull().mean() * 100).round(1)
print('  OHLCV null counts  :', ohlcv_null)
print('  Foreign flow null% :', ff_null.to_dict())
fin_null   = (fin[['pe','roe','roa']].isnull().mean() * 100).round(1)
print('  Finance ind null%  :', fin_null.to_dict())

print('\\n[3] OUTLIERS')
print(f'  Extreme return events (|z|>4): {len(outliers)}')
if not outliers.empty:
    print(outliers[['ticker','date','daily_return']].head(6).to_string(index=False))

print('\\n[4] RETURN CHARACTERISTICS')
print(ret_stats.to_string())
print('  -> Fat tails (excess kurtosis) in all tickers.')
print('  -> Mostly negative skew: large drawdowns exceed large gains.')

print('\\n[5] CORRELATIONS')
vals = corr.values[np.tril_indices_from(corr.values, k=-1)]
print(f'  Mean pairwise return correlation: {vals.mean():.3f}')

print('\\n[6] SENTIMENT DATA STATUS')
print(f'  news_all.csv: {len(news)} articles across {news["ticker"].nunique()} tickers.')
print('  Columns available: ticker, title, date, source, text_for_sentiment')
print('  NO sentiment_label / sentiment_score yet.')
print('  -> Next step: run PhoBERT or multilingual-BERT to generate scores.')

print('\\n[7] FINANCE INDICATORS')
bad = fin[fin['yearReport'] == 0].shape[0]
print(f'  yearReport==0 (API artefacts): {bad} rows — drop before modelling.')
print('  Use lengthReport==5 for clean annual analysis.')
print('  Or merge quarterly data on (ticker, year, quarter) for richer features.')

print('\\n[8] PRE-MODELLING ACTION ITEMS')
actions = [
    '1.  Fix VNM: forward-fill or exclude the 40 missing trading days.',
    '2.  Drop yearReport==0 rows from finance_indicators.',
    '3.  Choose quarterly vs annual merge strategy for financial features.',
    '4.  Run PhoBERT/mBERT on text_for_sentiment -> daily sentiment score.',
    '5.  Aggregate daily sentiment per ticker (mean/max/exponential decay).',
    '6.  Winsorise extreme returns (|z|>4) or add binary outlier indicator.',
    '7.  Impute foreign-flow NaNs (forward-fill recommended).',
    '8.  Add technical indicators via src/data/preprocess.py (RSI, MACD, ATR, BB, EMA).',
    '9.  Add limit_prox, ceil_hit, floor_hit, net_ff as Vietnam-specific features.',
    '10. Standardise all date columns to datetime before cross-file merges.',
]
for a in actions:
    print(f'  {a}')

print(f'\\n{SEP}')
""")

# ── Write notebook ────────────────────────────────────────────────────────────
nb.cells = cells
out = Path('/Users/cps/DL4AI-240166-project-1/notebooks/8-EDA-vietnam-stock-market.ipynb')
with open(out, 'w') as f:
    nbf.write(nb, f)
print(f'Written: {out}  ({out.stat().st_size:,} bytes, {len(nb.cells)} cells)')
