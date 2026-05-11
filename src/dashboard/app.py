# ============================================================
# VnAlpha — Vietnam Stock AI Dashboard
# Task 5.2: Streamlit SaaS
# Style: Modern FinTech / Dark Mode
# ============================================================

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
from datetime import datetime

API_BASE = "http://localhost:8000"

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title            = "VnAlpha — Vietnam Stock AI",
    page_icon             = "📈",
    layout                = "wide",
    initial_sidebar_state = "expanded"
)

# ── Global CSS ────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
<style>
    /* ── Base / Dark Background ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        background-color: #0d0d0d !important;
        color: #e0e0e0 !important;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #0d0d0d !important;
    }

    [data-testid="stMain"] {
        background-color: #0d0d0d !important;
    }

    /* Remove Streamlit default padding */
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 1rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        background-color: #0d0d0d !important;
    }

    /* ── Hide only menu/footer, KEEP header visible for toggle ── */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #111111 !important;
        border-right: 1px solid #2a2a2a !important;
        box-shadow: 2px 0 12px rgba(0,0,0,0.4) !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    /* ── Sidebar toggle button — always visible ── */
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        background: #1e1e1e !important;
        border-radius: 8px !important;
    }
    [data-testid="collapsedControl"] svg {
        fill: #5ba8ff !important;
    }

    /* ── Header bar — keep visible ── */
    [data-testid="stHeader"] {
        background-color: #0d0d0d !important;
        border-bottom: 1px solid #1e1e1e !important;
    }

    /* ── Cards ── */
    .vna-card {
        background: #1a1a1a !important;
        border-radius: 20px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 2px 16px rgba(0,0,0,0.4);
        margin-bottom: 1rem;
        border: 1px solid #2a2a2a !important;
    }
    .vna-card-accent {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 20px;
        padding: 1.4rem 1.6rem;
        color: white;
        margin-bottom: 1rem;
        border: 1px solid #2a2a3e;
    }

    /* ── Metric Cards ── */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #1a1a1a !important;
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        flex: 1;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        border: 1px solid #2a2a2a !important;
    }
    .metric-card .label {
        font-size: 0.75rem;
        color: #888888 !important;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    .metric-card .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff !important;
    }
    .metric-card .delta {
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 0.2rem;
    }
    .delta-up   { color: #27ae60 !important; }
    .delta-down { color: #e74c3c !important; }
    .delta-neu  { color: #888888 !important; }

    /* ── HTML Tables inside cards ── */
    table { width: 100%; border-collapse: collapse; }
    td, th { color: #e0e0e0 !important; background: transparent !important; }

    /* ── Signal Badge ── */
    .badge-buy  {
        background: #0d3320; color: #27ae60;
        padding: 4px 12px; border-radius: 20px;
        font-weight: 600; font-size: 0.85rem;
    }
    .badge-sell {
        background: #3a0d0d; color: #e74c3c;
        padding: 4px 12px; border-radius: 20px;
        font-weight: 600; font-size: 0.85rem;
    }
    .badge-hold {
        background: #3a2e0d; color: #f39c12;
        padding: 4px 12px; border-radius: 20px;
        font-weight: 600; font-size: 0.85rem;
    }

    /* ── Risk Badge ── */
    .risk-low      { color: #27ae60; font-weight: 600; }
    .risk-medium   { color: #f39c12; font-weight: 600; }
    .risk-high     { color: #e74c3c; font-weight: 600; }
    .risk-excluded { color: #8e44ad; font-weight: 600; }

    /* ── Page title ── */
    .page-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff !important;
        margin-bottom: 0.2rem;
    }
    .page-subtitle {
        font-size: 0.85rem;
        color: #888888 !important;
        margin-bottom: 1.5rem;
    }

    /* ── Brand ── */
    .brand-logo {
        font-size: 1.4rem;
        font-weight: 800;
        color: #ffffff !important;
        letter-spacing: -0.03em;
    }
    .brand-logo span { color: #5ba8ff; }

    /* ── Option menu override ── */
    .nav-link {
        border-radius: 12px !important;
        margin: 2px 8px !important;
        color: #aaaaaa !important;
    }
    .nav-link.active {
        background: #5ba8ff !important;
        color: white !important;
    }

    /* ── Highlight pill ── */
    .highlight-pill {
        background: #3a3800;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #fffa93;
    }

    /* ── Plotly chart background ── */
    .js-plotly-plot .plotly .main-svg {
        border-radius: 12px;
    }

    /* ── Streamlit dataframe ── */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: #5ba8ff !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: #3d8fe0 !important;
        box-shadow: 0 4px 12px rgba(91,168,255,0.4) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Slider ── */
    .stSlider [data-baseweb="slider"] {
        padding-top: 0.5rem;
    }

    /* ── Selectbox / Radio ── */
    [data-testid="stSelectbox"] > div,
    [data-testid="stRadio"] > div {
        color: #e0e0e0 !important;
    }

    /* ── Divider ── */
    hr { border-color: #2a2a2a !important; }

    /* ── Streamlit info/error boxes ── */
    .stAlert { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ── Helper: card wrapper ───────────────────────────────────────
def card(content_fn, title=None, accent=False):
    cls = "vna-card-accent" if accent else "vna-card"
    if title:
        st.markdown(
            f'<div class="{cls}">'
            f'<div style="font-weight:600;font-size:0.95rem;'
            f'color:#ffffff;margin-bottom:0.8rem;">{title}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
    content_fn()
    st.markdown('</div>', unsafe_allow_html=True)

def metric_cards(items):
    """items = list of (label, value, delta, delta_type)"""
    cols_html = ""
    for label, value, delta, dtype in items:
        arrow = "↑" if dtype == "up" else "↓" if dtype == "down" else "→"
        dcls  = f"delta-{dtype}"
        cols_html += f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="delta {dcls}">{arrow} {delta}</div>
        </div>"""
    st.markdown(
        f'<div class="metric-row">{cols_html}</div>',
        unsafe_allow_html=True
    )

# ── API Helper ────────────────────────────────────────────────
def call_api(method, endpoint, payload=None):
    try:
        url = f"{API_BASE}{endpoint}"
        r   = requests.get(url, timeout=30) if method == 'get' \
              else requests.post(url, json=payload, timeout=30)
        return (r.json(), None) if r.status_code == 200 \
               else (None, f"API Error {r.status_code}")
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API"
    except Exception as e:
        return None, str(e)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="brand-logo" style="padding:0.5rem 1rem 1rem;">'
        'Vn<span>Alpha</span></div>',
        unsafe_allow_html=True
    )

    health, err = call_api('get', '/health')
    if health:
        st.markdown(
            '<div style="margin:0 1rem 1rem;padding:8px 12px;'
            'background:#0d3320;border-radius:10px;font-size:0.8rem;'
            'color:#27ae60;font-weight:600;">'
            '🟢 API Connected</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="margin:0 1rem 1rem;padding:8px 12px;'
            'background:#3a0d0d;border-radius:10px;font-size:0.8rem;'
            'color:#e74c3c;font-weight:600;">'
            '🔴 API Offline</div>',
            unsafe_allow_html=True
        )

    page = option_menu(
        menu_title  = None,
        options     = [
            "Overview", "Price Prediction",
            "Signal Scanner", "Portfolio", "Risk Analysis"
        ],
        icons       = [
            "house-fill", "graph-up-arrow",
            "crosshair", "briefcase-fill",
            "exclamation-triangle-fill"
        ],
        default_index = 0,
        styles = {
            "container"  : {"padding": "0", "background": "#111111"},
            "icon"       : {"font-size": "0.9rem", "color": "#888888"},
            "nav-link"   : {
                "font-size"    : "0.88rem",
                "font-weight"  : "500",
                "color"        : "#aaaaaa",
                "border-radius": "12px",
                "margin"       : "2px 8px",
            },
            "nav-link-selected": {
                "background-color": "#5ba8ff",
                "color"           : "white",
                "font-weight"     : "600",
            },
        }
    )

    st.divider()
    st.markdown(
        '<div style="padding:0 1rem;font-size:0.75rem;color:#555555;">'
        'CS313 Deep Learning<br/>Student: DL4AI-240166</div>',
        unsafe_allow_html=True
    )

# ── PLOTLY DARK THEME ─────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor = 'rgba(0,0,0,0)',
    plot_bgcolor  = 'rgba(0,0,0,0)',
    font          = dict(family='Inter', color='#e0e0e0'),
    margin        = dict(l=10, r=10, t=40, b=10),
    xaxis         = dict(showgrid=True, gridcolor='#2a2a2a',
                         linecolor='#2a2a2a', color='#aaaaaa'),
    yaxis         = dict(showgrid=True, gridcolor='#2a2a2a',
                         linecolor='#2a2a2a', color='#aaaaaa'),
)

ALL_TICKERS = [
    'FPT','VCB','VHM','VNM','HPG','VIC','TCB',
    'MSN','MWG','VND','HDB','GAS','SAB','PNJ',
    'MBB','ACB','CTG','BID','SHB','TPB','KDH',
    'DXG','HSG','PDR','CMG','ELC','SGT'
]

# ============================================================
# PAGE 1: OVERVIEW
# ============================================================
if page == "Overview":
    st.markdown(
        '<div class="page-title">Good Morning 👋</div>'
        '<div class="page-subtitle">Here\'s your VnAlpha market '
        'intelligence summary</div>',
        unsafe_allow_html=True
    )

    metric_cards([
        ("Universe",        "27 Tickers",  "HOSE Blue Chips",   "neu"),
        ("Selection Alpha", "+53.59%",     "vs VNI Benchmark",  "up"),
        ("Best Sharpe",     "1.3969",      "Equal-Weight Port", "up"),
        ("Active Models",   "3 Models",    "MTL + XGB + GRU",   "neu"),
    ])

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown('<div class="vna-card">', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-weight:600;margin-bottom:1rem;color:#ffffff;">'
            'Portfolio Performance Comparison</div>',
            unsafe_allow_html=True
        )
        df_perf = pd.DataFrame({
            'Profile'     : ['Risk-Taking','Prudent','Equal-Weight','VNI Benchmark'],
            'Total Return': [35.73, 16.44, 53.93, 40.74],
            'Sharpe'      : [0.8596, 0.4550, 1.3969, 1.1694],
            'Max Drawdown': [-23.20, -21.97, -19.24, None]
        })
        fig = go.Figure()
        colors = ['#5ba8ff','#a8d8ea','#fffa93','#444444']
        for i, row in df_perf.iterrows():
            fig.add_trace(go.Bar(
                name=row['Profile'],
                x=[row['Profile']],
                y=[row['Total Return']],
                marker_color=colors[i],
                text=f"{row['Total Return']:.1f}%",
                textposition='outside',
                textfont=dict(color='#e0e0e0'),
                showlegend=False
            ))
        fig.update_layout(
            **PLOT_LAYOUT,
            title='Total Return by Portfolio Profile (%)',
            height=280, bargap=0.3
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="vna-card">', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-weight:600;margin-bottom:1rem;color:#ffffff;">'
            'Model Validation</div>',
            unsafe_allow_html=True
        )
        st.markdown("""
        <table style="width:100%;font-size:0.82rem;border-collapse:collapse;">
        <tr style="border-bottom:1px solid #2a2a2a;">
            <td style="padding:8px 0;color:#888888;">Spearman ρ</td>
            <td style="font-weight:600;text-align:right;color:#ffffff;">0.5562***</td>
        </tr>
        <tr style="border-bottom:1px solid #2a2a2a;">
            <td style="padding:8px 0;color:#888888;">Top 10 Overlap</td>
            <td style="font-weight:600;text-align:right;color:#ffffff;">9/10 (90%)</td>
        </tr>
        <tr style="border-bottom:1px solid #2a2a2a;">
            <td style="padding:8px 0;color:#888888;">Positive Precision</td>
            <td style="font-weight:600;text-align:right;color:#ffffff;">90% vs 35%</td>
        </tr>
        <tr style="border-bottom:1px solid #2a2a2a;">
            <td style="padding:8px 0;color:#888888;">Selection Alpha</td>
            <td style="font-weight:600;color:#27ae60;text-align:right;">+98.90%</td>
        </tr>
        <tr>
            <td style="padding:8px 0;color:#888888;">vs VNI</td>
            <td style="font-weight:600;color:#27ae60;text-align:right;">+53.59%</td>
        </tr>
        </table>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="vna-card">', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-weight:600;margin-bottom:0.8rem;color:#ffffff;">'
            'Pipeline</div>',
            unsafe_allow_html=True
        )
        steps = [
            ("📊", "Data",       "27 HOSE tickers"),
            ("🧠", "MTL Model",  "GRU + Attention"),
            ("🎯", "Signals",    "XGBoost 43 features"),
            ("⚖️",  "Portfolio", "Mean-Variance Opt."),
        ]
        for icon, title, desc in steps:
            st.markdown(
                f'<div style="display:flex;align-items:center;'
                f'gap:10px;padding:8px 0;border-bottom:1px solid #2a2a2a;">'
                f'<span style="font-size:1.1rem;">{icon}</span>'
                f'<div><div style="font-weight:600;font-size:0.82rem;color:#ffffff;">'
                f'{title}</div>'
                f'<div style="color:#888888;font-size:0.75rem;">'
                f'{desc}</div></div></div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PAGE 2: PRICE PREDICTION
# ============================================================
elif page == "Price Prediction":
    st.markdown(
        '<div class="page-title">📈 Price Prediction</div>'
        '<div class="page-subtitle">MTL Seq2Seq GRU + Attention '
        '— T+5 cumulative return forecast</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 2.5])

    with col1:
        st.markdown('<div class="vna-card">', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-weight:600;margin-bottom:1rem;color:#ffffff;">'
            'Configure</div>',
            unsafe_allow_html=True
        )
        ticker = st.selectbox("Ticker", ALL_TICKERS)
        n_days = st.slider("Lookback (days)", 20, 60, 20)
        predict_btn = st.button("🔮 Run Prediction", use_container_width=True)
        st.markdown(
            '<div style="margin-top:1rem;padding:10px;'
            'background:#111111;border-radius:10px;'
            'font-size:0.75rem;color:#888888;border:1px solid #2a2a2a;">'
            '<b style="color:#e0e0e0;">Model:</b> MTL Seq2Seq<br/>'
            '<b style="color:#e0e0e0;">Output:</b> 5-day return trajectory<br/>'
            '<b style="color:#e0e0e0;">Features:</b> 25 technical indicators'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if predict_btn:
            with st.spinner(f"Predicting {ticker}..."):
                result, err = call_api(
                    'post', '/predict/price',
                    {"ticker": ticker, "n_days_back": n_days}
                )
            if err:
                st.error(err)
            elif result:
                direction   = result['direction']
                confidence  = result['confidence']
                cur_price   = result['current_price']
                pred_prices = result['predicted_prices']
                pred_rets   = result['predicted_returns']

                dir_icon = "⬆️" if direction=="UP" else "⬇️" if direction=="DOWN" else "➡️"
                target   = pred_prices[-1] if pred_prices else cur_price
                chg      = (target - cur_price) / cur_price * 100
                dtype    = "up" if chg > 0 else "down" if chg < 0 else "neu"

                metric_cards([
                    ("Current Price", f"{cur_price:,.0f} đ", "Latest close", "neu"),
                    ("Direction",     f"{dir_icon} {direction}", f"Confidence {confidence:.1%}", dtype),
                    ("5-Day Target",  f"{target:,.0f} đ", f"{chg:+.2f}%", dtype),
                ])

                n_pred = len(pred_prices) if pred_prices else 1
                days   = ['Today', 'T+5 Target']
                prices = [cur_price] + (pred_prices or [cur_price])
                lc     = '#27ae60' if direction=='UP' else '#e74c3c' if direction=='DOWN' else '#f39c12'

                st.markdown('<div class="vna-card">', unsafe_allow_html=True)
                fig = go.Figure()
                fig.add_hline(
                    y=cur_price, line_dash='dash',
                    line_color='#555555', line_width=1,
                    annotation_text='Current Price',
                    annotation_font_size=10,
                    annotation_font_color='#888888'
                )
                fig.add_trace(go.Scatter(
                    x=days, y=prices,
                    mode='lines+markers',
                    name='Predicted',
                    line=dict(color=lc, width=3),
                    marker=dict(size=10, color=lc, line=dict(color='#0d0d0d', width=2)),
                ))
                fig.update_layout(
                    **PLOT_LAYOUT,
                    title=f"{ticker} — Price Forecast",
                    xaxis_title="Trading Day",
                    yaxis_title="Price (VND)",
                    height=320, showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                if pred_prices and pred_rets:
                    st.markdown('<div class="vna-card">', unsafe_allow_html=True)
                    st.markdown(
                        '<div style="font-weight:600;margin-bottom:0.8rem;color:#ffffff;">'
                        'Forecast Details</div>', unsafe_allow_html=True
                    )
                    rows = []
                    for i, (p, r) in enumerate(zip(pred_prices, pred_rets), 1):
                        chg_pct = (p - cur_price) / cur_price * 100
                        rows.append({
                            'Day'    : f"Day {i}",
                            'Price'  : f"{p:,.2f} VND",
                            'LogRet' : f"{r:.4f}",
                            'Change%': f"{chg_pct:+.2f}%"
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                    gauge_color = '#27ae60' if confidence > 0.6 else '#f39c12' if confidence > 0.4 else '#e74c3c'
                    fig_g = go.Figure(go.Indicator(
                        mode  = "gauge+number",
                        value = confidence * 100,
                        title = {'text': "Model Conviction (%)", 'font': {'color': '#e0e0e0'}},
                        number= {'font': {'color': '#e0e0e0'}},
                        gauge = {
                            'axis' : {'range': [0, 100], 'tickcolor': '#888888'},
                            'bar'  : {'color': gauge_color},
                            'bgcolor': '#1a1a1a',
                            'steps': [
                                {'range': [0,   40],  'color': '#2a1010'},
                                {'range': [40,  55],  'color': '#2a2510'},
                                {'range': [55, 100],  'color': '#102a18'},
                            ],
                            'threshold': {
                                'line': {'color': 'red', 'width': 3},
                                'thickness': 0.75, 'value': 55
                            }
                        }
                    ))
                    fig_g.update_layout(**PLOT_LAYOUT, height=220)
                    st.plotly_chart(fig_g, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PAGE 3: SIGNAL SCANNER
# ============================================================
elif page == "Signal Scanner":
    st.markdown(
        '<div class="page-title">🎯 Signal Scanner</div>'
        '<div class="page-subtitle">XGBoost Classifier — '
        'BUY/SELL/HOLD with conviction gating</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown('<div class="vna-card">', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-weight:600;margin-bottom:1rem;color:#ffffff;">'
            'Settings</div>', unsafe_allow_html=True
        )
        threshold = st.slider("Conviction Threshold", 0.40, 0.80, 0.55, 0.05)
        st.markdown(
            f'<div class="highlight-pill" style="margin-bottom:1rem;">'
            f'Active DA @ 0.55: 63.64%</div>',
            unsafe_allow_html=True
        )
        scan_all = st.button("🔍 Scan All", use_container_width=True)
        st.divider()
        single   = st.selectbox("Single Ticker", ALL_TICKERS)
        scan_one = st.button("🎯 Scan", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        to_scan = ALL_TICKERS if scan_all else [single] if scan_one else []

        if to_scan:
            results = []
            prog = st.progress(0, text="Scanning...")
            for i, t in enumerate(to_scan):
                r, _ = call_api('post', '/predict/signal', {"ticker": t, "threshold": threshold})
                if r: results.append(r)
                prog.progress((i+1)/len(to_scan), text=f"Scanning {t}...")
            prog.empty()

            if results:
                df    = pd.DataFrame(results)
                n_buy  = (df['signal']=='BUY').sum()
                n_sell = (df['signal']=='SELL').sum()
                n_hold = (df['signal']=='HOLD').sum()

                metric_cards([
                    ("BUY Signals",  str(n_buy),  "↑ Entry", "up"),
                    ("SELL Signals", str(n_sell), "↓ Exit",  "down"),
                    ("HOLD",         str(n_hold), "→ Wait",  "neu"),
                ])

                st.markdown('<div class="vna-card">', unsafe_allow_html=True)
                st.markdown(
                    '<div style="font-weight:600;margin-bottom:0.8rem;color:#ffffff;">'
                    'Signal Strength</div>', unsafe_allow_html=True
                )
                ds      = df.sort_values('conviction', ascending=False)
                buy_df  = ds[ds['signal']=='BUY']
                sell_df = ds[ds['signal']=='SELL']
                hold_df = ds[ds['signal']=='HOLD']
                fig = go.Figure()
                if len(buy_df):
                    fig.add_trace(go.Bar(x=buy_df['ticker'], y=buy_df['p_buy'],
                        name='BUY', marker_color='#27ae60',
                        text=buy_df['p_buy'].map(lambda x: f"{x:.0%}"),
                        textposition='outside', textfont=dict(color='#e0e0e0')))
                if len(sell_df):
                    fig.add_trace(go.Bar(x=sell_df['ticker'], y=-sell_df['p_sell'],
                        name='SELL', marker_color='#e74c3c',
                        text=sell_df['p_sell'].map(lambda x: f"{x:.0%}"),
                        textposition='outside', textfont=dict(color='#e0e0e0')))
                if len(hold_df):
                    fig.add_trace(go.Bar(x=hold_df['ticker'], y=hold_df['conviction'],
                        name='HOLD', marker_color='#f39c12', opacity=0.5))
                fig.add_hline(y= threshold, line_dash='dash', line_color='#27ae60',
                    line_width=1.5, annotation_text=f'BUY ≥{threshold}',
                    annotation_font_color='#27ae60')
                fig.add_hline(y=-threshold, line_dash='dash', line_color='#e74c3c',
                    line_width=1.5, annotation_text=f'SELL ≥{threshold}',
                    annotation_font_color='#e74c3c')
                fig.add_hline(y=0, line_color='#2a2a2a', line_width=1)
                fig.update_layout(**PLOT_LAYOUT, height=340, barmode='overlay',
                    yaxis_title='Signal Probability')
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="vna-card">', unsafe_allow_html=True)
                st.markdown(
                    '<div style="font-weight:600;margin-bottom:0.8rem;color:#ffffff;">'
                    'Signal Details</div>', unsafe_allow_html=True
                )
                disp   = ds[['ticker','signal','p_buy','p_sell','conviction','signal_date']].copy()
                styled = disp.style.map(
                    lambda v: (
                        'color:#27ae60;font-weight:600' if v=='BUY' else
                        'color:#e74c3c;font-weight:600' if v=='SELL' else
                        'color:#f39c12;font-weight:600'
                    ), subset=['signal']
                ).format({'p_buy':'{:.2%}','p_sell':'{:.2%}','conviction':'{:.2%}'})
                st.dataframe(styled, use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PAGE 4: PORTFOLIO
# ============================================================
elif page == "Portfolio":
    st.markdown(
        '<div class="page-title">💼 Portfolio</div>'
        '<div class="page-subtitle">Mean-Variance Optimization '
        '— 3 investor profiles</div>',
        unsafe_allow_html=True
    )

    profile_map = {
        '🚀 Risk-Taking (9 stocks)' : 'risk_taking',
        '🛡️ Prudent (3 stocks)'     : 'prudent',
        '⚖️ Equal-Weight (10 stocks)': 'equal_weight',
    }
    profile_label = st.radio("Profile", list(profile_map.keys()), horizontal=True)
    profile       = profile_map[profile_label]

    data, err = call_api('get', f'/portfolio/{profile}')
    if err:
        st.error(err)
    elif data:
        ret = data['expected_return']
        vol = data['expected_vol']
        sr  = data['sharpe_ratio']

        metric_cards([
            ("Expected Return", f"{ret:.1%}", "Annual",        "up"),
            ("Volatility",      f"{vol:.1%}", "Annual",        "down"),
            ("Sharpe Ratio",    f"{sr:.4f}",  "Risk-adjusted", "neu"),
            ("# Holdings", str(data['total_stocks']), "Positions", "neu"),
        ])

        stocks = pd.DataFrame(data['stocks'])
        cl, cr = st.columns(2)

        with cl:
            st.markdown('<div class="vna-card">', unsafe_allow_html=True)
            fig = px.pie(stocks, values='weight', names='ticker', title='Allocation',
                color_discrete_sequence=[
                    '#5ba8ff','#a8d8ea','#fffa93','#b8e6c1',
                    '#f9c784','#e8b4d8','#c4d4f0','#ffd3b6','#d4f0c4','#f0d4d4'
                ])
            fig.update_traces(textinfo='percent+label', textfont_color='#e0e0e0')
            fig.update_layout(**PLOT_LAYOUT, height=320, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with cr:
            st.markdown('<div class="vna-card">', unsafe_allow_html=True)
            sec  = stocks.groupby('sector')['weight'].sum().reset_index()
            fig2 = px.bar(sec, x='sector', y='weight', title='Sector Allocation',
                color='sector',
                color_discrete_sequence=['#5ba8ff','#fffa93','#27ae60','#e74c3c','#f39c12'])
            fig2.update_layout(**PLOT_LAYOUT, height=320, showlegend=False,
                yaxis_tickformat='.0%')
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="vna-card">', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-weight:600;margin-bottom:0.8rem;color:#ffffff;">'
            'Holdings</div>', unsafe_allow_html=True
        )
        styled = stocks.style.map(
            lambda v: (
                'color:#27ae60;font-weight:600' if v=='LOW' else
                'color:#f39c12;font-weight:600' if v=='MEDIUM' else
                'color:#e74c3c;font-weight:600' if v=='HIGH' else
                'color:#8e44ad;font-weight:600'
            ), subset=['risk_flag']
        ).format({'weight':'{:.2%}','risk_score':'{:.2f}'})
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PAGE 5: RISK ANALYSIS
# ============================================================
elif page == "Risk Analysis":
    st.markdown(
        '<div class="page-title">⚠️ Risk Analysis</div>'
        '<div class="page-subtitle">5-component scoring '
        '+ Sharpe stress test</div>',
        unsafe_allow_html=True
    )

    data, err = call_api('get', '/portfolio/scores/risk')
    if err:
        st.error(err)
    elif data:
        df    = pd.DataFrame(data['scores'])
        n_low  = (df['risk_flag']=='LOW').sum()
        n_med  = (df['risk_flag']=='MEDIUM').sum()
        n_high = (df['risk_flag']=='HIGH').sum()
        n_excl = (df['risk_flag']=='EXCLUDED').sum()

        metric_cards([
            ("🟢 LOW Risk",  str(n_low),  "Safe to hold",    "up"),
            ("🟡 MEDIUM",    str(n_med),  "Monitor closely", "neu"),
            ("🔴 HIGH Risk", str(n_high), "Reduce exposure", "down"),
            ("⛔ EXCLUDED",  str(n_excl), "Do not hold",     "down"),
        ])

        ds     = df.sort_values('composite_risk', ascending=True)
        cmap   = {'LOW':'#27ae60','MEDIUM':'#f39c12','HIGH':'#e74c3c','EXCLUDED':'#8e44ad'}
        colors = [cmap.get(f, 'gray') for f in ds['risk_flag']]

        st.markdown('<div class="vna-card">', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=ds['composite_risk'], y=ds['ticker'],
            orientation='h', marker_color=colors,
            text=ds['composite_risk'].round(2), textposition='outside',
            textfont=dict(color='#e0e0e0')
        ))
        fig.add_vline(x=5.0, line_dash='dash', line_color='#f39c12',
            annotation_text='Prudent threshold (5.0)',
            annotation_font_size=10, annotation_font_color='#f39c12')
        fig.add_vline(x=7.0, line_dash='dash', line_color='#e74c3c',
            annotation_text='Risk-Taking threshold (7.0)',
            annotation_font_size=10, annotation_font_color='#e74c3c')
        fig.update_layout(**PLOT_LAYOUT,
            title='Risk Score Ranking — 27 Tickers',
            xaxis_title='Composite Risk Score (0–10)', height=680)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="vna-card">', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-weight:600;margin-bottom:0.8rem;color:#ffffff;">'
            'Component Breakdown</div>', unsafe_allow_html=True
        )
        comps = ['volatility_risk','sell_risk','drawdown_risk','correlation_risk','reversal_risk']
        hmap  = df.set_index('ticker')[comps].sort_values('volatility_risk', ascending=False)
        fig2  = px.imshow(hmap, color_continuous_scale='RdYlGn_r',
            aspect='auto', labels=dict(color='Risk'), zmin=0, zmax=10)
        fig2.update_layout(**PLOT_LAYOUT, height=550)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="vna-card">', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-weight:600;margin-bottom:0.8rem;color:#ffffff;">'
            'Full Risk Scores</div>', unsafe_allow_html=True
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)