# ============================================================
# TASK 5.2 — Streamlit SaaS Dashboard
# Vietnam Stock Prediction & Portfolio Management
# Run: streamlit run src/dashboard/app.py
# ============================================================

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ── Config ────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title = "VN Stock AI Dashboard",
    page_icon  = "📈",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .signal-buy  { color: #27ae60; font-weight: bold; font-size: 1.5rem; }
    .signal-sell { color: #e74c3c; font-weight: bold; font-size: 1.5rem; }
    .signal-hold { color: #f39c12; font-weight: bold; font-size: 1.5rem; }
    .low-risk    { color: #27ae60; }
    .medium-risk { color: #f39c12; }
    .high-risk   { color: #e74c3c; }
</style>
""", unsafe_allow_html=True)


# ── API Helper ────────────────────────────────────────────────
def call_api(method, endpoint, payload=None):
    try:
        url = f"{API_BASE}{endpoint}"
        if method == 'get':
            r = requests.get(url, timeout=30)
        else:
            r = requests.post(url, json=payload, timeout=30)
        if r.status_code == 200:
            return r.json(), None
        return None, f"API Error {r.status_code}: {r.text}"
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Is the server running?"
    except Exception as e:
        return None, str(e)


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/"
             "thumb/2/2e/Flag_of_Vietnam.svg/320px-Flag_of_Vietnam.svg.png",
             width=80)
    st.title("VN Stock AI")
    st.caption("CS313 Deep Learning Project")
    st.divider()

    page = st.selectbox(
        "Navigate",
        [
            "🏠 Overview",
            "📈 Price Prediction",
            "🎯 Signal Scanner",
            "💼 Portfolio",
            "⚠️ Risk Analysis"
        ]
    )

    st.divider()

    # API Status
    health, err = call_api('get', '/health')
    if health:
        st.success("API Connected ✓")
        st.caption(f"Version: {health.get('version', 'N/A')}")
    else:
        st.error(f"API Offline ✗")
        st.caption(err)


# ============================================================
# PAGE 1: OVERVIEW
# ============================================================
if page == "🏠 Overview":
    st.markdown(
        '<div class="main-header">🇻🇳 Vietnam Stock AI Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    > *A deep learning-powered platform for Vietnamese stock market
    > prediction, trading signal identification, and portfolio management.*
    """)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Universe", "27 Tickers", "HOSE Blue Chips")
    with col2:
        st.metric("Models", "3 Active", "MTL + XGBoost + GRU")
    with col3:
        st.metric("Best Sharpe", "1.3969", "Equal-Weight Portfolio")
    with col4:
        st.metric("Selection Alpha", "+53.59%", "vs VNI Benchmark")

    st.divider()

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("📊 System Architecture")
        st.markdown("""
        | Component | Model | Task |
        |---|---|---|
        | Price Prediction | MTL Seq2Seq GRU | Task 2 |
        | Signal ID | XGBoost (43 features) | Task 3 |
        | Stock Selection | 5-Factor Scoring | Task 4.1 |
        | Risk Management | Sharpe + SELL signals | Task 4.2 |
        | Portfolio | Mean-Variance Opt. | Task 4.3 |
        """)

    with col_r:
        st.subheader("🏆 Portfolio Performance")
        perf_data = {
            'Profile'     : ['Risk-Taking', 'Prudent',
                             'Equal-Weight', 'VNI'],
            'Total Return': [35.73, 16.44, 53.93, 40.74],
            'Sharpe'      : [0.8596, 0.4550, 1.3969, 1.1694]
        }
        df_perf = pd.DataFrame(perf_data)
        fig = px.bar(
            df_perf, x='Profile', y='Total Return',
            color='Sharpe', color_continuous_scale='RdYlGn',
            title='Portfolio Total Returns (%)',
            text='Total Return'
        )
        fig.update_traces(texttemplate='%{text:.1f}%')
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 2: PRICE PREDICTION
# ============================================================
elif page == "📈 Price Prediction":
    st.title("📈 Price Prediction")
    st.caption("MTL Seq2Seq GRU + Attention — 5-day trajectory forecast")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Input")
        ticker = st.selectbox(
            "Select Ticker",
            ['FPT', 'VCB', 'VHM', 'VNM', 'HPG', 'VIC',
             'TCB', 'MSN', 'MWG', 'VND', 'HDB', 'GAS',
             'MBB', 'ACB', 'HPG', 'SAB', 'PNJ']
        )
        n_days = st.slider(
            "Historical window (days)", 20, 60, 20
        )
        predict_btn = st.button(
            "🔮 Predict", type="primary", use_container_width=True
        )

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
                # Direction badge
                direction = result['direction']
                confidence = result['confidence']

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Current Price",
                              f"{result['current_price']:,.0f} VND")
                with col_b:
                    delta_color = "normal" if direction == "UP" \
                                  else "inverse"
                    st.metric(
                        "Direction",
                        f"{'⬆' if direction=='UP' else '⬇' if direction=='DOWN' else '➡'} {direction}",
                        f"Confidence: {confidence:.1%}"
                    )
                with col_c:
                    target = result['predicted_prices'][-1] \
                             if result['predicted_prices'] else 0
                    change = (target - result['current_price']) / \
                             result['current_price'] * 100
                    st.metric(
                        "5-Day Target",
                        f"{target:,.0f} VND",
                        f"{change:+.2f}%"
                    )

                # Price trajectory chart
                days   = ['Today'] + [f"Day {i+1}" for i in range(5)]
                prices = [result['current_price']] + \
                          result['predicted_prices']

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=days, y=prices,
                    mode='lines+markers',
                    name='Predicted Price',
                    line=dict(
                        color='#27ae60' if direction=='UP'
                              else '#e74c3c',
                        width=2.5
                    ),
                    marker=dict(size=8)
                ))
                fig.add_hline(
                    y=result['current_price'],
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Current Price"
                )
                fig.update_layout(
                    title=f"{ticker} — 5-Day Price Forecast",
                    xaxis_title="Trading Day",
                    yaxis_title="Price (VND)",
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)

                # Returns table
                if result['predicted_returns']:
                    st.subheader("Daily Log Returns")
                    ret_df = pd.DataFrame({
                        'Day'       : [f"Day {i+1}" for i in range(
                            len(result['predicted_returns'])
                        )],
                        'Log Return': result['predicted_returns'],
                        'Price'     : result['predicted_prices']
                    })
                    st.dataframe(ret_df, use_container_width=True)


# ============================================================
# PAGE 3: SIGNAL SCANNER
# ============================================================
elif page == "🎯 Signal Scanner":
    st.title("🎯 Trading Signal Scanner")
    st.caption(
        "XGBoost Classifier — BUY/SELL/HOLD signals "
        "with conviction gating"
    )

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Settings")
        threshold = st.slider(
            "Conviction Threshold",
            min_value=0.40, max_value=0.80,
            value=0.55, step=0.05
        )
        scan_all = st.button(
            "🔍 Scan All Tickers",
            type="primary",
            use_container_width=True
        )

        st.divider()
        single_ticker = st.selectbox(
            "Or scan single ticker",
            ['FPT', 'VCB', 'VHM', 'VNM', 'HPG',
             'VIC', 'TCB', 'MSN', 'MWG', 'VND',
             'HDB', 'GAS', 'SAB', 'PNJ', 'MBB']
        )
        scan_one = st.button(
            "🎯 Scan",
            use_container_width=True
        )

    with col2:
        tickers_to_scan = []

        if scan_all:
            tickers_to_scan = [
                'FPT', 'VCB', 'VHM', 'VNM', 'HPG',
                'VIC', 'TCB', 'MSN', 'MWG', 'VND',
                'HDB', 'GAS', 'SAB', 'PNJ', 'MBB'
            ]
        elif scan_one:
            tickers_to_scan = [single_ticker]

        if tickers_to_scan:
            results = []
            progress = st.progress(0)
            for i, t in enumerate(tickers_to_scan):
                r, _ = call_api(
                    'post', '/predict/signal',
                    {"ticker": t, "threshold": threshold}
                )
                if r:
                    results.append(r)
                progress.progress((i + 1) / len(tickers_to_scan))
            progress.empty()

            if results:
                df_sig = pd.DataFrame(results)

                # Summary metrics
                n_buy  = (df_sig['signal'] == 'BUY').sum()
                n_sell = (df_sig['signal'] == 'SELL').sum()
                n_hold = (df_sig['signal'] == 'HOLD').sum()

                c1, c2, c3 = st.columns(3)
                c1.metric("🟢 BUY Signals",  n_buy)
                c2.metric("🔴 SELL Signals", n_sell)
                c3.metric("🟡 HOLD",         n_hold)

                # Signal table
                def color_signal(val):
                    if val == 'BUY':
                        return 'color: #27ae60; font-weight: bold'
                    elif val == 'SELL':
                        return 'color: #e74c3c; font-weight: bold'
                    return 'color: #f39c12'

                display_cols = [
                    'ticker', 'signal', 'p_buy',
                    'p_sell', 'conviction', 'signal_date'
                ]
                df_display = df_sig[display_cols].copy()
                df_display = df_display.sort_values(
                    'conviction', ascending=False
                )

                st.dataframe(
                    df_display.style.applymap(
                        color_signal, subset=['signal']
                    ).format({
                        'p_buy'     : '{:.2%}',
                        'p_sell'    : '{:.2%}',
                        'conviction': '{:.2%}'
                    }),
                    use_container_width=True
                )

                # Conviction chart
                fig = px.bar(
                    df_display.head(10),
                    x='ticker', y='conviction',
                    color='signal',
                    color_discrete_map={
                        'BUY' : '#27ae60',
                        'SELL': '#e74c3c',
                        'HOLD': '#f39c12'
                    },
                    title='Conviction Scores by Ticker',
                )
                fig.add_hline(
                    y=threshold, line_dash="dash",
                    line_color="red",
                    annotation_text=f"Threshold ({threshold})"
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 4: PORTFOLIO
# ============================================================
elif page == "💼 Portfolio":
    st.title("💼 Portfolio Composition")
    st.caption("Mean-Variance Optimization — 3 investor profiles")

    profile_choice = st.radio(
        "Select Profile",
        ["risk_taking", "prudent", "equal_weight"],
        format_func=lambda x: {
            'risk_taking' : '🚀 Risk-Taking (9 stocks)',
            'prudent'     : '🛡️ Prudent (3 stocks)',
            'equal_weight': '⚖️ Equal-Weight (10 stocks)'
        }[x],
        horizontal=True
    )

    data, err = call_api('get', f'/portfolio/{profile_choice}')

    if err:
        st.error(err)
    elif data:
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Expected Annual Return",
            f"{data['expected_return']:.1%}"
        )
        col2.metric(
            "Annual Volatility",
            f"{data['expected_vol']:.1%}"
        )
        col3.metric(
            "Sharpe Ratio",
            f"{data['sharpe_ratio']:.4f}"
        )

        stocks = pd.DataFrame(data['stocks'])

        col_l, col_r = st.columns(2)

        with col_l:
            # Pie chart
            fig = px.pie(
                stocks, values='weight',
                names='ticker',
                title='Portfolio Allocation',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textinfo='percent+label')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            # Sector breakdown
            sector_df = stocks.groupby('sector')[
                'weight'
            ].sum().reset_index()
            fig2 = px.bar(
                sector_df, x='sector', y='weight',
                title='Sector Allocation',
                color='sector',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig2.update_layout(
                height=350, showlegend=False,
                yaxis_tickformat='.0%'
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Holdings table
        st.subheader("Holdings")
        def color_risk(val):
            if val == 'LOW':    return 'color: #27ae60'
            if val == 'MEDIUM': return 'color: #f39c12'
            if val == 'HIGH':   return 'color: #e74c3c'
            return ''

        st.dataframe(
            stocks.style.applymap(
                color_risk, subset=['risk_flag']
            ).format({'weight': '{:.2%}', 'risk_score': '{:.2f}'}),
            use_container_width=True
        )


# ============================================================
# PAGE 5: RISK ANALYSIS
# ============================================================
elif page == "⚠️ Risk Analysis":
    st.title("⚠️ Risk Analysis")
    st.caption(
        "Five-component risk scoring + Sharpe stress test"
    )

    data, err = call_api('get', '/portfolio/scores/risk')

    if err:
        st.error(err)
    elif data:
        df_risk = pd.DataFrame(data['scores'])

        # Summary
        n_low      = (df_risk['risk_flag'] == 'LOW').sum()
        n_medium   = (df_risk['risk_flag'] == 'MEDIUM').sum()
        n_high     = (df_risk['risk_flag'] == 'HIGH').sum()
        n_excluded = (df_risk['risk_flag'] == 'EXCLUDED').sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🟢 LOW Risk",      n_low)
        c2.metric("🟡 MEDIUM Risk",   n_medium)
        c3.metric("🔴 HIGH Risk",     n_high)
        c4.metric("⛔ EXCLUDED",      n_excluded)

        # Risk bar chart
        df_sorted = df_risk.sort_values(
            'composite_risk', ascending=True
        )
        color_map = {
            'LOW'     : '#27ae60',
            'MEDIUM'  : '#f39c12',
            'HIGH'    : '#e74c3c',
            'EXCLUDED': '#8e44ad'
        }
        colors = [
            color_map.get(f, 'gray')
            for f in df_sorted['risk_flag']
        ]

        fig = go.Figure(go.Bar(
            x=df_sorted['composite_risk'],
            y=df_sorted['ticker'],
            orientation='h',
            marker_color=colors,
            text=df_sorted['composite_risk'].round(2),
            textposition='outside'
        ))
        fig.add_vline(x=5.0, line_dash="dash",
                      line_color="orange",
                      annotation_text="Prudent (5.0)")
        fig.add_vline(x=7.0, line_dash="dash",
                      line_color="red",
                      annotation_text="Risk-Taking (7.0)")
        fig.update_layout(
            title='Risk Score Ranking (27 Tickers)',
            xaxis_title='Composite Risk Score (0-10)',
            height=700
        )
        st.plotly_chart(fig, use_container_width=True)

        # Risk component heatmap
        st.subheader("Risk Component Breakdown")
        components = [
            'volatility_risk', 'sell_risk', 'drawdown_risk',
            'correlation_risk', 'reversal_risk'
        ]
        heatmap_df = df_risk.set_index('ticker')[
            components
        ].sort_values('volatility_risk', ascending=False)

        fig2 = px.imshow(
            heatmap_df,
            color_continuous_scale='RdYlGn_r',
            aspect='auto',
            title='Risk Components Heatmap',
            labels=dict(color='Risk Level'),
            zmin=0, zmax=10
        )
        fig2.update_layout(height=600)
        st.plotly_chart(fig2, use_container_width=True)

        # Full table
        st.subheader("Full Risk Scores")
        st.dataframe(df_risk, use_container_width=True)
