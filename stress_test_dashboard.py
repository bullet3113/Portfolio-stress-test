import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import datetime

# Set page configuration
st.set_page_config(
    page_title="Portfolio Stress Test",
    page_icon="📉",
    layout="wide"
)

# --- THEME COLORS (Grey, Blue, White) ---
THEME = {
    'normal': '#94a3b8',      # Slate Grey (Normal conditions)
    'stress': '#2563eb',      # Royal Blue (Stress conditions)
    'stress_dark': '#1e3a8a', # Dark Blue (Severe stress/Lines)
    'accent': '#60a5fa',      # Light Blue
    'bg': '#ffffff'           # White
}

# --- CONSTANTS & SCENARIOS ---
SCENARIOS = {
    'NORMAL': {
        'name': 'Normal Market Conditions',
        'vol_mult': 1.0,
        'corr_offset': 0.0,
        'drift': 0.05
    },
    'CRISIS_2008': {
        'name': '2008 Financial Crisis (Lehman)',
        'vol_mult': 3.5,
        'corr_offset': 0.4,
        'drift': -0.40
    },
    'COVID_19': {
        'name': 'COVID-19 Crash (Mar 2020)',
        'vol_mult': 5.0,
        'corr_offset': 0.2,
        'drift': -0.30
    },
    'TECH_BUBBLE': {
        'name': 'Tech Bubble Burst',
        'vol_mult': 2.5,
        'corr_offset': 0.1,
        'drift': -0.25
    },
    'RATE_SHOCK': {
        'name': 'Aggressive Rate Hike (+200bps)',
        'vol_mult': 1.5,
        'corr_offset': 0.15,
        'drift': -0.15
    }
}

# Default Portfolio
DEFAULT_DATA = [
    {"Ticker": "AAPL", "Weight (%)": 30},
    {"Ticker": "MSFT", "Weight (%)": 20},
    {"Ticker": "GOOGL", "Weight (%)": 20},
    {"Ticker": "AMZN", "Weight (%)": 15},
    {"Ticker": "TSLA", "Weight (%)": 15},
]

# --- HELPER FUNCTIONS ---

@st.cache_data(ttl=3600)
def fetch_market_data(tickers):
    """
    Fetches historical data from Yahoo Finance.
    """
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365*10) # 10 Years back
    
    try:
        # Download data
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)['Adj Close']
        
        # Handle single ticker case
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
            
        # Drop NaN to ensure we only use dates where ALL tickers have data
        data = data.dropna()
        
        if data.empty:
            return None, "No overlapping data found for these tickers."
            
        return data, None
    except Exception as e:
        return None, str(e)

def calculate_stats(historical_prices):
    """
    Calculates Annualized Volatility and Mean Return from history.
    """
    daily_returns = historical_prices.pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252)
    mean_return = daily_returns.mean() * 252
    corr_matrix = daily_returns.corr()
    
    n = len(corr_matrix)
    if n > 1:
        sum_corr = corr_matrix.sum().sum() - n
        avg_corr = sum_corr / (n * n - n)
    else:
        avg_corr = 1.0
        
    return daily_returns, volatility, mean_return, avg_corr

def run_simulation(weights, vols, current_avg_corr, scenario_params, iterations=5000, confidence=0.95):
    """
    Runs a Vectorized Monte Carlo Simulation.
    """
    stressed_vols = vols * scenario_params['vol_mult']
    
    if 'target_corr' in scenario_params:
        stressed_corr = scenario_params['target_corr']
    else:
        stressed_corr = np.clip(current_avg_corr + scenario_params['corr_offset'], -0.99, 0.99)
    
    # Market Factor
    market_drift_daily = scenario_params['drift'] / 252
    market_vol_daily = (0.15 * scenario_params['vol_mult']) / np.sqrt(252)
    
    z_market = np.random.normal(0, 1, iterations)
    market_returns = market_drift_daily + z_market * market_vol_daily
    
    # Idiosyncratic Factors
    z_idiosyncratic = np.random.normal(0, 1, (iterations, len(weights)))
    
    beta_weight = np.sqrt(abs(stressed_corr))
    if stressed_corr < 0: beta_weight = -beta_weight
    
    idio_weight = np.sqrt(1 - stressed_corr**2)
    
    stock_returns_daily = np.zeros((iterations, len(weights)))
    
    for i in range(len(weights)):
        stock_vol_daily = stressed_vols[i] / np.sqrt(252)
        term1 = market_returns * beta_weight
        term2 = (z_idiosyncratic[:, i] * stock_vol_daily) * idio_weight
        stock_returns_daily[:, i] = term1 + term2

    portfolio_returns = np.dot(stock_returns_daily, weights)
    
    sorted_returns = np.sort(portfolio_returns)
    index_cutoff = int((1 - confidence) * iterations)
    var_percent = sorted_returns[index_cutoff]
    
    tail_losses = sorted_returns[:index_cutoff]
    es_percent = tail_losses.mean() if len(tail_losses) > 0 else var_percent
    
    return portfolio_returns, var_percent, es_percent

# --- UI LAYOUT ---

def main():
    st.title("📉 Portfolio Stress Testing Dashboard")
    st.markdown("### Powered by Yahoo Finance Data")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        portfolio_val = st.number_input("Portfolio Value ($)", min_value=1000, value=1000000, step=10000)
        st.subheader("Risk Parameters")
        conf_level = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
        st.info(f"Simulates portfolio performance across {len(SCENARIOS)} different market stress scenarios simultaneously.")

    # Section 1: Inputs
    st.subheader("1. Asset Allocation")
    
    df_input = pd.DataFrame(DEFAULT_DATA)
    edited_df = st.data_editor(
        df_input, 
        num_rows="dynamic",
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker Symbol", required=True),
            "Weight (%)": st.column_config.NumberColumn("Weight (%)", min_value=0, max_value=100, format="%d %%")
        },
        use_container_width=True
    )
    
    total_weight = edited_df["Weight (%)"].sum()
    if total_weight != 100:
        st.warning(f"⚠️ Total weight is {total_weight}%. It should ideally sum to 100%.")
    
    # State Management
    if "hist_data" not in st.session_state:
        st.session_state.hist_data = None
        st.session_state.weights = None
        st.session_state.tickers = None

    if st.button("Fetch Data & Initialize", type="primary"):
        tickers = edited_df["Ticker"].astype(str).str.upper().tolist()
        weights = edited_df["Weight (%)"].values / 100.0
        
        if len(tickers) == 0:
            st.error("Please add at least one stock ticker.")
        else:
            with st.spinner('Fetching historical data from Yahoo Finance...'):
                hist_data, error = fetch_market_data(tickers)
            if error:
                st.error(f"Error fetching data: {error}")
            else:
                st.session_state.hist_data = hist_data
                st.session_state.weights = weights
                st.session_state.tickers = tickers
                st.session_state.run_static = True

    # Main Dashboard Logic
    if st.session_state.hist_data is not None:
        hist_data = st.session_state.hist_data
        weights = st.session_state.weights
        tickers = st.session_state.tickers
        
        # Calc Stats
        _, hist_vols, hist_means, hist_corr = calculate_stats(hist_data)
        
        st.success(f"Data Loaded: {len(hist_data)} trading days found. Portfolio Avg Correlation: {hist_corr:.2f}")

        # RESTORED: Stock Data Table
        st.subheader("1.1 Historical Risk Profile")
        stats_df = pd.DataFrame({
            "Ticker": tickers,
            "Allocated Weight": [f"{w*100:.0f}%" for w in weights],
            "Annualized Volatility": hist_vols.values,
            "Annualized Mean Return": hist_means.values
        })
        st.dataframe(
            stats_df.style.format({
                "Annualized Volatility": "{:.2%}",
                "Annualized Mean Return": "{:.2%}"
            }),
            use_container_width=True
        )

        # ADDED: Scenario Assumptions Table
        with st.expander("View Stress Scenario Assumptions"):
            st.markdown("These parameters dictate how the Monte Carlo engine modifies historical data for each stress test.")
            scenario_df = pd.DataFrame.from_dict(SCENARIOS, orient='index')
            scenario_df = scenario_df[['name', 'vol_mult', 'corr_offset', 'drift']]
            scenario_df.columns = ['Scenario Name', 'Volatility Multiplier', 'Correlation Bump', 'Annual Drift']
            st.table(scenario_df)
        
        # Section 2: Static Scenarios
        if st.session_state.get("run_static", False):
            st.markdown("---")
            st.subheader("2. Standard Stress Scenarios Results")
            
            results = []
            all_returns = {} 
            
            progress_bar = st.progress(0)
            for i, (key, scenario) in enumerate(SCENARIOS.items()):
                returns, var_pct, es_pct = run_simulation(
                    weights, hist_vols.values, hist_corr, scenario, confidence=conf_level
                )
                results.append({
                    "Scenario": scenario['name'],
                    "VaR ($)": abs(var_pct * portfolio_val),
                    "ES ($)": abs(es_pct * portfolio_val)
                })
                all_returns[scenario['name']] = returns * 100
                progress_bar.progress((i + 1) / len(SCENARIOS))
            progress_bar.empty()
            
            # Comparison Charts
            results_df = pd.DataFrame(results)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Potential Loss Magnitude**")
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(x=results_df["Scenario"], y=results_df["VaR ($)"], name='VaR', marker_color=THEME['stress']))
                fig_bar.add_trace(go.Bar(x=results_df["Scenario"], y=results_df["ES ($)"], name='ES', marker_color=THEME['stress_dark']))
                fig_bar.update_layout(barmode='group', template="plotly_white", legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig_bar, use_container_width=True)

            with col2:
                # RESTORED: Box Plot
                st.markdown("**Return Distribution Spread**")
                fig_box = go.Figure()
                for scenario_name, rets in all_returns.items():
                    color = THEME['normal'] if "Normal" in scenario_name else THEME['stress']
                    fig_box.add_trace(go.Box(
                        y=rets, 
                        name=scenario_name, 
                        marker_color=color,
                        boxpoints=False # Clean look
                    ))
                fig_box.update_layout(yaxis_title="Daily Return (%)", template="plotly_white", showlegend=False)
                st.plotly_chart(fig_box, use_container_width=True)

        # Section 3: Interactive Lab
        st.markdown("---")
        st.subheader("3. 🧪 Interactive Stress Laboratory")

        col_controls, col_viz = st.columns([1, 2])
        with col_controls:
            st.markdown("**Stress Controls**")
            target_corr = st.slider("Target Correlation", 0.0, 1.0, float(max(0.0, hist_corr)), 0.05)
            vol_mult = st.slider("Volatility Multiplier", 0.5, 5.0, 1.0, 0.1)
            rate_shock_bps = st.slider("Interest Rate Shock (bps)", 0, 500, 0, 25)
            
            drift_impact = -(rate_shock_bps / 100) * 0.05
            custom_drift = 0.05 + drift_impact
            st.caption(f"Resulting Annual Drift: {custom_drift*100:.2f}%")
            
            run_custom = st.button("Simulate Custom Scenario", type="secondary")

        with col_viz:
            if run_custom or st.session_state.get("run_static", False):
                custom_params = {'vol_mult': vol_mult, 'target_corr': target_corr, 'drift': custom_drift}
                
                cust_ret, cust_var, cust_es = run_simulation(
                    weights, hist_vols.values, hist_corr, custom_params, confidence=conf_level
                )
                norm_ret, norm_var, norm_es = run_simulation(
                    weights, hist_vols.values, hist_corr, SCENARIOS['NORMAL'], confidence=conf_level
                )

                fig = go.Figure()
                fig.add_trace(go.Histogram(x=norm_ret * 100, name='Normal Market', marker_color=THEME['normal'], opacity=0.7, nbinsx=50))
                fig.add_trace(go.Histogram(x=cust_ret * 100, name='Custom Stress', marker_color=THEME['stress'], opacity=0.6, nbinsx=50))
                fig.add_vline(x=cust_var*100, line_dash="dash", line_color=THEME['stress_dark'], annotation_text="Custom VaR")
                
                fig.update_layout(
                    title="Distribution Overlay: Normal vs Custom",
                    xaxis_title="Daily Return (%)", yaxis_title="Frequency",
                    barmode='overlay', template="plotly_white",
                    legend=dict(orientation="h", y=1.1)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Metrics
                c1, c2, c3 = st.columns(3)
                custom_var_val = abs(cust_var * portfolio_val)
                c1.metric("Custom VaR", f"${custom_var_val:,.0f}", delta_color="off")
                c2.metric("Custom ES", f"${abs(cust_es * portfolio_val):,.0f}", delta_color="off")
                c3.metric("Tail Ratio", f"{(cust_es/cust_var):.2f}")

if __name__ == "__main__":
    main()