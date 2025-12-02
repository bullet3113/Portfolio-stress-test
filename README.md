# **📉 Portfolio Stress Testing Dashboard**

A professional-grade financial analytics tool built with Python and Streamlit. This dashboard allows portfolio managers and individual investors to stress test equity portfolios against historical crises and hypothetical market shocks using Monte Carlo simulations.

## **🚀 Features**

### **1\. Real-Time Data Integration**

* **Yahoo Finance API:** Automatically fetches up to 10 years of historical adjusted close prices for user-defined tickers.  
* **Auto-Intersection:** Intelligently finds the common trading history across all assets in the portfolio.  
* **Statistical Analysis:** Calculates annualized volatility, mean returns, and the base correlation matrix of the portfolio.

### **2\. Standard Stress Scenarios**

Simulate portfolio performance against 5 pre-defined market conditions:

* **Normal Market:** Baseline projections based on historical data.  
* **2008 Financial Crisis:** Simulates a Lehman Brothers-style collapse (High Volatility, Correlation Convergence, Negative Drift).  
* **COVID-19 Crash:** Models extreme short-term volatility spikes.  
* **Tech Bubble:** Targets valuation compression.  
* **Rate Shock:** Simulates aggressive interest rate hikes (+200bps).

### **3\. Interactive Stress Laboratory 🧪**

A "Sandbox" mode allowing users to manually tweak risk drivers:

* **Correlation Control:** Force asset correlations to converge (testing diversification failure).  
* **Volatility Multiplier:** Scale the standard deviation of asset returns.  
* **Interest Rate Shock:** Apply basis point (bps) shocks to see the impact on portfolio drift.

### **4\. Advanced Risk Metrics**

* **Value at Risk (VaR):** The threshold loss at a specific confidence level (e.g., 95%).  
* **Expected Shortfall (ES):** Also known as Conditional VaR (CVaR); the average loss *beyond* the VaR threshold.  
* **Visualizations:**  
  * Distribution Histograms (Normal vs. Stressed).  
  * Comparative Bar Charts for Loss Magnitude.  
  * Box Plots for Volatility Spread.

## **🛠️ Installation & Setup**

### **Prerequisites**

Ensure you have Python 3.8+ installed.

### **1\. Clone the Repository**

### **2\. Install Dependencies**

### **3\. Run the Application**

The dashboard will open automatically in your default web browser at `http://localhost:8501`.

## **📖 Usage Guide**

1. **Define Portfolio:** In the "Asset Allocation" section, enter Stock Tickers (e.g., AAPL, MSFT) and their respective Weights. Ensure weights sum to 100%.  
2. **Fetch Data:** Click **"Fetch Data & Initialize"**. The app will download historical data and calculate the baseline risk profile.  
3. **Analyze Static Scenarios:** Scroll down to view the "Standard Stress Scenarios Results". Compare VaR and ES across different historical crises.  
4. **Interactive Lab:** Use the sliders in the "Interactive Stress Laboratory" to create custom "What-If" scenarios (e.g., "What if volatility doubles and rates rise by 100bps?").

## **🧮 Methodology**

### **Simulation Engine**

The application uses a **Vectorized Monte Carlo Simulation** (5,000 iterations per scenario).

**The Return Model:**

$$R\_{i,t} \= \\beta\_i \\cdot R\_{market} \+ \\sqrt{1 \- \\beta\_i^2} \\cdot \\epsilon\_i$$

Where:

* $R\_{market}$ is the systemic market factor (adjusted for scenario drift).  
* $\\epsilon\_i$ is the idiosyncratic (asset-specific) shock.  
* $\\beta\_i$ is derived from the target correlation parameter to simulate diversification breakdown during stress events.

### **Stress Assumptions**

| Scenario | Volatility Multiplier | Correlation Offset | Annual Drift |
| ----- | ----- | ----- | ----- |
| **Normal** | 1.0x | \+0.0 | \+5% |
| **2008 Crisis** | 3.5x | \+0.4 | \-40% |
| **COVID-19** | 5.0x | \+0.2 | \-30% |
| **Tech Bubble** | 2.5x | \+0.1 | \-25% |
| **Rate Shock** | 1.5x | \+0.15 | \-15% |

## **📦 Dependencies**

* `streamlit`: UI framework.  
* `yfinance`: Market data fetching.  
* `numpy`: Vectorized math and random number generation.  
* `pandas`: Data manipulation.  
* `plotly`: Interactive charting.

## **⚠️ Disclaimer**

This tool is for educational and informational purposes only. The simulations are based on historical approximations and mathematical models. They do not guarantee future performance. Always consult with a qualified financial advisor before making investment decisions.

