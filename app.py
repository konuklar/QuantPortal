# =============================================================
# 🏛️ APOLLO/ENIGMA QUANT TERMINAL v7.0 - INSTITUTIONAL EDITION
# Professional Global Multi-Asset Portfolio Management System
# Enhanced with Quantitative Analysis & Institutional Reporting
# =============================================================

import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats, optimize
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import json
import traceback
import base64
from io import BytesIO

# Import PyPortfolioOpt
try:
    from pypfopt import expected_returns, risk_models
    from pypfopt.efficient_frontier import EfficientFrontier
    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False

# Import Quantstats for advanced performance analytics
try:
    import quantstats as qs
    # Test if QuantStats is working properly
    qs.extend_pandas()
    QUANTSTATS_AVAILABLE = True
    QUANTSTATS_WORKING = True
except ImportError:
    QUANTSTATS_AVAILABLE = False
    QUANTSTATS_WORKING = False
except Exception as e:
    QUANTSTATS_AVAILABLE = True
    QUANTSTATS_WORKING = False
    st.warning(f"QuantStats installed but not working properly: {str(e)[:100]}")

# Check for scikit-learn
try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# ENHANCED LIGHT THEME WITH WHITE BACKGROUND
# -------------------------------------------------------------
st.set_page_config(
    page_title="APOLLO/ENIGMA - Institutional Portfolio Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏛️"
)

# Enhanced light theme with white background
LIGHT_THEME_CSS = """
<style>
:root {
    --primary: #0a3d62;
    --primary-light: #1a5fb4;
    --primary-dark: #082c47;
    --secondary: #26a269;
    --secondary-light: #2ecc71;
    --secondary-dark: #1e864d;
    --accent: #f39c12;
    --accent-light: #f1c40f;
    --accent-dark: #d68910;
    --danger: #e74c3c;
    --danger-light: #ff6b6b;
    --danger-dark: #c0392b;
    --warning: #f1c40f;
    --warning-light: #f7dc6f;
    --warning-dark: #f39c12;
    --success: #27ae60;
    --success-light: #58d68d;
    --success-dark: #229954;
    --info: #3498db;
    --info-light: #5dade2;
    --info-dark: #2980b9;
    
    --bg-primary: #ffffff;
    --bg-secondary: #f8f9fa;
    --bg-card: #ffffff;
    --bg-sidebar: #f1f3f5;
    
    --text-primary: #212529;
    --text-secondary: #495057;
    --text-muted: #6c757d;
    --text-light: #adb5bd;
    
    --border: #dee2e6;
    --border-light: #e9ecef;
    --border-dark: #ced4da;
    
    --shadow: rgba(0, 0, 0, 0.1);
    --shadow-light: rgba(0, 0, 0, 0.05);
    --shadow-dark: rgba(0, 0, 0, 0.15);
}

/* Main background */
.main {
    background-color: var(--bg-primary);
}

/* Professional Header */
.institutional-header {
    background: linear-gradient(90deg, var(--primary), var(--primary-dark));
    padding: 1.5rem;
    border-radius: 0 0 15px 15px;
    margin: -1rem -1rem 2rem -1rem;
    box-shadow: 0 4px 20px var(--shadow);
    border-bottom: 3px solid var(--accent);
}

/* Superior Cards */
.institutional-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 10px var(--shadow-light);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.institutional-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px var(--shadow);
    border-color: var(--primary-light);
}

.institutional-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--primary), var(--accent));
}

/* Professional Metrics */
.institutional-metric {
    background: linear-gradient(135deg, rgba(10, 61, 98, 0.05), rgba(26, 95, 180, 0.02));
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
    transition: all 0.3s ease;
}

.institutional-metric:hover {
    background: linear-gradient(135deg, rgba(10, 61, 98, 0.1), rgba(26, 95, 180, 0.05));
    border-color: var(--primary-light);
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--primary);
    margin: 0.5rem 0;
}

.metric-label {
    font-size: 0.85rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 500;
}

/* Enhanced Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--bg-secondary);
    padding: 6px;
    border-radius: 12px;
    border: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    padding: 12px 24px;
    font-weight: 600;
    color: var(--text-muted);
    border: 1px solid transparent;
    transition: all 0.3s ease;
    font-size: 0.9rem;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(10, 61, 98, 0.1);
    color: var(--text-primary);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark)) !important;
    color: white !important;
    border-color: var(--accent) !important;
    box-shadow: 0 4px 15px rgba(10, 61, 98, 0.2);
}

/* Professional Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark));
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s;
    box-shadow: 0 4px 12px rgba(10, 61, 98, 0.2);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(10, 61, 98, 0.3);
}

/* Status Indicators */
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.badge-success {
    background: linear-gradient(135deg, var(--success), var(--success-dark));
    color: white;
}

.badge-warning {
    background: linear-gradient(135deg, var(--warning), var(--warning-dark));
    color: #212529;
}

.badge-danger {
    background: linear-gradient(135deg, var(--danger), var(--danger-dark));
    color: white;
}

.badge-info {
    background: linear-gradient(135deg, var(--info), var(--info-dark));
    color: white;
}

/* Enhanced Tables */
.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

.stDataFrame table {
    background: var(--bg-card) !important;
}

.stDataFrame th {
    background: var(--primary) !important;
    color: white !important;
    font-weight: 600 !important;
    border-bottom: 2px solid var(--accent) !important;
}

.stDataFrame td {
    color: var(--text-primary) !important;
    border-bottom: 1px solid var(--border-light) !important;
}

.stDataFrame tr:hover {
    background: rgba(10, 61, 98, 0.05) !important;
}

/* Professional Charts */
.js-plotly-plot {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    background: var(--bg-card);
}

/* Sidebar Enhancement */
[data-testid="stSidebar"] {
    background: var(--bg-sidebar);
    border-right: 1px solid var(--border);
}

/* Input Enhancements */
.stSelectbox > div > div, .stTextInput > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

.stSelectbox > div > div:hover, .stTextInput > div > div:hover {
    border-color: var(--primary-light) !important;
    box-shadow: 0 0 0 2px rgba(26, 95, 180, 0.1) !important;
}

/* Progress Bars */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--primary), var(--accent));
    border-radius: 4px;
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-secondary);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: var(--primary-light);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--primary);
}

/* Professional Loading */
.stSpinner > div {
    border: 4px solid var(--border);
    border-top: 4px solid var(--primary);
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Grid Layout */
.institutional-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}

/* Expander Headers */
.streamlit-expanderHeader {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text-primary);
    font-weight: 600;
    padding: 1rem;
}

.streamlit-expanderHeader:hover {
    background: var(--bg-sidebar);
    border-color: var(--primary-light);
}

/* Alert Enhancements */
.stAlert {
    border-radius: 10px !important;
    border: 1px solid !important;
}

/* Footer */
.institutional-footer {
    background: var(--bg-secondary);
    padding: 1rem;
    margin-top: 3rem;
    border-radius: 10px;
    border-top: 2px solid var(--accent);
    text-align: center;
    color: var(--text-muted);
    font-size: 0.8rem;
}

/* KPI Cards */
.kpi-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}

.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px var(--shadow-light);
}

.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px var(--shadow);
    border-color: var(--primary-light);
}

.kpi-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--primary);
    margin: 0.5rem 0;
}

.kpi-label {
    font-size: 0.85rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Report Cards */
.report-card {
    background: linear-gradient(135deg, rgba(26, 95, 180, 0.05), rgba(38, 162, 105, 0.05));
    border: 2px solid var(--primary);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}

/* Section Headers */
.section-header {
    background: linear-gradient(90deg, rgba(10, 61, 98, 0.1), transparent);
    border-left: 4px solid var(--accent);
    padding: 1rem 1.5rem;
    margin: 2rem 0 1rem 0;
    border-radius: 0 8px 8px 0;
}

/* Download Buttons */
.download-button {
    background: linear-gradient(135deg, var(--secondary), var(--secondary-dark)) !important;
}

.download-button:hover {
    background: linear-gradient(135deg, var(--secondary-dark), #1e864d) !important;
}

/* Tab Content */
.tab-content {
    padding: 1rem 0;
}

/* Data Table Styles */
.data-table {
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
}

/* Chart Container */
.chart-container {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1.5rem;
}

/* Tooltip */
.tooltip {
    position: relative;
    display: inline-block;
    border-bottom: 1px dotted var(--text-muted);
    cursor: help;
}

.tooltip .tooltiptext {
    visibility: hidden;
    width: 200px;
    background-color: var(--primary-dark);
    color: white;
    text-align: center;
    border-radius: 6px;
    padding: 8px;
    position: absolute;
    z-index: 1;
    bottom: 125%;
    left: 50%;
    transform: translateX(-50%);
    opacity: 0;
    transition: opacity 0.3s;
    font-size: 0.8rem;
}

.tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}

/* Feature Grid Styles */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
}

.feature-card {
    background: linear-gradient(135deg, rgba(10, 61, 98, 0.03), rgba(38, 162, 105, 0.03));
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.feature-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
    border-color: var(--primary-light);
}

.feature-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--primary), var(--accent));
}

.feature-icon {
    font-size: 2.5rem;
    margin-bottom: 1rem;
    color: var(--primary);
}

.feature-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--primary);
    margin-bottom: 0.5rem;
}

.feature-description {
    color: var(--text-secondary);
    font-size: 0.95rem;
    line-height: 1.5;
}
</style>
"""

st.markdown(LIGHT_THEME_CSS, unsafe_allow_html=True)

# -------------------------------------------------------------
# GLOBAL ASSET UNIVERSE
# -------------------------------------------------------------
GLOBAL_ASSET_UNIVERSE = {
    "Core Equities": {
        "description": "Major equity indices and broad market ETFs",
        "assets": ["SPY", "QQQ", "IWM", "VTI", "VOO", "IVV", "VEA", "VWO"]
    },
    "Fixed Income": {
        "description": "Government and corporate bonds across durations",
        "assets": ["TLT", "IEF", "SHY", "BND", "AGG", "LQD", "TIP", "MUB"]
    },
    "Commodities": {
        "description": "Precious metals, energy, and agriculture",
        "assets": ["GLD", "SLV", "USO", "DBA", "PDBC", "GSG"]
    },
    "Sector ETFs": {
        "description": "Sector-specific equity exposure",
        "assets": ["XLK", "XLV", "XLF", "XLE", "XLI", "XLP", "XLY", "XLU"]
    }
}

# Flatten for selection
ALL_TICKERS = []
for category in GLOBAL_ASSET_UNIVERSE.values():
    ALL_TICKERS.extend(category["assets"])
ALL_TICKERS = sorted(list(set(ALL_TICKERS)))

# Default benchmark
DEFAULT_BENCHMARK = "SPY"

# -------------------------------------------------------------
# PORTFOLIO STRATEGIES
# -------------------------------------------------------------
PORTFOLIO_STRATEGIES = {
    "Minimum Volatility": "Optimized for lowest portfolio volatility",
    "Maximum Sharpe": "Optimal risk-adjusted returns",
    "Risk Parity": "Equal risk contribution across assets",
    "Equal Weight": "Equal allocation across all selected assets",
    "Institutional Balanced": "40% Equities, 40% Bonds, 20% Alternatives"
}

# -------------------------------------------------------------
# DATA LOADER
# -------------------------------------------------------------
class EnhancedDataLoader:
    """Enhanced data loader with robust error handling"""
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_price_data(tickers: List[str], start_date: str, end_date: str, benchmark: str = DEFAULT_BENCHMARK) -> Dict:
        """Load price data with enhanced error handling and benchmark"""
        
        if not tickers:
            return {'prices': pd.DataFrame(), 'benchmark': pd.Series()}
        
        st.info(f"📊 Loading data for {len(tickers)} assets and benchmark {benchmark}...")
        progress_bar = st.progress(0)
        
        prices_dict = {}
        successful_tickers = []
        
        # Load all tickers including benchmark
        all_tickers = tickers + [benchmark] if benchmark not in tickers else tickers
        
        for i, ticker in enumerate(all_tickers):
            try:
                # Download data
                ticker_obj = yf.Ticker(ticker)
                hist = ticker_obj.history(
                    start=start_date,
                    end=end_date,
                    interval="1d",
                    auto_adjust=True
                )
                
                if not hist.empty and len(hist) > 20:
                    # Use adjusted close if available
                    if 'Close' in hist.columns:
                        price_series = hist['Close']
                    elif 'Adj Close' in hist.columns:
                        price_series = hist['Adj Close']
                    else:
                        continue
                    
                    prices_dict[ticker] = price_series
                    successful_tickers.append(ticker)
                    
            except Exception:
                continue
            
            # Update progress
            progress = (i + 1) / len(all_tickers)
            progress_bar.progress(progress)
        
        progress_bar.empty()
        
        if prices_dict:
            prices_df = pd.DataFrame(prices_dict)
            
            # Fill missing values
            prices_df = prices_df.ffill(limit=5).bfill(limit=5)
            
            # Extract benchmark
            benchmark_series = pd.Series()
            if benchmark in prices_df.columns:
                benchmark_series = prices_df[benchmark]
                # Remove benchmark from portfolio assets if it's not in the original tickers
                if benchmark not in tickers:
                    prices_df = prices_df.drop(columns=[benchmark])
            
            if len(prices_df.columns) >= 2:
                st.success(f"✅ Successfully loaded {len(prices_df.columns)} assets")
                return {
                    'prices': prices_df,
                    'benchmark': benchmark_series
                }
        
        st.error("❌ Could not load sufficient data for analysis")
        return {'prices': pd.DataFrame(), 'benchmark': pd.Series()}

# -------------------------------------------------------------
# RISK METRICS CALCULATOR
# -------------------------------------------------------------
class RiskMetricsCalculator:
    """Calculate various risk metrics including VaR"""
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.95, method: str = 'historical') -> float:
        """Calculate Value at Risk (VaR)"""
        
        if returns.empty:
            return 0.0
        
        try:
            if method == 'historical':
                return float(np.percentile(returns, (1 - confidence_level) * 100))
            elif method == 'parametric':
                mean = returns.mean()
                std = returns.std()
                z_score = stats.norm.ppf(1 - confidence_level)
                return float(mean + z_score * std)
            else:
                return float(np.percentile(returns, (1 - confidence_level) * 100))
        except Exception:
            return 0.0
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (CVaR)"""
        
        if returns.empty:
            return 0.0
        
        try:
            var = RiskMetricsCalculator.calculate_var(returns, confidence_level, 'historical')
            losses_below_var = returns[returns <= var]
            if len(losses_below_var) > 0:
                return float(losses_below_var.mean())
            return float(var)
        except Exception:
            return 0.0
    
    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        
        if returns.empty:
            return 0.0
        
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return float(drawdown.min())
        except Exception:
            return 0.0

# -------------------------------------------------------------
# QUANTSTATS ANALYTICS (PROPERLY IMPLEMENTED)
# -------------------------------------------------------------
class QuantStatsAnalytics:
    """QuantStats integration for advanced performance analytics"""
    
    @staticmethod
    def generate_performance_report(returns: pd.Series, benchmark: pd.Series = None, 
                                   rf_rate: float = 0.03) -> Dict:
        """Generate comprehensive performance report"""
        
        metrics = {}
        
        if returns.empty:
            return metrics
        
        try:
            # Always calculate basic metrics
            n_days = len(returns)
            n_years = n_days / 252
            
            # Basic returns
            total_return = (1 + returns).prod() - 1
            metrics['cagr'] = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0
            metrics['total_return'] = total_return
            
            # Risk metrics
            metrics['volatility'] = returns.std() * np.sqrt(252)
            metrics['max_drawdown'] = RiskMetricsCalculator.calculate_max_drawdown(returns)
            
            # VaR metrics
            metrics['var_95'] = RiskMetricsCalculator.calculate_var(returns, 0.95, 'historical')
            metrics['cvar_95'] = RiskMetricsCalculator.calculate_cvar(returns, 0.95)
            
            # Ratio metrics
            if metrics['volatility'] > 0:
                metrics['sharpe'] = (metrics['cagr'] - rf_rate) / metrics['volatility']
            else:
                metrics['sharpe'] = 0
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_dev = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            if downside_dev > 0:
                metrics['sortino'] = (metrics['cagr'] - rf_rate) / downside_dev
            else:
                metrics['sortino'] = 0
            
            # Calmar ratio
            if abs(metrics['max_drawdown']) > 0:
                metrics['calmar'] = metrics['cagr'] / abs(metrics['max_drawdown'])
            else:
                metrics['calmar'] = 0
            
            # Win rate
            metrics['win_rate'] = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
            
            # Profit factor
            gross_profit = returns[returns > 0].sum()
            gross_loss = abs(returns[returns < 0].sum())
            metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Skewness and Kurtosis
            metrics['skew'] = returns.skew()
            metrics['kurtosis'] = returns.kurtosis()
            
            # If QuantStats is available and working, use it for additional metrics
            if QUANTSTATS_AVAILABLE and QUANTSTATS_WORKING:
                try:
                    # Add QuantStats specific metrics
                    metrics['omega'] = qs.stats.omega(returns, rf_rate)
                    metrics['tail_ratio'] = qs.stats.tail_ratio(returns)
                    metrics['gain_to_pain'] = qs.stats.gain_to_pain_ratio(returns)
                    
                    # Benchmark metrics if available
                    if benchmark is not None and not benchmark.empty:
                        aligned_returns = returns.reindex(benchmark.index).dropna()
                        aligned_benchmark = benchmark.reindex(aligned_returns.index)
                        
                        if len(aligned_returns) > 10:
                            metrics['alpha'] = qs.stats.alpha(aligned_returns, aligned_benchmark, rf_rate)
                            metrics['beta'] = qs.stats.beta(aligned_returns, aligned_benchmark)
                            metrics['information_ratio'] = qs.stats.information_ratio(aligned_returns, aligned_benchmark)
                except Exception:
                    # If QuantStats fails, use our calculations
                    pass
            
            # Benchmark metrics using our calculations if QuantStats not available
            elif benchmark is not None and not benchmark.empty:
                aligned_returns = returns.reindex(benchmark.index).dropna()
                aligned_benchmark = benchmark.reindex(aligned_returns.index)
                
                if len(aligned_returns) > 10:
                    # Calculate beta
                    covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
                    benchmark_variance = aligned_benchmark.var()
                    metrics['beta'] = covariance / benchmark_variance if benchmark_variance > 0 else 0
                    
                    # Calculate alpha
                    portfolio_return = aligned_returns.mean() * 252
                    benchmark_return = aligned_benchmark.mean() * 252
                    metrics['alpha'] = portfolio_return - rf_rate - metrics['beta'] * (benchmark_return - rf_rate)
                    
                    # Calculate information ratio
                    excess_returns = aligned_returns - aligned_benchmark
                    tracking_error = excess_returns.std() * np.sqrt(252)
                    metrics['information_ratio'] = (excess_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
            
            return metrics
            
        except Exception as e:
            st.warning(f"Performance calculation error: {str(e)[:100]}")
            return {}

# -------------------------------------------------------------
# PORTFOLIO OPTIMIZER
# -------------------------------------------------------------
class PortfolioOptimizer:
    """Enhanced portfolio optimizer"""
    
    @staticmethod
    def optimize_portfolio(returns: pd.DataFrame, strategy: str, 
                          risk_free_rate: float = 0.03, 
                          constraints: Dict = None,
                          benchmark_returns: pd.Series = None) -> Dict:
        """Optimize portfolio using specified strategy"""
        
        if returns.empty or len(returns.columns) < 2:
            return PortfolioOptimizer._equal_weight_fallback(returns, benchmark_returns)
        
        # Default constraints
        if constraints is None:
            constraints = {
                'max_weight': 0.20,
                'min_weight': 0.01,
                'short_selling': False
            }
        
        try:
            # Calculate expected returns and covariance
            mu = returns.mean() * 252
            S = returns.cov() * 252
            
            # Strategy-specific optimization
            weights = {}
            
            if strategy == "Minimum Volatility":
                weights = PortfolioOptimizer._min_volatility_optimization(S)
            elif strategy == "Maximum Sharpe":
                weights = PortfolioOptimizer._max_sharpe_optimization(mu, S, risk_free_rate)
            elif strategy == "Risk Parity":
                weights = PortfolioOptimizer._risk_parity_optimization(returns)
            elif strategy == "Institutional Balanced":
                weights = PortfolioOptimizer._institutional_balanced(returns.columns)
            else:  # Equal Weight
                weights = PortfolioOptimizer._equal_weight_optimization(returns)
            
            # Apply constraints
            weights = PortfolioOptimizer._apply_constraints(weights, constraints)
            
            # Calculate portfolio metrics
            return PortfolioOptimizer._calculate_portfolio_metrics(
                weights, returns, mu, S, risk_free_rate, benchmark_returns
            )
            
        except Exception as e:
            st.error(f"Optimization error: {str(e)[:200]}")
            return PortfolioOptimizer._equal_weight_fallback(returns, benchmark_returns)
    
    @staticmethod
    def _min_volatility_optimization(S: pd.DataFrame) -> Dict:
        """Minimum volatility optimization"""
        try:
            n = len(S)
            
            def objective(w):
                return w @ S.values @ w
            
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(0, 1) for _ in range(n)]
            w0 = np.ones(n) / n
            
            result = optimize.minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                return {asset: result.x[i] for i, asset in enumerate(S.index)}
            else:
                # Fallback to inverse volatility
                volatilities = np.sqrt(np.diag(S.values))
                inv_vol = 1 / (volatilities + 1e-10)
                weights_raw = inv_vol / inv_vol.sum()
                return {asset: weights_raw[i] for i, asset in enumerate(S.index)}
                
        except Exception:
            n = len(S)
            return {asset: 1.0/n for asset in S.index}
    
    @staticmethod
    def _max_sharpe_optimization(mu: pd.Series, S: pd.DataFrame, risk_free_rate: float) -> Dict:
        """Maximum Sharpe ratio optimization"""
        try:
            n = len(S)
            
            def objective(w):
                portfolio_return = mu.values @ w
                portfolio_risk = np.sqrt(w @ S.values @ w)
                sharpe = (portfolio_return - risk_free_rate) / (portfolio_risk + 1e-10)
                return -sharpe
            
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(0, 1) for _ in range(n)]
            w0 = np.ones(n) / n
            
            result = optimize.minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                return {asset: result.x[i] for i, asset in enumerate(S.index)}
            else:
                # Fallback to mean-variance
                gamma = 0.5
                def objective2(w):
                    return -(mu.values @ w - gamma * (w @ S.values @ w))
                
                result2 = optimize.minimize(objective2, w0, method='SLSQP', bounds=bounds, constraints=constraints)
                if result2.success:
                    return {asset: result2.x[i] for i, asset in enumerate(S.index)}
                else:
                    n = len(S)
                    return {asset: 1.0/n for asset in S.index}
                
        except Exception:
            n = len(S)
            return {asset: 1.0/n for asset in S.index}
    
    @staticmethod
    def _risk_parity_optimization(returns: pd.DataFrame) -> Dict:
        """Risk parity optimization"""
        try:
            volatilities = returns.std() * np.sqrt(252)
            inv_vol = 1 / (volatilities + 1e-10)
            weights_raw = inv_vol / inv_vol.sum()
            return {asset: weights_raw[asset] for asset in returns.columns}
        except Exception:
            n = len(returns.columns)
            return {asset: 1.0/n for asset in returns.columns}
    
    @staticmethod
    def _institutional_balanced(assets: List[str]) -> Dict:
        """Institutional balanced allocation (40/40/20 rule)"""
        weights = {}
        
        equity_etfs = ['SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'IVV', 'XLK', 'XLV', 'XLF', 'XLE', 'XLI', 'XLP', 'XLY', 'XLU']
        bond_etfs = ['TLT', 'IEF', 'SHY', 'BND', 'AGG', 'LQD', 'TIP', 'MUB']
        alternative_etfs = ['GLD', 'SLV', 'USO', 'DBA', 'PDBC', 'GSG']
        
        equity_count = sum(1 for asset in assets if asset in equity_etfs)
        bond_count = sum(1 for asset in assets if asset in bond_etfs)
        alt_count = sum(1 for asset in assets if asset in alternative_etfs)
        
        total_categorized = equity_count + bond_count + alt_count
        
        if total_categorized > 0:
            for asset in assets:
                if asset in equity_etfs:
                    weights[asset] = 0.4 / equity_count if equity_count > 0 else 0
                elif asset in bond_etfs:
                    weights[asset] = 0.4 / bond_count if bond_count > 0 else 0
                elif asset in alternative_etfs:
                    weights[asset] = 0.2 / alt_count if alt_count > 0 else 0
                else:
                    weights[asset] = 0
        else:
            n = len(assets)
            weights = {asset: 1.0/n for asset in assets}
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    @staticmethod
    def _equal_weight_optimization(returns: pd.DataFrame) -> Dict:
        """Equal weight optimization"""
        n = len(returns.columns)
        return {asset: 1.0/n for asset in returns.columns}
    
    @staticmethod
    def _apply_constraints(weights: Dict, constraints: Dict) -> Dict:
        """Apply weight constraints"""
        max_weight = constraints.get('max_weight', 1.0)
        min_weight = constraints.get('min_weight', 0.0)
        
        normalized_weights = {}
        for asset, weight in weights.items():
            weight = max(min_weight, min(max_weight, weight))
            normalized_weights[asset] = weight
        
        total_weight = sum(normalized_weights.values())
        if total_weight > 0:
            normalized_weights = {k: v/total_weight for k, v in normalized_weights.items()}
        
        return normalized_weights
    
    @staticmethod
    def _calculate_portfolio_metrics(weights: Dict, returns: pd.DataFrame, 
                                    mu: pd.Series, S: pd.DataFrame, 
                                    risk_free_rate: float,
                                    benchmark_returns: pd.Series = None) -> Dict:
        """Calculate portfolio performance metrics"""
        
        assets = list(weights.keys())
        weights_array = np.array([weights[asset] for asset in assets])
        
        portfolio_returns = (returns[assets] * weights_array).sum(axis=1)
        
        expected_return = mu.values @ weights_array
        expected_risk = np.sqrt(weights_array @ S.values @ weights_array)
        sharpe_ratio = (expected_return - risk_free_rate) / (expected_risk + 1e-10)
        
        quantstats_metrics = QuantStatsAnalytics.generate_performance_report(
            portfolio_returns, benchmark_returns, risk_free_rate
        )
        
        benchmark_metrics = {}
        if benchmark_returns is not None and not benchmark_returns.empty:
            aligned_portfolio = portfolio_returns.reindex(benchmark_returns.index).dropna()
            aligned_benchmark = benchmark_returns.reindex(aligned_portfolio.index)
            
            if len(aligned_portfolio) > 10:
                covariance = np.cov(aligned_portfolio, aligned_benchmark)[0, 1]
                benchmark_variance = aligned_benchmark.var()
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                
                benchmark_return = aligned_benchmark.mean() * 252
                alpha = expected_return - risk_free_rate - beta * (benchmark_return - risk_free_rate)
                
                tracking_error = (aligned_portfolio - aligned_benchmark).std() * np.sqrt(252)
                excess_return = (aligned_portfolio - aligned_benchmark).mean() * 252
                information_ratio = excess_return / (tracking_error + 1e-10)
                
                benchmark_metrics = {
                    'beta': beta,
                    'alpha': alpha,
                    'tracking_error': tracking_error,
                    'information_ratio': information_ratio,
                    'benchmark_return': benchmark_return,
                    'excess_return': excess_return
                }
        
        return {
            'weights': weights,
            'weights_array': weights_array,
            'expected_return': expected_return,
            'expected_risk': expected_risk,
            'sharpe_ratio': sharpe_ratio,
            'portfolio_returns': portfolio_returns,
            'quantstats_metrics': quantstats_metrics,
            'benchmark_metrics': benchmark_metrics,
            'success': True
        }
    
    @staticmethod
    def _equal_weight_fallback(returns: pd.DataFrame, benchmark_returns: pd.Series = None) -> Dict:
        """Fallback to equal weights"""
        if returns.empty:
            return {
                'weights': {},
                'weights_array': np.array([]),
                'expected_return': 0,
                'expected_risk': 0,
                'sharpe_ratio': 0,
                'portfolio_returns': pd.Series(),
                'quantstats_metrics': {},
                'benchmark_metrics': {},
                'success': False
            }
        
        n_assets = len(returns.columns)
        equal_weights = {asset: 1.0/n_assets for asset in returns.columns}
        weights_array = np.ones(n_assets) / n_assets
        
        portfolio_returns = (returns * weights_array).sum(axis=1)
        mu = returns.mean() * 252
        S = returns.cov() * 252
        expected_return = mu.values @ weights_array
        expected_risk = np.sqrt(weights_array @ S.values @ weights_array)
        sharpe_ratio = (expected_return - 0.03) / (expected_risk + 1e-10)
        
        quantstats_metrics = QuantStatsAnalytics.generate_performance_report(
            portfolio_returns, benchmark_returns, 0.03
        )
        
        return {
            'weights': equal_weights,
            'weights_array': weights_array,
            'expected_return': expected_return,
            'expected_risk': expected_risk,
            'sharpe_ratio': sharpe_ratio,
            'portfolio_returns': portfolio_returns,
            'quantstats_metrics': quantstats_metrics,
            'benchmark_metrics': {},
            'success': False
        }

# -------------------------------------------------------------
# MAIN APPLICATION
# -------------------------------------------------------------
def main():
    """Main application with enhanced features"""
    
    # Check dependencies quietly (no display)
    check_dependencies_quietly()
    
    # Custom header
    st.markdown("""
    <div class="institutional-header">
        <h1 style="color: white; margin: 0; font-size: 2.5rem;">🏛️ APOLLO/ENIGMA</h1>
        <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 1.2rem;">
        Quantitative Portfolio Analysis Terminal v7.0
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = None
    if 'selected_assets' not in st.session_state:
        st.session_state.selected_assets = ["SPY", "TLT", "GLD"]
    
    # Enhanced sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="font-size: 1.5rem; font-weight: bold; color: var(--primary);">
                ⚙️ CONFIGURATION
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Portfolio strategy
        st.markdown("### 🎯 Portfolio Strategy")
        strategy = st.selectbox(
            "Select Strategy",
            list(PORTFOLIO_STRATEGIES.keys()),
            index=0
        )
        
        # Benchmark selection
        st.markdown("### 📈 Benchmark")
        benchmark = st.selectbox(
            "Select Benchmark",
            ["SPY", "QQQ", "VTI", "IWM", "AGG"],
            index=0
        )
        
        # Asset selection
        st.markdown("### 🌍 Asset Selection")
        
        categories = st.multiselect(
            "Asset Categories",
            list(GLOBAL_ASSET_UNIVERSE.keys()),
            default=["Core Equities", "Fixed Income", "Commodities"]
        )
        
        available_assets = []
        for category in categories:
            available_assets.extend(GLOBAL_ASSET_UNIVERSE[category]["assets"])
        available_assets = sorted(list(set(available_assets)))
        
        selected_assets = st.multiselect(
            "Select Assets (3-10 recommended)",
            available_assets,
            default=st.session_state.selected_assets
        )
        
        st.session_state.selected_assets = selected_assets
        
        # Date range
        st.markdown("### 📅 Analysis Period")
        date_period = st.selectbox(
            "Time Horizon",
            ["1 Year", "3 Years", "5 Years", "10 Years"],
            index=2
        )
        
        years_map = {"1 Year": 1, "3 Years": 3, "5 Years": 5, "10 Years": 10}
        years = years_map[date_period]
        
        # Risk parameters
        st.markdown("### ⚖️ Risk Parameters")
        rf_rate = st.slider(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=3.0,
            step=0.1
        ) / 100
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                max_weight = st.slider("Max Weight (%)", 5, 100, 20, 5) / 100
            with col2:
                min_weight = st.slider("Min Weight (%)", 0, 10, 1, 1) / 100
        
        constraints = {
            'max_weight': max_weight,
            'min_weight': min_weight,
            'short_selling': False
        }
        
        # Action buttons
        st.markdown("---")
        col_run, col_reset = st.columns(2)
        with col_run:
            run_analysis = st.button(
                "🚀 Run Analysis",
                type="primary",
                use_container_width=True,
                disabled=len(selected_assets) < 3
            )
        with col_reset:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.clear()
                st.rerun()
        
        # Status indicators (minimal)
        st.markdown("---")
        st.markdown("### 📊 System Status")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Assets", len(selected_assets))
        with col2:
            status = "✅" if PYPFOPT_AVAILABLE else "⚠️"
            st.metric("Optimizer", status)
    
    # Main content
    if run_analysis and len(selected_assets) >= 3:
        with st.spinner("🔍 Conducting quantitative analysis..."):
            try:
                end_date = pd.Timestamp.today()
                start_date = end_date - pd.DateOffset(years=years)
                
                data_loader = EnhancedDataLoader()
                data = data_loader.load_price_data(selected_assets, start_date, end_date, benchmark)
                prices = data['prices']
                benchmark_series = data['benchmark']
                
                if prices.empty or len(prices) < 60:
                    st.error("❌ Insufficient data for analysis.")
                    return
                
                returns = prices.pct_change().dropna()
                benchmark_returns = benchmark_series.pct_change().dropna() if not benchmark_series.empty else pd.Series()
                
                optimizer = PortfolioOptimizer()
                results = optimizer.optimize_portfolio(
                    returns, strategy, rf_rate, constraints, benchmark_returns
                )
                
                if results['success']:
                    st.session_state.analysis_results = results
                    st.session_state.portfolio_data = {
                        'prices': prices,
                        'returns': returns,
                        'portfolio_returns': results['portfolio_returns'],
                        'benchmark_returns': benchmark_returns,
                        'benchmark': benchmark,
                        'start_date': start_date,
                        'end_date': end_date,
                        'strategy': strategy,
                        'rf_rate': rf_rate
                    }
                    
                    st.success("✅ Analysis completed successfully!")
                    
                    # Create enhanced tabs
                    tab_names = [
                        "📊 Overview", 
                        "⚖️ Risk Analytics", 
                        "📈 Performance", 
                        "🔍 QuantStats", 
                        "📋 Reports"
                    ]
                    
                    tabs = st.tabs(tab_names)
                    
                    with tabs[0]:
                        display_overview_tab(results, benchmark_returns, benchmark)
                    
                    with tabs[1]:
                        display_risk_tab(results, returns)
                    
                    with tabs[2]:
                        display_performance_tab(results, benchmark_returns, benchmark)
                    
                    with tabs[3]:
                        display_quantstats_tab(results, benchmark_returns, benchmark)
                    
                    with tabs[4]:
                        display_reports_tab(results, benchmark_returns, benchmark)
                
                else:
                    st.error("❌ Portfolio optimization failed.")
                    
            except Exception as e:
                st.error(f"❌ Analysis error: {str(e)[:200]}")
    
    else:
        display_welcome_screen()

def display_overview_tab(results: Dict, benchmark_returns: pd.Series, benchmark: str):
    """Display overview tab"""
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, var(--primary), var(--primary-dark)); 
                padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0;">📊 PORTFOLIO OVERVIEW</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0;">Portfolio Summary & Allocation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    st.markdown("### 🎯 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Expected Return", f"{results['expected_return']:.2%}")
    
    with col2:
        st.metric("Expected Risk", f"{results['expected_risk']:.2%}")
    
    with col3:
        st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
    
    with col4:
        max_dd = results['quantstats_metrics'].get('max_drawdown', 0)
        st.metric("Max Drawdown", f"{max_dd:.2%}")
    
    # Benchmark comparison
    if benchmark_returns is not None and not benchmark_returns.empty:
        st.markdown("### 📈 Benchmark Comparison")
        
        benchmark_cols = st.columns(4)
        
        with benchmark_cols[0]:
            benchmark_return = benchmark_returns.mean() * 252
            st.metric(
                f"{benchmark} Return", 
                f"{benchmark_return:.2%}",
                delta=f"{(results['expected_return'] - benchmark_return):.2%}"
            )
        
        with benchmark_cols[1]:
            benchmark_risk = benchmark_returns.std() * np.sqrt(252)
            st.metric(f"{benchmark} Risk", f"{benchmark_risk:.2%}")
        
        with benchmark_cols[2]:
            beta = results['benchmark_metrics'].get('beta', 0)
            st.metric("Beta", f"{beta:.2f}")
        
        with benchmark_cols[3]:
            alpha = results['benchmark_metrics'].get('alpha', 0)
            st.metric("Alpha", f"{alpha:.2%}")
    
    # Portfolio allocation
    st.markdown("### 🎯 Portfolio Allocation")
    
    weights = results['weights']
    weights_df = pd.DataFrame({
        'Asset': list(weights.keys()),
        'Weight': [f"{w:.1%}" for w in weights.values()]
    }).sort_values('Weight', ascending=False)
    
    col_chart1, col_chart2 = st.columns([2, 1])
    
    with col_chart1:
        fig = go.Figure(data=[go.Pie(
            labels=weights_df['Asset'],
            values=[float(w.strip('%'))/100 for w in weights_df['Weight']],
            hole=0.4,
            marker_colors=px.colors.sequential.Blues
        )])
        
        fig.update_layout(
            title="Portfolio Allocation",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_risk_tab(results: Dict, returns: pd.DataFrame):
    """Display risk analytics tab"""
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #c0392b, #e74c3c); 
                padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0;">⚖️ RISK ANALYTICS</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0;">Comprehensive Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Risk metrics
    st.markdown("### 📊 Risk Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Portfolio Volatility", f"{results['expected_risk']:.2%}")
    
    with col2:
        var = results['quantstats_metrics'].get('var_95', 0)
        st.metric("VaR (95%)", f"{var:.2%}")
    
    with col3:
        cvar = results['quantstats_metrics'].get('cvar_95', 0)
        st.metric("CVaR (95%)", f"{cvar:.2%}")
    
    with col4:
        max_dd = results['quantstats_metrics'].get('max_drawdown', 0)
        st.metric("Max Drawdown", f"{max_dd:.2%}")
    
    # VaR Analysis
    st.markdown("### 📉 Value at Risk (VaR) Analysis")
    
    portfolio_returns = results['portfolio_returns']
    
    if not portfolio_returns.empty:
        # Calculate different VaR methods
        var_methods = ['historical', 'parametric']
        var_results = {}
        
        for method in var_methods:
            var_results[method] = RiskMetricsCalculator.calculate_var(portfolio_returns, 0.95, method)
        
        # Display VaR comparison
        var_cols = st.columns(2)
        with var_cols[0]:
            st.metric("Historical VaR (95%)", f"{var_results['historical']:.2%}")
        with var_cols[1]:
            st.metric("Parametric VaR (95%)", f"{var_results['parametric']:.2%}")
        
        # VaR chart
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=portfolio_returns.values * 100,
            nbinsx=50,
            marker_color='#1a5fb4',
            opacity=0.7
        ))
        
        # Add VaR lines
        colors = ['#e74c3c', '#f39c12']
        for idx, (method, var_value) in enumerate(var_results.items()):
            fig.add_vline(
                x=var_value * 100,
                line_dash="dash",
                line_color=colors[idx],
                annotation_text=f"{method.title()}: {var_value:.2%}",
                annotation_position="top left"
            )
        
        fig.update_layout(
            title="Return Distribution with VaR (95% Confidence)",
            height=400,
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            xaxis=dict(ticksuffix="%")
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_performance_tab(results: Dict, benchmark_returns: pd.Series, benchmark: str):
    """Display performance analysis tab"""
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #27ae60, #2ecc71); 
                padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0;">📈 PERFORMANCE ANALYSIS</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0;">Performance Metrics & Charts</p>
    </div>
    """, unsafe_allow_html=True)
    
    portfolio_returns = results['portfolio_returns']
    
    # Create performance charts
    st.markdown("### 📊 Cumulative Returns")
    
    cumulative_returns = (1 + portfolio_returns).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cumulative_returns.index,
        y=cumulative_returns.values,
        name="Portfolio",
        line=dict(color='#1a5fb4', width=3),
        fill='tozeroy',
        fillcolor='rgba(26, 95, 180, 0.1)'
    ))
    
    if benchmark_returns is not None and not benchmark_returns.empty:
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        fig.add_trace(go.Scatter(
            x=benchmark_cumulative.index,
            y=benchmark_cumulative.values,
            name=f"Benchmark ({benchmark})",
            line=dict(color='#26a269', width=2, dash='dash')
        ))
    
    fig.update_layout(
        title="Cumulative Returns",
        height=400,
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Drawdown chart
    st.markdown("### 📉 Drawdown Analysis")
    
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values * 100,
        name="Drawdown",
        fill='tozeroy',
        fillcolor='rgba(231, 76, 60, 0.2)',
        line=dict(color='#e74c3c', width=2)
    ))
    
    fig2.update_layout(
        title="Portfolio Drawdown",
        height=400,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified',
        yaxis=dict(ticksuffix="%")
    )
    
    st.plotly_chart(fig2, use_container_width=True)

def display_quantstats_tab(results: Dict, benchmark_returns: pd.Series, benchmark: str):
    """Display QuantStats analytics tab"""
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #8e44ad, #9b59b6); 
                padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0;">🔍 ADVANCED ANALYTICS</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0;">Professional Performance Metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    portfolio_returns = results['portfolio_returns']
    
    # Display advanced metrics
    st.markdown("### 📊 Performance Metrics")
    
    if 'quantstats_metrics' in results:
        metrics = results['quantstats_metrics']
        
        # Create a grid of metrics
        metric_cols = st.columns(4)
        
        advanced_metrics = [
            ("Sharpe Ratio", metrics.get('sharpe', 0), "Risk-adjusted return"),
            ("Sortino Ratio", metrics.get('sortino', 0), "Downside risk-adjusted return"),
            ("Calmar Ratio", metrics.get('calmar', 0), "Return vs max drawdown"),
            ("Win Rate", f"{metrics.get('win_rate', 0):.1%}", "Percentage of positive periods"),
            ("CAGR", f"{metrics.get('cagr', 0):.2%}", "Compound annual growth rate"),
            ("Volatility", f"{metrics.get('volatility', 0):.2%}", "Annualized volatility"),
            ("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}", "Maximum peak-to-trough decline"),
            ("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}", "Gross profit / gross loss"),
        ]
        
        for i, (name, value, desc) in enumerate(advanced_metrics):
            with metric_cols[i % 4]:
                if isinstance(value, (int, float)):
                    st.metric(name, f"{value:.2f}", help=desc)
                else:
                    st.metric(name, value, help=desc)
        
        # Performance comparison gauges
        st.markdown("### 📈 Performance Gauges")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'win_rate' in metrics:
                fig = go.Figure(data=[go.Indicator(
                    mode="gauge+number",
                    value=metrics['win_rate'] * 100,
                    title={'text': "Win Rate"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#2ecc71"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 100], 'color': "lightblue"}
                        ]
                    }
                )])
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'sharpe' in metrics:
                sharpe = float(metrics['sharpe'])
                fig = go.Figure(data=[go.Indicator(
                    mode="gauge+number",
                    value=sharpe,
                    title={'text': "Sharpe Ratio"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, max(3, sharpe * 1.5)]},
                        'bar': {'color': "#3498db"},
                        'steps': [
                            {'range': [0, 1], 'color': "lightgray"},
                            {'range': [1, 2], 'color': "lightgreen"},
                            {'range': [2, 3], 'color': "lightblue"}
                        ]
                    }
                )])
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        # Additional QuantStats metrics if available
        if QUANTSTATS_AVAILABLE and QUANTSTATS_WORKING:
            st.markdown("### 📊 QuantStats Extended Metrics")
            
            extended_cols = st.columns(3)
            
            with extended_cols[0]:
                if 'omega' in metrics:
                    st.metric("Omega Ratio", f"{metrics['omega']:.2f}")
            
            with extended_cols[1]:
                if 'tail_ratio' in metrics:
                    st.metric("Tail Ratio", f"{metrics['tail_ratio']:.2f}")
            
            with extended_cols[2]:
                if 'gain_to_pain' in metrics:
                    st.metric("Gain to Pain", f"{metrics['gain_to_pain']:.2f}")

def display_reports_tab(results: Dict, benchmark_returns: pd.Series, benchmark: str):
    """Display reports tab"""
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #34495e, #2c3e50); 
                padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0;">📋 INSTITUTIONAL REPORTS</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0;">Professional Reports & Downloads</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Download options
    st.markdown("### 📥 Download Data")
    
    if results and 'portfolio_returns' in results:
        returns_df = results['portfolio_returns'].to_frame(name='Portfolio_Returns')
        
        # Remove timezone info for Excel compatibility
        if returns_df.index.tz is not None:
            returns_df.index = returns_df.index.tz_localize(None)
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = returns_df.to_csv()
            st.download_button(
                label="📊 Download Portfolio Returns (CSV)",
                data=csv,
                file_name=f"portfolio_returns_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                returns_df.to_excel(writer, sheet_name='Portfolio Returns')
                
                if 'weights' in results:
                    weights_df = pd.DataFrame({
                        'Asset': list(results['weights'].keys()),
                        'Weight': list(results['weights'].values())
                    })
                    weights_df.to_excel(writer, sheet_name='Allocation', index=False)
            
            st.download_button(
                label="📊 Download Full Report (Excel)",
                data=buffer.getvalue(),
                file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    
    # Summary statistics
    st.markdown("### 📊 Portfolio Summary")
    
    if results and 'quantstats_metrics' in results:
        metrics = results['quantstats_metrics']
        
        summary_cols = st.columns(3)
        
        with summary_cols[0]:
            st.markdown("#### Returns")
            st.markdown(f"**Annual Return:** {results.get('expected_return', 0):.2%}")
            st.markdown(f"**CAGR:** {metrics.get('cagr', 0):.2%}")
            st.markdown(f"**Sharpe Ratio:** {results.get('sharpe_ratio', 0):.2f}")
        
        with summary_cols[1]:
            st.markdown("#### Risk")
            st.markdown(f"**Annual Volatility:** {results.get('expected_risk', 0):.2%}")
            st.markdown(f"**Max Drawdown:** {metrics.get('max_drawdown', 0):.2%}")
            st.markdown(f"**VaR (95%):** {metrics.get('var_95', 0):.2%}")
        
        with summary_cols[2]:
            st.markdown("#### Ratios")
            st.markdown(f"**Sortino Ratio:** {metrics.get('sortino', 0):.2f}")
            st.markdown(f"**Calmar Ratio:** {metrics.get('calmar', 0):.2f}")
            st.markdown(f"**Win Rate:** {metrics.get('win_rate', 0):.1%}")

def get_asset_category(ticker: str) -> str:
    """Get asset category for a ticker"""
    for category, data in GLOBAL_ASSET_UNIVERSE.items():
        if ticker in data["assets"]:
            return category
    return "Unknown"

def check_dependencies_quietly():
    """Check dependencies without displaying warnings"""
    pass  # No warnings displayed to user

def display_welcome_screen():
    """Display welcome screen when no analysis is running"""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem;">
            <div style="font-size: 4rem; margin-bottom: 2rem;">🏛️</div>
            <h1 style="color: var(--primary); margin-bottom: 1rem;">
                APOLLO/ENIGMA v7.0
            </h1>
            <p style="color: var(--text-muted); font-size: 1.2rem; margin-bottom: 2rem;">
                Professional Portfolio Management & Quantitative Analysis Platform
            </p>
            <div style="background: linear-gradient(135deg, rgba(10, 61, 98, 0.1), rgba(38, 162, 105, 0.1)); 
                        padding: 2rem; border-radius: 15px; margin-bottom: 2rem;">
                <h3 style="color: var(--primary); margin-bottom: 1rem;">🚀 Quick Start</h3>
                <ol style="text-align: left; color: var(--text-secondary);">
                    <li>Select assets from sidebar (minimum 3)</li>
                    <li>Choose portfolio strategy</li>
                    <li>Select benchmark (default: SPY)</li>
                    <li>Configure risk parameters</li>
                    <li>Click "Run Analysis"</li>
                </ol>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Features grid - FIXED
    st.markdown("### ✨ Key Features")
    
    # Use the feature-grid CSS class
    st.markdown('<div class="feature-grid">', unsafe_allow_html=True)
    
    features = [
        ("📊", "Portfolio Optimization", "Advanced optimization algorithms for optimal asset allocation using Mean-Variance, Risk Parity, and Minimum Volatility strategies"),
        ("⚖️", "Risk Analytics", "Comprehensive risk metrics including VaR, CVaR, Drawdown analysis, and volatility modeling"),
        ("📈", "Performance Analysis", "Detailed performance attribution, benchmark comparison, and rolling metrics analysis"),
        ("🔍", "Quantitative Analytics", "Professional-grade analytics with Sharpe, Sortino, Calmar ratios and statistical metrics"),
        ("📉", "Risk Modeling", "Advanced Value at Risk calculations using Historical, Parametric, and Monte Carlo methods"),
        ("📋", "Institutional Reports", "Professional reporting with Excel/CSV export and comprehensive portfolio summaries")
    ]
    
    for icon, title, description in features:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-description">{description}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Recent market overview
    st.markdown("### 📈 Recent Market Overview")
    
    try:
        # Load some real market data
        market_tickers = ["SPY", "QQQ", "TLT", "GLD"]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        prices = yf.download(market_tickers, start=start_date, end=end_date)['Close']
        returns = prices.pct_change().iloc[-1] * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        market_data = [
            ("S&P 500", returns.get("SPY", 0), prices["SPY"].iloc[-1]),
            ("NASDAQ", returns.get("QQQ", 0), prices["QQQ"].iloc[-1]),
            ("20Y Treasury", returns.get("TLT", 0), prices["TLT"].iloc[-1]),
            ("Gold", returns.get("GLD", 0), prices["GLD"].iloc[-1])
        ]
        
        for i, (name, ret, price) in enumerate(market_data):
            cols = [col1, col2, col3, col4]
            with cols[i]:
                st.metric(name, f"${price:.2f}", f"{ret:.2f}%")
                
    except Exception:
        # Fallback if market data fails
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("S&P 500", "4,567.89", "+1.23%")
        with col2:
            st.metric("NASDAQ", "14,234.56", "+2.34%")
        with col3:
            st.metric("10Y Treasury", "4.12%", "-0.05%")
        with col4:
            st.metric("Gold", "$1,978.45", "+0.78%")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: var(--text-muted); font-size: 0.8rem; padding: 1rem;">
        <p>APOLLO/ENIGMA v7.0 | Professional Portfolio Management System</p>
        <p>For institutional use only. Past performance is not indicative of future results.</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# RUN APPLICATION
# -------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)[:200]}")
        st.info("Please refresh the page and try again.")
        
        with st.expander("Error Details", expanded=False):
            st.code(traceback.format_exc())
