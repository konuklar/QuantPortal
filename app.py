# =============================================================
# 🏛️ Institutional Apollo / ENIGMA – Quant Terminal v5.1
# Professional Portfolio Optimization & Global Multi-Asset Edition
# Complete Version with Full PyPortfolioOpt Integration
# Fixed: Removed networkx dependency
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
import plotly.figure_factory as ff
from scipy import stats, optimize
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import json
import concurrent.futures
from functools import lru_cache
import traceback
import time
import hashlib
import inspect
from pathlib import Path
import pickle
import tempfile

# Import PyPortfolioOpt with all required components
try:
    from pypfopt import expected_returns, risk_models, objective_functions, black_litterman, hierarchical_portfolio
    from pypfopt.efficient_frontier import EfficientFrontier, EfficientSemivariance
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    from pypfopt.objective_functions import L2_reg, transaction_cost
    from pypfopt.cla import CLA
    from pypfopt.hierarchical_portfolio import HRPOpt
    from pypfopt.risk_models import CovarianceShrinkage
    PYPFOPT_AVAILABLE = True
except ImportError as e:
    PYPFOPT_AVAILABLE = False
    st.warning(f"⚠️ PyPortfolioOpt not installed or has import errors: {str(e)[:100]}. Some optimization features will be limited.")
    # Define dummy classes for fallback
    class EfficientFrontier:
        def __init__(self, *args, **kwargs):
            pass
    class HRPOpt:
        def __init__(self, *args, **kwargs):
            pass

# Additional imports for enhanced functionality
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from itertools import combinations
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# ENHANCED GLOBAL ASSET UNIVERSE WITH SYMBOL MAPPING
# -------------------------------------------------------------
GLOBAL_ASSET_UNIVERSE = {
    # US Major Indices & ETFs
    "US_Indices": [
        "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "IVV", 
        "VEA", "VWO", "VUG", "VO", "VB", "VTV", "XLK", "XLV",
        "XLF", "XLE", "XLI", "XLP", "XLY", "XLU", "XLB", "XLC"
    ],
    
    # Bonds & Fixed Income
    "Bonds": [
        "TLT", "IEF", "SHY", "BND", "AGG", "HYG", "JNK",
        "MUB", "TIP", "LQD", "EMB", "BIL", "GOVT", "MBB",
        "VCIT", "VCSH", "VTIP", "SCHZ", "SCHO", "SCHR"
    ],
    
    # Commodities
    "Commodities": [
        "GLD", "SLV", "USO", "UNG", "DBA", "PDBC", "GSG",
        "WEAT", "CORN", "SOYB", "CPER", "PALL", "PPLT",
        "DBB", "DBC", "JJG", "JJN", "JJC"
    ],
    
    # Cryptocurrencies
    "Cryptocurrencies": [
        "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD",
        "SOL-USD", "DOT-USD", "DOGE-USD", "MATIC-USD", "AVAX-USD",
        "LTC-USD", "UNI-USD", "LINK-USD", "ATOM-USD", "ETC-USD",
        "FIL-USD", "XLM-USD", "VET-USD", "TRX-USD", "ALGO-USD"
    ],
    
    # US Stocks - Technology
    "US_Tech_Stocks": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META",
        "AVGO", "ASML", "ORCL", "AMD", "INTC", "CSCO", "CRM",
        "ADBE", "NFLX", "PYPL", "QCOM", "TXN", "IBM"
    ],
    
    # US Stocks - Finance
    "US_Finance_Stocks": [
        "JPM", "BAC", "WFC", "C", "GS", "MS", "SCHW",
        "BLK", "AXP", "V", "MA", "PYPL", "COF", "DFS",
        "USB", "PNC", "TFC", "BK", "STT", "MMC"
    ],
    
    # US Stocks - Healthcare
    "US_Healthcare_Stocks": [
        "JNJ", "UNH", "PFE", "MRK", "ABT", "TMO", "ABBV",
        "LLY", "BMY", "AMGN", "CVS", "CI", "DHR", "SYK",
        "BDX", "ISRG", "GILD", "VRTX", "REGN", "HCA"
    ],
    
    # European Stocks
    "Europe_Stocks": [
        "ASML.AS", "SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE",
        "NOVN.SW", "ROG.SW", "NESN.SW", "UBSG.SW", "CSGN.SW",
        "SAN.PA", "BNP.PA", "AIR.PA", "MC.PA", "OR.PA",
        "ENEL.MI", "ENI.MI", "ISP.MI", "UCG.MI", "AI.PA",
        "VOW3.DE", "BMW.DE", "DBK.DE", "BAS.DE", "LIN.DE"
    ],
    
    # UK Stocks
    "UK_Stocks": [
        "HSBA.L", "BP.L", "GSK.L", "RIO.L", "AAL.L",
        "AZN.L", "ULVR.L", "DGE.L", "BATS.L", "NG.L",
        "VOD.L", "LLOY.L", "BARCL.L", "BARC.L", "TSCO.L",
        "SHEL.L", "GLEN.L", "REL.L", "PRU.L", "LGEN.L"
    ],
    
    # Asia Pacific Stocks
    "Asia_Stocks": [
        "9988.HK", "0700.HK", "0388.HK", "0005.HK", "1299.HK",
        "7203.T", "8306.T", "9984.T", "6758.T", "6861.T",
        "BABA", "JD", "BIDU", "NTES", "TCEHY",
        "TSM", "SONY", "SNE", "MFG", "MUFG"
    ],
    
    # Emerging Markets
    "Emerging_Stocks": [
        "HDB", "INFY", "TCS.NS", "SBIN.NS", "RELIANCE.NS",
        "VALE", "ITUB", "BBD", "GGB", "ABEV", "SBS",
        "FMX", "AMX", "TEO", "PBR", "BSBR",
        "CHL", "CHU", "CEO", "PTR", "SNP"
    ],
    
    # Australia
    "Australia_Stocks": [
        "BHP.AX", "RIO.AX", "CBA.AX", "WBC.AX", "ANZ.AX",
        "NAB.AX", "CSL.AX", "WES.AX", "WOW.AX", "TLS.AX",
        "FMG.AX", "GMG.AX", "MQG.AX", "TCL.AX", "WPL.AX"
    ],
    
    # Singapore
    "Singapore_Stocks": [
        "D05.SI", "O39.SI", "U11.SI", "Z74.SI", "C09.SI",
        "G13.SI", "BN4.SI", "F34.SI", "C38U.SI", "ME8U.SI"
    ],
    
    # Turkey
    "Turkey_Stocks": [
        "AKBNK.IS", "GARAN.IS", "ISCTR.IS", "KOZAA.IS", "SAHOL.IS",
        "THYAO.IS", "TCELL.IS", "TUPRS.IS", "ARCLK.IS", "BIMAS.IS",
        "ASELS.IS", "HALKB.IS", "VAKBN.IS", "YKBNK.IS", "GARAN.IS"
    ],
    
    # Currencies & Forex with alternative symbols
    "Currencies": [
        "EURUSD=X", "EURUSD", "GBPUSD=X", "GBPUSD", 
        "USDJPY=X", "USDJPY", "USDCHF=X", "USDCHF",
        "AUDUSD=X", "AUDUSD", "USDCAD=X", "USDCAD", 
        "NZDUSD=X", "NZDUSD", "USDTRY=X", "USDTRY",
        "USDCNY=X", "USDCNY", "USDSGD=X", "USDSGD", 
        "USDHKD=X", "USDHKD", "USDINR=X", "USDINR",
        "USDBRL=X", "USDBRL", "USDZAR=X", "USDZAR", 
        "USDMXN=X", "USDMXN", "USDKRW=X", "USDKRW"
    ],
    
    # Volatility & Alternatives
    "Alternatives": [
        "^VIX", "VIXY", "UVXY", "SVXY", "SH",
        "TMF", "UPRO", "TQQQ", "SQQQ", "SPXU",
        "GDX", "GDXJ", "SLV", "USLV", "AGQ",
        "REET", "VNQ", "IYR", "REM", "MORT"
    ],
    
    # Real Estate
    "Real_Estate": [
        "VNQ", "IYR", "XLRE", "SCHH", "FREL",
        "REZ", "ROOF", "SRET", "MORT", "REM",
        "O", "AMT", "PLD", "CCI", "EQIX",
        "PSA", "AVB", "EQR", "DLR", "WELL"
    ],
    
    # Infrastructure
    "Infrastructure": [
        "IFRA", "PAVE", "MLPX", "UTG", "UTF",
        "NLR", "URA", "REMX", "PICK", "WOOD",
        "TAN", "ICLN", "PBW", "QCLN", "FAN"
    ]
}

# Enhanced Symbol mapping for common variations
SYMBOL_MAPPING = {
    "BTC-USD": ["BTC-USD", "BTCUSD", "BTCUSDT", "BTC-USD"],
    "ETH-USD": ["ETH-USD", "ETHUSD", "ETHUSDT", "ETH-USD"],
    "BRK-B": ["BRK-B", "BRK.B", "BRK_B"],
    "GOOGL": ["GOOGL", "GOOG"],
    "EURUSD=X": ["EURUSD=X", "EURUSD", "EUR/USD"],
    "GBPUSD=X": ["GBPUSD=X", "GBPUSD", "GBP/USD"],
    "USDJPY=X": ["USDJPY=X", "USDJPY", "USD/JPY"],
    "USDCHF=X": ["USDCHF=X", "USDCHF", "USD/CHF"],
    "AUDUSD=X": ["AUDUSD=X", "AUDUSD", "AUD/USD"],
    "USDCAD=X": ["USDCAD=X", "USDCAD", "USD/CAD"],
    "NZDUSD=X": ["NZDUSD=X", "NZDUSD", "NZD/USD"],
    "VOW3.DE": ["VOW3.DE", "VOW.DE", "VOW3.DE"],
    "BMW.DE": ["BMW.DE", "BMW.DE"],
    "DBK.DE": ["DBK.DE", "DBK.DE"],
}

# Flatten universe for selection
ALL_TICKERS = []
for category in GLOBAL_ASSET_UNIVERSE.values():
    ALL_TICKERS.extend(category)
ALL_TICKERS = list(dict.fromkeys(ALL_TICKERS))  # Remove duplicates while preserving order

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
APP_TITLE = "🏛️ Apollo/ENIGMA - Global Portfolio Terminal v5.1"
DEFAULT_RF_ANNUAL = 0.03
TRADING_DAYS = 252
MONTE_CARLO_SIMULATIONS = 10000
MAX_CACHE_SIZE = 100
CACHE_DIR = tempfile.gettempdir() + "/apollo_cache/"

os.environ["NUMEXPR_MAX_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# Create cache directory if it doesn't exist
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

# Initialize Streamlit with wide layout
st.set_page_config(
    page_title="Apollo/ENIGMA - Global Portfolio Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# -------------------------------------------------------------
# ENHANCED INSTITUTIONAL THEME
# -------------------------------------------------------------
st.markdown("""
<style>
:root {
    --primary: #1a5fb4;
    --primary-dark: #0d4fa0;
    --secondary: #26a269;
    --secondary-dark: #1e864d;
    --danger: #c01c28;
    --danger-dark: #9a1a23;
    --warning: #f5a623;
    --warning-dark: #d48e1e;
    --dark-bg: #0f172a;
    --dark-bg-light: #1e293b;
    --card-bg: #1e293b;
    --card-bg-light: #2d3748;
    --border: #334155;
    --border-light: #475569;
    --text: #f1f5f9;
    --text-muted: #94a3b8;
    --text-light: #e2e8f0;
    --success: #16a34a;
    --success-dark: #15803d;
    --info: #3b82f6;
    --info-dark: #2563eb;
}

/* Enhanced tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background-color: var(--dark-bg-light);
    padding: 4px;
    border-radius: 10px;
    border: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 500;
    color: var(--text-muted);
    border: 1px solid transparent;
    transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: var(--card-bg);
    color: var(--text-light);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark)) !important;
    color: white !important;
    border-color: var(--primary) !important;
    box-shadow: 0 4px 12px rgba(26, 95, 180, 0.3);
}

/* Professional KPI cards */
.kpi-card {
    background: linear-gradient(145deg, var(--card-bg), #1a2332);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    transition: all 0.3s ease;
    height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    position: relative;
    overflow: hidden;
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--primary), var(--secondary));
}

.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
    border-color: var(--primary);
}

.kpi-value {
    font-size: 28px;
    font-weight: 700;
    color: var(--text);
    margin: 8px 0;
    line-height: 1.2;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

.kpi-label {
    font-size: 13px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 500;
}

/* Enhanced metrics */
[data-testid="metric-container"] {
    background: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    transition: all 0.3s ease;
}

[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
    border-color: var(--primary) !important;
}

[data-testid="metric-container"] label {
    color: var(--text-muted) !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}

[data-testid="metric-container"] div {
    color: var(--text) !important;
    font-size: 24px !important;
    font-weight: 700 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark));
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s;
    width: 100%;
    box-shadow: 0 4px 12px rgba(26, 95, 180, 0.3);
    position: relative;
    overflow: hidden;
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    transition: 0.5s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(26, 95, 180, 0.4);
}

.stButton > button:hover::before {
    left: 100%;
}

/* Secondary buttons */
.secondary-button > button {
    background: linear-gradient(135deg, var(--secondary), var(--secondary-dark));
}

.danger-button > button {
    background: linear-gradient(135deg, var(--danger), var(--danger-dark));
}

.warning-button > button {
    background: linear-gradient(135deg, var(--warning), var(--warning-dark));
}

/* Status indicators */
.status-success { 
    color: var(--success) !important;
    font-weight: 600;
}
.status-warning { 
    color: var(--warning) !important;
    font-weight: 600;
}
.status-error { 
    color: var(--danger) !important;
    font-weight: 600;
}
.status-info {
    color: var(--info) !important;
    font-weight: 600;
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--dark-bg);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(var(--primary), var(--primary-dark));
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(var(--primary-dark), var(--primary));
}

/* Card enhancements */
.card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}

.card-header {
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 15px;
    color: var(--text);
    border-bottom: 1px solid var(--border);
    padding-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.card-header::before {
    content: '•';
    color: var(--primary);
    font-size: 1.5em;
}

/* Progress bars */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--primary), var(--secondary));
    border-radius: 4px;
}

/* Select boxes and inputs */
.stSelectbox > div > div, .stTextInput > div > div {
    background: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

.stSelectbox > div > div:hover, .stTextInput > div > div:hover {
    border-color: var(--primary) !important;
}

/* Data tables */
.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

.stDataFrame table {
    background: var(--card-bg) !important;
}

.stDataFrame thead {
    background: var(--dark-bg-light) !important;
}

.stDataFrame th {
    color: var(--text) !important;
    font-weight: 600 !important;
    border-bottom: 2px solid var(--border) !important;
}

.stDataFrame td {
    color: var(--text-light) !important;
    border-bottom: 1px solid var(--border-light) !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-weight: 600 !important;
}

.streamlit-expanderHeader:hover {
    background: var(--card-bg-light) !important;
    border-color: var(--primary) !important;
}

/* Alerts and info boxes */
.stAlert {
    border-radius: 10px !important;
    border: 1px solid !important;
}

/* Plotly chart container */
.js-plotly-plot {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--border);
}

/* Tooltips */
[title] {
    position: relative;
}

[title]:hover::after {
    content: attr(title);
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    background: var(--dark-bg);
    color: var(--text);
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 12px;
    white-space: nowrap;
    z-index: 1000;
    border: 1px solid var(--border);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

/* Badges */
.badge {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.badge-primary {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark));
    color: white;
}

.badge-success {
    background: linear-gradient(135deg, var(--success), var(--success-dark));
    color: white;
}

.badge-warning {
    background: linear-gradient(135deg, var(--warning), var(--warning-dark));
    color: white;
}

.badge-danger {
    background: linear-gradient(135deg, var(--danger), var(--danger-dark));
    color: white;
}

/* Grid layout */
.grid-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    margin-bottom: 20px;
}

/* Loading spinner */
.stSpinner > div {
    border: 3px solid var(--border);
    border-top: 3px solid var(--primary);
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .grid-container {
        grid-template-columns: 1fr;
    }
    
    .kpi-card {
        height: auto;
        min-height: 100px;
    }
    
    .kpi-value {
        font-size: 24px;
    }
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# PORTFOLIO STRATEGIES
# -------------------------------------------------------------
PORTFOLIO_STRATEGIES = {
    "Equal Weight": "Equal allocation across all selected assets",
    "Minimum Volatility": "Optimized for lowest portfolio volatility",
    "Maximum Sharpe Ratio": "Optimized for highest risk-adjusted returns",
    "Maximum Quadratic Utility": "Balances return and risk based on risk aversion",
    "Efficient Risk": "Optimizes for maximum return at given risk level",
    "Efficient Return": "Optimizes for minimum risk at given return target",
    "Risk Parity": "Equal risk contribution from each asset",
    "Maximum Diversification": "Maximizes diversification ratio",
    "Mean-Variance Optimal": "Classical Markowitz optimization",
    "Hierarchical Risk Parity": "Uses hierarchical clustering for diversification",
    "Minimum CVaR": "Minimizes Conditional Value at Risk",
    "Black-Litterman": "Combines market equilibrium with investor views",
    "Custom Weights": "Manually specify asset weights"
}

# -------------------------------------------------------------
# ENHANCED CACHING SYSTEM
# -------------------------------------------------------------
class EnhancedCache:
    """Enhanced caching system for better performance with persistence"""
    
    @staticmethod
    def get_cache_key(*args, **kwargs):
        """Generate a unique cache key from function arguments"""
        call_frame = inspect.currentframe().f_back
        func_name = call_frame.f_code.co_name
        
        # Create a string representation of arguments
        args_str = ''.join([str(arg) for arg in args])
        kwargs_str = ''.join([f"{k}{v}" for k, v in sorted(kwargs.items())])
        
        # Generate MD5 hash
        key_string = f"{func_name}_{args_str}_{kwargs_str}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    @staticmethod
    def cache_data(data, key, ttl_seconds=3600, metadata=None):
        """Cache data with TTL and metadata"""
        cache_file = Path(CACHE_DIR) / f"{key}.pkl"
        
        cache_entry = {
            'data': data,
            'timestamp': time.time(),
            'ttl': ttl_seconds,
            'metadata': metadata or {}
        }
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_entry, f, protocol=pickle.HIGHEST_PROTOCOL)
            return True
        except Exception as e:
            st.warning(f"Cache write failed: {str(e)[:100]}")
            return False
    
    @staticmethod
    def get_cached_data(key):
        """Retrieve cached data if not expired"""
        cache_file = Path(CACHE_DIR) / f"{key}.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cache_entry = pickle.load(f)
            
            # Check if cache is expired
            if time.time() - cache_entry['timestamp'] > cache_entry['ttl']:
                try:
                    cache_file.unlink(missing_ok=True)
                except:
                    pass
                return None
            
            return cache_entry['data']
        except Exception as e:
            try:
                cache_file.unlink(missing_ok=True)
            except:
                pass
            return None
    
    @staticmethod
    def clear_cache():
        """Clear all cache files"""
        cache_dir = Path(CACHE_DIR)
        if cache_dir.exists():
            for file in cache_dir.glob("*.pkl"):
                try:
                    file.unlink()
                except:
                    pass
            return True
        return False
    
    @staticmethod
    def get_cache_stats():
        """Get cache statistics"""
        cache_dir = Path(CACHE_DIR)
        if not cache_dir.exists():
            return {"total_files": 0, "total_size_mb": 0}
        
        files = list(cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in files if f.exists())
        
        return {
            "total_files": len(files),
            "total_size_mb": total_size / (1024 * 1024)
        }

# -------------------------------------------------------------
# ENHANCED DATA LOADING WITH PARALLEL PROCESSING
# -------------------------------------------------------------
def download_single_ticker(ticker, start_date, end_date, max_retries=3, timeout=10):
    """Download data for a single ticker with retry logic and symbol fallback"""
    for attempt in range(max_retries):
        try:
            # Try alternative symbols first
            symbols_to_try = [ticker]
            if ticker in SYMBOL_MAPPING:
                symbols_to_try = SYMBOL_MAPPING[ticker] + symbols_to_try
            
            for symbol in symbols_to_try:
                try:
                    ticker_obj = yf.Ticker(symbol)
                    
                    # Get info to check if ticker exists
                    info = ticker_obj.info
                    if not info:
                        continue
                    
                    # Try different intervals if daily fails
                    hist = ticker_obj.history(
                        start=start_date,
                        end=end_date,
                        interval="1d",
                        auto_adjust=True,
                        prepost=False,
                        timeout=timeout
                    )
                    
                    if not hist.empty and len(hist) > 10:
                        # Use adjusted close if available, otherwise close
                        if 'Close' in hist.columns:
                            price_series = hist['Close'].copy()
                        elif 'Adj Close' in hist.columns:
                            price_series = hist['Adj Close'].copy()
                        else:
                            continue
                        
                        # Remove any extreme outliers
                        median_price = price_series.median()
                        std_price = price_series.std()
                        if std_price > 0:
                            # Cap at 10 standard deviations
                            lower_bound = median_price - 10 * std_price
                            upper_bound = median_price + 10 * std_price
                            price_series = price_series.clip(lower_bound, upper_bound)
                        
                        return symbol, price_series
                        
                except Exception as e:
                    if attempt == max_retries - 1:
                        st.debug(f"Failed to download {symbol}: {str(e)[:100]}")
                    continue
            
            # Try batch download as fallback
            try:
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    threads=False,
                    auto_adjust=True,
                    timeout=timeout
                )
                if not data.empty and 'Close' in data.columns:
                    return ticker, data['Close']
            except Exception as e:
                if attempt == max_retries - 1:
                    st.debug(f"Batch download failed for {ticker}: {str(e)[:100]}")
                
        except Exception as e:
            if attempt == max_retries - 1:
                st.debug(f"Final attempt failed for {ticker}: {str(e)[:100]}")
            time.sleep(0.5 * (attempt + 1))  # Exponential backoff
    
    return ticker, None

@st.cache_data(show_spinner=False, ttl=3600, max_entries=20)
def load_global_prices_enhanced(tickers: List[str], start_date: str, end_date: str, 
                               use_parallel: bool = True) -> pd.DataFrame:
    """Load global asset prices with enhanced error handling and parallel downloads"""
    
    if not tickers:
        return pd.DataFrame()
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    prices_dict = {}
    successful_tickers = []
    failed_tickers = []
    
    start_time = time.time()
    
    if use_parallel and len(tickers) > 5:
        # Parallel download for larger sets
        max_workers = min(8, len(tickers))  # Reduced to avoid rate limiting
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_ticker = {
                executor.submit(download_single_ticker, ticker, start_date, end_date): ticker 
                for ticker in tickers
            }
            
            # Process completed tasks
            completed = 0
            total = len(tickers)
            
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                completed += 1
                
                try:
                    symbol, price_data = future.result(timeout=15)
                    if price_data is not None and len(price_data) > 10:
                        prices_dict[symbol] = price_data
                        successful_tickers.append(symbol)
                    else:
                        failed_tickers.append(ticker)
                except Exception as e:
                    failed_tickers.append(ticker)
                
                # Update progress
                progress = completed / total
                progress_bar.progress(progress)
                status_text.text(f"Loaded {completed}/{total} assets ({len(successful_tickers)} successful)")
    
    else:
        # Sequential download for smaller sets
        total = len(tickers)
        for i, ticker in enumerate(tickers):
            symbol, price_data = download_single_ticker(ticker, start_date, end_date)
            
            if price_data is not None and len(price_data) > 10:
                prices_dict[symbol] = price_data
                successful_tickers.append(symbol)
            else:
                failed_tickers.append(ticker)
            
            # Update progress
            progress = (i + 1) / total
            progress_bar.progress(progress)
            status_text.text(f"Loaded {i + 1}/{total} assets...")
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    elapsed_time = time.time() - start_time
    
    # Report results
    if successful_tickers:
        st.success(f"✅ Successfully loaded {len(successful_tickers)} out of {len(tickers)} assets in {elapsed_time:.1f}s")
        
        if failed_tickers:
            with st.expander(f"⚠️ Failed to load {len(failed_tickers)} assets", expanded=False):
                st.write(", ".join(failed_tickers[:20]))
                if len(failed_tickers) > 20:
                    st.write(f"... and {len(failed_tickers) - 20} more")
    
    # Create DataFrame
    if prices_dict:
        prices_df = pd.DataFrame(prices_dict)
        
        # Sort columns to maintain order as much as possible
        ordered_columns = []
        for ticker in tickers:
            if ticker in prices_df.columns:
                ordered_columns.append(ticker)
            else:
                # Check if any mapped symbol is in the dataframe
                for mapped in SYMBOL_MAPPING.get(ticker, []):
                    if mapped in prices_df.columns and mapped not in ordered_columns:
                        ordered_columns.append(mapped)
                        break
        
        # Add any remaining columns
        remaining_cols = [col for col in prices_df.columns if col not in ordered_columns]
        ordered_columns.extend(remaining_cols)
        
        prices_df = prices_df[ordered_columns]
        
        # Handle missing data
        original_columns = len(prices_df.columns)
        prices_df = prices_df.dropna(axis=1, how='all')
        
        if prices_df.empty:
            st.error("❌ No data available after cleaning. All tickers failed.")
            return pd.DataFrame()
        
        # Fill small gaps (limit to 5 consecutive NaNs)
        prices_df = prices_df.ffill(limit=5).bfill(limit=5)
        
        # Remove assets with too much missing data (>25%)
        missing_pct = prices_df.isna().mean()
        valid_columns = missing_pct[missing_pct < 0.25].index.tolist()
        prices_df = prices_df[valid_columns]
        
        removed_columns = original_columns - len(prices_df.columns)
        if removed_columns > 0:
            st.info(f"📊 Removed {removed_columns} assets with excessive missing data")
        
        if len(prices_df.columns) < 2:
            st.error("❌ Insufficient data for portfolio analysis. Need at least 2 assets with valid data.")
            return pd.DataFrame()
        
        # Ensure we have enough rows
        if len(prices_df) < 20:
            st.warning(f"⚠️ Limited data: only {len(prices_df)} data points available")
        
        # Normalize column names to original ticker names where possible
        column_mapping = {}
        for col in prices_df.columns:
            for ticker in tickers:
                if ticker == col or (ticker in SYMBOL_MAPPING and col in SYMBOL_MAPPING[ticker]):
                    column_mapping[col] = ticker
                    break
        
        prices_df = prices_df.rename(columns=column_mapping)
        
        return prices_df
    
    else:
        st.error("❌ Could not load any data. Please check your ticker symbols and internet connection.")
        return pd.DataFrame()

def generate_demo_data(tickers, start_date, end_date):
    """Generate realistic synthetic data for demonstration when real data fails"""
    st.info("📊 Generating realistic demo data for testing purposes...")
    
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    if len(dates) < 100:
        dates = pd.date_range(end=pd.Timestamp.today(), periods=252, freq='B')
    
    prices_dict = {}
    
    # Realistic asset parameters based on actual market data
    asset_params = {
        # ETFs
        'SPY': {'base': 400, 'vol': 0.18, 'trend': 0.08, 'jump_prob': 0.01, 'jump_size': 0.03},
        'QQQ': {'base': 350, 'vol': 0.22, 'trend': 0.10, 'jump_prob': 0.015, 'jump_size': 0.04},
        'IWM': {'base': 180, 'vol': 0.20, 'trend': 0.06, 'jump_prob': 0.012, 'jump_size': 0.035},
        'DIA': {'base': 340, 'vol': 0.17, 'trend': 0.07, 'jump_prob': 0.01, 'jump_size': 0.03},
        
        # Bonds
        'TLT': {'base': 100, 'vol': 0.12, 'trend': 0.02, 'jump_prob': 0.005, 'jump_size': 0.02},
        'IEF': {'base': 105, 'vol': 0.08, 'trend': 0.015, 'jump_prob': 0.004, 'jump_size': 0.015},
        'SHY': {'base': 85, 'vol': 0.04, 'trend': 0.01, 'jump_prob': 0.003, 'jump_size': 0.01},
        
        # Commodities
        'GLD': {'base': 180, 'vol': 0.15, 'trend': 0.04, 'jump_prob': 0.008, 'jump_size': 0.025},
        'SLV': {'base': 22, 'vol': 0.25, 'trend': 0.03, 'jump_prob': 0.01, 'jump_size': 0.04},
        'USO': {'base': 70, 'vol': 0.35, 'trend': 0.05, 'jump_prob': 0.02, 'jump_size': 0.06},
        
        # Cryptocurrencies
        'BTC-USD': {'base': 30000, 'vol': 0.65, 'trend': 0.20, 'jump_prob': 0.03, 'jump_size': 0.10},
        'ETH-USD': {'base': 2000, 'vol': 0.70, 'trend': 0.18, 'jump_prob': 0.025, 'jump_size': 0.09},
        
        # Stocks
        'AAPL': {'base': 150, 'vol': 0.25, 'trend': 0.12, 'jump_prob': 0.01, 'jump_size': 0.04},
        'MSFT': {'base': 300, 'vol': 0.22, 'trend': 0.10, 'jump_prob': 0.01, 'jump_size': 0.035},
        'GOOGL': {'base': 120, 'vol': 0.24, 'trend': 0.09, 'jump_prob': 0.01, 'jump_size': 0.04},
        'AMZN': {'base': 130, 'vol': 0.28, 'trend': 0.08, 'jump_prob': 0.012, 'jump_size': 0.045},
        'TSLA': {'base': 220, 'vol': 0.45, 'trend': 0.15, 'jump_prob': 0.02, 'jump_size': 0.08},
        'NVDA': {'base': 450, 'vol': 0.40, 'trend': 0.25, 'jump_prob': 0.02, 'jump_size': 0.09},
        'META': {'base': 300, 'vol': 0.30, 'trend': 0.14, 'jump_prob': 0.015, 'jump_size': 0.06},
        'JPM': {'base': 140, 'vol': 0.20, 'trend': 0.07, 'jump_prob': 0.008, 'jump_size': 0.03},
        'JNJ': {'base': 160, 'vol': 0.15, 'trend': 0.05, 'jump_prob': 0.006, 'jump_size': 0.025},
        'V': {'base': 220, 'vol': 0.18, 'trend': 0.08, 'jump_prob': 0.01, 'jump_size': 0.035},
    }
    
    # Generate correlated returns
    np.random.seed(42)
    n_assets = len(tickers)
    n_days = len(dates)
    
    # Create realistic correlation matrix
    base_corr = 0.4
    corr_matrix = np.eye(n_assets) * (1 - base_corr) + base_corr
    
    # Add sector correlations for realism
    for i, ticker_i in enumerate(tickers):
        for j, ticker_j in enumerate(tickers):
            if i != j:
                # Tech stocks are more correlated
                if ticker_i in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META'] and \
                   ticker_j in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META']:
                    corr_matrix[i, j] = 0.7
                # Financial stocks
                elif ticker_i in ['JPM', 'BAC', 'WFC', 'C', 'GS'] and \
                     ticker_j in ['JPM', 'BAC', 'WFC', 'C', 'GS']:
                    corr_matrix[i, j] = 0.65
                # Bonds have negative correlation with stocks
                elif (ticker_i in ['TLT', 'IEF', 'SHY'] and ticker_j not in ['TLT', 'IEF', 'SHY']) or \
                     (ticker_j in ['TLT', 'IEF', 'SHY'] and ticker_i not in ['TLT', 'IEF', 'SHY']):
                    corr_matrix[i, j] = -0.3
    
    # Ensure correlation matrix is positive definite
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    corr_matrix = np.maximum(corr_matrix, -0.99)
    corr_matrix = np.minimum(corr_matrix, 0.99)
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Generate correlated random numbers
    try:
        L = np.linalg.cholesky(corr_matrix)
    except:
        # If not positive definite, use nearest correlation matrix
        eigvals, eigvecs = np.linalg.eigh(corr_matrix)
        eigvals = np.maximum(eigvals, 0.01)
        corr_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
        L = np.linalg.cholesky(corr_matrix)
    
    uncorrelated = np.random.randn(n_assets, n_days)
    correlated = L @ uncorrelated
    
    for idx, ticker in enumerate(tickers):
        # Get parameters
        params = asset_params.get(ticker, {'base': 100, 'vol': 0.25, 'trend': 0.06, 
                                          'jump_prob': 0.01, 'jump_size': 0.03})
        
        # Generate returns with trend and volatility
        daily_trend = (1 + params['trend']) ** (1/252) - 1
        daily_vol = params['vol'] / np.sqrt(252)
        
        # Create base returns
        returns = daily_trend + daily_vol * correlated[idx]
        
        # Add realistic autocorrelation (momentum)
        for i in range(1, len(returns)):
            returns[i] = 0.1 * returns[i-1] + 0.9 * returns[i]
        
        # Add jump diffusion (market crashes and rallies)
        jump_mask = np.random.rand(len(returns)) < params['jump_prob']
        jump_returns = np.random.randn(np.sum(jump_mask)) * params['jump_size']
        returns[jump_mask] += jump_returns
        
        # Add day of week effects
        day_of_week = dates.dayofweek.values
        returns[day_of_week == 4] *= 1.1  # Fridays often have different behavior
        returns[day_of_week == 0] *= 0.95  # Mondays
        
        # Calculate prices
        cumulative_returns = np.exp(np.cumsum(returns))
        prices = params['base'] * cumulative_returns
        
        # Add micro-structure noise
        prices = prices * (1 + np.random.randn(len(prices)) * 0.001)
        
        # Ensure prices are positive
        prices = np.maximum(prices, 0.01)
        
        prices_dict[ticker] = pd.Series(prices, index=dates)
    
    return pd.DataFrame(prices_dict)

@st.cache_data(show_spinner=False, ttl=3600, max_entries=50)
def load_global_prices_cached(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Load global asset prices with enhanced caching"""
    cache_key = f"prices_{hashlib.md5((str(sorted(tickers)) + start_date + end_date).encode()).hexdigest()}"
    
    # Try to get from enhanced cache first
    cached_data = EnhancedCache.get_cached_data(cache_key)
    if cached_data is not None:
        st.info(f"📚 Loaded {len(cached_data.columns)} assets from cache")
        return cached_data
    
    # If not in cache, load from original function
    data = load_global_prices_enhanced(tickers, start_date, end_date)
    
    # Cache the result
    if not data.empty:
        EnhancedCache.cache_data(data, cache_key, ttl_seconds=7200)  # 2 hour TTL
        st.info(f"💾 Cached {len(data.columns)} assets for future use")
    
    return data

# -------------------------------------------------------------
# ENHANCED PERFORMANCE METRICS CALCULATOR
# -------------------------------------------------------------
class PerformanceMetrics:
    """Enhanced performance metrics calculator with all major financial ratios"""
    
    @staticmethod
    def calculate_metrics(returns: pd.Series, benchmark_returns: pd.Series = None, 
                         risk_free_rate: float = 0.03, trading_days: int = 252) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        if returns is None or returns.empty:
            return {}
        
        returns_clean = returns.dropna()
        if len(returns_clean) < 5:
            return {}
        
        metrics = {}
        
        try:
            # Basic metrics
            metrics['annual_return'] = returns_clean.mean() * trading_days
            metrics['annual_volatility'] = returns_clean.std() * np.sqrt(trading_days)
            
            if metrics['annual_volatility'] > 0:
                metrics['sharpe_ratio'] = (metrics['annual_return'] - risk_free_rate) / metrics['annual_volatility']
            else:
                metrics['sharpe_ratio'] = 0
            
            # Downside metrics
            downside_returns = returns_clean[returns_clean < 0]
            if len(downside_returns) > 0:
                downside_deviation = downside_returns.std() * np.sqrt(trading_days)
                if downside_deviation > 0:
                    metrics['sortino_ratio'] = (metrics['annual_return'] - risk_free_rate) / downside_deviation
                else:
                    metrics['sortino_ratio'] = np.nan
            else:
                metrics['sortino_ratio'] = np.nan
            
            # Maximum Drawdown
            cumulative = (1 + returns_clean).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            metrics['max_drawdown'] = drawdown.min() if not drawdown.empty else 0
            
            # Drawdown statistics
            if not drawdown.empty:
                metrics['avg_drawdown'] = drawdown.mean()
                metrics['drawdown_std'] = drawdown.std()
                metrics['drawdown_skewness'] = stats.skew(drawdown.dropna())
                metrics['drawdown_kurtosis'] = stats.kurtosis(drawdown.dropna())
            
            # Recovery metrics
            if metrics['max_drawdown'] < 0:
                try:
                    # Find drawdown recovery
                    underwater = cumulative < running_max
                    if underwater.any():
                        recovery_start = drawdown.idxmin() if not drawdown.empty else None
                        
                        if recovery_start:
                            # Find when portfolio recovers to previous high
                            recovery_index = cumulative.index.get_loc(recovery_start)
                            recovery_periods = 0
                            for i in range(recovery_index, len(cumulative)):
                                if cumulative.iloc[i] >= running_max.iloc[i]:
                                    recovery_periods = i - recovery_index
                                    break
                            metrics['recovery_period'] = recovery_periods
                        else:
                            metrics['recovery_period'] = np.nan
                except:
                    metrics['recovery_period'] = np.nan
            
            # Calmar Ratio
            if metrics['max_drawdown'] < 0:
                metrics['calmar_ratio'] = metrics['annual_return'] / abs(metrics['max_drawdown'])
            else:
                metrics['calmar_ratio'] = np.nan
            
            # Information Ratio (if benchmark provided)
            if benchmark_returns is not None and not benchmark_returns.empty:
                # Align dates
                aligned_returns = returns_clean.copy()
                aligned_benchmark = benchmark_returns.reindex(returns_clean.index).dropna()
                aligned_returns = aligned_returns.reindex(aligned_benchmark.index)
                
                if len(aligned_returns) > 5 and len(aligned_benchmark) > 5:
                    active_returns = aligned_returns - aligned_benchmark
                    tracking_error = active_returns.std() * np.sqrt(trading_days)
                    if tracking_error > 0:
                        metrics['information_ratio'] = (
                            aligned_returns.mean() * trading_days - 
                            aligned_benchmark.mean() * trading_days
                        ) / tracking_error
                    else:
                        metrics['information_ratio'] = np.nan
        
            # Skewness and Kurtosis
            if len(returns_clean) > 10:
                metrics['skewness'] = stats.skew(returns_clean.dropna())
                metrics['kurtosis'] = stats.kurtosis(returns_clean.dropna())
            else:
                metrics['skewness'] = np.nan
                metrics['kurtosis'] = np.nan
            
            # Value at Risk (Historical)
            if len(returns_clean) > 5:
                metrics['var_95'] = np.percentile(returns_clean, 5)
                metrics['var_99'] = np.percentile(returns_clean, 1)
                metrics['var_90'] = np.percentile(returns_clean, 10)
            else:
                metrics['var_95'] = np.nan
                metrics['var_99'] = np.nan
                metrics['var_90'] = np.nan
            
            # Conditional VaR (Expected Shortfall)
            if not pd.isna(metrics['var_95']):
                tail_95 = returns_clean[returns_clean <= metrics['var_95']]
                metrics['cvar_95'] = tail_95.mean() if len(tail_95) > 0 else metrics['var_95']
            else:
                metrics['cvar_95'] = np.nan
                
            if not pd.isna(metrics['var_99']):
                tail_99 = returns_clean[returns_clean <= metrics['var_99']]
                metrics['cvar_99'] = tail_99.mean() if len(tail_99) > 0 else metrics['var_99']
            else:
                metrics['cvar_99'] = np.nan
            
            # Gain/Loss Ratio
            positive_returns = returns_clean[returns_clean > 0]
            negative_returns = returns_clean[returns_clean < 0]
            if len(positive_returns) > 0 and len(negative_returns) > 0:
                metrics['gain_loss_ratio'] = abs(positive_returns.mean() / negative_returns.mean())
            else:
                metrics['gain_loss_ratio'] = np.nan
            
            # Win Rate
            metrics['win_rate'] = (returns_clean > 0).mean()
            
            # Average Win/Loss
            if len(positive_returns) > 0:
                metrics['avg_win'] = positive_returns.mean()
            else:
                metrics['avg_win'] = 0
            
            if len(negative_returns) > 0:
                metrics['avg_loss'] = negative_returns.mean()
            else:
                metrics['avg_loss'] = 0
            
            # Profit Factor
            gross_profit = positive_returns.sum()
            gross_loss = abs(negative_returns.sum())
            if gross_loss > 0:
                metrics['profit_factor'] = gross_profit / gross_loss
            else:
                metrics['profit_factor'] = np.nan
            
            # Omega Ratio
            threshold = risk_free_rate / trading_days
            excess_returns = returns_clean - threshold
            positive_excess = excess_returns[excess_returns > 0].sum()
            negative_excess = abs(excess_returns[excess_returns < 0].sum())
            if negative_excess > 0:
                metrics['omega_ratio'] = positive_excess / negative_excess
            else:
                metrics['omega_ratio'] = np.nan
            
            # Ulcer Index
            if len(drawdown) > 0:
                metrics['ulcer_index'] = np.sqrt((drawdown ** 2).mean())
            else:
                metrics['ulcer_index'] = np.nan
            
            # Pain Index
            if len(drawdown) > 0:
                metrics['pain_index'] = abs(drawdown).mean()
            else:
                metrics['pain_index'] = np.nan
            
            # Tail Ratio
            if not pd.isna(metrics['var_95']) and not pd.isna(metrics['var_99']) and metrics['var_99'] != 0:
                metrics['tail_ratio'] = abs(metrics['var_95'] / metrics['var_99'])
            else:
                metrics['tail_ratio'] = np.nan
            
            # Burke Ratio
            if metrics['max_drawdown'] < 0:
                metrics['burke_ratio'] = metrics['annual_return'] / np.sqrt(
                    sum(sorted(drawdown.dropna().tolist())[:5]) ** 2 if len(drawdown) >= 5 else metrics['max_drawdown'] ** 2
                )
            else:
                metrics['burke_ratio'] = np.nan
            
            # Sterling Ratio
            if metrics['max_drawdown'] < 0:
                metrics['sterling_ratio'] = metrics['annual_return'] / abs(metrics['max_drawdown'])
            else:
                metrics['sterling_ratio'] = np.nan
            
            # Treynor Ratio (requires beta)
            if benchmark_returns is not None and not benchmark_returns.empty:
                aligned_returns = returns_clean.copy()
                aligned_benchmark = benchmark_returns.reindex(returns_clean.index).dropna()
                aligned_returns = aligned_returns.reindex(aligned_benchmark.index)
                
                if len(aligned_returns) > 10:
                    cov_matrix = np.cov(aligned_returns, aligned_benchmark)
                    if cov_matrix[1, 1] > 0:
                        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
                        metrics['beta'] = beta
                        if beta != 0:
                            metrics['treynor_ratio'] = (metrics['annual_return'] - risk_free_rate) / beta
                        else:
                            metrics['treynor_ratio'] = np.nan
            
            # M2 Measure
            if metrics['annual_volatility'] > 0:
                metrics['m2_measure'] = risk_free_rate + (metrics['sharpe_ratio'] * metrics['annual_volatility'])
            else:
                metrics['m2_measure'] = np.nan
            
            # Capture Ratios
            if benchmark_returns is not None and not benchmark_returns.empty:
                aligned_returns = returns_clean.copy()
                aligned_benchmark = benchmark_returns.reindex(returns_clean.index).dropna()
                aligned_returns = aligned_returns.reindex(aligned_benchmark.index)
                
                if len(aligned_returns) > 10:
                    # Up capture
                    up_market = aligned_benchmark > 0
                    if up_market.any():
                        metrics['up_capture'] = (
                            aligned_returns[up_market].mean() / 
                            aligned_benchmark[up_market].mean()
                        ) if aligned_benchmark[up_market].mean() != 0 else np.nan
                    
                    # Down capture
                    down_market = aligned_benchmark < 0
                    if down_market.any():
                        metrics['down_capture'] = (
                            aligned_returns[down_market].mean() / 
                            aligned_benchmark[down_market].mean()
                        ) if aligned_benchmark[down_market].mean() != 0 else np.nan
            
            # Hit Rate
            if benchmark_returns is not None and not benchmark_returns.empty:
                aligned_returns = returns_clean.copy()
                aligned_benchmark = benchmark_returns.reindex(returns_clean.index).dropna()
                aligned_returns = aligned_returns.reindex(aligned_benchmark.index)
                
                if len(aligned_returns) > 0:
                    metrics['hit_rate'] = (aligned_returns > aligned_benchmark).mean()
            
            # Return over Maximum Drawdown (RoMaD)
            if metrics['max_drawdown'] < 0:
                metrics['romad'] = metrics['annual_return'] / abs(metrics['max_drawdown'])
            else:
                metrics['romad'] = np.nan
            
        except Exception as e:
            st.warning(f"Error calculating some metrics: {str(e)[:100]}")
            # Return basic metrics even if others fail
            pass
        
        return metrics
    
    @staticmethod
    def calculate_rolling_metrics(returns: pd.Series, window: int = 63, 
                                 risk_free_rate: float = 0.03) -> pd.DataFrame:
        """Calculate rolling performance metrics"""
        if returns.empty or len(returns) < window:
            return pd.DataFrame()
        
        rolling_metrics = []
        
        for i in range(window, len(returns) + 1):
            window_returns = returns.iloc[i-window:i]
            
            # Basic rolling metrics
            rolling_return = window_returns.mean() * 252
            rolling_vol = window_returns.std() * np.sqrt(252)
            
            # Sharpe ratio
            if rolling_vol > 0:
                sharpe = (rolling_return - risk_free_rate) / rolling_vol
            else:
                sharpe = 0
            
            # Sortino ratio
            downside = window_returns[window_returns < 0]
            if len(downside) > 0:
                downside_vol = downside.std() * np.sqrt(252)
                if downside_vol > 0:
                    sortino = (rolling_return - risk_free_rate) / downside_vol
                else:
                    sortino = np.nan
            else:
                sortino = np.nan
            
            rolling_metrics.append({
                'date': returns.index[i-1],
                'return': rolling_return,
                'volatility': rolling_vol,
                'sharpe': sharpe,
                'sortino': sortino
            })
        
        return pd.DataFrame(rolling_metrics).set_index('date')

# -------------------------------------------------------------
# FIXED PYPORTFOLIOOPT INTEGRATION (NO SERIES BOOLEAN AMBIGUITY)
# -------------------------------------------------------------
class PortfolioOptimizer:
    """Enhanced portfolio optimization using PyPortfolioOpt with fixed boolean ambiguity"""
    
    @staticmethod
    def optimize_portfolio(returns_df: pd.DataFrame, strategy: str, 
                          target_return: float = None, target_risk: float = None,
                          risk_free_rate: float = 0.03, 
                          risk_aversion: float = 1.0,
                          short_allowed: bool = False,
                          view_confidences: Optional[List[float]] = None,
                          market_weights: Optional[np.ndarray] = None) -> Dict:
        """Optimize portfolio using PyPortfolioOpt with FIXED error handling"""
        
        # Validate input
        if returns_df is None or returns_df.empty:
            return PortfolioOptimizer._fallback_weights([], risk_free_rate)
        
        if len(returns_df.columns) < 2:
            n_assets = len(returns_df.columns)
            equal_weights = np.ones(n_assets) / n_assets if n_assets > 0 else np.array([])
            return {
                'weights': equal_weights,
                'expected_return': 0,
                'expected_risk': 0,
                'sharpe_ratio': 0,
                'method': 'Single Asset',
                'cleaned_weights': {asset: equal_weights[i] for i, asset in enumerate(returns_df.columns)},
                'success': True
            }
        
        if not PYPFOPT_AVAILABLE:
            return PortfolioOptimizer._fallback_weights(returns_df.columns.tolist(), risk_free_rate, returns_df)
        
        try:
            # Clean returns data
            returns_clean = returns_df.copy()
            
            # FIXED: Remove extreme outliers (>10 standard deviations) - NO SERIES BOOLEAN AMBIGUITY
            for col in returns_clean.columns:
                col_data = returns_clean[col].dropna()
                if len(col_data) > 10:
                    mean = col_data.mean()
                    std = col_data.std()
                    if std > 0:
                        # FIX: Use numpy array for comparison to avoid Series boolean ambiguity
                        values = returns_clean[col].values
                        mask_array = np.abs(values - mean) < (10 * std)
                        # Apply the mask properly
                        outliers_mask = ~mask_array
                        if np.any(outliers_mask):
                            # Replace outliers with mean
                            returns_clean.loc[outliers_mask, col] = mean
            
            # Calculate expected returns and covariance
            try:
                mu = expected_returns.mean_historical_return(returns_clean)
            except Exception as e:
                # Fallback to simple mean
                mu = returns_clean.mean() * TRADING_DAYS
                st.warning(f"Using simple mean returns: {str(e)[:100]}")
            
            try:
                S = risk_models.sample_cov(returns_clean)
                
                # Check if covariance matrix is positive definite
                try:
                    np.linalg.cholesky(S)
                except np.linalg.LinAlgError:
                    # Use shrinkage estimator if not positive definite
                    st.info("Covariance matrix not positive definite. Using Ledoit-Wolf shrinkage.")
                    S = risk_models.CovarianceShrinkage(returns_clean).ledoit_wolf()
            except Exception as e:
                # Fallback to diagonal covariance
                st.warning(f"Using diagonal covariance: {str(e)[:100]}")
                var_values = returns_clean.var()
                S = pd.DataFrame(np.diag(var_values), 
                                index=returns_clean.columns, 
                                columns=returns_clean.columns)
            
            # Create efficient frontier
            ef = EfficientFrontier(mu, S, weight_bounds=(0, 1) if not short_allowed else (-1, 1))
            
            # Apply strategy
            weights = None
            
            if strategy == "Minimum Volatility":
                try:
                    weights = ef.min_volatility()
                except Exception as e:
                    st.warning(f"Min volatility failed: {str(e)[:100]}. Using equal weights.")
                    weights = PortfolioOptimizer._equal_weights(returns_clean)
                    
            elif strategy == "Maximum Sharpe Ratio":
                try:
                    weights = ef.max_sharpe(risk_free_rate=risk_free_rate/TRADING_DAYS)
                except Exception as e:
                    st.warning(f"Max Sharpe failed: {str(e)[:100]}. Using Min Volatility.")
                    weights = ef.min_volatility()
                
            elif strategy == "Maximum Quadratic Utility":
                try:
                    weights = ef.max_quadratic_utility(risk_aversion=risk_aversion)
                except Exception as e:
                    st.warning(f"Max Quadratic Utility failed: {str(e)[:100]}. Using Max Sharpe.")
                    weights = ef.max_sharpe(risk_free_rate=risk_free_rate/TRADING_DAYS)
                
            elif strategy == "Efficient Risk":
                if target_risk:
                    try:
                        weights = ef.efficient_risk(target_risk=target_risk/np.sqrt(TRADING_DAYS))
                    except Exception as e:
                        st.warning(f"Efficient Risk failed: {str(e)[:100]}. Using Min Volatility.")
                        weights = ef.min_volatility()
                else:
                    weights = ef.min_volatility()
                    
            elif strategy == "Efficient Return":
                if target_return:
                    try:
                        weights = ef.efficient_return(target_return=target_return/TRADING_DAYS)
                    except Exception as e:
                        st.warning(f"Efficient Return failed: {str(e)[:100]}. Using Max Sharpe.")
                        weights = ef.max_sharpe(risk_free_rate=risk_free_rate/TRADING_DAYS)
                else:
                    weights = ef.max_sharpe(risk_free_rate=risk_free_rate/TRADING_DAYS)
                    
            elif strategy == "Risk Parity":
                # Simple risk parity implementation
                volatilities = returns_clean.std()
                # FIX: Handle zero or NaN volatilities
                valid_vols = volatilities[~volatilities.isna() & (volatilities > 0)]
                if len(valid_vols) > 0:
                    inv_vol = 1 / valid_vols
                    weights_series = inv_vol / inv_vol.sum()
                    # Create weights dictionary with proper alignment
                    weights = {asset: weights_series.get(asset, 0) for asset in returns_clean.columns}
                else:
                    weights = PortfolioOptimizer._equal_weights(returns_clean)
                
            elif strategy == "Maximum Diversification":
                try:
                    # Calculate diversification ratio
                    weights = ef.max_sharpe(risk_free_rate=risk_free_rate/TRADING_DAYS)
                    # Note: PyPortfolioOpt doesn't have max diversification directly
                    # This is an approximation
                except Exception as e:
                    st.warning(f"Max Diversification failed: {str(e)[:100]}. Using Min Volatility.")
                    weights = ef.min_volatility()
                
            elif strategy == "Mean-Variance Optimal":
                try:
                    weights = ef.max_sharpe(risk_free_rate=risk_free_rate/TRADING_DAYS)
                except Exception as e:
                    st.warning(f"Mean-Variance failed: {str(e)[:100]}. Using Min Volatility.")
                    weights = ef.min_volatility()
            
            elif strategy == "Hierarchical Risk Parity" and PYPFOPT_AVAILABLE:
                try:
                    # Use Hierarchical Risk Parity
                    hrp = HRPOpt(returns_clean)
                    weights = hrp.optimize()
                except Exception as e:
                    st.warning(f"HRP failed: {str(e)[:100]}. Using Min Volatility.")
                    weights = ef.min_volatility()
            
            elif strategy == "Minimum CVaR":
                try:
                    # Use Efficient Semivariance for CVaR approximation
                    ef_semi = EfficientSemivariance(mu, returns_clean)
                    ef_semi.efficient_return(target_return=target_return/TRADING_DAYS if target_return else mu.mean()/TRADING_DAYS)
                    weights = ef_semi.clean_weights()
                except Exception as e:
                    st.warning(f"Min CVaR failed: {str(e)[:100]}. Using Min Volatility.")
                    weights = ef.min_volatility()
            
            elif strategy == "Black-Litterman":
                try:
                    # Simplified Black-Litterman implementation
                    if market_weights is None:
                        # Use market cap weights as proxy
                        market_weights = np.ones(len(returns_clean.columns)) / len(returns_clean.columns)
                    
                    # Create implied returns
                    delta = 2.5  # Risk aversion parameter
                    pi = delta * S @ market_weights
                    
                    # Create view matrix (simplified - equal views)
                    P = np.eye(len(returns_clean.columns))
                    Q = pi * 1.1  # 10% higher returns expected
                    
                    if view_confidences is None:
                        view_confidences = np.ones(len(returns_clean.columns)) * 0.1
                    
                    omega = np.diag(view_confidences)
                    
                    # Calculate Black-Litterman returns
                    tau = 0.05
                    M = tau * S
                    try:
                        pi_bl = pi + tau * S @ P.T @ np.linalg.inv(P @ tau * S @ P.T + omega) @ (Q - P @ pi)
                        S_bl = S + M - M @ P.T @ np.linalg.inv(P @ M @ P.T + omega) @ P @ M
                        
                        # Recreate efficient frontier with BL returns
                        ef_bl = EfficientFrontier(pi_bl, S_bl, weight_bounds=(0, 1) if not short_allowed else (-1, 1))
                        weights = ef_bl.max_sharpe(risk_free_rate=risk_free_rate/TRADING_DAYS)
                    except:
                        # Fallback if matrix inversion fails
                        weights = ef.max_sharpe(risk_free_rate=risk_free_rate/TRADING_DAYS)
                        
                except Exception as e:
                    st.warning(f"Black-Litterman failed: {str(e)[:100]}. Using Max Sharpe.")
                    weights = ef.max_sharpe(risk_free_rate=risk_free_rate/TRADING_DAYS)
                    
            else:
                # Default to minimum volatility
                weights = ef.min_volatility()
            
            # Clean weights
            if isinstance(weights, dict):
                cleaned_weights = ef.clean_weights() if hasattr(ef, 'clean_weights') else weights
            else:
                cleaned_weights = ef.clean_weights()
            
            # Convert to array
            weights_array = np.array([cleaned_weights.get(asset, 0) for asset in returns_df.columns])
            
            # Normalize weights (handle rounding errors)
            total_weight = weights_array.sum()
            if abs(total_weight - 1.0) > 0.001:
                weights_array = weights_array / (total_weight + 1e-10)
            
            # Calculate performance metrics
            try:
                if hasattr(ef, 'portfolio_performance'):
                    expected_return, expected_risk, sharpe_ratio = ef.portfolio_performance(
                        risk_free_rate=risk_free_rate/TRADING_DAYS
                    )
                else:
                    # Fallback calculation
                    portfolio_returns = (returns_clean * weights_array).sum(axis=1)
                    expected_return = portfolio_returns.mean() * TRADING_DAYS
                    expected_risk = portfolio_returns.std() * np.sqrt(TRADING_DAYS)
                    if expected_risk > 0:
                        sharpe_ratio = (expected_return - risk_free_rate) / expected_risk
                    else:
                        sharpe_ratio = 0
            except Exception as e:
                # Fallback calculation
                portfolio_returns = (returns_clean * weights_array).sum(axis=1)
                expected_return = portfolio_returns.mean() * TRADING_DAYS
                expected_risk = portfolio_returns.std() * np.sqrt(TRADING_DAYS)
                if expected_risk > 0:
                    sharpe_ratio = (expected_return - risk_free_rate) / expected_risk
                else:
                    sharpe_ratio = 0
            
            return {
                'weights': weights_array,
                'expected_return': expected_return,
                'expected_risk': expected_risk,
                'sharpe_ratio': sharpe_ratio,
                'method': strategy,
                'cleaned_weights': cleaned_weights,
                'success': True,
                'covariance_matrix': S,
                'expected_returns': mu
            }
            
        except Exception as e:
            st.error(f"Optimization error: {str(e)[:200]}")
            import traceback
            traceback.print_exc()
            return PortfolioOptimizer._fallback_weights(returns_df.columns.tolist(), risk_free_rate, returns_df)
    
    @staticmethod
    def _equal_weights(returns_df):
        """Generate equal weights"""
        n_assets = len(returns_df.columns)
        weights = {asset: 1.0/n_assets for asset in returns_df.columns}
        return weights
    
    @staticmethod
    def _fallback_weights(tickers, risk_free_rate, returns_df=None):
        """Fallback to equal weights"""
        n_assets = len(tickers)
        if n_assets == 0:
            return {
                'weights': np.array([]),
                'expected_return': 0,
                'expected_risk': 0,
                'sharpe_ratio': 0,
                'method': 'No Assets',
                'cleaned_weights': {},
                'success': False
            }
        
        equal_weights = np.ones(n_assets) / n_assets
        
        if returns_df is not None and not returns_df.empty:
            port_returns = (returns_df * equal_weights).sum(axis=1)
            ann_return = port_returns.mean() * TRADING_DAYS
            ann_risk = port_returns.std() * np.sqrt(TRADING_DAYS)
            if ann_risk > 0:
                sharpe = (ann_return - risk_free_rate) / ann_risk
            else:
                sharpe = 0
        else:
            ann_return = 0
            ann_risk = 0
            sharpe = 0
        
        return {
            'weights': equal_weights,
            'expected_return': ann_return,
            'expected_risk': ann_risk,
            'sharpe_ratio': sharpe,
            'method': 'Equal Weight (Fallback)',
            'cleaned_weights': {asset: equal_weights[i] for i, asset in enumerate(tickers)},
            'success': False
        }
    
    @staticmethod
    def calculate_efficient_frontier(returns_df: pd.DataFrame, risk_free_rate: float = 0.03, 
                                    n_points: int = 20) -> pd.DataFrame:
        """Calculate efficient frontier points"""
        if not PYPFOPT_AVAILABLE or returns_df.empty or len(returns_df.columns) < 2:
            return pd.DataFrame()
        
        try:
            # Calculate expected returns and covariance
            mu = expected_returns.mean_historical_return(returns_df)
            S = risk_models.sample_cov(returns_df)
            
            ef = EfficientFrontier(mu, S)
            
            # Get min vol portfolio
            min_vol_weights = ef.min_volatility()
            ef.set_weights(min_vol_weights)
            min_vol_return, min_vol_risk, _ = ef.portfolio_performance()
            
            # Get max sharpe portfolio
            max_sharpe_weights = ef.max_sharpe(risk_free_rate=risk_free_rate/TRADING_DAYS)
            ef.set_weights(max_sharpe_weights)
            max_sharpe_return, max_sharpe_risk, _ = ef.portfolio_performance()
            
            # Generate efficient frontier
            target_returns = np.linspace(min_vol_return, max_sharpe_return * 1.5, n_points)
            
            frontier_points = []
            
            for target_return in target_returns:
                try:
                    ef.efficient_return(target_return=target_return/TRADING_DAYS)
                    ret, risk, sharpe = ef.portfolio_performance(
                        risk_free_rate=risk_free_rate/TRADING_DAYS
                    )
                    frontier_points.append({
                        'return': ret,
                        'risk': risk,
                        'sharpe': sharpe,
                        'target_return': target_return
                    })
                except:
                    continue
            
            return pd.DataFrame(frontier_points)
            
        except Exception as e:
            st.warning(f"Could not calculate efficient frontier: {str(e)[:100]}")
            return pd.DataFrame()

# -------------------------------------------------------------
# ENHANCED RISK ANALYTICS (UPDATED)
# -------------------------------------------------------------
class EnhancedRiskAnalytics:
    """Professional risk analytics with multiple VaR methods"""
    
    @staticmethod
    def calculate_var(returns: pd.Series, method: str = "historical", 
                     confidence_level: float = 0.95, 
                     window: int = None, 
                     params: Dict = None) -> Dict:
        """Calculate Value at Risk using multiple methods with enhanced error handling"""
        
        if returns is None or returns.empty:
            return {
                'VaR': np.nan,
                'CVaR': np.nan,
                'method': method,
                'confidence': confidence_level,
                'error': 'Empty returns series',
                'success': False
            }
        
        returns_clean = returns.dropna()
        
        if len(returns_clean) < 20:
            return {
                'VaR': np.nan,
                'CVaR': np.nan,
                'method': method,
                'confidence': confidence_level,
                'error': f'Insufficient data: {len(returns_clean)} points',
                'success': False
            }
        
        try:
            if method == "historical":
                return EnhancedRiskAnalytics._historical_var(returns_clean, confidence_level)
            
            elif method == "parametric":
                return EnhancedRiskAnalytics._parametric_var(returns_clean, confidence_level)
            
            elif method == "ewma":
                return EnhancedRiskAnalytics._ewma_var(returns_clean, confidence_level, window, params)
            
            elif method == "monte_carlo":
                return EnhancedRiskAnalytics._monte_carlo_var(returns_clean, confidence_level, params)
            
            elif method == "modified":
                return EnhancedRiskAnalytics._modified_var(returns_clean, confidence_level)
            
            else:
                return {
                    'VaR': np.nan,
                    'CVaR': np.nan,
                    'method': method,
                    'confidence': confidence_level,
                    'error': 'Unknown method',
                    'success': False
                }
                
        except Exception as e:
            return {
                'VaR': np.nan,
                'CVaR': np.nan,
                'method': method,
                'confidence': confidence_level,
                'error': f'Calculation failed: {str(e)[:100]}',
                'success': False
            }
    
    @staticmethod
    def _historical_var(returns, confidence_level):
        """Historical Simulation VaR"""
        try:
            var = np.percentile(returns, (1 - confidence_level) * 100)
            tail = returns[returns <= var]
            cvar = tail.mean() if len(tail) > 0 else var
            
            return {
                'VaR': var,
                'CVaR': cvar,
                'method': 'Historical Simulation',
                'confidence': confidence_level,
                'observations': len(returns),
                'percentile': (1 - confidence_level) * 100,
                'success': True
            }
        except:
            return {
                'VaR': np.nan,
                'CVaR': np.nan,
                'method': 'Historical Simulation',
                'confidence': confidence_level,
                'error': 'Percentile calculation failed',
                'success': False
            }
    
    @staticmethod
    def _parametric_var(returns, confidence_level):
        """Parametric (Normal Distribution) VaR"""
        try:
            mu = returns.mean()
            sigma = returns.std()
            
            if sigma == 0 or pd.isna(sigma):
                return {
                    'VaR': np.nan,
                    'CVaR': np.nan,
                    'method': 'Parametric (Normal)',
                    'confidence': confidence_level,
                    'error': 'Zero or NaN volatility',
                    'success': False
                }
            
            z_score = stats.norm.ppf(1 - confidence_level)
            var = mu + z_score * sigma
            cvar = mu - (sigma / (1 - confidence_level)) * stats.norm.pdf(z_score)
            
            return {
                'VaR': var,
                'CVaR': cvar,
                'method': 'Parametric (Normal)',
                'confidence': confidence_level,
                'mu': mu,
                'sigma': sigma,
                'z_score': z_score,
                'success': True
            }
        except Exception as e:
            return {
                'VaR': np.nan,
                'CVaR': np.nan,
                'method': 'Parametric (Normal)',
                'confidence': confidence_level,
                'error': f'Calculation failed: {str(e)[:100]}',
                'success': False
            }
    
    @staticmethod
    def _ewma_var(returns, confidence_level, window=None, params=None):
        """EWMA VaR with proper initialization"""
        try:
            if window is None:
                window = min(252, len(returns))
            
            lambda_param = params.get('lambda', 0.94) if params else 0.94
            
            # Calculate EWMA variance with proper initialization
            returns_squared = returns ** 2
            
            # Initialize with simple average of first window
            if len(returns) < window:
                return {
                    'VaR': np.nan,
                    'CVaR': np.nan,
                    'method': 'EWMA Parametric',
                    'confidence': confidence_level,
                    'error': f'Insufficient data for window {window}',
                    'success': False
                }
            
            ewma_var = pd.Series(index=returns.index, dtype=float)
            ewma_var.iloc[:window] = returns_squared.iloc[:window].mean()
            
            # Recursive EWMA calculation
            for i in range(window, len(returns)):
                ewma_var.iloc[i] = lambda_param * ewma_var.iloc[i-1] + (1 - lambda_param) * returns_squared.iloc[i-1]
            
            current_vol = np.sqrt(ewma_var.iloc[-1])
            
            if current_vol == 0 or pd.isna(current_vol):
                return {
                    'VaR': np.nan,
                    'CVaR': np.nan,
                    'method': 'EWMA Parametric',
                    'confidence': confidence_level,
                    'error': 'Zero or NaN volatility',
                    'success': False
                }
            
            z_score = stats.norm.ppf(1 - confidence_level)
            var = z_score * current_vol
            cvar = - (current_vol / (1 - confidence_level)) * stats.norm.pdf(z_score)
            
            return {
                'VaR': var,
                'CVaR': cvar,
                'method': 'EWMA Parametric',
                'confidence': confidence_level,
                'lambda': lambda_param,
                'ewma_vol': current_vol,
                'z_score': z_score,
                'success': True
            }
        except Exception as e:
            return {
                'VaR': np.nan,
                'CVaR': np.nan,
                'method': 'EWMA Parametric',
                'confidence': confidence_level,
                'error': f'Calculation failed: {str(e)[:100]}',
                'success': False
            }
    
    @staticmethod
    def _monte_carlo_var(returns, confidence_level, params=None):
        """Monte Carlo Simulation VaR"""
        try:
            n_simulations = params.get('n_simulations', 10000) if params else 10000
            days = params.get('days', 1) if params else 1
            
            mu = returns.mean()
            sigma = returns.std()
            
            if sigma == 0 or pd.isna(sigma):
                return {
                    'VaR': np.nan,
                    'CVaR': np.nan,
                    'method': 'Monte Carlo Simulation',
                    'confidence': confidence_level,
                    'error': 'Zero or NaN volatility',
                    'success': False
                }
            
            # Set seed for reproducibility
            np.random.seed(42)
            
            # Simulate returns using GBM
            dt = 1/252
            simulations = np.random.normal(mu*dt, sigma*np.sqrt(dt), (days, n_simulations))
            
            # Calculate cumulative returns
            cumulative_returns = np.ones(n_simulations)
            for day in range(days):
                cumulative_returns *= (1 + simulations[day])
            
            final_returns = cumulative_returns - 1
            
            var = np.percentile(final_returns, (1 - confidence_level) * 100)
            tail = final_returns[final_returns <= var]
            cvar = tail.mean() if len(tail) > 0 else var
            
            return {
                'VaR': var,
                'CVaR': cvar,
                'method': 'Monte Carlo Simulation',
                'confidence': confidence_level,
                'simulations': n_simulations,
                'days': days,
                'mu': mu,
                'sigma': sigma,
                'success': True
            }
        except Exception as e:
            return {
                'VaR': np.nan,
                'CVaR': np.nan,
                'method': 'Monte Carlo Simulation',
                'confidence': confidence_level,
                'error': f'Calculation failed: {str(e)[:100]}',
                'success': False
            }
    
    @staticmethod
    def _modified_var(returns, confidence_level):
        """Modified VaR using Cornish-Fisher expansion"""
        try:
            mu = returns.mean()
            sigma = returns.std()
            skew = stats.skew(returns)
            kurt = stats.kurtosis(returns)
            
            if sigma == 0 or pd.isna(sigma):
                return {
                    'VaR': np.nan,
                    'CVaR': np.nan,
                    'method': 'Modified (Cornish-Fisher)',
                    'confidence': confidence_level,
                    'error': 'Zero or NaN volatility',
                    'success': False
                }
            
            # Cornish-Fisher expansion
            z = stats.norm.ppf(1 - confidence_level)
            z_cf = z + (1/6) * (z**2 - 1) * skew + (1/24) * (z**3 - 3*z) * kurt - (1/36) * (2*z**3 - 5*z) * skew**2
            
            var = mu + z_cf * sigma
            # CVaR approximation for modified distribution
            cvar = mu - sigma * (stats.norm.pdf(z) / (1 - confidence_level)) * (1 + (skew/6)*(z**2 - 1) + (kurt/24)*(z**3 - 3*z))
            
            return {
                'VaR': var,
                'CVaR': cvar,
                'method': 'Modified (Cornish-Fisher)',
                'confidence': confidence_level,
                'mu': mu,
                'sigma': sigma,
                'skewness': skew,
                'kurtosis': kurt,
                'z_cf': z_cf,
                'success': True
            }
        except Exception as e:
            return {
                'VaR': np.nan,
                'CVaR': np.nan,
                'method': 'Modified (Cornish-Fisher)',
                'confidence': confidence_level,
                'error': f'Calculation failed: {str(e)[:100]}',
                'success': False
            }
    
    @staticmethod
    def calculate_all_var_methods(returns: pd.Series, confidence_level: float = 0.95) -> pd.DataFrame:
        """Calculate VaR using all available methods"""
        methods = ["historical", "parametric", "ewma", "monte_carlo", "modified"]
        results = []
        
        for method in methods:
            result = EnhancedRiskAnalytics.calculate_var(returns, method, confidence_level)
            results.append({
                'Method': result['method'],
                'VaR': result['VaR'],
                'CVaR': result['CVaR'],
                'Status': '✅ Success' if result.get('success', False) else '❌ Failed',
                'Details': result.get('error', 'Calculated successfully')
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def calculate_stress_test(returns: pd.DataFrame, stress_scenarios: Dict = None) -> pd.DataFrame:
        """Calculate stress test results for different scenarios"""
        if stress_scenarios is None:
            stress_scenarios = {
                'Market Crash (-10%)': -0.10,
                'Market Correction (-5%)': -0.05,
                'Bear Market (-20%)': -0.20,
                'Flash Crash (-30%)': -0.30,
                'Moderate Decline (-3%)': -0.03
            }
        
        results = []
        for scenario_name, stress_return in stress_scenarios.items():
            # Apply stress scenario to returns
            stressed_returns = returns * (1 + stress_return)
            
            # Calculate portfolio impact
            portfolio_returns = stressed_returns.mean(axis=1)
            total_return = (1 + portfolio_returns).prod() - 1
            
            results.append({
                'Scenario': scenario_name,
                'Stress Return': f"{stress_return:.1%}",
                'Portfolio Impact': f"{total_return:.2%}",
                'Worst Asset': returns.columns[returns.min().idxmin()],
                'Best Asset': returns.columns[returns.max().idxmax()]
            })
        
        return pd.DataFrame(results)

# -------------------------------------------------------------
# FIXED EWMA ANALYSIS (NO RENDERING ISSUES)
# -------------------------------------------------------------
class EWMAAnalysis:
    """Enhanced EWMA volatility analysis with proper error handling"""
    
    @staticmethod
    def calculate_ewma_volatility(returns: pd.DataFrame, lambda_param: float = 0.94) -> pd.DataFrame:
        """Calculate EWMA volatility for multiple assets with robust initialization"""
        if returns.empty:
            return pd.DataFrame()
        
        ewma_vol = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        for asset in returns.columns:
            r = returns[asset].dropna()
            if len(r) < 20:
                ewma_vol[asset] = np.nan
                continue
            
            try:
                # Initialize EWMA variance with simple average of squared returns
                init_window = min(30, len(r))
                if init_window < 10:
                    ewma_vol[asset] = r.rolling(20).std()
                    continue
                
                init_var = r.iloc[:init_window].var()
                
                if pd.isna(init_var) or init_var <= 0:
                    ewma_vol[asset] = r.rolling(20).std()
                    continue
                
                ewma_var = pd.Series(index=r.index, dtype=float)
                ewma_var.iloc[:init_window] = init_var
                
                # Calculate squared returns
                r_squared = r ** 2
                
                # Recursive EWMA calculation
                for i in range(init_window, len(r)):
                    prev_var = ewma_var.iloc[i-1]
                    if not pd.isna(prev_var):
                        ewma_var.iloc[i] = lambda_param * prev_var + (1 - lambda_param) * r_squared.iloc[i-1]
                    else:
                        ewma_var.iloc[i] = init_var
                
                # Calculate volatility (sqrt of variance)
                ewma_vol[asset] = np.sqrt(ewma_var) * np.sqrt(252)  # Annualize
                
                # Handle any remaining NaN values
                ewma_vol[asset] = ewma_vol[asset].ffill().bfill()
                
            except Exception as e:
                # Fallback to rolling standard deviation
                ewma_vol[asset] = r.rolling(20).std() * np.sqrt(252)
        
        # Drop columns that are all NaN
        ewma_vol = ewma_vol.dropna(axis=1, how='all')
        
        # Fill any remaining NaN values with column mean
        for col in ewma_vol.columns:
            if ewma_vol[col].isna().any():
                col_mean = ewma_vol[col].mean()
                if not pd.isna(col_mean):
                    ewma_vol[col] = ewma_vol[col].fillna(col_mean)
        
        return ewma_vol
    
    @staticmethod
    def calculate_ewma_correlation(returns: pd.DataFrame, lambda_param: float = 0.94) -> pd.DataFrame:
        """Calculate EWMA correlation matrix"""
        if returns.empty or len(returns.columns) < 2:
            return pd.DataFrame()
        
        # Calculate EWMA covariance
        n_assets = len(returns.columns)
        ewma_cov = pd.DataFrame(np.zeros((n_assets, n_assets)), 
                               index=returns.columns, columns=returns.columns)
        
        # Initialize with simple covariance
        init_window = min(30, len(returns))
        if init_window > 10:
            init_cov = returns.iloc[:init_window].cov()
        else:
            init_cov = returns.cov()
        
        # Calculate EWMA for each pair
        for i, asset_i in enumerate(returns.columns):
            for j, asset_j in enumerate(returns.columns):
                if i <= j:  # Calculate only upper triangle
                    r_i = returns[asset_i].dropna()
                    r_j = returns[asset_j].dropna()
                    
                    # Align indices
                    common_idx = r_i.index.intersection(r_j.index)
                    if len(common_idx) < 20:
                        continue
                    
                    r_i_aligned = r_i.loc[common_idx]
                    r_j_aligned = r_j.loc[common_idx]
                    
                    # Calculate product of returns
                    product_returns = r_i_aligned * r_j_aligned
                    
                    # EWMA of product
                    ewma_product = pd.Series(index=common_idx, dtype=float)
                    init_value = product_returns.iloc[:init_window].mean()
                    ewma_product.iloc[:init_window] = init_value
                    
                    for k in range(init_window, len(common_idx)):
                        ewma_product.iloc[k] = lambda_param * ewma_product.iloc[k-1] + (1 - lambda_param) * product_returns.iloc[k-1]
                    
                    ewma_cov.iloc[i, j] = ewma_product.iloc[-1]
                    ewma_cov.iloc[j, i] = ewma_cov.iloc[i, j]
        
        # Calculate correlation from covariance
        ewma_corr = pd.DataFrame(np.zeros((n_assets, n_assets)), 
                                index=returns.columns, columns=returns.columns)
        
        for i, asset_i in enumerate(returns.columns):
            for j, asset_j in enumerate(returns.columns):
                if i <= j:
                    var_i = ewma_cov.iloc[i, i]
                    var_j = ewma_cov.iloc[j, j]
                    if var_i > 0 and var_j > 0:
                        corr = ewma_cov.iloc[i, j] / np.sqrt(var_i * var_j)
                        ewma_corr.iloc[i, j] = corr
                        ewma_corr.iloc[j, i] = corr
        
        return ewma_corr

# -------------------------------------------------------------
# ENHANCED PORTFOLIO ANALYZER
# -------------------------------------------------------------
class PortfolioAnalyzer:
    """Comprehensive portfolio analysis with enhanced features"""
    
    @staticmethod
    def analyze_portfolio(prices: pd.DataFrame, weights: np.ndarray, 
                         benchmark_prices: pd.Series = None) -> Dict:
        """Comprehensive portfolio analysis"""
        
        analysis = {}
        
        try:
            # Calculate returns
            returns = prices.pct_change().dropna()
            portfolio_returns = (returns * weights).sum(axis=1)
            
            analysis['returns'] = portfolio_returns
            analysis['cumulative_returns'] = (1 + portfolio_returns).cumprod()
            
            # Calculate metrics
            benchmark_returns = None
            if benchmark_prices is not None:
                benchmark_returns = benchmark_prices.pct_change().dropna()
            
            analysis['metrics'] = PerformanceMetrics.calculate_metrics(
                portfolio_returns, benchmark_returns
            )
            
            # Calculate rolling metrics
            if len(portfolio_returns) > 63:
                analysis['rolling_volatility'] = portfolio_returns.rolling(window=63).std() * np.sqrt(252)
                analysis['rolling_sharpe'] = (portfolio_returns.rolling(window=63).mean() * 252 - 0.03) / (
                    analysis['rolling_volatility'] + 1e-10
                )
                analysis['rolling_sortino'] = PortfolioAnalyzer._calculate_rolling_sortino(
                    portfolio_returns, window=63
                )
            else:
                analysis['rolling_volatility'] = pd.Series(dtype=float)
                analysis['rolling_sharpe'] = pd.Series(dtype=float)
                analysis['rolling_sortino'] = pd.Series(dtype=float)
            
            # Calculate drawdowns
            cumulative = analysis['cumulative_returns']
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            
            analysis['drawdown'] = drawdown
            analysis['underwater'] = cumulative < running_max
            
            # Calculate monthly returns
            if len(portfolio_returns) > 20:
                monthly_returns = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                analysis['monthly_returns'] = monthly_returns
                
                # Monthly performance statistics
                analysis['monthly_stats'] = {
                    'mean': monthly_returns.mean(),
                    'std': monthly_returns.std(),
                    'min': monthly_returns.min(),
                    'max': monthly_returns.max(),
                    'positive_months': (monthly_returns > 0).sum(),
                    'total_months': len(monthly_returns)
                }
            else:
                analysis['monthly_returns'] = pd.Series(dtype=float)
                analysis['monthly_stats'] = {}
            
            # Calculate quarterly returns
            if len(portfolio_returns) > 60:
                quarterly_returns = portfolio_returns.resample('Q').apply(lambda x: (1 + x).prod() - 1)
                analysis['quarterly_returns'] = quarterly_returns
            else:
                analysis['quarterly_returns'] = pd.Series(dtype=float)
            
            # Calculate annual returns
            if len(portfolio_returns) > 252:
                annual_returns = portfolio_returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
                analysis['annual_returns'] = annual_returns
                
                # Annual performance statistics
                analysis['annual_stats'] = {
                    'mean': annual_returns.mean(),
                    'std': annual_returns.std(),
                    'min': annual_returns.min(),
                    'max': annual_returns.max(),
                    'positive_years': (annual_returns > 0).sum(),
                    'total_years': len(annual_returns)
                }
            else:
                analysis['annual_returns'] = pd.Series(dtype=float)
                analysis['annual_stats'] = {}
            
            # Calculate rolling correlations with benchmark
            if benchmark_returns is not None and len(portfolio_returns) > 20:
                aligned_benchmark = benchmark_returns.reindex(portfolio_returns.index).dropna()
                aligned_portfolio = portfolio_returns.reindex(aligned_benchmark.index)
                
                if len(aligned_portfolio) > 60:
                    rolling_corr = aligned_portfolio.rolling(window=60).corr(aligned_benchmark)
                    analysis['rolling_correlation'] = rolling_corr
            
            # Calculate beta
            if benchmark_returns is not None:
                aligned_benchmark = benchmark_returns.reindex(portfolio_returns.index).dropna()
                aligned_portfolio = portfolio_returns.reindex(aligned_benchmark.index)
                
                if len(aligned_portfolio) > 10:
                    cov_matrix = np.cov(aligned_portfolio, aligned_benchmark)
                    if cov_matrix[1, 1] > 0:
                        analysis['beta'] = cov_matrix[0, 1] / cov_matrix[1, 1]
            
            # Calculate tracking error
            if benchmark_returns is not None:
                aligned_benchmark = benchmark_returns.reindex(portfolio_returns.index).dropna()
                aligned_portfolio = portfolio_returns.reindex(aligned_benchmark.index)
                
                if len(aligned_portfolio) > 10:
                    active_returns = aligned_portfolio - aligned_benchmark
                    analysis['tracking_error'] = active_returns.std() * np.sqrt(252)
            
            # Calculate turnover (simplified)
            if len(weights) > 1 and 'previous_weights' in analysis:
                turnover = np.abs(weights - analysis['previous_weights']).sum() / 2
                analysis['turnover'] = turnover
            
        except Exception as e:
            st.error(f"Error in portfolio analysis: {str(e)[:200]}")
        
        return analysis
    
    @staticmethod
    def _calculate_rolling_sortino(returns, window=63, risk_free_rate=0.03):
        """Calculate rolling Sortino ratio"""
        if len(returns) < window:
            return pd.Series(dtype=float)
        
        rolling_sortino = pd.Series(index=returns.index, dtype=float)
        
        for i in range(window, len(returns) + 1):
            window_returns = returns.iloc[i-window:i]
            
            # Calculate downside deviation
            downside_returns = window_returns[window_returns < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std() * np.sqrt(252)
                if downside_std > 0:
                    excess_return = window_returns.mean() * 252 - risk_free_rate
                    rolling_sortino.iloc[i-1] = excess_return / downside_std
        
        return rolling_sortino

# -------------------------------------------------------------
# FIXED PERFORMANCE ATTRIBUTION (NO LENGTH MISMATCH)
# -------------------------------------------------------------
class PerformanceAttribution:
    """Professional performance attribution analysis with error handling"""
    
    @staticmethod
    def calculate_brinson_attribution(portfolio_returns: pd.Series, 
                                     benchmark_returns: pd.Series,
                                     portfolio_weights: np.ndarray,
                                     benchmark_weights: np.ndarray,
                                     asset_returns: pd.DataFrame) -> Dict:
        """Calculate Brinson attribution analysis with validation - FIXED LENGTH MISMATCH"""
        
        # Validate inputs
        if (portfolio_returns is None or benchmark_returns is None or 
            portfolio_weights is None or benchmark_weights is None or 
            asset_returns is None):
            return {
                'total_active_return': np.nan,
                'allocation_effect': np.nan,
                'selection_effect': np.nan,
                'interaction_effect': np.nan,
                'residual': np.nan,
                'success': False,
                'error': 'Missing input data'
            }
        
        try:
            # Ensure weights sum to 1
            portfolio_weights = portfolio_weights / (portfolio_weights.sum() + 1e-10)
            benchmark_weights = benchmark_weights / (benchmark_weights.sum() + 1e-10)
            
            # FIX: Align lengths properly
            n_assets = len(portfolio_weights)
            asset_names = asset_returns.columns.tolist()
            
            if len(asset_names) != n_assets:
                # If mismatch, try to align
                st.warning(f"Weight length ({n_assets}) doesn't match asset count ({len(asset_names)}). Attempting alignment.")
                # Use only common assets
                common_assets = [col for col in asset_names if col in asset_names[:n_assets]]
                if len(common_assets) < 2:
                    return {
                        'total_active_return': np.nan,
                        'allocation_effect': np.nan,
                        'selection_effect': np.nan,
                        'interaction_effect': np.nan,
                        'residual': np.nan,
                        'success': False,
                        'error': f'Cannot align: weights={n_assets}, assets={len(asset_names)}'
                    }
                asset_returns = asset_returns[common_assets]
                portfolio_weights = portfolio_weights[:len(common_assets)]
                benchmark_weights = benchmark_weights[:len(common_assets)]
                n_assets = len(common_assets)
            
            # Calculate total active return
            total_active_return = portfolio_returns.mean() - benchmark_returns.mean()
            
            # Calculate asset returns mean
            asset_means = asset_returns.mean()
            benchmark_mean = benchmark_returns.mean()
            
            # Ensure proper alignment
            if len(asset_means) != n_assets:
                asset_means = asset_means.iloc[:n_assets]
            
            # Allocation effect
            allocation_effect = np.sum(
                (portfolio_weights - benchmark_weights) * 
                (asset_means.values - benchmark_mean)
            )
            
            # Selection effect
            selection_effect = np.sum(
                benchmark_weights * 
                (asset_means.values - benchmark_mean)
            )
            
            # Interaction effect
            interaction_effect = np.sum(
                (portfolio_weights - benchmark_weights) * 
                (asset_means.values - benchmark_mean)
            ) - allocation_effect
            
            # Residual (should be close to 0)
            residual = total_active_return - (allocation_effect + selection_effect + interaction_effect)
            
            return {
                'total_active_return': total_active_return,
                'allocation_effect': allocation_effect,
                'selection_effect': selection_effect,
                'interaction_effect': interaction_effect,
                'residual': residual,
                'success': True,
                'asset_count': n_assets
            }
            
        except Exception as e:
            return {
                'total_active_return': np.nan,
                'allocation_effect': np.nan,
                'selection_effect': np.nan,
                'interaction_effect': np.nan,
                'residual': np.nan,
                'success': False,
                'error': f'Calculation failed: {str(e)[:100]}'
            }

# -------------------------------------------------------------
# MONTE CARLO SIMULATION ENGINE (UPDATED)
# -------------------------------------------------------------
class MonteCarloSimulator:
    """Professional Monte Carlo simulation engine with error handling"""
    
    @staticmethod
    def simulate_gbm(S0: float, mu: float, sigma: float, 
                    T: float = 1.0, n_steps: int = 252,
                    n_sims: int = 10000, seed: int = 42) -> np.ndarray:
        """Geometric Brownian Motion simulation with validation"""
        
        # Validate inputs
        if S0 <= 0 or sigma < 0 or n_steps <= 0 or n_sims <= 0:
            raise ValueError("Invalid input parameters for GBM simulation")
        
        np.random.seed(seed)
        
        dt = T / n_steps
        paths = np.zeros((n_steps + 1, n_sims))
        paths[0] = S0
        
        # Pre-calculate drift and diffusion terms
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt)
        
        # Vectorized simulation for speed
        rand_nums = np.random.randn(n_steps, n_sims)
        for t in range(1, n_steps + 1):
            paths[t] = paths[t-1] * np.exp(drift + diffusion * rand_nums[t-1])
        
        return paths
    
    @staticmethod
    def simulate_multivariate_gbm(S0: np.ndarray, mu: np.ndarray, cov: np.ndarray,
                                T: float = 1.0, n_steps: int = 252,
                                n_sims: int = 10000, seed: int = 42) -> np.ndarray:
        """Multivariate GBM simulation with correlation"""
        
        n_assets = len(S0)
        
        np.random.seed(seed)
        dt = T / n_steps
        
        # Decompose covariance matrix
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            # If not positive definite, use eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-10)
            L = eigvecs @ np.diag(np.sqrt(eigvals))
        
        # Initialize paths
        paths = np.zeros((n_assets, n_steps + 1, n_sims))
        paths[:, 0, :] = S0.reshape(-1, 1)
        
        # Vectorized simulation
        for t in range(1, n_steps + 1):
            # Generate correlated random numbers
            Z = np.random.randn(n_assets, n_sims)
            correlated_Z = L @ Z
            
            # Update prices
            for i in range(n_assets):
                drift = (mu[i] - 0.5 * cov[i, i]) * dt
                diffusion = np.sqrt(cov[i, i]) * np.sqrt(dt)
                paths[i, t, :] = paths[i, t-1, :] * np.exp(drift + diffusion * correlated_Z[i, :])
        
        return paths
    
    @staticmethod
    def calculate_var_cvar(paths: np.ndarray, alpha: float = 0.95) -> Tuple[float, float]:
        """Calculate VaR and CVaR from simulation paths"""
        if paths is None or paths.size == 0:
            return np.nan, np.nan
        
        try:
            final_returns = (paths[-1] / paths[0]) - 1
            var = np.percentile(final_returns, (1 - alpha) * 100)
            tail = final_returns[final_returns <= var]
            cvar = tail.mean() if len(tail) > 0 else var
            
            return var, cvar
        except:
            return np.nan, np.nan
    
    @staticmethod
    def simulate_portfolio_scenarios(portfolio_weights: np.ndarray, 
                                   asset_returns: pd.DataFrame,
                                   n_sims: int = 10000, 
                                   horizon_days: int = 252,
                                   seed: int = 42) -> Dict:
        """Simulate portfolio scenarios using Monte Carlo"""
        
        n_assets = len(portfolio_weights)
        
        if n_assets == 0 or asset_returns.empty:
            return {}
        
        # Calculate asset parameters
        asset_means = asset_returns.mean() * 252  # Annualize
        asset_cov = asset_returns.cov() * 252  # Annualize
        
        # Simulate multivariate GBM
        S0 = np.ones(n_assets)  # Starting with unit prices
        paths = MonteCarloSimulator.simulate_multivariate_gbm(
            S0, asset_means.values, asset_cov.values,
            T=horizon_days/252, n_steps=horizon_days,
            n_sims=n_sims, seed=seed
        )
        
        # Calculate portfolio paths
        portfolio_paths = np.zeros((horizon_days + 1, n_sims))
        for t in range(horizon_days + 1):
            portfolio_values = paths[:, t, :].T @ portfolio_weights
            portfolio_paths[t, :] = portfolio_values
        
        # Calculate statistics
        final_returns = (portfolio_paths[-1] / portfolio_paths[0]) - 1
        
        results = {
            'paths': portfolio_paths,
            'final_returns': final_returns,
            'mean_return': np.mean(final_returns),
            'median_return': np.median(final_returns),
            'std_return': np.std(final_returns),
            'min_return': np.min(final_returns),
            'max_return': np.max(final_returns),
            'var_95': np.percentile(final_returns, 5),
            'var_99': np.percentile(final_returns, 1),
            'cvar_95': np.mean(final_returns[final_returns <= np.percentile(final_returns, 5)]),
            'cvar_99': np.mean(final_returns[final_returns <= np.percentile(final_returns, 1)]),
            'probability_loss': np.mean(final_returns < 0),
            'probability_gain_10': np.mean(final_returns > 0.10),
            'probability_gain_20': np.mean(final_returns > 0.20)
        }
        
        return results

# -------------------------------------------------------------
# ENHANCED REPORT GENERATOR
# -------------------------------------------------------------
class ReportGenerator:
    """Generate comprehensive portfolio reports with export functionality"""
    
    @staticmethod
    def generate_summary_report(portfolio_analysis: Dict, benchmark_name: str = "Benchmark") -> str:
        """Generate markdown summary report"""
        
        metrics = portfolio_analysis.get('metrics', {})
        monthly_stats = portfolio_analysis.get('monthly_stats', {})
        annual_stats = portfolio_analysis.get('annual_stats', {})
        
        # Format numbers for display
        def format_pct(value):
            if pd.isna(value):
                return "N/A"
            return f"{value:.2%}"
        
        def format_num(value):
            if pd.isna(value):
                return "N/A"
            return f"{value:.2f}"
        
        report = f"""
# 📊 Portfolio Performance Summary
## Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 📈 Key Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Annual Return** | {format_pct(metrics.get('annual_return', 0))} | Average annual compounded return |
| **Annual Volatility** | {format_pct(metrics.get('annual_volatility', 0))} | Standard deviation of returns (risk measure) |
| **Sharpe Ratio** | {format_num(metrics.get('sharpe_ratio', 0))} | Risk-adjusted return (higher is better) |
| **Sortino Ratio** | {format_num(metrics.get('sortino_ratio', 0))} | Downside risk-adjusted return |
| **Maximum Drawdown** | {format_pct(metrics.get('max_drawdown', 0))} | Worst historical peak-to-trough decline |
| **Calmar Ratio** | {format_num(metrics.get('calmar_ratio', 0))} | Return relative to maximum drawdown |
| **Win Rate** | {format_pct(metrics.get('win_rate', 0))} | Percentage of profitable periods |

## ⚠️ Risk Metrics

| Risk Measure | 95% Confidence | 99% Confidence |
|--------------|----------------|----------------|
| **Value at Risk (VaR)** | {format_pct(metrics.get('var_95', 0))} | {format_pct(metrics.get('var_99', 0))} |
| **Conditional VaR (CVaR)** | {format_pct(metrics.get('cvar_95', 0))} | {format_pct(metrics.get('cvar_99', 0))} |
| **Ulcer Index** | {format_num(metrics.get('ulcer_index', 0))} | Measure of downside volatility |
| **Pain Index** | {format_pct(metrics.get('pain_index', 0))} | Average drawdown severity |

## 📊 Distribution Statistics

| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| **Skewness** | {format_num(metrics.get('skewness', 0))} | {ReportGenerator._interpret_skewness(metrics.get('skewness', 0))} |
| **Kurtosis** | {format_num(metrics.get('kurtosis', 0))} | {ReportGenerator._interpret_kurtosis(metrics.get('kurtosis', 0))} |
| **Profit Factor** | {format_num(metrics.get('profit_factor', 0))} | Gross profit / gross loss |
| **Gain/Loss Ratio** | {format_num(metrics.get('gain_loss_ratio', 0))} | Average gain / average loss |
| **Omega Ratio** | {format_num(metrics.get('omega_ratio', 0))} | Probability-weighted ratio of gains vs losses |

## 📅 Time Period Performance

### Monthly Statistics
- **Average Monthly Return**: {format_pct(monthly_stats.get('mean', 0))}
- **Monthly Volatility**: {format_pct(monthly_stats.get('std', 0))}
- **Best Month**: {format_pct(monthly_stats.get('max', 0))}
- **Worst Month**: {format_pct(monthly_stats.get('min', 0))}
- **Positive Months**: {monthly_stats.get('positive_months', 0)} / {monthly_stats.get('total_months', 0)} ({format_pct(monthly_stats.get('positive_months', 0)/max(monthly_stats.get('total_months', 1), 1))})

### Annual Statistics
- **Average Annual Return**: {format_pct(annual_stats.get('mean', 0))}
- **Annual Volatility**: {format_pct(annual_stats.get('std', 0))}
- **Best Year**: {format_pct(annual_stats.get('max', 0))}
- **Worst Year**: {format_pct(annual_stats.get('min', 0))}
- **Positive Years**: {annual_stats.get('positive_years', 0)} / {annual_stats.get('total_years', 0)}

## 🎯 Portfolio Characteristics

| Characteristic | Value |
|----------------|-------|
| **Beta (vs {benchmark_name})** | {format_num(portfolio_analysis.get('beta', 0))} |
| **Tracking Error** | {format_pct(portfolio_analysis.get('tracking_error', 0))} |
| **Information Ratio** | {format_num(metrics.get('information_ratio', 0))} |
| **Up Capture** | {format_num(metrics.get('up_capture', 0))} |
| **Down Capture** | {format_num(metrics.get('down_capture', 0))} |
| **Treynor Ratio** | {format_num(metrics.get('treynor_ratio', 0))} |

## 📈 Return Profile Analysis

| Metric | Value |
|--------|-------|
| **Average Win** | {format_pct(metrics.get('avg_win', 0))} |
| **Average Loss** | {format_pct(metrics.get('avg_loss', 0))} |
| **Recovery Period** | {metrics.get('recovery_period', 'N/A')} days |
| **Burke Ratio** | {format_num(metrics.get('burke_ratio', 0))} |
| **Sterling Ratio** | {format_num(metrics.get('sterling_ratio', 0))} |
| **M2 Measure** | {format_pct(metrics.get('m2_measure', 0))} |

---

### 📋 Report Notes:
- All returns are annualized where applicable
- Risk-free rate assumed at 3.00% annual
- Trading days per year: 252
- Report generated for informational purposes only
- Past performance does not guarantee future results

### 🎯 Key Takeaways:
1. **Risk-Adjusted Performance**: {ReportGenerator._assess_sharpe(metrics.get('sharpe_ratio', 0))}
2. **Downside Protection**: {ReportGenerator._assess_sortino(metrics.get('sortino_ratio', 0))}
3. **Drawdown Management**: {ReportGenerator._assess_drawdown(metrics.get('max_drawdown', 0))}
4. **Consistency**: {ReportGenerator._assess_consistency(metrics.get('win_rate', 0))}

---

*This report was generated by Apollo/ENIGMA Portfolio Terminal v5.1*
*For institutional use only. Not financial advice.*
"""
        
        return report
    
    @staticmethod
    def _interpret_skewness(value: float) -> str:
        if pd.isna(value):
            return "Not available"
        if value > 0.5:
            return "Right-skewed (more frequent large gains)"
        elif value < -0.5:
            return "Left-skewed (more frequent large losses)"
        else:
            return "Approximately symmetric"
    
    @staticmethod
    def _interpret_kurtosis(value: float) -> str:
        if pd.isna(value):
            return "Not available"
        if value > 3.5:
            return "Leptokurtic (fat tails, more extreme events)"
        elif value < 2.5:
            return "Platykurtic (thin tails, fewer extremes)"
        else:
            return "Normal distribution (kurtosis ≈ 3)"
    
    @staticmethod
    def _assess_sharpe(sharpe: float) -> str:
        if pd.isna(sharpe):
            return "Sharpe ratio not available"
        if sharpe > 1.5:
            return "Excellent risk-adjusted returns"
        elif sharpe > 1.0:
            return "Good risk-adjusted returns"
        elif sharpe > 0.5:
            return "Moderate risk-adjusted returns"
        elif sharpe > 0:
            return "Poor risk-adjusted returns"
        else:
            return "Negative risk-adjusted returns"
    
    @staticmethod
    def _assess_sortino(sortino: float) -> str:
        if pd.isna(sortino):
            return "Sortino ratio not available"
        if sortino > 2.0:
            return "Excellent downside protection"
        elif sortino > 1.5:
            return "Good downside protection"
        elif sortino > 1.0:
            return "Moderate downside protection"
        elif sortino > 0:
            return "Poor downside protection"
        else:
            return "Negative downside protection"
    
    @staticmethod
    def _assess_drawdown(drawdown: float) -> str:
        if pd.isna(drawdown):
            return "Drawdown data not available"
        if drawdown > -0.10:
            return "Excellent drawdown control"
        elif drawdown > -0.20:
            return "Good drawdown control"
        elif drawdown > -0.30:
            return "Moderate drawdown control"
        else:
            return "Poor drawdown control"
    
    @staticmethod
    def _assess_consistency(win_rate: float) -> str:
        if pd.isna(win_rate):
            return "Consistency data not available"
        if win_rate > 0.60:
            return "Highly consistent positive returns"
        elif win_rate > 0.55:
            return "Consistent positive returns"
        elif win_rate > 0.50:
            return "Moderately consistent"
        elif win_rate > 0.45:
            return "Somewhat inconsistent"
        else:
            return "Inconsistent performance"
    
    @staticmethod
    def generate_html_report(portfolio_analysis: Dict, weights: Dict, 
                           benchmark_name: str = "Benchmark") -> str:
        """Generate HTML report for download"""
        
        metrics = portfolio_analysis.get('metrics', {})
        
        html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Portfolio Performance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 30px; }}
        .section {{ margin-bottom: 30px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }}
        .metric-card {{ background: #f5f5f5; padding: 15px; border-radius: 5px; border-left: 4px solid #1a5fb4; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #1a5fb4; }}
        .metric-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f5f5f5; }}
        .footer {{ margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Portfolio Performance Report</h1>
        <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    
    <div class="section">
        <h2>📈 Key Performance Metrics</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{metrics.get('annual_return', 0):.2%}</div>
                <div class="metric-label">Annual Return</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('annual_volatility', 0):.2%}</div>
                <div class="metric-label">Annual Volatility</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('sharpe_ratio', 0):.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>⚖️ Risk Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>95% Confidence</th>
                <th>99% Confidence</th>
            </tr>
            <tr>
                <td>Value at Risk (VaR)</td>
                <td>{metrics.get('var_95', 0):.2%}</td>
                <td>{metrics.get('var_99', 0):.2%}</td>
            </tr>
            <tr>
                <td>Conditional VaR (CVaR)</td>
                <td>{metrics.get('cvar_95', 0):.2%}</td>
                <td>{metrics.get('cvar_99', 0):.2%}</td>
            </tr>
        </table>
    </div>
    
    <div class="section">
        <h2>📊 Portfolio Weights</h2>
        <table>
            <tr>
                <th>Asset</th>
                <th>Weight</th>
            </tr>
            {"".join([f"<tr><td>{asset}</td><td>{weight:.2%}</td></tr>" for asset, weight in weights.items()])}
        </table>
    </div>
    
    <div class="footer">
        <p>This report was generated by Apollo/ENIGMA Portfolio Terminal v5.1</p>
        <p>For informational purposes only. Past performance does not guarantee future results.</p>
    </div>
</body>
</html>
"""
        
        return html_report

# -------------------------------------------------------------
# ENHANCED ERROR HANDLING AND LOGGING
# -------------------------------------------------------------
class ErrorHandler:
    """Enhanced error handling and logging with session state management"""
    
    @staticmethod
    def handle_error(e: Exception, context: str = ""):
        """Handle errors gracefully and log them"""
        error_msg = f"Error in {context}: {str(e)}"
        
        # Log to session state for debugging
        if 'error_log' not in st.session_state:
            st.session_state.error_log = []
        
        st.session_state.error_log.append({
            'timestamp': datetime.now(),
            'context': context,
            'error': str(e),
            'traceback': traceback.format_exc()
        })
        
        # Keep only last 100 errors
        if len(st.session_state.error_log) > 100:
            st.session_state.error_log = st.session_state.error_log[-100:]
        
        # Display user-friendly error
        st.error(f"❌ {error_msg}")
        
        # Add expander for technical details
        with st.expander("🔧 Technical Details"):
            st.code(traceback.format_exc())
    
    @staticmethod
    def get_error_log():
        """Get error log from session state"""
        return st.session_state.get('error_log', [])
    
    @staticmethod
    def clear_error_log():
        """Clear error log"""
        if 'error_log' in st.session_state:
            st.session_state.error_log = []
    
    @staticmethod
    def log_performance(context: str, start_time: float):
        """Log performance metrics"""
        elapsed = time.time() - start_time
        if 'performance_log' not in st.session_state:
            st.session_state.performance_log = []
        
        st.session_state.performance_log.append({
            'timestamp': datetime.now(),
            'context': context,
            'elapsed_seconds': elapsed
        })
        
        # Keep only last 100 performance logs
        if len(st.session_state.performance_log) > 100:
            st.session_state.performance_log = st.session_state.performance_log[-100:]

# -------------------------------------------------------------
# ENHANCED VISUALIZATION UTILITIES
# -------------------------------------------------------------
class VisualizationUtils:
    """Enhanced visualization utilities for better charts"""
    
    @staticmethod
    def create_performance_chart(portfolio_cumulative: pd.Series, 
                                benchmark_cumulative: pd.Series,
                                title: str = "Portfolio Performance") -> go.Figure:
        """Create enhanced performance chart"""
        
        fig = go.Figure()
        
        # Portfolio line
        fig.add_trace(go.Scatter(
            x=portfolio_cumulative.index,
            y=portfolio_cumulative.values,
            name="Portfolio",
            line=dict(color='#1a5fb4', width=3),
            fill='tozeroy',
            fillcolor='rgba(26, 95, 180, 0.1)',
            hovertemplate='%{x|%Y-%m-%d}<br>Value: %{y:.2f}<extra></extra>'
        ))
        
        # Benchmark line
        if benchmark_cumulative is not None and not benchmark_cumulative.empty:
            fig.add_trace(go.Scatter(
                x=benchmark_cumulative.index,
                y=benchmark_cumulative.values,
                name="Benchmark",
                line=dict(color='#26a269', width=2, dash='dash'),
                hovertemplate='%{x|%Y-%m-%d}<br>Value: %{y:.2f}<extra></extra>'
            ))
        
        # Add drawdown shading
        running_max = portfolio_cumulative.expanding().max()
        drawdown = (portfolio_cumulative - running_max) / running_max
        
        # Add drawdown areas
        fig.add_trace(go.Scatter(
            x=portfolio_cumulative.index,
            y=running_max.values,
            name='Peak',
            line=dict(color='rgba(255, 255, 255, 0)'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title=title,
            height=500,
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # Add range selector
        fig.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=False),
            type="date"
        )
        
        return fig
    
    @staticmethod
    def create_drawdown_chart(drawdown_series: pd.Series) -> go.Figure:
        """Create enhanced drawdown chart"""
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drawdown_series.index,
            y=drawdown_series.values * 100,
            name="Drawdown",
            fill='tozeroy',
            fillcolor='rgba(220, 53, 69, 0.3)',
            line=dict(color='#dc3545', width=2),
            hovertemplate='%{x|%Y-%m-%d}<br>Drawdown: %{y:.1f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title="Portfolio Drawdown",
            height=400,
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode='x unified',
            yaxis=dict(ticksuffix="%"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    @staticmethod
    def create_correlation_heatmap(correlation_matrix: pd.DataFrame) -> go.Figure:
        """Create enhanced correlation heatmap"""
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Asset Correlation Matrix",
            height=600,
            template="plotly_dark",
            xaxis_title="Assets",
            yaxis_title="Assets",
            xaxis=dict(tickangle=45),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig

# -------------------------------------------------------------
# MAIN APPLICATION
# -------------------------------------------------------------
def main():
    st.title(APP_TITLE)
    
    # Initialize session state with enhanced defaults
    default_tickers = ["SPY", "TLT", "GLD", "AAPL", "MSFT"]
    
    session_defaults = {
        'selected_tickers': default_tickers,
        'portfolio_strategy': "Equal Weight",
        'custom_weights': {},
        'data_loaded': False,
        'current_weights': None,
        'use_demo_data': False,
        'show_advanced': False,
        'optimization_results': None,
        'last_refresh': None,
        'portfolio_history': []
    }
    
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Sidebar configuration
    with st.sidebar:
        st.title("⚙️ Portfolio Configuration")
        
        # Add quick action buttons at the top
        col_qa1, col_qa2 = st.columns(2)
        with col_qa1:
            if st.button("📊 Quick Demo", use_container_width=True, type="secondary"):
                st.session_state.selected_tickers = ["SPY", "QQQ", "TLT", "GLD", "BTC-USD", "AAPL", "MSFT"]
                st.session_state.portfolio_strategy = "Maximum Sharpe Ratio"
                st.success("Demo portfolio loaded!")
        
        with col_qa2:
            if st.button("🔄 Reset All", use_container_width=True, type="secondary"):
                for key in list(st.session_state.keys()):
                    if key not in ['selected_tickers', 'portfolio_strategy']:
                        st.session_state[key] = session_defaults.get(key, None)
                EnhancedCache.clear_cache()
                st.success("Application reset and cache cleared!")
                st.rerun()
        
        # Asset selection with enhanced filtering
        st.subheader("🌍 Asset Selection")
        
        # Enhanced category filter with search
        category_search = st.text_input("🔍 Search Categories", "", 
                                       key="sidebar_category_search",
                                       help="Search for specific asset categories")
        
        if category_search:
            filtered_categories = [cat for cat in GLOBAL_ASSET_UNIVERSE.keys() 
                                 if category_search.lower() in cat.lower()]
        else:
            filtered_categories = list(GLOBAL_ASSET_UNIVERSE.keys())
        
        reliable_categories = ["US_Indices", "US_Tech_Stocks", "Bonds", "Commodities", "Cryptocurrencies"]
        default_cats = [cat for cat in reliable_categories if cat in filtered_categories]
        
        category_filter = st.multiselect(
            "Filter by Category",
            filtered_categories,
            default=default_cats,
            key="sidebar_category_filter",
            help="Select categories to filter available assets"
        )
        
        # Enhanced ticker selection with search
        asset_search = st.text_input("🔍 Search Assets", "", 
                                    key="sidebar_asset_search",
                                    help="Search for specific assets by ticker or name")
        
        # Filter tickers based on selected categories and search
        filtered_tickers = []
        if category_filter:
            for category in category_filter:
                filtered_tickers.extend(GLOBAL_ASSET_UNIVERSE[category])
        else:
            filtered_tickers = ALL_TICKERS
        
        # Remove duplicates while preserving order
        filtered_tickers = list(dict.fromkeys(filtered_tickers))
        
        # Apply search filter
        if asset_search:
            filtered_tickers = [ticker for ticker in filtered_tickers 
                              if asset_search.upper() in ticker.upper()]
        
        # Get valid default values
        current_selected = st.session_state.get('selected_tickers', default_tickers)
        valid_defaults = [t for t in current_selected if t in filtered_tickers]
        
        if not valid_defaults and filtered_tickers:
            valid_defaults = filtered_tickers[:min(10, len(filtered_tickers))]
        
        # Enhanced asset selector with virtualization for large lists
        selected_tickers = st.multiselect(
            "Select Assets (3-15 recommended)",
            filtered_tickers,
            default=valid_defaults,
            help="Select 3-15 assets for optimal diversification and performance",
            key="sidebar_asset_selector"
        )
        
        # Display selection statistics
        if selected_tickers:
            st.caption(f"✅ Selected: {len(selected_tickers)} assets")
            
            # Show categories of selected assets
            selected_categories = {}
            for ticker in selected_tickers:
                for category, assets in GLOBAL_ASSET_UNIVERSE.items():
                    if ticker in assets:
                        selected_categories[category] = selected_categories.get(category, 0) + 1
                        break
            
            if selected_categories:
                with st.expander("📊 Selection Breakdown", expanded=False):
                    for category, count in selected_categories.items():
                        st.write(f"{category.replace('_', ' ')}: {count} assets")
        
        # Validate selection
        if len(selected_tickers) < 2:
            st.error("Please select at least 2 assets for portfolio analysis")
        elif len(selected_tickers) > 20:
            st.warning("⚠️ More than 20 assets selected. This may impact optimization performance.")
        
        st.session_state.selected_tickers = selected_tickers
        
        # Enhanced benchmark selection
        benchmark_options = ["SPY", "QQQ", "VTI", "IWM", "DIA", "BND", "GLD"] + \
                           [t for t in ALL_TICKERS if t not in selected_tickers]
        benchmark_options = list(dict.fromkeys(benchmark_options))
        
        default_benchmark = "SPY" if "SPY" in benchmark_options else benchmark_options[0]
        
        benchmark = st.selectbox(
            "📊 Benchmark Index",
            benchmark_options,
            index=benchmark_options.index(default_benchmark) if default_benchmark in benchmark_options else 0,
            help="Primary benchmark for performance comparison",
            key="sidebar_benchmark_selector"
        )
        
        # Enhanced date range selector
        st.subheader("📅 Date Range")
        
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            date_preset = st.selectbox(
                "Time Period",
                ["1 Month", "3 Months", "6 Months", "1 Year", "3 Years", "5 Years", "10 Years", "Max", "Custom"],
                index=4,  # Default to 3 years
                key="sidebar_date_preset"
            )
        
        with col_d2:
            if date_preset == "Custom":
                period_years = st.slider("Years", 1, 20, 5)
                date_preset = f"{period_years} Years"
        
        end_date = pd.Timestamp.today()
        
        date_mapping = {
            "1 Month": pd.DateOffset(months=1),
            "3 Months": pd.DateOffset(months=3),
            "6 Months": pd.DateOffset(months=6),
            "1 Year": pd.DateOffset(years=1),
            "3 Years": pd.DateOffset(years=3),
            "5 Years": pd.DateOffset(years=5),
            "10 Years": pd.DateOffset(years=10),
            "Max": pd.DateOffset(years=20)  # Max 20 years for performance
        }
        
        if date_preset in date_mapping:
            start_date = end_date - date_mapping[date_preset]
        else:
            # Custom years
            years = int(date_preset.split()[0])
            start_date = end_date - pd.DateOffset(years=years)
        
        # Enhanced portfolio strategy selector
        st.subheader("🎯 Portfolio Strategy")
        
        strategy = st.selectbox(
            "Construction Method",
            list(PORTFOLIO_STRATEGIES.keys()),
            index=0,
            help="Select portfolio optimization strategy",
            key="sidebar_strategy_selector"
        )
        
        st.session_state.portfolio_strategy = strategy
        
        # Strategy description
        if strategy in PORTFOLIO_STRATEGIES:
            st.caption(f"📝 {PORTFOLIO_STRATEGIES[strategy]}")
        
        # Enhanced advanced settings
        with st.expander("⚙️ Advanced Settings", expanded=st.session_state.show_advanced):
            # Risk parameters in columns
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                rf_annual = st.number_input(
                    "Risk-Free Rate (%)",
                    value=3.0,
                    min_value=0.0,
                    max_value=20.0,
                    step=0.5,
                    format="%.1f",
                    key="sidebar_rf_annual"
                ) / 100
            
            with col_a2:
                confidence_level = st.slider(
                    "VaR Confidence (%)",
                    min_value=90,
                    max_value=99,
                    value=95,
                    step=1,
                    key="sidebar_confidence_level"
                ) / 100
            
            # Optimization constraints
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                short_selling = st.checkbox(
                    "Allow Short Selling",
                    value=False,
                    help="Allow negative portfolio weights (advanced)",
                    key="sidebar_short_selling"
                )
            
            with col_b2:
                risk_aversion = st.slider(
                    "Risk Aversion",
                    min_value=0.5,
                    max_value=5.0,
                    value=2.0,
                    step=0.5,
                    help="Higher values mean more risk aversion",
                    key="sidebar_risk_aversion"
                )
            
            # Data options
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                use_parallel = st.checkbox(
                    "Parallel Download",
                    value=True,
                    help="Use parallel processing for faster data loading"
                )
            
            with col_c2:
                use_demo_data = st.checkbox(
                    "Demo Mode",
                    value=False,
                    help="Use synthetic data for testing (when real data fails)"
                )
                st.session_state.use_demo_data = use_demo_data
            
            # Cache management
            if st.button("🗑️ Clear Cache", key="sidebar_clear_cache"):
                if EnhancedCache.clear_cache():
                    st.success("Cache cleared successfully!")
                else:
                    st.error("Failed to clear cache")
            
            # Show cache statistics
            cache_stats = EnhancedCache.get_cache_stats()
            st.caption(f"Cache: {cache_stats['total_files']} files ({cache_stats['total_size_mb']:.1f} MB)")
        
        # Enhanced action buttons
        st.markdown("---")
        
        col_run1, col_run2 = st.columns([2, 1])
        
        with col_run1:
            run_analysis = st.button(
                "🚀 Run Portfolio Analysis",
                type="primary",
                use_container_width=True,
                key="sidebar_run_analysis",
                disabled=len(selected_tickers) < 2
            )
        
        with col_run2:
            if st.button("📊 View Cache", key="sidebar_view_cache"):
                cache_stats = EnhancedCache.get_cache_stats()
                st.info(f"Cache Stats: {cache_stats['total_files']} files, {cache_stats['total_size_mb']:.2f} MB")
        
        # System status
        st.markdown("---")
        st.caption("System Status")
        
        status_col1, status_col2, status_col3 = st.columns(3)
        
        with status_col1:
            st.metric("PyPortfolioOpt", "✅" if PYPFOPT_AVAILABLE else "❌")
        
        with status_col2:
            st.metric("Assets", len(ALL_TICKERS))
        
        with status_col3:
            st.metric("Version", "5.1")
    
    # Main content area - Enhanced with loading states
    if not run_analysis or len(selected_tickers) < 2:
        # Enhanced welcome screen
        welcome_col1, welcome_col2 = st.columns([2, 1])
        
        with welcome_col1:
            st.info("👈 Configure your portfolio in the sidebar and click 'Run Portfolio Analysis' to begin")
        
        with welcome_col2:
            if st.button("🔄 Refresh Data", key="welcome_refresh"):
                EnhancedCache.clear_cache()
                st.success("Cache cleared. Ready for fresh data.")
        
        # Dashboard-style statistics
        with st.expander("📊 Global Market Overview", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Assets", len(ALL_TICKERS), "Global Coverage")
            
            with col2:
                st.metric("Categories", len(GLOBAL_ASSET_UNIVERSE), "Diversified")
            
            with col3:
                st.metric("Geographic Markets", "20+", "Worldwide")
            
            with col4:
                st.metric("Asset Classes", "12+", "Multi-Asset")
            
            # Market heat map visualization
            st.subheader("🌍 Asset Distribution by Category")
            
            category_counts = {cat: len(assets) for cat, assets in GLOBAL_ASSET_UNIVERSE.items()}
            categories_df = pd.DataFrame({
                'Category': list(category_counts.keys()),
                'Count': list(category_counts.values())
            }).sort_values('Count', ascending=False)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=categories_df['Category'].str.replace('_', ' '),
                    y=categories_df['Count'],
                    marker_color='#1a5fb4',
                    text=categories_df['Count'],
                    textposition='auto',
                    hovertemplate='%{x}<br>Assets: %{y}<extra></extra>'
                )
            ])
            
            fig.update_layout(
                height=400,
                template="plotly_dark",
                xaxis_title="Category",
                yaxis_title="Number of Assets",
                xaxis_tickangle=45,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced quick start guide
        with st.expander("🚀 Getting Started Guide", expanded=True):
            st.markdown("""
            ### 📋 Quick Start Guide
            
            1. **Select Assets**: Choose 3-15 diverse assets across different categories
            2. **Set Timeframe**: Pick appropriate historical period for analysis
            3. **Choose Strategy**: Select optimization method based on your objectives
            4. **Configure Settings**: Adjust risk parameters and constraints
            5. **Run Analysis**: Generate comprehensive insights and reports
            
            ### 🎯 Recommended Portfolios
            
            **Balanced Portfolio** (Moderate Risk):
            - SPY (US Stocks) - 40%
            - TLT (US Bonds) - 30%
            - GLD (Gold) - 10%
            - BND (Aggregate Bonds) - 20%
            
            **Growth Portfolio** (Higher Risk):
            - QQQ (Tech) - 50%
            - ARKK (Innovation) - 15%
            - ICLN (Clean Energy) - 15%
            - MSFT (Individual Stock) - 20%
            
            **Conservative Portfolio** (Lower Risk):
            - SHY (Short-term Bonds) - 40%
            - TIP (Inflation Protected) - 30%
            - Utilities Sector - 20%
            - Consumer Staples - 10%
            
            ### ⚠️ Important Notes
            
            - Start with 3-10 assets for optimal diversification
            - Use at least 3 years of data for reliable statistics
            - Consider correlation between assets when selecting
            - Rebalance portfolio periodically based on strategy
            """)
        
        # Feature highlights
        with st.expander("✨ Key Features", expanded=False):
            col_f1, col_f2, col_f3 = st.columns(3)
            
            with col_f1:
                st.markdown("""
                **📈 Portfolio Optimization**
                - 13+ optimization strategies
                - Efficient frontier analysis
                - Real-time performance metrics
                - Custom weight allocation
                """)
            
            with col_f2:
                st.markdown("""
                **⚖️ Risk Management**
                - Advanced VaR calculations (5 methods)
                - Monte Carlo simulations
                - Stress testing scenarios
                - Correlation analysis
                """)
            
            with col_f3:
                st.markdown("""
                **📊 Analytics & Reporting**
                - Performance attribution
                - EWMA volatility
                - Comprehensive reports
                - Export functionality
                """)
        
        # System requirements
        with st.expander("🔧 System Requirements", expanded=False):
            st.markdown("""
            ### Required Packages
            
            ```bash
            streamlit>=1.28.0
            pandas>=2.0.0
            numpy>=1.24.0
            plotly>=5.17.0
            yfinance>=0.2.28
            scipy>=1.11.0
            pypfopt>=1.5.0
            seaborn>=0.12.0
            ```
            
            ### Installation
            
            ```bash
            pip install -r requirements.txt
            streamlit run app.py
            ```
            
            ### Troubleshooting
            
            1. **Data loading issues**: Check internet connection and try demo mode
            2. **PyPortfolioOpt errors**: Reinstall with `pip install pypfopt --upgrade`
            3. **Memory issues**: Reduce number of assets or time period
            4. **Performance issues**: Clear cache from advanced settings
            """)
        
        st.stop()
    
    # Load data with enhanced progress tracking
    st.subheader("📊 Loading Market Data...")
    
    try:
        # Combine selected tickers with benchmark
        all_tickers_to_load = list(dict.fromkeys(selected_tickers + [benchmark]))
        
        # Create progress container
        progress_container = st.container()
        
        with progress_container:
            # Create progress bars
            progress_overall = st.progress(0, text="Initializing data download...")
            progress_status = st.empty()
            
            # Load data with progress updates
            with st.spinner(f"Downloading market data for {len(all_tickers_to_load)} assets..."):
                # Update progress
                progress_status.text("Connecting to data sources...")
                progress_overall.progress(0.1)
                
                # Use cached loader for better performance
                prices = load_global_prices_cached(all_tickers_to_load, start_date, end_date)
                
                # Update progress
                progress_status.text("Processing and validating data...")
                progress_overall.progress(0.5)
                
                # If data loading fails and demo data is enabled
                if (prices.empty or len(prices.columns) < 2) and st.session_state.use_demo_data:
                    progress_status.text("Generating realistic demo data...")
                    prices = generate_demo_data(all_tickers_to_load, start_date, end_date)
                    st.info("📊 Using demo data for testing purposes")
                
                # Validate loaded data
                if prices.empty:
                    progress_overall.progress(0, text="Failed to load data")
                    st.error("❌ No data loaded. Please check your selections and try again.")
                    if not st.session_state.use_demo_data:
                        st.info("💡 Try enabling 'Demo Mode' in Advanced Settings for synthetic data")
                    st.stop()
                
                if len(prices.columns) < 2:
                    st.error("❌ Insufficient data for analysis. Need at least 2 assets with valid data.")
                    st.stop()
                
                # Update progress
                progress_status.text("Calculating returns and statistics...")
                progress_overall.progress(0.8)
                
                # Check which tickers were successfully loaded
                loaded_tickers = prices.columns.tolist()
                missing_tickers = [t for t in all_tickers_to_load if t not in loaded_tickers]
                
                # Update selected tickers to only include loaded ones
                selected_tickers = [t for t in selected_tickers if t in loaded_tickers]
                
                if benchmark not in loaded_tickers:
                    benchmark = loaded_tickers[0] if loaded_tickers else "SPY"
                    st.warning(f"Benchmark not available, using {benchmark} instead")
                
                if len(selected_tickers) < 2:
                    st.error("❌ Need at least 2 selected assets with valid data.")
                    st.stop()
                
                # Calculate returns
                portfolio_tickers = [t for t in selected_tickers if t in prices.columns]
                prices = prices[portfolio_tickers + [benchmark]]
                returns = prices.pct_change().dropna()
                
                # Final progress update
                progress_overall.progress(1.0, text="Data loaded successfully!")
                time.sleep(0.3)  # Brief pause to show completion
                
                # Clear progress indicators
                progress_overall.empty()
                progress_status.empty()
            
            # Store in session state
            st.session_state.prices = prices
            st.session_state.returns = returns
            st.session_state.benchmark = benchmark
            st.session_state.data_loaded = True
            st.session_state.last_refresh = datetime.now()
            
            # Success message with statistics
            success_col1, success_col2, success_col3 = st.columns(3)
            
            with success_col1:
                st.success(f"✅ {len(loaded_tickers)} assets loaded")
            
            with success_col2:
                days_loaded = len(returns)
                st.info(f"📅 {days_loaded} trading days")
            
            with success_col3:
                date_range = f"{returns.index[0].date()} to {returns.index[-1].date()}"
                st.info(f"📊 {date_range}")
            
            if missing_tickers:
                with st.expander(f"⚠️ {len(missing_tickers)} assets failed to load", expanded=False):
                    st.write(", ".join(missing_tickers[:10]))
                    if len(missing_tickers) > 10:
                        st.write(f"... and {len(missing_tickers) - 10} more")
                    
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)[:200]}")
        ErrorHandler.handle_error(e, "Data Loading")
        st.info("💡 Try reducing the number of assets or enabling Demo Mode")
        st.stop()
    
    # Create enhanced tabs with icons and better organization
    tab_names = [
        "📈 Overview & Weights",
        "⚖️ Risk Analytics", 
        "🎯 Portfolio Optimization",
        "🔗 Correlation Matrix",
        "📊 EWMA Analysis",
        "🎲 Monte Carlo VaR",
        "📊 Performance Attribution",
        "📋 Report & Export"
    ]
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(tab_names)
    
    # Tab 1: Overview & Weights
    with tab1:
        st.header("📊 Portfolio Overview & Analysis")
        
        try:
            # Get data from session state
            returns = st.session_state.returns
            prices = st.session_state.prices
            selected_tickers = st.session_state.selected_tickers
            benchmark = st.session_state.benchmark
            
            # Calculate portfolio based on strategy
            if st.session_state.portfolio_strategy == "Equal Weight":
                n_assets = len(selected_tickers)
                weights = np.ones(n_assets) / n_assets
                portfolio_returns = returns[selected_tickers].mean(axis=1)
                method_desc = "Equal Weight Portfolio"
                
            elif st.session_state.portfolio_strategy == "Custom Weights":
                # Custom weights editor
                st.subheader("✏️ Custom Portfolio Weights Editor")
                
                # Initialize or update custom weights
                if selected_tickers != list(st.session_state.custom_weights.keys()):
                    st.session_state.custom_weights = {
                        ticker: 1.0/len(selected_tickers) 
                        for ticker in selected_tickers
                    }
                
                # Create weight editor with columns
                cols = st.columns(4)  # Increased to 4 columns for better layout
                weight_inputs = {}
                
                for idx, ticker in enumerate(selected_tickers):
                    with cols[idx % 4]:
                        default_weight = st.session_state.custom_weights.get(ticker, 0.0) * 100
                        weight = st.number_input(
                            f"{ticker} Weight (%)",
                            min_value=0.0,
                            max_value=100.0,
                            value=default_weight,
                            step=1.0,
                            key=f"tab1_weight_{ticker}",
                            help=f"Set weight for {ticker}"
                        ) / 100
                        weight_inputs[ticker] = weight
                
                # Display current weight sum
                total_weight = sum(weight_inputs.values())
                st.info(f"Current total: {total_weight*100:.1f}% {'✅' if 99.9 <= total_weight*100 <= 100.1 else '⚠️'}")
                
                # Normalize weights if needed
                if total_weight > 0:
                    weights = np.array([weight_inputs[t] / total_weight for t in selected_tickers])
                    st.session_state.custom_weights = {t: weights[i] for i, t in enumerate(selected_tickers)}
                else:
                    weights = np.ones(len(selected_tickers)) / len(selected_tickers)
                    st.warning("Weights must sum to > 0. Using equal weights.")
                
                portfolio_returns = (returns[selected_tickers] * weights).sum(axis=1)
                method_desc = "Custom Weight Portfolio"
                
            else:
                # Use PyPortfolioOpt optimization
                with st.spinner("🔧 Optimizing portfolio..."):
                    optimizer = PortfolioOptimizer()
                    result = optimizer.optimize_portfolio(
                        returns[selected_tickers],
                        st.session_state.portfolio_strategy,
                        risk_free_rate=rf_annual,
                        risk_aversion=risk_aversion,
                        short_allowed=short_selling
                    )
                    
                    weights = result['weights']
                    portfolio_returns = (returns[selected_tickers] * weights).sum(axis=1)
                    method_desc = result['method']
                    
                    # Store optimization results
                    st.session_state.optimization_results = result
            
            # Store weights for use in other tabs
            st.session_state.current_weights = weights
            st.session_state.portfolio_returns = portfolio_returns
            st.session_state.asset_returns = returns[selected_tickers]
            
            # Display portfolio summary
            col_summary1, col_summary2, col_summary3 = st.columns(3)
            
            with col_summary1:
                st.metric("Strategy", method_desc)
            
            with col_summary2:
                st.metric("Assets", len(selected_tickers))
            
            with col_summary3:
                # Calculate portfolio value if starting with $10,000
                initial_value = 10000
                cumulative_returns = (1 + portfolio_returns).cumprod()
                current_value = initial_value * cumulative_returns.iloc[-1] if len(cumulative_returns) > 0 else initial_value
                st.metric("Portfolio Value", f"${current_value:,.0f}", 
                         f"from ${initial_value:,.0f}")
            
            # Display weights and performance in columns
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("📊 Portfolio Allocation")
                
                weights_df = pd.DataFrame({
                    'Asset': selected_tickers,
                    'Weight': weights,
                    'Category': [
                        next((cat for cat, assets in GLOBAL_ASSET_UNIVERSE.items() if t in assets), 'Other') 
                        for t in selected_tickers
                    ]
                })
                
                # Sort by weight
                weights_df = weights_df.sort_values('Weight', ascending=False)
                
                # Display weights table with formatting
                st.dataframe(
                    weights_df.style.format({'Weight': '{:.2%}'}).background_gradient(
                        subset=['Weight'], cmap='Blues'
                    ).bar(subset=['Weight'], color='#1a5fb4'),
                    use_container_width=True,
                    height=400
                )
                
                # Add pie chart option
                if st.checkbox("Show Pie Chart", key="tab1_show_pie"):
                    fig = px.pie(
                        weights_df,
                        values='Weight',
                        names='Asset',
                        title='Portfolio Allocation',
                        color_discrete_sequence=px.colors.sequential.Blues
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📈 Portfolio Performance")
                
                # Benchmark returns
                benchmark_returns = returns[benchmark]
                
                # Calculate cumulative performance
                portfolio_cumulative = (1 + portfolio_returns).cumprod()
                benchmark_cumulative = (1 + benchmark_returns).cumprod()
                
                # Create performance chart using enhanced utility
                fig = VisualizationUtils.create_performance_chart(
                    portfolio_cumulative, benchmark_cumulative,
                    title=f"Portfolio vs {benchmark} Benchmark"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Key performance metrics in a grid
                st.subheader("🎯 Key Performance Metrics")
                
                # Calculate metrics
                metrics_calculator = PerformanceMetrics()
                portfolio_metrics = metrics_calculator.calculate_metrics(
                    portfolio_returns, benchmark_returns, rf_annual
                )
                
                # Display metrics in columns
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    ann_return = portfolio_metrics.get('annual_return', 0)
                    benchmark_return = benchmark_returns.mean() * TRADING_DAYS
                    active_return = ann_return - benchmark_return
                    delta_color = "inverse" if active_return < 0 else "normal"
                    st.metric("Annual Return", f"{ann_return:.2%}", 
                             f"{active_return:+.2%}", delta_color=delta_color)
                
                with col_b:
                    ann_vol = portfolio_metrics.get('annual_volatility', 0)
                    benchmark_vol = benchmark_returns.std() * np.sqrt(TRADING_DAYS)
                    vol_diff = ann_vol - benchmark_vol
                    delta_color = "inverse" if vol_diff > 0 else "normal"
                    st.metric("Annual Volatility", f"{ann_vol:.2%}", 
                             f"{vol_diff:+.2%}", delta_color=delta_color)
                
                with col_c:
                    sharpe = portfolio_metrics.get('sharpe_ratio', 0)
                    benchmark_sharpe = (benchmark_return - rf_annual) / (benchmark_vol + 1e-10)
                    sharpe_diff = sharpe - benchmark_sharpe
                    delta_color = "inverse" if sharpe_diff < 0 else "normal"
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}", 
                             f"{sharpe_diff:+.2f}", delta_color=delta_color)
                
                with col_d:
                    max_dd = portfolio_metrics.get('max_drawdown', 0)
                    benchmark_cum = (1 + benchmark_returns).cumprod()
                    benchmark_running_max = benchmark_cum.expanding().max()
                    benchmark_drawdown = (benchmark_cum - benchmark_running_max) / benchmark_running_max
                    benchmark_max_dd = benchmark_drawdown.min() if not benchmark_drawdown.empty else 0
                    dd_diff = max_dd - benchmark_max_dd
                    delta_color = "inverse" if dd_diff < 0 else "normal"
                    st.metric("Max Drawdown", f"{max_dd:.2%}", 
                             f"{dd_diff:+.2%}", delta_color=delta_color)
                
                # Drawdown chart
                st.subheader("📉 Drawdown Analysis")
                cumulative = (1 + portfolio_returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                
                fig = VisualizationUtils.create_drawdown_chart(drawdown)
                st.plotly_chart(fig, use_container_width=True)
                
                # Drawdown statistics
                if not drawdown.empty:
                    col_dd1, col_dd2, col_dd3 = st.columns(3)
                    with col_dd1:
                        avg_drawdown = drawdown.mean()
                        st.metric("Average Drawdown", f"{avg_drawdown:.2%}")
                    
                    with col_dd2:
                        recovery_days = portfolio_metrics.get('recovery_period', 'N/A')
                        if isinstance(recovery_days, (int, float)):
                            st.metric("Recovery Period", f"{recovery_days} days")
                        else:
                            st.metric("Recovery Period", recovery_days)
                    
                    with col_dd3:
                        ulcer_index = portfolio_metrics.get('ulcer_index', 0)
                        st.metric("Ulcer Index", f"{ulcer_index:.3f}")
        
        except Exception as e:
            st.error(f"Error in Overview tab: {str(e)[:200]}")
            ErrorHandler.handle_error(e, "Tab 1: Overview")
            st.info("💡 Try different assets or check your internet connection")
    
    # Tab 2: Enhanced Risk Analytics
    with tab2:
        st.header("⚖️ Comprehensive Risk Analytics")
        
        # Ensure we have portfolio data
        if 'portfolio_returns' not in st.session_state:
            st.error("Please run portfolio analysis first in the Overview tab.")
            st.stop()
        
        try:
            portfolio_returns = st.session_state.portfolio_returns
            risk_analytics = EnhancedRiskAnalytics()
            
            # Risk overview metrics
            st.subheader("📊 Risk Overview")
            
            # Calculate basic risk metrics
            col_risk1, col_risk2, col_risk3, col_risk4 = st.columns(4)
            
            with col_risk1:
                volatility = portfolio_returns.std() * np.sqrt(252)
                st.metric("Annual Volatility", f"{volatility:.2%}")
            
            with col_risk2:
                var_95 = np.percentile(portfolio_returns, 5)
                st.metric("1-Day VaR (95%)", f"{var_95:.2%}")
            
            with col_risk3:
                tail_95 = portfolio_returns[portfolio_returns <= var_95]
                cvar_95 = tail_95.mean() if len(tail_95) > 0 else var_95
                st.metric("1-Day CVaR (95%)", f"{cvar_95:.2%}")
            
            with col_risk4:
                max_dd = ((1 + portfolio_returns).cumprod() / 
                         (1 + portfolio_returns).cumprod().cummax() - 1).min()
                st.metric("Max Drawdown", f"{max_dd:.2%}")
            
            # VaR Method Selection
            st.subheader("📈 Value at Risk Analysis")
            
            col_var1, col_var2, col_var3 = st.columns(3)
            
            with col_var1:
                var_method = st.selectbox(
                    "VaR Calculation Method",
                    ["Historical Simulation", "Parametric (Normal)", "EWMA", 
                     "Monte Carlo Simulation", "Modified (Cornish-Fisher)", "Compare All Methods"],
                    index=0,
                    key="tab2_var_method"
                )
            
            with col_var2:
                var_horizon = st.selectbox(
                    "Time Horizon",
                    ["1 Day", "5 Days", "10 Days", "1 Month", "3 Months"],
                    index=0,
                    key="tab2_var_horizon"
                )
                
                # Convert horizon to days
                horizon_map = {"1 Day": 1, "5 Days": 5, "10 Days": 10, "1 Month": 21, "3 Months": 63}
                horizon_days = horizon_map[var_horizon]
            
            with col_var3:
                var_confidence = st.slider(
                    "Confidence Level",
                    min_value=0.90,
                    max_value=0.99,
                    value=0.95,
                    step=0.01,
                    key="tab2_var_confidence",
                    format="%.2f"
                )
            
            # Calculate portfolio returns for horizon
            if horizon_days > 1:
                # Aggregate returns for horizon
                horizon_returns = portfolio_returns.rolling(horizon_days).apply(
                    lambda x: np.prod(1 + x) - 1, raw=True
                ).dropna()
            else:
                horizon_returns = portfolio_returns
            
            # Calculate VaR based on selected method
            if var_method == "Compare All Methods":
                st.subheader("📊 VaR Method Comparison")
                
                var_results = risk_analytics.calculate_all_var_methods(
                    horizon_returns, 
                    var_confidence
                )
                
                # Display results in columns
                col_compare1, col_compare2 = st.columns(2)
                
                with col_compare1:
                    st.dataframe(
                        var_results.style.format({
                            'VaR': '{:.4%}',
                            'CVaR': '{:.4%}'
                        }).applymap(
                            lambda x: 'color: green' if isinstance(x, float) and x < 0 else '', 
                            subset=['VaR', 'CVaR']
                        ),
                        use_container_width=True,
                        height=300
                    )
                
                with col_compare2:
                    # Visualization
                    fig = go.Figure()
                    
                    # Filter successful results
                    successful_results = var_results[var_results['Status'].str.contains('Success')]
                    
                    if not successful_results.empty:
                        fig.add_trace(go.Bar(
                            name='VaR',
                            x=successful_results['Method'],
                            y=successful_results['VaR'].abs() * 100,
                            marker_color='#1a5fb4',
                            text=successful_results['VaR'].apply(lambda x: f"{x:.2%}"),
                            textposition='auto',
                            hovertemplate='%{x}<br>VaR: %{text}<extra></extra>'
                        ))
                        
                        fig.add_trace(go.Bar(
                            name='CVaR',
                            x=successful_results['Method'],
                            y=successful_results['CVaR'].abs() * 100,
                            marker_color='#26a269',
                            text=successful_results['CVaR'].apply(lambda x: f"{x:.2%}"),
                            textposition='auto',
                            hovertemplate='%{x}<br>CVaR: %{text}<extra></extra>'
                        ))
                    
                    fig.update_layout(
                        title=f"VaR & CVaR Comparison ({var_horizon}, {var_confidence*100:.0f}% Confidence)",
                        height=400,
                        template="plotly_dark",
                        yaxis_title="Value (%)",
                        barmode='group',
                        xaxis_tickangle=45,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                # Single method analysis
                method_map = {
                    "Historical Simulation": "historical",
                    "Parametric (Normal)": "parametric",
                    "EWMA": "ewma",
                    "Monte Carlo Simulation": "monte_carlo",
                    "Modified (Cornish-Fisher)": "modified"
                }
                
                method_key = method_map[var_method]
                
                # Calculate VaR
                var_result = risk_analytics.calculate_var(
                    horizon_returns,
                    method_key,
                    var_confidence,
                    params={'n_simulations': 10000, 'days': horizon_days} if method_key == "monte_carlo" else None
                )
                
                # Display results
                col_result1, col_result2, col_result3, col_result4 = st.columns(4)
                
                with col_result1:
                    if var_result.get('success', False) and not pd.isna(var_result.get('VaR')):
                        st.metric(
                            f"{var_horizon} VaR",
                            f"{var_result['VaR']:.4%}",
                            f"{var_confidence*100:.0f}% Confidence"
                        )
                    else:
                        st.metric(f"{var_horizon} VaR", "N/A")
                
                with col_result2:
                    if var_result.get('success', False) and not pd.isna(var_result.get('CVaR')):
                        st.metric(
                            f"{var_horizon} CVaR",
                            f"{var_result['CVaR']:.4%}",
                            f"{var_confidence*100:.0f}% Confidence"
                        )
                    else:
                        st.metric(f"{var_horizon} CVaR", "N/A")
                
                with col_result3:
                    if var_result.get('success', False):
                        st.metric("Method", var_result['method'])
                
                with col_result4:
                    if var_result.get('success', False):
                        st.metric("Status", "✅ Success")
                    else:
                        st.metric("Status", "❌ Failed")
                
                # Distribution plot with VaR
                if var_result.get('success', False) and not pd.isna(var_result.get('VaR')):
                    st.subheader("📊 Return Distribution with Risk Measures")
                    
                    fig = go.Figure()
                    
                    # Histogram of returns
                    fig.add_trace(go.Histogram(
                        x=horizon_returns,
                        nbinsx=50,
                        name="Returns",
                        marker_color='#1a5fb4',
                        opacity=0.7,
                        histnorm='probability density',
                        hovertemplate='Return: %{x:.4%}<br>Density: %{y:.4f}<extra></extra>'
                    ))
                    
                    # Add VaR line
                    if not pd.isna(var_result.get('VaR')):
                        fig.add_vline(
                            x=var_result['VaR'],
                            line_dash="dash",
                            line_color="#f5a623",
                            annotation_text=f"VaR: {var_result['VaR']:.4%}",
                            annotation_position="top left"
                        )
                    
                    # Add CVaR line
                    if not pd.isna(var_result.get('CVaR')):
                        fig.add_vline(
                            x=var_result['CVaR'],
                            line_dash="dot",
                            line_color="#c01c28",
                            annotation_text=f"CVaR: {var_result['CVaR']:.4%}",
                            annotation_position="top right"
                        )
                    
                    # Add mean line
                    mean_return = horizon_returns.mean()
                    fig.add_vline(
                        x=mean_return,
                        line_dash="solid",
                        line_color="#26a269",
                        annotation_text=f"Mean: {mean_return:.4%}",
                        annotation_position="bottom right"
                    )
                    
                    fig.update_layout(
                        height=500,
                        title=f"Return Distribution ({var_horizon})",
                        template="plotly_dark",
                        xaxis_title="Return",
                        yaxis_title="Density",
                        showlegend=True,
                        hovermode='x unified',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(tickformat=".2%"),
                        bargap=0.1
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Risk Metrics by Asset
            st.subheader("📊 Detailed Risk Metrics by Asset")
            
            # Calculate metrics for each asset
            risk_metrics_data = []
            asset_returns = st.session_state.asset_returns
            
            for asset in selected_tickers:
                if asset in asset_returns.columns:
                    asset_returns_series = asset_returns[asset]
                    asset_metrics = PerformanceMetrics.calculate_metrics(asset_returns_series)
                    
                    risk_metrics_data.append({
                        'Asset': asset,
                        'Category': next((cat for cat, assets in GLOBAL_ASSET_UNIVERSE.items() if asset in assets), 'Other'),
                        'Annual Vol': asset_metrics.get('annual_volatility', 0),
                        'Max DD': asset_metrics.get('max_drawdown', 0),
                        'Sharpe': asset_metrics.get('sharpe_ratio', 0),
                        'VaR 95%': asset_metrics.get('var_95', 0),
                        'CVaR 95%': asset_metrics.get('cvar_95', 0),
                        'Skewness': asset_metrics.get('skewness', 0),
                        'Kurtosis': asset_metrics.get('kurtosis', 0)
                    })
            
            risk_metrics_df = pd.DataFrame(risk_metrics_data)
            
            # Display risk metrics table
            st.dataframe(
                risk_metrics_df.style.format({
                    'Annual Vol': '{:.2%}',
                    'Max DD': '{:.2%}',
                    'Sharpe': '{:.2f}',
                    'VaR 95%': '{:.2%}',
                    'CVaR 95%': '{:.2%}',
                    'Skewness': '{:.2f}',
                    'Kurtosis': '{:.2f}'
                }).background_gradient(
                    subset=['Annual Vol', 'Max DD'], 
                    cmap='Reds'
                ).background_gradient(
                    subset=['Sharpe'], 
                    cmap='Greens'
                ),
                use_container_width=True,
                height=400
            )
            
            # Rolling Volatility Chart
            st.subheader("📈 Rolling Volatility Analysis")
            
            if len(portfolio_returns) > 63:
                rolling_vol = portfolio_returns.rolling(window=63).std() * np.sqrt(252)
                rolling_mean = portfolio_returns.rolling(window=63).mean() * 252
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("63-Day Rolling Volatility", "63-Day Rolling Returns"),
                    vertical_spacing=0.1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=rolling_vol.index,
                        y=rolling_vol.values * 100,
                        name="Rolling Volatility",
                        line=dict(color='#1a5fb4', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(26, 95, 180, 0.1)',
                        hovertemplate='%{x|%Y-%m-%d}<br>Volatility: %{y:.1f}%<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=rolling_mean.index,
                        y=rolling_mean.values * 100,
                        name="Rolling Returns",
                        line=dict(color='#26a269', width=2),
                        hovertemplate='%{x|%Y-%m-%d}<br>Return: %{y:.1f}%<extra></extra>'
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(
                    height=600,
                    template="plotly_dark",
                    showlegend=True,
                    hovermode='x unified',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                fig.update_yaxes(title_text="Volatility (%)", row=1, col=1)
                fig.update_yaxes(title_text="Return (%)", row=2, col=1)
                fig.update_xaxes(title_text="Date", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Stress Testing
            st.subheader("⚠️ Portfolio Stress Testing")
            
            if 'asset_returns' in st.session_state:
                stress_test_results = risk_analytics.calculate_stress_test(
                    st.session_state.asset_returns
                )
                
                st.dataframe(
                    stress_test_results.style.format({
                        'Portfolio Impact': '{}'
                    }).background_gradient(
                        subset=['Portfolio Impact'], 
                        cmap='RdBu_r',
                        vmin=-30, 
                        vmax=0
                    ),
                    use_container_width=True,
                    height=300
                )
            
            # Advanced Risk Statistics
            with st.expander("🔍 Advanced Risk Statistics", expanded=False):
                col_adv1, col_adv2, col_adv3 = st.columns(3)
                
                with col_adv1:
                    # Calculate tail risk metrics
                    tail_ratio = portfolio_metrics.get('tail_ratio', 0)
                    st.metric("Tail Ratio", f"{tail_ratio:.2f}")
                    
                    # Calculate Omega Ratio
                    omega_ratio = portfolio_metrics.get('omega_ratio', 0)
                    st.metric("Omega Ratio", f"{omega_ratio:.2f}")
                
                with col_adv2:
                    # Calculate Burke Ratio
                    burke_ratio = portfolio_metrics.get('burke_ratio', 0)
                    st.metric("Burke Ratio", f"{burke_ratio:.2f}")
                    
                    # Calculate Sterling Ratio
                    sterling_ratio = portfolio_metrics.get('sterling_ratio', 0)
                    st.metric("Sterling Ratio", f"{sterling_ratio:.2f}")
                
                with col_adv3:
                    # Calculate Gain/Loss Ratio
                    gain_loss = portfolio_metrics.get('gain_loss_ratio', 0)
                    st.metric("Gain/Loss Ratio", f"{gain_loss:.2f}")
                    
                    # Calculate Ulcer Index
                    ulcer_index = portfolio_metrics.get('ulcer_index', 0)
                    st.metric("Ulcer Index", f"{ulcer_index:.3f}")
        
        except Exception as e:
            st.error(f"Error in Risk Analytics tab: {str(e)[:200]}")
            ErrorHandler.handle_error(e, "Tab 2: Risk Analytics")
    
    # Tab 3: Portfolio Optimization
    with tab3:
        st.header("🎯 Advanced Portfolio Optimization")
        
        try:
            returns = st.session_state.returns
            selected_tickers = st.session_state.selected_tickers
            
            # Optimization Strategy Configuration
            st.subheader("⚙️ Optimization Configuration")
            
            col_opt1, col_opt2, col_opt3 = st.columns(3)
            
            with col_opt1:
                opt_method = st.selectbox(
                    "Optimization Method",
                    list(PORTFOLIO_STRATEGIES.keys()),
                    index=list(PORTFOLIO_STRATEGIES.keys()).index(st.session_state.portfolio_strategy)
                    if st.session_state.portfolio_strategy in PORTFOLIO_STRATEGIES else 0,
                    key="tab3_opt_method"
                )
            
            with col_opt2:
                target_type = st.selectbox(
                    "Target Type",
                    ["Max Sharpe Ratio", "Min Volatility", "Target Return", "Target Risk"],
                    index=0,
                    key="tab3_target_type"
                )
            
            with col_opt3:
                if target_type == "Target Return":
                    target_value = st.slider(
                        "Target Annual Return (%)",
                        min_value=0.0,
                        max_value=50.0,
                        value=10.0,
                        step=0.5,
                        key="tab3_target_return"
                    ) / 100
                    target_param = target_value
                elif target_type == "Target Risk":
                    target_value = st.slider(
                        "Target Annual Volatility (%)",
                        min_value=5.0,
                        max_value=50.0,
                        value=15.0,
                        step=0.5,
                        key="tab3_target_risk"
                    ) / 100
                    target_param = target_value
                else:
                    target_param = None
            
            # Advanced constraints
            with st.expander("🔧 Advanced Constraints", expanded=False):
                col_const1, col_const2, col_const3 = st.columns(3)
                
                with col_const1:
                    min_weight = st.slider(
                        "Minimum Weight (%)",
                        min_value=0.0,
                        max_value=20.0,
                        value=0.0,
                        step=0.5,
                        key="tab3_min_weight"
                    ) / 100
                
                with col_const2:
                    max_weight = st.slider(
                        "Maximum Weight (%)",
                        min_value=10.0,
                        max_value=100.0,
                        value=100.0,
                        step=0.5,
                        key="tab3_max_weight"
                    ) / 100
                
                with col_const3:
                    risk_aversion_opt = st.slider(
                        "Risk Aversion",
                        min_value=0.5,
                        max_value=10.0,
                        value=2.0,
                        step=0.1,
                        key="tab3_risk_aversion"
                    )
            
            # Run Optimization
            if st.button("🚀 Run Optimization", type="primary", key="tab3_run_optimization"):
                with st.spinner("Optimizing portfolio..."):
                    # Prepare returns data
                    returns_data = returns[selected_tickers]
                    
                    # Create optimizer instance
                    optimizer = PortfolioOptimizer()
                    
                    # Determine optimization parameters
                    if target_type == "Max Sharpe Ratio":
                        strategy_name = "Maximum Sharpe Ratio"
                        target_return = None
                        target_risk = None
                    elif target_type == "Min Volatility":
                        strategy_name = "Minimum Volatility"
                        target_return = None
                        target_risk = None
                    elif target_type == "Target Return":
                        strategy_name = "Efficient Return"
                        target_return = target_param
                        target_risk = None
                    else:  # Target Risk
                        strategy_name = "Efficient Risk"
                        target_return = None
                        target_risk = target_param
                    
                    # Run optimization
                    result = optimizer.optimize_portfolio(
                        returns_data,
                        strategy_name,
                        target_return=target_return,
                        target_risk=target_risk,
                        risk_free_rate=rf_annual,
                        risk_aversion=risk_aversion_opt,
                        short_allowed=short_selling
                    )
                    
                    # Store results
                    st.session_state.optimization_results = result
                    st.session_state.current_weights = result['weights']
                    
                    # Display results
                    st.success(f"✅ Optimization complete using {result['method']}")
            
            # Display optimization results if available
            if st.session_state.optimization_results:
                result = st.session_state.optimization_results
                
                # Performance metrics
                col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
                
                with col_perf1:
                    st.metric("Expected Return", f"{result['expected_return']:.2%}")
                
                with col_perf2:
                    st.metric("Expected Risk", f"{result['expected_risk']:.2%}")
                
                with col_perf3:
                    st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")
                
                with col_perf4:
                    st.metric("Method", result['method'])
                
                # Display optimized weights
                st.subheader("📊 Optimized Portfolio Weights")
                
                weights_df = pd.DataFrame({
                    'Asset': selected_tickers,
                    'Weight': result['weights'],
                    'Category': [
                        next((cat for cat, assets in GLOBAL_ASSET_UNIVERSE.items() if t in assets), 'Other') 
                        for t in selected_tickers
                    ]
                }).sort_values('Weight', ascending=False)
                
                # Create two columns for weights display
                col_weights1, col_weights2 = st.columns([2, 1])
                
                with col_weights1:
                    st.dataframe(
                        weights_df.style.format({'Weight': '{:.2%}'}).background_gradient(
                            subset=['Weight'], cmap='Blues'
                        ).bar(subset=['Weight'], color='#1a5fb4'),
                        use_container_width=True,
                        height=400
                    )
                
                with col_weights2:
                    # Pie chart
                    fig = px.pie(
                        weights_df,
                        values='Weight',
                        names='Asset',
                        title='Portfolio Allocation',
                        color_discrete_sequence=px.colors.sequential.Blues
                    )
                    fig.update_traces(
                        textposition='inside', 
                        textinfo='percent+label',
                        hole=0.3
                    )
                    fig.update_layout(
                        height=400,
                        showlegend=False,
                        margin=dict(t=50, b=0, l=0, r=0)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Efficient Frontier Analysis
                st.subheader("📈 Efficient Frontier Analysis")
                
                with st.spinner("Calculating efficient frontier..."):
                    # Calculate efficient frontier points
                    frontier_df = optimizer.calculate_efficient_frontier(
                        returns_data,
                        risk_free_rate=rf_annual,
                        n_points=50
                    )
                    
                    if not frontier_df.empty:
                        fig = go.Figure()
                        
                        # Efficient frontier line
                        fig.add_trace(go.Scatter(
                            x=frontier_df['risk'],
                            y=frontier_df['return'],
                            mode='lines',
                            name='Efficient Frontier',
                            line=dict(color='#1a5fb4', width=3),
                            hovertemplate='Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
                        ))
                        
                        # Individual assets
                        asset_returns = returns_data.mean() * TRADING_DAYS
                        asset_risks = returns_data.std() * np.sqrt(TRADING_DAYS)
                        
                        fig.add_trace(go.Scatter(
                            x=asset_risks,
                            y=asset_returns,
                            mode='markers+text',
                            name='Individual Assets',
                            marker=dict(
                                size=12,
                                color='#26a269',
                                line=dict(width=2, color='white')
                            ),
                            text=returns_data.columns,
                            textposition="top center",
                            hovertemplate='%{text}<br>Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
                        ))
                        
                        # Optimized portfolio
                        fig.add_trace(go.Scatter(
                            x=[result['expected_risk']],
                            y=[result['expected_return']],
                            mode='markers',
                            name='Optimized Portfolio',
                            marker=dict(
                                size=20,
                                color='#f5a623',
                                symbol='star',
                                line=dict(width=2, color='white')
                            ),
                            hovertemplate='Optimized Portfolio<br>Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
                        ))
                        
                        # Capital Market Line
                        max_sharpe_idx = frontier_df['sharpe'].idxmax()
                        max_sharpe_point = frontier_df.loc[max_sharpe_idx]
                        
                        # Plot CML
                        cml_x = np.linspace(0, max_sharpe_point['risk'] * 1.5, 50)
                        cml_y = rf_annual + (max_sharpe_point['sharpe'] * cml_x)
                        
                        fig.add_trace(go.Scatter(
                            x=cml_x,
                            y=cml_y,
                            mode='lines',
                            name='Capital Market Line',
                            line=dict(color='#c01c28', width=2, dash='dash'),
                            hovertemplate='Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
                        ))
                        
                        fig.update_layout(
                            title="Efficient Frontier",
                            height=600,
                            template="plotly_dark",
                            xaxis_title="Annual Risk (Volatility)",
                            yaxis_title="Annual Return",
                            hovermode='closest',
                            showlegend=True,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(tickformat=".0%"),
                            yaxis=dict(tickformat=".0%")
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Frontier statistics
                        with st.expander("📊 Efficient Frontier Statistics", expanded=False):
                            col_ef1, col_ef2, col_ef3, col_ef4 = st.columns(4)
                            
                            with col_ef1:
                                min_vol_idx = frontier_df['risk'].idxmin()
                                min_vol = frontier_df.loc[min_vol_idx]
                                st.metric("Minimum Volatility", f"{min_vol['risk']:.2%}")
                            
                            with col_ef2:
                                st.metric("Minimum Return", f"{frontier_df['return'].min():.2%}")
                            
                            with col_ef3:
                                st.metric("Maximum Return", f"{frontier_df['return'].max():.2%}")
                            
                            with col_ef4:
                                max_sharpe = frontier_df['sharpe'].max()
                                st.metric("Maximum Sharpe", f"{max_sharpe:.2f}")
            
            # Optimization Comparison
            st.subheader("🔍 Optimization Method Comparison")
            
            if st.button("🔄 Compare Methods", key="tab3_compare_methods"):
                with st.spinner("Comparing optimization methods..."):
                    # Test different optimization methods
                    methods_to_test = [
                        "Minimum Volatility",
                        "Maximum Sharpe Ratio",
                        "Maximum Quadratic Utility",
                        "Risk Parity",
                        "Hierarchical Risk Parity"
                    ]
                    
                    comparison_results = []
                    
                    for method in methods_to_test:
                        try:
                            result = optimizer.optimize_portfolio(
                                returns_data,
                                method,
                                risk_free_rate=rf_annual,
                                risk_aversion=risk_aversion_opt,
                                short_allowed=short_selling
                            )
                            
                            if result['success']:
                                comparison_results.append({
                                    'Method': method,
                                    'Return': result['expected_return'],
                                    'Risk': result['expected_risk'],
                                    'Sharpe': result['sharpe_ratio'],
                                    'Max Weight': result['weights'].max(),
                                    'Min Weight': result['weights'].min()
                                })
                        except Exception as e:
                            st.debug(f"Method {method} failed: {str(e)[:100]}")
                    
                    if comparison_results:
                        comparison_df = pd.DataFrame(comparison_results)
                        
                        # Display comparison table
                        st.dataframe(
                            comparison_df.style.format({
                                'Return': '{:.2%}',
                                'Risk': '{:.2%}',
                                'Sharpe': '{:.2f}',
                                'Max Weight': '{:.2%}',
                                'Min Weight': '{:.2%}'
                            }).background_gradient(
                                subset=['Return', 'Sharpe'], 
                                cmap='Greens'
                            ).background_gradient(
                                subset=['Risk'], 
                                cmap='Reds_r'
                            ),
                            use_container_width=True,
                            height=300
                        )
                        
                        # Visualization
                        fig = go.Figure()
                        
                        for idx, row in comparison_df.iterrows():
                            fig.add_trace(go.Scatter(
                                x=[row['Risk']],
                                y=[row['Return']],
                                mode='markers+text',
                                name=row['Method'],
                                marker=dict(size=row['Sharpe'] * 10 + 10),
                                text=row['Method'],
                                textposition="top center",
                                hovertemplate=(
                                    f"{row['Method']}<br>"
                                    f"Return: {row['Return']:.2%}<br>"
                                    f"Risk: {row['Risk']:.2%}<br>"
                                    f"Sharpe: {row['Sharpe']:.2f}<extra></extra>"
                                )
                            ))
                        
                        fig.update_layout(
                            title="Optimization Method Comparison",
                            height=500,
                            template="plotly_dark",
                            xaxis_title="Annual Risk",
                            yaxis_title="Annual Return",
                            hovermode='closest',
                            showlegend=True,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(tickformat=".0%"),
                            yaxis=dict(tickformat=".0%")
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error in Portfolio Optimization tab: {str(e)[:200]}")
            ErrorHandler.handle_error(e, "Tab 3: Portfolio Optimization")
    
    # Tab 4: Correlation Matrix
    with tab4:
        st.header("🔗 Asset Correlation Analysis")
        
        try:
            returns = st.session_state.returns
            selected_tickers = st.session_state.selected_tickers
            
            # Calculate correlation matrix
            correlation_matrix = returns[selected_tickers].corr()
            
            # Correlation matrix display
            col_corr1, col_corr2 = st.columns([3, 1])
            
            with col_corr1:
                st.subheader("🌐 Correlation Heatmap")
                
                # Create enhanced correlation heatmap
                fig = VisualizationUtils.create_correlation_heatmap(correlation_matrix)
                st.plotly_chart(fig, use_container_width=True)
            
            with col_corr2:
                st.subheader("📊 Correlation Statistics")
                
                # Calculate correlation statistics
                correlation_values = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)]
                
                if len(correlation_values) > 0:
                    st.metric("Average Correlation", f"{correlation_values.mean():.3f}")
                    st.metric("Median Correlation", f"{np.median(correlation_values):.3f}")
                    st.metric("Minimum Correlation", f"{correlation_values.min():.3f}")
                    st.metric("Maximum Correlation", f"{correlation_values.max():.3f}")
                    
                    # Correlation distribution
                    with st.expander("📈 Distribution", expanded=False):
                        fig = px.histogram(
                            x=correlation_values,
                            nbins=30,
                            title="Correlation Distribution",
                            labels={'x': 'Correlation Coefficient'},
                            color_discrete_sequence=['#1a5fb4']
                        )
                        fig.update_layout(
                            height=300,
                            template="plotly_dark",
                            showlegend=False,
                            bargap=0.1
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Hierarchical Clustering
            st.subheader("🌳 Hierarchical Clustering Analysis")
            
            if len(selected_tickers) >= 3:
                # Calculate distance matrix (1 - absolute correlation)
                distance_matrix = 1 - np.abs(correlation_matrix.values)
                
                # Perform hierarchical clustering
                linkage_matrix = linkage(squareform(distance_matrix), method='ward')
                
                # Create dendrogram
                fig = go.Figure()
                
                # Create dendrogram trace
                dendro_go = ff.create_dendrogram(
                    distance_matrix,
                    orientation='left',
                    labels=selected_tickers,
                    colorscale=['#1a5fb4', '#26a269', '#f5a623', '#c01c28'],
                    color_threshold=0.7 * max(linkage_matrix[:, 2])
                )
                
                for trace in dendro_go.data:
                    fig.add_trace(trace)
                
                fig.update_layout(
                    title="Asset Hierarchical Clustering",
                    height=600,
                    template="plotly_dark",
                    xaxis_title="Distance",
                    yaxis_title="Assets",
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Cluster interpretation
                num_clusters = st.slider(
                    "Number of Clusters",
                    min_value=2,
                    max_value=min(10, len(selected_tickers)),
                    value=min(4, len(selected_tickers)),
                    key="tab4_num_clusters"
                )
                
                # Assign clusters
                clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
                
                # Display cluster assignments
                cluster_df = pd.DataFrame({
                    'Asset': selected_tickers,
                    'Cluster': clusters,
                    'Category': [
                        next((cat for cat, assets in GLOBAL_ASSET_UNIVERSE.items() if t in assets), 'Other') 
                        for t in selected_tickers
                    ]
                }).sort_values('Cluster')
                
                st.dataframe(
                    cluster_df.style.background_gradient(
                        subset=['Cluster'], 
                        cmap='Set3'
                    ),
                    use_container_width=True,
                    height=300
                )
            
            # Rolling Correlation Analysis
            st.subheader("📈 Rolling Correlation Analysis")
            
            if len(returns) > 63:
                # Select two assets for rolling correlation
                col_roll1, col_roll2 = st.columns(2)
                
                with col_roll1:
                    asset1 = st.selectbox(
                        "First Asset",
                        selected_tickers,
                        index=0,
                        key="tab4_asset1"
                    )
                
                with col_roll2:
                    # Exclude the first asset from second selection
                    other_assets = [a for a in selected_tickers if a != asset1]
                    asset2 = st.selectbox(
                        "Second Asset",
                        other_assets,
                        index=0,
                        key="tab4_asset2"
                    )
                
                # Calculate rolling correlation
                window = st.slider(
                    "Rolling Window (days)",
                    min_value=20,
                    max_value=252,
                    value=63,
                    key="tab4_rolling_window"
                )
                
                rolling_corr = returns[asset1].rolling(window=window).corr(returns[asset2])
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=rolling_corr.index,
                    y=rolling_corr.values,
                    name=f"{asset1} vs {asset2}",
                    line=dict(color='#1a5fb4', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(26, 95, 180, 0.1)',
                    hovertemplate='%{x|%Y-%m-%d}<br>Correlation: %{y:.3f}<extra></extra>'
                ))
                
                # Add zero line
                fig.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="gray",
                    opacity=0.5
                )
                
                fig.update_layout(
                    title=f"Rolling Correlation: {asset1} vs {asset2}",
                    height=400,
                    template="plotly_dark",
                    xaxis_title="Date",
                    yaxis_title="Correlation Coefficient",
                    hovermode='x unified',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(range=[-1, 1])
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Rolling correlation statistics
                col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                
                with col_stats1:
                    avg_corr = rolling_corr.mean()
                    st.metric("Average", f"{avg_corr:.3f}")
                
                with col_stats2:
                    std_corr = rolling_corr.std()
                    st.metric("Std Dev", f"{std_corr:.3f}")
                
                with col_stats3:
                    min_corr = rolling_corr.min()
                    st.metric("Minimum", f"{min_corr:.3f}")
                
                with col_stats4:
                    max_corr = rolling_corr.max()
                    st.metric("Maximum", f"{max_corr:.3f}")
        
        except Exception as e:
            st.error(f"Error in Correlation Matrix tab: {str(e)[:200]}")
            ErrorHandler.handle_error(e, "Tab 4: Correlation Matrix")
    
    # Tab 5: EWMA Analysis
    with tab5:
        st.header("📊 EWMA Volatility Analysis")
        
        try:
            returns = st.session_state.returns
            selected_tickers = st.session_state.selected_tickers
            
            # EWMA Configuration
            st.subheader("⚙️ EWMA Configuration")
            
            col_ewma1, col_ewma2, col_ewma3 = st.columns(3)
            
            with col_ewma1:
                lambda_param = st.slider(
                    "Decay Factor (λ)",
                    min_value=0.85,
                    max_value=0.99,
                    value=0.94,
                    step=0.01,
                    key="tab5_lambda_param",
                    help="Higher values give more weight to older observations"
                )
            
            with col_ewma2:
                ewma_window = st.slider(
                    "Initialization Window",
                    min_value=20,
                    max_value=100,
                    value=30,
                    step=5,
                    key="tab5_ewma_window",
                    help="Number of days for initial variance calculation"
                )
            
            with col_ewma3:
                ewma_annualize = st.checkbox(
                    "Annualize Volatility",
                    value=True,
                    key="tab5_annualize"
                )
            
            # Calculate EWMA volatility
            with st.spinner("Calculating EWMA volatility..."):
                ewma_analyzer = EWMAAnalysis()
                ewma_vol = ewma_analyzer.calculate_ewma_volatility(
                    returns[selected_tickers],
                    lambda_param=lambda_param
                )
                
                if ewma_annualize:
                    multiplier = np.sqrt(252)
                else:
                    multiplier = 1
                
                # Display EWMA volatility chart
                st.subheader("📈 EWMA Volatility Over Time")
                
                # Select assets to display
                assets_to_display = st.multiselect(
                    "Select assets to display",
                    selected_tickers,
                    default=selected_tickers[:min(5, len(selected_tickers))],
                    key="tab5_assets_display"
                )
                
                if assets_to_display:
                    fig = go.Figure()
                    
                    colors = px.colors.qualitative.Set1
                    
                    for i, asset in enumerate(assets_to_display):
                        if asset in ewma_vol.columns:
                            fig.add_trace(go.Scatter(
                                x=ewma_vol.index,
                                y=ewma_vol[asset].values * multiplier,
                                name=asset,
                                line=dict(color=colors[i % len(colors)], width=2),
                                hovertemplate='%{x|%Y-%m-%d}<br>%{y:.1%} volatility<extra></extra>'
                            ))
                    
                    fig.update_layout(
                        title=f"EWMA Volatility (λ={lambda_param})",
                        height=500,
                        template="plotly_dark",
                        xaxis_title="Date",
                        yaxis_title=f"{'Annualized ' if ewma_annualize else ''}Volatility",
                        hovermode='x unified',
                        showlegend=True,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        yaxis=dict(tickformat=".0%")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # EWMA volatility statistics
                    st.subheader("📊 EWMA Volatility Statistics")
                    
                    # Calculate statistics
                    recent_vol = ewma_vol.iloc[-min(20, len(ewma_vol)):].mean() * multiplier
                    avg_vol = ewma_vol.mean() * multiplier
                    max_vol = ewma_vol.max() * multiplier
                    min_vol = ewma_vol.min() * multiplier
                    
                    stats_df = pd.DataFrame({
                        'Asset': selected_tickers,
                        f'Recent {"Annual " if ewma_annualize else ""}Vol': recent_vol,
                        f'Avg {"Annual " if ewma_annualize else ""}Vol': avg_vol,
                        f'Max {"Annual " if ewma_annualize else ""}Vol': max_vol,
                        f'Min {"Annual " if ewma_annualize else ""}Vol': min_vol
                    })
                    
                    st.dataframe(
                        stats_df.style.format({
                            f'Recent {"Annual " if ewma_annualize else ""}Vol': '{:.2%}',
                            f'Avg {"Annual " if ewma_annualize else ""}Vol': '{:.2%}',
                            f'Max {"Annual " if ewma_annualize else ""}Vol': '{:.2%}',
                            f'Min {"Annual " if ewma_annualize else ""}Vol': '{:.2%}'
                        }).background_gradient(
                            subset=[
                                f'Recent {"Annual " if ewma_annualize else ""}Vol',
                                f'Avg {"Annual " if ewma_annualize else ""}Vol'
                            ], 
                            cmap='Reds'
                        ),
                        use_container_width=True,
                        height=300
                    )
            
            # EWMA Correlation Matrix
            st.subheader("🔗 EWMA Dynamic Correlation")
            
            if st.button("🔄 Calculate EWMA Correlation", key="tab5_calc_ewma_corr"):
                with st.spinner("Calculating EWMA correlation matrix..."):
                    ewma_corr = ewma_analyzer.calculate_ewma_correlation(
                        returns[selected_tickers],
                        lambda_param=lambda_param
                    )
                    
                    if not ewma_corr.empty:
                        # Display heatmap
                        fig = VisualizationUtils.create_correlation_heatmap(ewma_corr)
                        fig.update_layout(title="EWMA Correlation Matrix")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Compare with simple correlation
                        simple_corr = returns[selected_tickers].corr()
                        
                        # Calculate difference
                        if ewma_corr.shape == simple_corr.shape:
                            corr_diff = ewma_corr - simple_corr
                            
                            st.subheader("📊 Correlation Comparison")
                            
                            col_comp1, col_comp2 = st.columns(2)
                            
                            with col_comp1:
                                st.metric(
                                    "Avg EWMA Correlation",
                                    f"{ewma_corr.values[np.triu_indices_from(ewma_corr.values, k=1)].mean():.3f}"
                                )
                            
                            with col_comp2:
                                st.metric(
                                    "Avg Simple Correlation", 
                                    f"{simple_corr.values[np.triu_indices_from(simple_corr.values, k=1)].mean():.3f}"
                                )
                            
                            # Difference heatmap
                            fig = go.Figure(data=go.Heatmap(
                                z=corr_diff.values,
                                x=corr_diff.columns,
                                y=corr_diff.index,
                                colorscale='RdBu',
                                zmid=0,
                                text=corr_diff.round(3).values,
                                texttemplate='%{text}',
                                textfont={"size": 10},
                                hoverongaps=False,
                                hovertemplate='%{y} vs %{x}<br>Difference: %{z:.3f}<extra></extra>'
                            ))
                            
                            fig.update_layout(
                                title="EWMA - Simple Correlation Difference",
                                height=500,
                                template="plotly_dark",
                                xaxis_title="Assets",
                                yaxis_title="Assets",
                                xaxis=dict(tickangle=45),
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error in EWMA Analysis tab: {str(e)[:200]}")
            ErrorHandler.handle_error(e, "Tab 5: EWMA Analysis")
    
    # Tab 6: Monte Carlo VaR
    with tab6:
        st.header("🎲 Monte Carlo Simulation & VaR")
        
        try:
            returns = st.session_state.returns
            selected_tickers = st.session_state.selected_tickers
            
            # Simulation Configuration
            st.subheader("⚙️ Simulation Configuration")
            
            col_mc1, col_mc2, col_mc3 = st.columns(3)
            
            with col_mc1:
                mc_simulations = st.selectbox(
                    "Number of Simulations",
                    [1000, 5000, 10000, 25000, 50000],
                    index=2,
                    key="tab6_simulations"
                )
            
            with col_mc2:
                mc_horizon = st.selectbox(
                    "Time Horizon",
                    ["1 Day", "1 Week", "1 Month", "3 Months", "6 Months", "1 Year"],
                    index=2,
                    key="tab6_horizon"
                )
                
                # Convert to days
                horizon_days_map = {
                    "1 Day": 1,
                    "1 Week": 5,
                    "1 Month": 21,
                    "3 Months": 63,
                    "6 Months": 126,
                    "1 Year": 252
                }
                mc_horizon_days = horizon_days_map[mc_horizon]
            
            with col_mc3:
                mc_confidence = st.slider(
                    "Confidence Level",
                    min_value=0.90,
                    max_value=0.99,
                    value=0.95,
                    step=0.01,
                    key="tab6_confidence",
                    format="%.2f"
                )
            
            # Portfolio weights for simulation
            if 'current_weights' not in st.session_state:
                st.warning("Please run portfolio optimization first")
                st.stop()
            
            weights = st.session_state.current_weights
            
            # Run Monte Carlo Simulation
            if st.button("🚀 Run Monte Carlo Simulation", type="primary", key="tab6_run_simulation"):
                with st.spinner(f"Running {mc_simulations:,} Monte Carlo simulations..."):
                    simulator = MonteCarloSimulator()
                    
                    # Get asset returns
                    asset_returns = returns[selected_tickers]
                    
                    # Run simulation
                    simulation_results = simulator.simulate_portfolio_scenarios(
                        weights,
                        asset_returns,
                        n_sims=mc_simulations,
                        horizon_days=mc_horizon_days,
                        seed=42
                    )
                    
                    # Store results
                    st.session_state.mc_results = simulation_results
                    st.session_state.mc_horizon = mc_horizon
                    st.session_state.mc_confidence = mc_confidence
                    
                    st.success(f"✅ Simulation complete with {mc_simulations:,} scenarios")
            
            # Display simulation results if available
            if 'mc_results' in st.session_state:
                results = st.session_state.mc_results
                
                # Key results metrics
                st.subheader("📊 Simulation Results")
                
                col_res1, col_res2, col_res3, col_res4 = st.columns(4)
                
                with col_res1:
                    st.metric(
                        f"{mc_horizon} VaR ({mc_confidence*100:.0f}%)",
                        f"{results['var_95']:.2%}",
                        f"Value at Risk"
                    )
                
                with col_res2:
                    st.metric(
                        f"{mc_horizon} CVaR ({mc_confidence*100:.0f}%)",
                        f"{results['cvar_95']:.2%}",
                        f"Conditional VaR"
                    )
                
                with col_res3:
                    st.metric(
                        "Probability of Loss",
                        f"{results['probability_loss']:.2%}",
                        f"P(Portfolio < 0)"
                    )
                
                with col_res4:
                    st.metric(
                        f"Probability > 10%",
                        f"{results['probability_gain_10']:.2%}",
                        f"P(Portfolio > 10%)"
                    )
                
                # Additional statistics
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                
                with col_stat1:
                    st.metric("Mean Return", f"{results['mean_return']:.2%}")
                
                with col_stat2:
                    st.metric("Median Return", f"{results['median_return']:.2%}")
                
                with col_stat3:
                    st.metric("Std Dev", f"{results['std_return']:.2%}")
                
                # Distribution of final returns
                st.subheader("📈 Return Distribution")
                
                fig = go.Figure()
                
                # Histogram of final returns
                fig.add_trace(go.Histogram(
                    x=results['final_returns'],
                    nbinsx=50,
                    name="Return Distribution",
                    marker_color='#1a5fb4',
                    opacity=0.7,
                    histnorm='probability density',
                    hovertemplate='Return: %{x:.2%}<br>Density: %{y:.4f}<extra></extra>'
                ))
                
                # Add VaR and CVaR lines
                fig.add_vline(
                    x=results['var_95'],
                    line_dash="dash",
                    line_color="#f5a623",
                    annotation_text=f"VaR: {results['var_95']:.2%}",
                    annotation_position="top left"
                )
                
                fig.add_vline(
                    x=results['cvar_95'],
                    line_dash="dot",
                    line_color="#c01c28",
                    annotation_text=f"CVaR: {results['cvar_95']:.2%}",
                    annotation_position="top right"
                )
                
                # Add mean and median
                fig.add_vline(
                    x=results['mean_return'],
                    line_dash="solid",
                    line_color="#26a269",
                    annotation_text=f"Mean: {results['mean_return']:.2%}",
                    annotation_position="bottom right"
                )
                
                fig.add_vline(
                    x=results['median_return'],
                    line_dash="solid",
                    line_color="#c01c28",
                    annotation_text=f"Median: {results['median_return']:.2%}",
                    annotation_position="bottom left"
                )
                
                fig.update_layout(
                    height=500,
                    title=f"Monte Carlo Simulation Results ({mc_horizon}, {mc_simulations:,} simulations)",
                    template="plotly_dark",
                    xaxis_title=f"{mc_horizon} Return",
                    yaxis_title="Density",
                    showlegend=False,
                    hovermode='x unified',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(tickformat=".0%"),
                    bargap=0.1
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Cumulative probability plot
                st.subheader("📊 Cumulative Probability Distribution")
                
                sorted_returns = np.sort(results['final_returns'])
                cumulative_prob = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=sorted_returns,
                    y=cumulative_prob,
                    mode='lines',
                    name="CDF",
                    line=dict(color='#1a5fb4', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(26, 95, 180, 0.1)',
                    hovertemplate='Return: %{x:.2%}<br>Cumulative Probability: %{y:.2%}<extra></extra>'
                ))
                
                # Add confidence level lines
                fig.add_hline(
                    y=mc_confidence,
                    line_dash="dash",
                    line_color="#f5a623",
                    annotation_text=f"{mc_confidence*100:.0f}% Confidence",
                    annotation_position="top right"
                )
                
                # Find VaR from CDF
                var_idx = np.searchsorted(cumulative_prob, 1 - mc_confidence)
                if var_idx < len(sorted_returns):
                    var_value = sorted_returns[var_idx]
                    fig.add_vline(
                        x=var_value,
                        line_dash="dash",
                        line_color="#f5a623",
                        annotation_text=f"VaR: {var_value:.2%}",
                        annotation_position="bottom left"
                    )
                
                fig.update_layout(
                    height=400,
                    title="Cumulative Distribution Function",
                    template="plotly_dark",
                    xaxis_title=f"{mc_horizon} Return",
                    yaxis_title="Cumulative Probability",
                    hovermode='x unified',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(tickformat=".0%"),
                    yaxis=dict(tickformat=".0%")
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Path visualization
                st.subheader("📈 Simulated Portfolio Paths")
                
                # Select number of paths to display
                n_paths_display = st.slider(
                    "Number of paths to display",
                    min_value=10,
                    max_value=100,
                    value=50,
                    step=10,
                    key="tab6_n_paths"
                )
                
                paths = results['paths']
                
                fig = go.Figure()
                
                # Display selected paths
                for i in range(min(n_paths_display, mc_simulations)):
                    path_idx = np.random.randint(0, mc_simulations) if n_paths_display < mc_simulations else i
                    fig.add_trace(go.Scatter(
                        x=np.arange(len(paths)),
                        y=paths[:, path_idx],
                        mode='lines',
                        line=dict(width=1, color='rgba(26, 95, 180, 0.2)'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                # Add average path
                avg_path = paths.mean(axis=1)
                fig.add_trace(go.Scatter(
                    x=np.arange(len(paths)),
                    y=avg_path,
                    mode='lines',
                    name='Average Path',
                    line=dict(width=3, color='#f5a623'),
                    hovertemplate='Day %{x}<br>Value: %{y:.2f}<extra></extra>'
                ))
                
                # Add median path
                median_path = np.median(paths, axis=1)
                fig.add_trace(go.Scatter(
                    x=np.arange(len(paths)),
                    y=median_path,
                    mode='lines',
                    name='Median Path',
                    line=dict(width=2, color='#26a269', dash='dash'),
                    hovertemplate='Day %{x}<br>Value: %{y:.2f}<extra></extra>'
                ))
                
                # Add confidence bands
                lower_band = np.percentile(paths, 5, axis=1)
                upper_band = np.percentile(paths, 95, axis=1)
                
                fig.add_trace(go.Scatter(
                    x=np.arange(len(paths)),
                    y=upper_band,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=np.arange(len(paths)),
                    y=lower_band,
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(26, 95, 180, 0.1)',
                    name='90% Confidence Band',
                    line=dict(width=0),
                    hovertemplate='Day %{x}<br>Upper: %{y:.2f}<extra></extra>'
                ))
                
                fig.update_layout(
                    height=500,
                    title=f"Simulated Portfolio Paths ({n_paths_display} of {mc_simulations:,} shown)",
                    template="plotly_dark",
                    xaxis_title="Days",
                    yaxis_title="Portfolio Value (Starting at 1.0)",
                    hovermode='x unified',
                    showlegend=True,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Sensitivity analysis
                st.subheader("🔍 Sensitivity Analysis")
                
                col_sens1, col_sens2 = st.columns(2)
                
                with col_sens1:
                    st.write("**VaR by Confidence Level**")
                    confidence_levels = [0.90, 0.95, 0.99]
                    var_by_conf = []
                    
                    for conf in confidence_levels:
                        var = np.percentile(results['final_returns'], (1 - conf) * 100)
                        var_by_conf.append({
                            'Confidence': f'{conf*100:.0f}%',
                            'VaR': var,
                            'CVaR': np.mean(results['final_returns'][results['final_returns'] <= var])
                        })
                    
                    var_df = pd.DataFrame(var_by_conf)
                    st.dataframe(
                        var_df.style.format({
                            'VaR': '{:.4%}',
                            'CVaR': '{:.4%}'
                        }),
                        use_container_width=True,
                        height=150
                    )
                
                with col_sens2:
                    st.write("**Return Percentiles**")
                    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
                    percentile_returns = []
                    
                    for p in percentiles:
                        value = np.percentile(results['final_returns'], p)
                        percentile_returns.append({
                            'Percentile': f'{p}th',
                            'Return': value
                        })
                    
                    perc_df = pd.DataFrame(percentile_returns)
                    st.dataframe(
                        perc_df.style.format({
                            'Return': '{:.4%}'
                        }),
                        use_container_width=True,
                        height=150
                    )
        
        except Exception as e:
            st.error(f"Error in Monte Carlo VaR tab: {str(e)[:200]}")
            ErrorHandler.handle_error(e, "Tab 6: Monte Carlo VaR")
    
    # Tab 7: Performance Attribution
    with tab7:
        st.header("📊 Performance Attribution Analysis")
        
        try:
            returns = st.session_state.returns
            selected_tickers = st.session_state.selected_tickers
            
            if 'current_weights' not in st.session_state:
                st.warning("Please run portfolio optimization in Tab 1 first")
                st.stop()
            
            weights = st.session_state.current_weights
            portfolio_returns = st.session_state.portfolio_returns
            benchmark = st.session_state.benchmark
            benchmark_returns = returns[benchmark]
            
            # Performance Attribution Configuration
            st.subheader("⚙️ Attribution Configuration")
            
            col_att1, col_att2 = st.columns(2)
            
            with col_att1:
                attribution_method = st.selectbox(
                    "Attribution Method",
                    ["Brinson-Fachler", "Brinson-Hood-Beebower", "Top-Down", "Bottom-Up"],
                    index=0,
                    key="tab7_attribution_method"
                )
            
            with col_att2:
                attribution_period = st.selectbox(
                    "Attribution Period",
                    ["Daily", "Weekly", "Monthly", "Quarterly", "Annually"],
                    index=2,
                    key="tab7_attribution_period"
                )
            
            # Benchmark weights (can be market-cap weighted or equal-weighted)
            st.subheader("📊 Benchmark Configuration")
            
            benchmark_weight_method = st.radio(
                "Benchmark Weighting Method",
                ["Equal Weight", "Market Cap Proxy", "Custom Weights"],
                index=0,
                key="tab7_benchmark_method"
            )
            
            if benchmark_weight_method == "Market Cap Proxy":
                # Use volatility inverse as proxy for market cap
                volatilities = returns[selected_tickers].std()
                benchmark_weights = (1 / volatilities) / (1 / volatilities).sum()
            elif benchmark_weight_method == "Custom Weights":
                st.write("Enter custom benchmark weights:")
                custom_weights = {}
                for ticker in selected_tickers:
                    weight = st.number_input(
                        f"{ticker} weight",
                        min_value=0.0,
                        max_value=100.0,
                        value=100.0/len(selected_tickers),
                        step=1.0,
                        key=f"tab7_custom_{ticker}"
                    ) / 100
                    custom_weights[ticker] = weight
                
                total_weight = sum(custom_weights.values())
                if total_weight > 0:
                    benchmark_weights = np.array([custom_weights[t] / total_weight for t in selected_tickers])
                else:
                    benchmark_weights = np.ones(len(selected_tickers)) / len(selected_tickers)
                    st.warning("Weights sum to 0, using equal weights")
            else:  # Equal Weight
                benchmark_weights = np.ones(len(selected_tickers)) / len(selected_tickers)
            
            # Run Attribution Analysis
            if st.button("🚀 Run Attribution Analysis", type="primary", key="tab7_run_attribution"):
                with st.spinner("Running performance attribution analysis..."):
                    attribution = PerformanceAttribution()
                    
                    # Calculate Brinson attribution
                    attribution_result = attribution.calculate_brinson_attribution(
                        portfolio_returns,
                        benchmark_returns,
                        weights,
                        benchmark_weights,
                        returns[selected_tickers]
                    )
                    
                    if attribution_result['success']:
                        st.session_state.attribution_result = attribution_result
                        st.success("✅ Attribution analysis complete")
                    else:
                        st.error(f"Attribution failed: {attribution_result.get('error', 'Unknown error')}")
            
            # Display Attribution Results
            if 'attribution_result' in st.session_state:
                result = st.session_state.attribution_result
                
                st.subheader("📈 Attribution Results")
                
                # Overall attribution metrics
                col_attr1, col_attr2, col_attr3, col_attr4 = st.columns(4)
                
                with col_attr1:
                    st.metric(
                        "Total Active Return",
                        f"{result['total_active_return']:.4%}",
                        "vs Benchmark"
                    )
                
                with col_attr2:
                    allocation_pct = (result['allocation_effect'] / result['total_active_return'] * 100 
                                    if result['total_active_return'] != 0 else 0)
                    st.metric(
                        "Allocation Effect",
                        f"{result['allocation_effect']:.4%}",
                        f"{allocation_pct:.1f}% of total"
                    )
                
                with col_attr3:
                    selection_pct = (result['selection_effect'] / result['total_active_return'] * 100 
                                   if result['total_active_return'] != 0 else 0)
                    st.metric(
                        "Selection Effect",
                        f"{result['selection_effect']:.4%}",
                        f"{selection_pct:.1f}% of total"
                    )
                
                with col_attr4:
                    interaction_pct = (result['interaction_effect'] / result['total_active_return'] * 100 
                                     if result['total_active_return'] != 0 else 0)
                    st.metric(
                        "Interaction Effect",
                        f"{result['interaction_effect']:.4%}",
                        f"{interaction_pct:.1f}% of total"
                    )
                
                # Attribution breakdown visualization
                st.subheader("📊 Attribution Breakdown")
                
                attribution_data = {
                    'Component': ['Allocation', 'Selection', 'Interaction', 'Residual'],
                    'Value': [
                        result['allocation_effect'],
                        result['selection_effect'],
                        result['interaction_effect'],
                        result['residual']
                    ],
                    'Percentage': [
                        allocation_pct,
                        selection_pct,
                        interaction_pct,
                        (result['residual'] / result['total_active_return'] * 100 
                         if result['total_active_return'] != 0 else 0)
                    ]
                }
                
                fig = go.Figure()
                
                colors = ['#1a5fb4', '#26a269', '#f5a623', '#c01c28']
                
                fig.add_trace(go.Bar(
                    x=attribution_data['Component'],
                    y=attribution_data['Value'],
                    marker_color=colors,
                    text=attribution_data['Value'],
                    texttemplate='%{text:.4%}',
                    textposition='auto',
                    hovertemplate='%{x}<br>Value: %{y:.4%}<br>Percentage: %{customdata:.1f}%<extra></extra>',
                    customdata=attribution_data['Percentage']
                ))
                
                fig.update_layout(
                    height=400,
                    title="Performance Attribution Components",
                    template="plotly_dark",
                    xaxis_title="Component",
                    yaxis_title="Contribution",
                    hovermode='x unified',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(tickformat=".2%")
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed attribution by asset
                st.subheader("🔍 Detailed Asset-Level Attribution")
                
                # Calculate asset-level contributions
                asset_returns = returns[selected_tickers].mean() * 252
                benchmark_return = benchmark_returns.mean() * 252
                
                asset_contributions = []
                for i, asset in enumerate(selected_tickers):
                    # Allocation effect
                    allocation = (weights[i] - benchmark_weights[i]) * (asset_returns[asset] - benchmark_return)
                    
                    # Selection effect
                    selection = benchmark_weights[i] * (asset_returns[asset] - benchmark_return)
                    
                    asset_contributions.append({
                        'Asset': asset,
                        'Weight': weights[i],
                        'Benchmark Weight': benchmark_weights[i],
                        'Asset Return': asset_returns[asset],
                        'Allocation': allocation,
                        'Selection': selection,
                        'Total': allocation + selection
                    })
                
                asset_df = pd.DataFrame(asset_contributions).sort_values('Total', ascending=False)
                
                st.dataframe(
                    asset_df.style.format({
                        'Weight': '{:.2%}',
                        'Benchmark Weight': '{:.2%}',
                        'Asset Return': '{:.2%}',
                        'Allocation': '{:.4%}',
                        'Selection': '{:.4%}',
                        'Total': '{:.4%}'
                    }).background_gradient(
                        subset=['Total'], 
                        cmap='RdYlGn',
                        vmin=-0.05,
                        vmax=0.05
                    ),
                    use_container_width=True,
                    height=400
                )
                
                # Rolling attribution analysis
                st.subheader("📈 Rolling Attribution Analysis")
                
                if len(portfolio_returns) > 126:
                    window = st.slider(
                        "Rolling Window (days)",
                        min_value=20,
                        max_value=252,
                        value=63,
                        key="tab7_rolling_window"
                    )
                    
                    rolling_allocation = []
                    rolling_selection = []
                    dates = []
                    
                    for i in range(window, len(portfolio_returns)):
                        window_returns = returns[selected_tickers].iloc[i-window:i]
                        window_port_returns = portfolio_returns.iloc[i-window:i]
                        window_bench_returns = benchmark_returns.iloc[i-window:i]
                        
                        # Calculate window averages
                        asset_means = window_returns.mean() * 252
                        port_mean = window_port_returns.mean() * 252
                        bench_mean = window_bench_returns.mean() * 252
                        
                        # Simple attribution calculation
                        allocation = 0
                        selection = 0
                        
                        for j, asset in enumerate(selected_tickers):
                            allocation += (weights[j] - benchmark_weights[j]) * (asset_means[asset] - bench_mean)
                            selection += benchmark_weights[j] * (asset_means[asset] - bench_mean)
                        
                        rolling_allocation.append(allocation)
                        rolling_selection.append(selection)
                        dates.append(portfolio_returns.index[i])
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=rolling_allocation,
                        name='Allocation Effect',
                        line=dict(color='#1a5fb4', width=2),
                        hovertemplate='%{x|%Y-%m-%d}<br>Allocation: %{y:.4%}<extra></extra>'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=rolling_selection,
                        name='Selection Effect',
                        line=dict(color='#26a269', width=2),
                        hovertemplate='%{x|%Y-%m-%d}<br>Selection: %{y:.4%}<extra></extra>'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=np.array(rolling_allocation) + np.array(rolling_selection),
                        name='Total Active',
                        line=dict(color='#f5a623', width=3),
                        hovertemplate='%{x|%Y-%m-%d}<br>Total Active: %{y:.4%}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        height=500,
                        title=f"Rolling Attribution Analysis ({window}-day window)",
                        template="plotly_dark",
                        xaxis_title="Date",
                        yaxis_title="Contribution",
                        hovermode='x unified',
                        showlegend=True,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        yaxis=dict(tickformat=".2%")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error in Performance Attribution tab: {str(e)[:200]}")
            ErrorHandler.handle_error(e, "Tab 7: Performance Attribution")
    
    # Tab 8: Report & Export
    with tab8:
        st.header("📋 Comprehensive Report & Export")
        
        try:
            # Check if we have all necessary data
            if 'portfolio_returns' not in st.session_state:
                st.warning("Please run portfolio analysis in Tab 1 first")
                st.stop()
            
            portfolio_analysis = {}
            weights = {}
            
            if 'current_weights' in st.session_state:
                weights_array = st.session_state.current_weights
                selected_tickers = st.session_state.selected_tickers
                weights = {asset: weights_array[i] for i, asset in enumerate(selected_tickers)}
            
            # Report Configuration
            st.subheader("⚙️ Report Configuration")
            
            col_rep1, col_rep2, col_rep3 = st.columns(3)
            
            with col_rep1:
                report_type = st.selectbox(
                    "Report Type",
                    ["Summary Report", "Detailed Analysis", "Executive Summary", "Risk Report"],
                    index=0,
                    key="tab8_report_type"
                )
            
            with col_rep2:
                include_charts = st.checkbox(
                    "Include Charts",
                    value=True,
                    key="tab8_include_charts"
                )
            
            with col_rep3:
                include_export = st.checkbox(
                    "Enable Export",
                    value=True,
                    key="tab8_include_export"
                )
            
            # Generate Report
            if st.button("📊 Generate Report", type="primary", key="tab8_generate_report"):
                with st.spinner("Generating comprehensive report..."):
                    # Create report generator instance
                    report_gen = ReportGenerator()
                    
                    # Generate summary report
                    benchmark = st.session_state.benchmark
                    summary_report = report_gen.generate_summary_report(
                        portfolio_analysis,
                        benchmark
                    )
                    
                    # Display report
                    st.subheader("📄 Generated Report")
                    
                    # Create tabs for different report formats
                    report_tab1, report_tab2, report_tab3 = st.tabs(["📝 Summary", "📊 Metrics", "📈 Charts"])
                    
                    with report_tab1:
                        st.markdown(summary_report)
                    
                    with report_tab2:
                        # Display detailed metrics
                        if 'portfolio_returns' in st.session_state:
                            portfolio_returns = st.session_state.portfolio_returns
                            benchmark_returns = st.session_state.returns[benchmark]
                            
                            metrics_calculator = PerformanceMetrics()
                            metrics = metrics_calculator.calculate_metrics(
                                portfolio_returns,
                                benchmark_returns,
                                rf_annual
                            )
                            
                            # Display metrics in a nice format
                            col_met1, col_met2 = st.columns(2)
                            
                            with col_met1:
                                st.subheader("📈 Return Metrics")
                                for key in ['annual_return', 'sharpe_ratio', 'sortino_ratio', 
                                           'calmar_ratio', 'omega_ratio', 'm2_measure']:
                                    if key in metrics and not pd.isna(metrics[key]):
                                        if 'return' in key or 'measure' in key:
                                            st.metric(key.replace('_', ' ').title(), 
                                                     f"{metrics[key]:.2%}")
                                        else:
                                            st.metric(key.replace('_', ' ').title(), 
                                                     f"{metrics[key]:.2f}")
                            
                            with col_met2:
                                st.subheader("⚖️ Risk Metrics")
                                for key in ['annual_volatility', 'max_drawdown', 'var_95', 
                                           'cvar_95', 'ulcer_index', 'pain_index']:
                                    if key in metrics and not pd.isna(metrics[key]):
                                        st.metric(key.replace('_', ' ').title(), 
                                                 f"{metrics[key]:.2%}")
                    
                    with report_tab3:
                        if include_charts and 'portfolio_returns' in st.session_state:
                            portfolio_returns = st.session_state.portfolio_returns
                            
                            # Cumulative returns chart
                            portfolio_cumulative = (1 + portfolio_returns).cumprod()
                            benchmark_cumulative = (1 + st.session_state.returns[benchmark]).cumprod()
                            
                            fig1 = VisualizationUtils.create_performance_chart(
                                portfolio_cumulative,
                                benchmark_cumulative,
                                title="Portfolio vs Benchmark Performance"
                            )
                            st.plotly_chart(fig1, use_container_width=True)
                            
                            # Drawdown chart
                            running_max = portfolio_cumulative.expanding().max()
                            drawdown = (portfolio_cumulative - running_max) / running_max
                            
                            fig2 = VisualizationUtils.create_drawdown_chart(drawdown)
                            st.plotly_chart(fig2, use_container_width=True)
                            
                            # Rolling metrics
                            if len(portfolio_returns) > 63:
                                rolling_vol = portfolio_returns.rolling(window=63).std() * np.sqrt(252)
                                rolling_sharpe = (portfolio_returns.rolling(window=63).mean() * 252 - rf_annual) / (rolling_vol + 1e-10)
                                
                                fig3 = make_subplots(
                                    rows=2, cols=1,
                                    subplot_titles=("63-Day Rolling Volatility", "63-Day Rolling Sharpe"),
                                    vertical_spacing=0.1
                                )
                                
                                fig3.add_trace(
                                    go.Scatter(
                                        x=rolling_vol.index,
                                        y=rolling_vol.values * 100,
                                        name="Rolling Volatility",
                                        line=dict(color='#1a5fb4', width=2),
                                        hovertemplate='%{x|%Y-%m-%d}<br>Volatility: %{y:.1f}%<extra></extra>'
                                    ),
                                    row=1, col=1
                                )
                                
                                fig3.add_trace(
                                    go.Scatter(
                                        x=rolling_sharpe.index,
                                        y=rolling_sharpe.values,
                                        name="Rolling Sharpe",
                                        line=dict(color='#26a269', width=2),
                                        hovertemplate='%{x|%Y-%m-%d}<br>Sharpe: %{y:.2f}<extra></extra>'
                                    ),
                                    row=2, col=1
                                )
                                
                                fig3.update_layout(
                                    height=600,
                                    template="plotly_dark",
                                    showlegend=False,
                                    hovermode='x unified'
                                )
                                
                                fig3.update_yaxes(title_text="Volatility (%)", row=1, col=1)
                                fig3.update_yaxes(title_text="Sharpe Ratio", row=2, col=1)
                                fig3.update_xaxes(title_text="Date", row=2, col=1)
                                
                                st.plotly_chart(fig3, use_container_width=True)
                    
                    # Export Options
                    if include_export:
                        st.subheader("💾 Export Options")
                        
                        col_exp1, col_exp2, col_exp3 = st.columns(3)
                        
                        with col_exp1:
                            # Export as CSV
                            if st.button("📁 Export as CSV", key="tab8_export_csv"):
                                # Create DataFrame with weights
                                weights_df = pd.DataFrame({
                                    'Asset': list(weights.keys()),
                                    'Weight': list(weights.values())
                                })
                                
                                csv = weights_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Weights CSV",
                                    data=csv,
                                    file_name="portfolio_weights.csv",
                                    mime="text/csv"
                                )
                        
                        with col_exp2:
                            # Export as JSON
                            if st.button("📄 Export as JSON", key="tab8_export_json"):
                                export_data = {
                                    'portfolio': {
                                        'weights': weights,
                                        'strategy': st.session_state.portfolio_strategy,
                                        'assets': selected_tickers,
                                        'benchmark': benchmark
                                    },
                                    'metrics': metrics if 'metrics' in locals() else {}
                                }
                                
                                json_str = json.dumps(export_data, indent=2, default=str)
                                st.download_button(
                                    label="Download Portfolio JSON",
                                    data=json_str,
                                    file_name="portfolio_config.json",
                                    mime="application/json"
                                )
                        
                        with col_exp3:
                            # Export as HTML Report
                            if st.button("🌐 Export as HTML", key="tab8_export_html"):
                                html_report = report_gen.generate_html_report(
                                    portfolio_analysis,
                                    weights,
                                    benchmark
                                )
                                
                                st.download_button(
                                    label="Download HTML Report",
                                    data=html_report,
                                    file_name="portfolio_report.html",
                                    mime="text/html"
                                )
                        
                        # Export all data
                        st.subheader("📊 Export All Data")
                        
                        if st.button("💾 Export Complete Dataset", key="tab8_export_all"):
                            # Create a ZIP file with all data
                            import zipfile
                            import io
                            
                            zip_buffer = io.BytesIO()
                            
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                # Add weights
                                weights_df = pd.DataFrame({
                                    'Asset': list(weights.keys()),
                                    'Weight': list(weights.values())
                                })
                                zip_file.writestr('weights.csv', weights_df.to_csv(index=False))
                                
                                # Add returns data
                                returns_df = st.session_state.returns
                                zip_file.writestr('returns.csv', returns_df.to_csv())
                                
                                # Add prices data
                                prices_df = st.session_state.prices
                                zip_file.writestr('prices.csv', prices_df.to_csv())
                                
                                # Add metrics
                                if 'metrics' in locals():
                                    metrics_df = pd.DataFrame([metrics])
                                    zip_file.writestr('metrics.csv', metrics_df.to_csv(index=False))
                                
                                # Add summary report
                                zip_file.writestr('summary_report.md', summary_report)
                            
                            zip_buffer.seek(0)
                            
                            st.download_button(
                                label="📦 Download Complete Dataset (ZIP)",
                                data=zip_buffer,
                                file_name="portfolio_data.zip",
                                mime="application/zip"
                            )
            
            # Portfolio Snapshot
            st.subheader("📸 Portfolio Snapshot")
            
            if 'current_weights' in st.session_state:
                # Display quick snapshot
                col_snap1, col_snap2 = st.columns(2)
                
                with col_snap1:
                    st.write("**Current Allocation**")
                    weights_df = pd.DataFrame({
                        'Asset': list(weights.keys()),
                        'Weight': list(weights.values())
                    }).sort_values('Weight', ascending=False)
                    
                    # Top 5 holdings
                    top_5 = weights_df.head(5)
                    other_weight = weights_df.iloc[5:]['Weight'].sum() if len(weights_df) > 5 else 0
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=list(top_5['Asset']) + (['Other'] if other_weight > 0 else []),
                        values=list(top_5['Weight']) + ([other_weight] if other_weight > 0 else []),
                        hole=0.3,
                        marker_colors=px.colors.sequential.Blues
                    )])
                    
                    fig.update_layout(
                        height=300,
                        showlegend=True,
                                                margin=dict(t=0, b=0, l=0, r=0),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_snap2:
                    st.write("**Quick Stats**")
                    
                    if 'portfolio_returns' in st.session_state:
                        portfolio_returns = st.session_state.portfolio_returns
                        benchmark_returns = st.session_state.returns[benchmark]
                        
                        # Calculate quick metrics
                        ann_return = portfolio_returns.mean() * 252
                        ann_vol = portfolio_returns.std() * np.sqrt(252)
                        
                        if ann_vol > 0:
                            sharpe = (ann_return - rf_annual) / ann_vol
                        else:
                            sharpe = 0
                        
                        cumulative = (1 + portfolio_returns).cumprod()
                        running_max = cumulative.expanding().max()
                        max_dd = ((cumulative - running_max) / running_max).min()
                        
                        # Display metrics
                        st.metric("Annual Return", f"{ann_return:.2%}")
                        st.metric("Annual Volatility", f"{ann_vol:.2%}")
                        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                        st.metric("Max Drawdown", f"{max_dd:.2%}")
            
            # System Information
            with st.expander("🔧 System Information", expanded=False):
                col_sys1, col_sys2 = st.columns(2)
                
                with col_sys1:
                    st.write("**Application Info**")
                    st.write(f"Version: 5.1")
                    st.write(f"Last Refresh: {st.session_state.last_refresh if 'last_refresh' in st.session_state else 'N/A'}")
                    st.write(f"Assets Loaded: {len(st.session_state.selected_tickers) if 'selected_tickers' in st.session_state else 0}")
                    st.write(f"Data Points: {len(st.session_state.returns) if 'returns' in st.session_state else 0}")
                
                with col_sys2:
                    st.write("**Cache Status**")
                    cache_stats = EnhancedCache.get_cache_stats()
                    st.write(f"Cache Files: {cache_stats['total_files']}")
                    st.write(f"Cache Size: {cache_stats['total_size_mb']:.2f} MB")
                    
                    if st.button("Clear All Cache", key="tab8_clear_cache"):
                        EnhancedCache.clear_cache()
                        st.success("Cache cleared successfully!")
                        st.rerun()
        
        except Exception as e:
            st.error(f"Error in Report & Export tab: {str(e)[:200]}")
            ErrorHandler.handle_error(e, "Tab 8: Report & Export")
    
    # Footer
    st.markdown("---")
    col_footer1, col_footer2, col_footer3 = st.columns(3)
    
    with col_footer1:
        st.caption("Apollo/ENIGMA Portfolio Terminal v5.1")
    
    with col_footer2:
        st.caption("For Institutional Use Only")
    
    with col_footer3:
        st.caption("© 2024 Apollo ENIGMA Systems")

# -------------------------------------------------------------
# APPLICATION ENTRY POINT
# -------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        ErrorHandler.handle_error(e, "Main Application")
        
        # Show recovery options
        col_rec1, col_rec2 = st.columns(2)
        with col_rec1:
            if st.button("🔄 Restart Application"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        with col_rec2:
            if st.button("📊 Demo Mode"):
                st.session_state.use_demo_data = True
                st.session_state.selected_tickers = ["SPY", "QQQ", "TLT", "GLD", "AAPL", "MSFT"]
                st.rerun()
