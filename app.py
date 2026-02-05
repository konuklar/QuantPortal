# =============================================================
# 🏛️ Apollo/ENIGMA - Quant Portfolio Terminal v6.0
# Professional Portfolio Optimization with Enhanced Analytics
# Complete Quantitative Finance Implementation
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
from scipy import stats, optimize, special
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Callable
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
import base64
from io import BytesIO
from enum import Enum
import itertools

# Advanced imports for quantitative analysis
try:
    import arch
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    st.warning("ARCH package not installed. GARCH modeling will be limited.")

try:
    from pypfopt import expected_returns, risk_models, objective_functions, black_litterman
    from pypfopt.efficient_frontier import EfficientFrontier, EfficientCVaR, EfficientSemivariance
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    from pypfopt.objective_functions import L2_reg, transaction_cost
    from pypfopt.cla import CLA
    from pypfopt.hierarchical_portfolio import HRPOpt
    from pypfopt.risk_models import CovarianceShrinkage
    PYPFOPT_AVAILABLE = True
except ImportError as e:
    PYPFOPT_AVAILABLE = False
    st.warning(f"PyPortfolioOpt import error: {e}")

# Additional imports
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.stats import norm, t, skew, kurtosis
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
np.set_printoptions(precision=4, suppress=True)

# =============================================================
# ENHANCED GLOBAL ASSET UNIVERSE
# =============================================================
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
    
    # Technology Stocks
    "US_Tech_Stocks": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META",
        "AVGO", "ASML", "ORCL", "AMD", "INTC", "CSCO", "CRM",
        "ADBE", "NFLX", "PYPL", "QCOM", "TXN", "IBM"
    ],
    
    # Financial Stocks
    "US_Finance_Stocks": [
        "JPM", "BAC", "WFC", "C", "GS", "MS", "SCHW",
        "BLK", "AXP", "V", "MA", "PYPL", "COF", "DFS",
        "USB", "PNC", "TFC", "BK", "STT", "MMC"
    ],
    
    # Healthcare Stocks
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
        "ENEL.MI", "ENI.MI", "ISP.MI", "UCG.MI", "AI.PA"
    ]
}

# Flatten universe
ALL_TICKERS = []
for category in GLOBAL_ASSET_UNIVERSE.values():
    ALL_TICKERS.extend(category)
ALL_TICKERS = list(dict.fromkeys(ALL_TICKERS))

# =============================================================
# ENHANCED CONFIGURATION
# =============================================================
APP_TITLE = "🏛️ Apollo/ENIGMA - Quantitative Portfolio Terminal v6.0"
DEFAULT_RF_ANNUAL = 0.03
TRADING_DAYS = 252
MONTE_CARLO_SIMULATIONS = 10000
MAX_CACHE_SIZE = 100
CACHE_DIR = tempfile.gettempdir() + "/apollo_cache_v6/"

# Create cache directory
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

# Streamlit config
st.set_page_config(
    page_title="Apollo/ENIGMA - Quantitative Portfolio Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# =============================================================
# ENHANCED QUANTITATIVE ANALYTICS ENGINE
# =============================================================
class QuantitativeAnalytics:
    """Comprehensive quantitative analytics engine with advanced metrics"""
    
    class RiskMethod(Enum):
        HISTORICAL = "historical"
        PARAMETRIC = "parametric"
        MONTE_CARLO = "monte_carlo"
        EWMA = "ewma"
        GARCH = "garch"
        MODIFIED = "modified"
        
    @staticmethod
    def calculate_all_metrics(returns: pd.Series, benchmark_returns: pd.Series = None, 
                            risk_free_rate: float = 0.03) -> Dict:
        """Calculate comprehensive quantitative metrics"""
        
        metrics = {}
        returns_clean = returns.dropna()
        
        if len(returns_clean) < 5:
            return metrics
        
        # Basic Return Metrics
        metrics['total_return'] = (1 + returns_clean).prod() - 1
        metrics['annualized_return'] = (1 + metrics['total_return']) ** (TRADING_DAYS/len(returns_clean)) - 1
        metrics['arithmetic_mean'] = returns_clean.mean() * TRADING_DAYS
        metrics['geometric_mean'] = ((1 + returns_clean).prod() ** (TRADING_DAYS/len(returns_clean))) - 1
        
        # Volatility Metrics
        metrics['annualized_volatility'] = returns_clean.std() * np.sqrt(TRADING_DAYS)
        metrics['downside_deviation'] = returns_clean[returns_clean < 0].std() * np.sqrt(TRADING_DAYS)
        metrics['upside_deviation'] = returns_clean[returns_clean > 0].std() * np.sqrt(TRADING_DAYS)
        metrics['semi_deviation'] = returns_clean[returns_clean < returns_clean.mean()].std() * np.sqrt(TRADING_DAYS)
        
        # Risk-Adjusted Return Ratios
        if metrics['annualized_volatility'] > 0:
            metrics['sharpe_ratio'] = (metrics['annualized_return'] - risk_free_rate) / metrics['annualized_volatility']
            metrics['sharpe_ratio_arithmetic'] = (metrics['arithmetic_mean'] - risk_free_rate) / metrics['annualized_volatility']
        
        if metrics['downside_deviation'] > 0:
            metrics['sortino_ratio'] = (metrics['annualized_return'] - risk_free_rate) / metrics['downside_deviation']
            metrics['upside_ratio'] = (metrics['annualized_return'] - risk_free_rate) / metrics['upside_deviation']
        
        # Maximum Drawdown Metrics
        cumulative = (1 + returns_clean).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        metrics['max_drawdown'] = drawdown.min()
        metrics['max_drawdown_duration'] = QuantitativeAnalytics._calculate_drawdown_duration(drawdown)
        metrics['avg_drawdown'] = drawdown.mean()
        metrics['drawdown_std'] = drawdown.std()
        
        # Calmar and Sterling Ratios
        if abs(metrics['max_drawdown']) > 0:
            metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown'])
            metrics['sterling_ratio'] = metrics['annualized_return'] / abs(metrics['avg_drawdown'])
        
        # Value at Risk Metrics (Multiple Methods)
        var_metrics = QuantitativeAnalytics.calculate_all_var(returns_clean, confidence_level=0.95)
        metrics.update(var_metrics)
        
        # Conditional VaR (Expected Shortfall)
        metrics['cvar_95'] = QuantitativeAnalytics._calculate_cvar(returns_clean, 0.95)
        metrics['cvar_99'] = QuantitativeAnalytics._calculate_cvar(returns_clean, 0.99)
        
        # Higher Moments
        metrics['skewness'] = returns_clean.skew()
        metrics['kurtosis'] = returns_clean.kurtosis()
        metrics['excess_kurtosis'] = metrics['kurtosis'] - 3
        
        # Gain/Loss Statistics
        positive_returns = returns_clean[returns_clean > 0]
        negative_returns = returns_clean[returns_clean < 0]
        
        if len(positive_returns) > 0:
            metrics['avg_gain'] = positive_returns.mean()
            metrics['max_gain'] = positive_returns.max()
            metrics['gain_std'] = positive_returns.std()
        
        if len(negative_returns) > 0:
            metrics['avg_loss'] = negative_returns.mean()
            metrics['max_loss'] = negative_returns.min()
            metrics['loss_std'] = negative_returns.std()
        
        # Win/Loss Ratios
        metrics['win_rate'] = len(positive_returns) / len(returns_clean)
        metrics['profit_factor'] = abs(positive_returns.sum() / negative_returns.sum()) if len(negative_returns) > 0 else np.inf
        metrics['gain_loss_ratio'] = abs(metrics['avg_gain'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else np.inf
        
        # Specialized Ratios
        metrics['omega_ratio'] = QuantitativeAnalytics._calculate_omega_ratio(returns_clean, risk_free_rate/TRADING_DAYS)
        metrics['burke_ratio'] = QuantitativeAnalytics._calculate_burke_ratio(returns_clean, risk_free_rate)
        metrics['m2_measure'] = risk_free_rate + metrics.get('sharpe_ratio', 0) * metrics['annualized_volatility'] if 'sharpe_ratio' in metrics else 0
        metrics['treynor_ratio'] = QuantitativeAnalytics._calculate_treynor_ratio(returns_clean, benchmark_returns, risk_free_rate) if benchmark_returns is not None else np.nan
        metrics['information_ratio'] = QuantitativeAnalytics._calculate_information_ratio(returns_clean, benchmark_returns) if benchmark_returns is not None else np.nan
        
        # Tail Risk Metrics
        metrics['var_ratio'] = abs(metrics['var_95'] / metrics['var_99']) if metrics['var_99'] != 0 else np.nan
        metrics['conditional_skewness'] = QuantitativeAnalytics._calculate_conditional_skewness(returns_clean)
        metrics['conditional_kurtosis'] = QuantitativeAnalytics._calculate_conditional_kurtosis(returns_clean)
        
        # Ulcer and Pain Metrics
        metrics['ulcer_index'] = np.sqrt((drawdown ** 2).sum() / len(drawdown))
        metrics['pain_index'] = abs(drawdown).sum() / len(drawdown)
        metrics['martin_ratio'] = metrics['annualized_return'] / metrics['ulcer_index'] if metrics['ulcer_index'] > 0 else 0
        
        # Drawdown-based Ratios
        metrics['recovery_factor'] = metrics['total_return'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] < 0 else np.nan
        metrics['romad_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] < 0 else np.nan
        
        # Statistical Tests
        metrics['jarque_bera'] = QuantitativeAnalytics._jarque_bera_test(returns_clean)
        metrics['shapiro_wilk'] = QuantitativeAnalytics._shapiro_wilk_test(returns_clean)
        metrics['lilliefors'] = QuantitativeAnalytics._lilliefors_test(returns_clean)
        
        # Serial Correlation
        metrics['autocorrelation_1'] = returns_clean.autocorr(lag=1)
        metrics['autocorrelation_5'] = returns_clean.autocorr(lag=5)
        
        # Risk Decomposition
        metrics['var_contribution'] = QuantitativeAnalytics._calculate_var_contribution(returns_clean)
        metrics['cvar_contribution'] = QuantitativeAnalytics._calculate_cvar_contribution(returns_clean, 0.95)
        
        return metrics
    
    @staticmethod
    def calculate_all_var(returns: pd.Series, confidence_level: float = 0.95) -> Dict:
        """Calculate Value at Risk using multiple methods"""
        
        var_results = {}
        
        # Historical VaR
        var_results['var_historical'] = np.percentile(returns, (1 - confidence_level) * 100)
        
        # Parametric VaR (Normal)
        mu, sigma = returns.mean(), returns.std()
        z_score = norm.ppf(1 - confidence_level)
        var_results['var_parametric_normal'] = mu + z_score * sigma
        
        # Parametric VaR (Student's t)
        try:
            df, mu_t, sigma_t = t.fit(returns)
            t_score = t.ppf(1 - confidence_level, df)
            var_results['var_parametric_t'] = mu_t + t_score * sigma_t
        except:
            var_results['var_parametric_t'] = np.nan
        
        # Modified VaR (Cornish-Fisher)
        skew_val = returns.skew()
        kurt_val = returns.kurtosis()
        z_cf = z_score + (z_score**2 - 1) * skew_val/6 + (z_score**3 - 3*z_score) * (kurt_val-3)/24 - (2*z_score**3 - 5*z_score) * skew_val**2/36
        var_results['var_modified'] = mu + z_cf * sigma
        
        # EWMA VaR
        try:
            var_results['var_ewma'] = QuantitativeAnalytics._calculate_ewma_var(returns, confidence_level)
        except:
            var_results['var_ewma'] = np.nan
        
        # Monte Carlo VaR
        try:
            var_results['var_monte_carlo'] = QuantitativeAnalytics._calculate_monte_carlo_var(returns, confidence_level)
        except:
            var_results['var_monte_carlo'] = np.nan
        
        # GARCH VaR (if ARCH available)
        if ARCH_AVAILABLE:
            try:
                var_results['var_garch'] = QuantitativeAnalytics._calculate_garch_var(returns, confidence_level)
            except:
                var_results['var_garch'] = np.nan
        
        return var_results
    
    @staticmethod
    def _calculate_ewma_var(returns: pd.Series, confidence_level: float = 0.95, lambda_param: float = 0.94) -> float:
        """Calculate EWMA VaR"""
        returns_squared = returns ** 2
        ewma_var = returns_squared.ewm(alpha=1-lambda_param, adjust=False).mean()
        ewma_vol = np.sqrt(ewma_var.iloc[-1])
        z_score = norm.ppf(1 - confidence_level)
        return z_score * ewma_vol
    
    @staticmethod
    def _calculate_monte_carlo_var(returns: pd.Series, confidence_level: float = 0.95, 
                                 n_simulations: int = 10000, horizon: int = 1) -> float:
        """Calculate Monte Carlo VaR"""
        mu, sigma = returns.mean(), returns.std()
        simulations = np.random.normal(mu, sigma, (horizon, n_simulations))
        portfolio_values = np.ones(n_simulations)
        
        for t in range(horizon):
            portfolio_values *= (1 + simulations[t])
        
        final_returns = portfolio_values - 1
        return np.percentile(final_returns, (1 - confidence_level) * 100)
    
    @staticmethod
    def _calculate_garch_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate GARCH VaR"""
        if not ARCH_AVAILABLE:
            return np.nan
        
        try:
            # Fit GARCH(1,1) model
            model = arch.arch_model(returns * 100, vol='Garch', p=1, q=1)
            res = model.fit(disp='off')
            forecast = res.forecast(horizon=1)
            conditional_vol = np.sqrt(forecast.variance.values[-1, 0]) / 100  # Convert back from percentage
            z_score = norm.ppf(1 - confidence_level)
            return z_score * conditional_vol
        except Exception as e:
            return np.nan
    
    @staticmethod
    def _calculate_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = np.percentile(returns, (1 - confidence_level) * 100)
        tail_returns = returns[returns <= var]
        return tail_returns.mean() if len(tail_returns) > 0 else var
    
    @staticmethod
    def _calculate_omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
        """Calculate Omega Ratio"""
        excess_returns = returns - threshold
        gains = excess_returns[excess_returns > 0].sum()
        losses = abs(excess_returns[excess_returns < 0].sum())
        return gains / losses if losses > 0 else np.inf
    
    @staticmethod
    def _calculate_burke_ratio(returns: pd.Series, risk_free_rate: float = 0.03) -> float:
        """Calculate Burke Ratio"""
        annual_return = returns.mean() * TRADING_DAYS
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # Sort drawdowns and take worst 5
        worst_drawdowns = np.sort(drawdown.values)[:5]
        sum_squared_drawdowns = np.sum(worst_drawdowns ** 2)
        
        if sum_squared_drawdowns > 0:
            return (annual_return - risk_free_rate) / np.sqrt(sum_squared_drawdowns)
        return 0
    
    @staticmethod
    def _calculate_treynor_ratio(returns: pd.Series, benchmark_returns: pd.Series, 
                               risk_free_rate: float = 0.03) -> float:
        """Calculate Treynor Ratio"""
        aligned_returns = returns.dropna()
        aligned_benchmark = benchmark_returns.reindex(aligned_returns.index).dropna()
        aligned_returns = aligned_returns.reindex(aligned_benchmark.index)
        
        if len(aligned_returns) > 10:
            cov_matrix = np.cov(aligned_returns, aligned_benchmark)
            beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 0
            annual_return = aligned_returns.mean() * TRADING_DAYS
            return (annual_return - risk_free_rate) / beta if beta != 0 else np.nan
        return np.nan
    
    @staticmethod
    def _calculate_information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate Information Ratio"""
        aligned_returns = returns.dropna()
        aligned_benchmark = benchmark_returns.reindex(aligned_returns.index).dropna()
        aligned_returns = aligned_returns.reindex(aligned_benchmark.index)
        
        if len(aligned_returns) > 10:
            active_returns = aligned_returns - aligned_benchmark
            tracking_error = active_returns.std() * np.sqrt(TRADING_DAYS)
            active_return = active_returns.mean() * TRADING_DAYS
            return active_return / tracking_error if tracking_error > 0 else np.nan
        return np.nan
    
    @staticmethod
    def _calculate_drawdown_duration(drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in days"""
        if len(drawdown) == 0:
            return 0
        
        underwater = drawdown < 0
        durations = []
        current_duration = 0
        
        for is_underwater in underwater:
            if is_underwater:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0
        
        if current_duration > 0:
            durations.append(current_duration)
        
        return max(durations) if durations else 0
    
    @staticmethod
    def _calculate_conditional_skewness(returns: pd.Series) -> float:
        """Calculate conditional skewness (skewness of negative returns)"""
        negative_returns = returns[returns < 0]
        return negative_returns.skew() if len(negative_returns) > 10 else np.nan
    
    @staticmethod
    def _calculate_conditional_kurtosis(returns: pd.Series) -> float:
        """Calculate conditional kurtosis (kurtosis of negative returns)"""
        negative_returns = returns[returns < 0]
        return negative_returns.kurtosis() if len(negative_returns) > 10 else np.nan
    
    @staticmethod
    def _jarque_bera_test(returns: pd.Series) -> Dict:
        """Perform Jarque-Bera normality test"""
        if len(returns) < 5:
            return {'statistic': np.nan, 'pvalue': np.nan, 'is_normal': False}
        
        jb_stat, p_value = stats.jarque_bera(returns.values)
        return {
            'statistic': jb_stat,
            'pvalue': p_value,
            'is_normal': p_value > 0.05
        }
    
    @staticmethod
    def _shapiro_wilk_test(returns: pd.Series) -> Dict:
        """Perform Shapiro-Wilk normality test"""
        if len(returns) < 3 or len(returns) > 5000:
            return {'statistic': np.nan, 'pvalue': np.nan, 'is_normal': False}
        
        stat, p_value = stats.shapiro(returns.values)
        return {
            'statistic': stat,
            'pvalue': p_value,
            'is_normal': p_value > 0.05
        }
    
    @staticmethod
    def _lilliefors_test(returns: pd.Series) -> Dict:
        """Perform Lilliefors normality test"""
        if len(returns) < 5:
            return {'statistic': np.nan, 'pvalue': np.nan, 'is_normal': False}
        
        # Lilliefors test implementation
        normed_data = (returns.values - returns.mean()) / returns.std()
        stat, p_value = stats.kstest(normed_data, 'norm')
        return {
            'statistic': stat,
            'pvalue': p_value,
            'is_normal': p_value > 0.05
        }
    
    @staticmethod
    def _calculate_var_contribution(returns: pd.Series) -> float:
        """Calculate VaR contribution using historical method"""
        sorted_returns = np.sort(returns.values)
        n = len(sorted_returns)
        var_level = int(np.floor(0.05 * n))  # 95% VaR
        
        if var_level < 1:
            return 0
        
        var_threshold = sorted_returns[var_level]
        returns_below_var = sorted_returns[:var_level+1]
        
        return np.mean(returns_below_var)
    
    @staticmethod
    def _calculate_cvar_contribution(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate CVaR contribution"""
        var = np.percentile(returns.values, (1 - confidence_level) * 100)
        returns_below_var = returns[returns <= var]
        
        return returns_below_var.mean() if len(returns_below_var) > 0 else var
    
    @staticmethod
    def calculate_garch_volatility(returns: pd.DataFrame, p: int = 1, q: int = 1) -> pd.DataFrame:
        """Calculate GARCH volatility for multiple assets"""
        if not ARCH_AVAILABLE:
            return pd.DataFrame()
        
        garch_vol = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        for asset in returns.columns:
            try:
                asset_returns = returns[asset].dropna() * 100  # Convert to percentage
                if len(asset_returns) < 50:
                    continue
                
                # Fit GARCH model
                model = arch.arch_model(asset_returns, vol='Garch', p=p, q=q)
                res = model.fit(disp='off', show_warning=False)
                
                # Get conditional volatility
                conditional_vol = res.conditional_volatility / 100  # Convert back
                garch_vol[asset] = conditional_vol.reindex(returns.index)
                
            except Exception as e:
                continue
        
        return garch_vol.dropna(axis=1, how='all')
    
    @staticmethod
    def calculate_ewma_volatility(returns: pd.DataFrame, lambda_param: float = 0.94) -> pd.DataFrame:
        """Calculate EWMA volatility for multiple assets"""
        ewma_vol = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        for asset in returns.columns:
            try:
                returns_squared = returns[asset] ** 2
                ewma_var = returns_squared.ewm(alpha=1-lambda_param, adjust=False).mean()
                ewma_vol[asset] = np.sqrt(ewma_var)
            except:
                continue
        
        return ewma_vol.dropna(axis=1, how='all')
    
    @staticmethod
    def perform_stress_test(returns: pd.DataFrame, stress_scenarios: Dict) -> pd.DataFrame:
        """Perform stress testing with historical scenarios"""
        
        results = []
        
        for scenario_name, stress_params in stress_scenarios.items():
            # Apply stress scenario
            stressed_returns = returns.copy()
            
            if 'shock' in stress_params:
                # Apply uniform shock
                stressed_returns = stressed_returns * (1 + stress_params['shock'])
            elif 'var_shock' in stress_params:
                # Apply VaR-based shock
                var_shock = stress_params['var_shock']
                for asset in returns.columns:
                    asset_var = np.percentile(returns[asset].dropna(), var_shock * 100)
                    stressed_returns[asset] = returns[asset] + asset_var
            
            # Calculate portfolio impact (assuming equal weights for simplicity)
            portfolio_returns = stressed_returns.mean(axis=1)
            total_return = (1 + portfolio_returns).prod() - 1
            
            # Calculate other metrics
            worst_asset = stressed_returns.min().idxmin() if not stressed_returns.empty else 'N/A'
            best_asset = stressed_returns.max().idxmax() if not stressed_returns.empty else 'N/A'
            max_loss = stressed_returns.min().min()
            avg_loss = stressed_returns.mean().min()
            
            results.append({
                'Scenario': scenario_name,
                'Description': stress_params.get('description', ''),
                'Portfolio Impact': total_return,
                'Worst Asset': worst_asset,
                'Best Asset': best_asset,
                'Max Loss': max_loss,
                'Avg Loss': avg_loss
            })
        
        return pd.DataFrame(results)

# =============================================================
# HISTORICAL STRESS TESTING SCENARIOS
# =============================================================
HISTORICAL_STRESS_SCENARIOS = {
    '2008 Financial Crisis': {
        'description': 'Global financial crisis 2007-2008',
        'shock': -0.50,
        'period': '2007-10-09 to 2009-03-09',
        'market_drop': -56.4,
        'duration': '17 months'
    },
    '2020 COVID Crash': {
        'description': 'COVID-19 pandemic market crash',
        'shock': -0.35,
        'period': '2020-02-19 to 2020-03-23',
        'market_drop': -33.9,
        'duration': '1 month'
    },
    'Dot-com Bubble Burst': {
        'description': 'Technology bubble burst 2000-2002',
        'shock': -0.45,
        'period': '2000-03-24 to 2002-10-09',
        'market_drop': -49.1,
        'duration': '2.5 years'
    },
    '2011 European Debt Crisis': {
        'description': 'European sovereign debt crisis',
        'shock': -0.22,
        'period': '2011-04-29 to 2011-10-03',
        'market_drop': -19.4,
        'duration': '5 months'
    },
    '2015-16 China Slowdown': {
        'description': 'Chinese market crash and economic slowdown',
        'shock': -0.15,
        'period': '2015-06-12 to 2016-02-11',
        'market_drop': -13.3,
        'duration': '8 months'
    },
    '1987 Black Monday': {
        'description': 'Largest one-day market decline',
        'shock': -0.25,
        'period': '1987-10-19',
        'market_drop': -22.6,
        'duration': '1 day'
    },
    '1997 Asian Financial Crisis': {
        'description': 'Asian currency and market crisis',
        'shock': -0.40,
        'period': '1997-07-02 to 1998-08-31',
        'market_drop': -54.0,
        'duration': '14 months'
    },
    '1998 Russian Crisis': {
        'description': 'Russian financial crisis and LTCM collapse',
        'shock': -0.20,
        'period': '1998-08-17 to 1998-10-08',
        'market_drop': -19.3,
        'duration': '2 months'
    },
    '2018 Q4 Market Correction': {
        'description': 'Fed tightening and trade war fears',
        'shock': -0.20,
        'period': '2018-10-03 to 2018-12-24',
        'market_drop': -19.8,
        'duration': '3 months'
    },
    '2022 Inflation Shock': {
        'description': 'High inflation and aggressive Fed tightening',
        'shock': -0.25,
        'period': '2022-01-03 to 2022-10-12',
        'market_drop': -25.4,
        'duration': '9 months'
    },
    '1973-74 Oil Crisis': {
        'description': 'Oil embargo and stagflation',
        'shock': -0.48,
        'period': '1973-01-11 to 1974-10-03',
        'market_drop': -48.2,
        'duration': '21 months'
    },
    '2010 Flash Crash': {
        'description': 'High-frequency trading flash crash',
        'shock': -0.10,
        'period': '2010-05-06',
        'market_drop': -9.0,
        'duration': '36 minutes'
    },
    '1994 Bond Market Massacre': {
        'description': 'Sudden interest rate hikes',
        'shock': -0.10,
        'period': '1994-02-04 to 1994-12-07',
        'market_drop': -8.9,
        'duration': '10 months'
    },
    '2002 Accounting Scandals': {
        'description': 'Enron, WorldCom accounting scandals',
        'shock': -0.25,
        'period': '2002-03-19 to 2002-10-09',
        'market_drop': -31.5,
        'duration': '7 months'
    },
    '2013 Taper Tantrum': {
        'description': 'Fed taper announcement shock',
        'shock': -0.06,
        'period': '2013-05-22 to 2013-06-24',
        'market_drop': -5.8,
        'duration': '1 month'
    }
}

# =============================================================
# ENHANCED PORTFOLIO OPTIMIZATION ENGINE
# =============================================================
class PortfolioOptimizerEnhanced:
    """Enhanced portfolio optimization with PyPortfolioOpt integration"""
    
    class Strategy(Enum):
        MIN_VOLATILITY = "Minimum Volatility"
        MAX_SHARPE = "Maximum Sharpe Ratio"
        MAX_QUADRATIC_UTILITY = "Maximum Quadratic Utility"
        EFFICIENT_RISK = "Efficient Risk"
        EFFICIENT_RETURN = "Efficient Return"
        MAX_DIVERSIFICATION = "Maximum Diversification"
        MIN_CVAR = "Minimum CVaR"
        HRP = "Hierarchical Risk Parity"
        RISK_PARITY = "Risk Parity"
        MEAN_VARIANCE = "Mean-Variance Optimal"
        EQUAL_WEIGHT = "Equal Weight"
        BLACK_LITTERMAN = "Black-Litterman"
        CUSTOM = "Custom Weights"
    
    def __init__(self):
        self.analytics = QuantitativeAnalytics()
    
    def optimize(self, returns: pd.DataFrame, strategy: Strategy, **kwargs) -> Dict:
        """Optimize portfolio using selected strategy"""
        
        if strategy == self.Strategy.EQUAL_WEIGHT:
            return self._equal_weight(returns)
        
        if not PYPFOPT_AVAILABLE:
            st.warning("PyPortfolioOpt not available. Using equal weight fallback.")
            return self._equal_weight(returns)
        
        try:
            # Clean returns
            returns_clean = returns.dropna()
            if returns_clean.empty:
                return self._equal_weight(returns)
            
            # Calculate expected returns and covariance
            mu = expected_returns.mean_historical_return(returns_clean)
            
            # Use different covariance estimators based on strategy
            if strategy in [self.Strategy.HRP, self.Strategy.MIN_CVAR]:
                S = risk_models.sample_cov(returns_clean)
            else:
                # Use shrinkage for better stability
                S = CovarianceShrinkage(returns_clean).ledoit_wolf()
            
            # Apply optimization strategy
            if strategy == self.Strategy.MIN_VOLATILITY:
                return self._min_volatility(mu, S, **kwargs)
            
            elif strategy == self.Strategy.MAX_SHARPE:
                return self._max_sharpe(mu, S, **kwargs)
            
            elif strategy == self.Strategy.MAX_QUADRATIC_UTILITY:
                return self._max_quadratic_utility(mu, S, **kwargs)
            
            elif strategy == self.Strategy.EFFICIENT_RISK:
                return self._efficient_risk(mu, S, **kwargs)
            
            elif strategy == self.Strategy.EFFICIENT_RETURN:
                return self._efficient_return(mu, S, **kwargs)
            
            elif strategy == self.Strategy.MAX_DIVERSIFICATION:
                return self._max_diversification(mu, S, **kwargs)
            
            elif strategy == self.Strategy.MIN_CVAR:
                return self._min_cvar(returns_clean, **kwargs)
            
            elif strategy == self.Strategy.HRP:
                return self._hierarchical_risk_parity(returns_clean, **kwargs)
            
            elif strategy == self.Strategy.RISK_PARITY:
                return self._risk_parity(returns_clean, **kwargs)
            
            elif strategy == self.Strategy.MEAN_VARIANCE:
                return self._mean_variance(mu, S, **kwargs)
            
            elif strategy == self.Strategy.BLACK_LITTERMAN:
                return self._black_litterman(mu, S, returns_clean, **kwargs)
            
            else:
                return self._equal_weight(returns)
                
        except Exception as e:
            st.error(f"Optimization error: {str(e)[:200]}")
            return self._equal_weight(returns)
    
    def _equal_weight(self, returns: pd.DataFrame) -> Dict:
        """Equal weight portfolio"""
        n_assets = len(returns.columns)
        weights = np.ones(n_assets) / n_assets
        
        return {
            'weights': weights,
            'weights_dict': dict(zip(returns.columns, weights)),
            'strategy': 'Equal Weight',
            'expected_return': returns.mean().dot(weights) * TRADING_DAYS,
            'expected_risk': np.sqrt(weights @ returns.cov().dot(weights) * TRADING_DAYS),
            'success': True
        }
    
    def _min_volatility(self, mu, S, **kwargs) -> Dict:
        """Minimum volatility portfolio"""
        ef = EfficientFrontier(mu, S)
        weights = ef.min_volatility()
        ret, vol, sharpe = ef.portfolio_performance()
        
        return {
            'weights': np.array(list(weights.values())),
            'weights_dict': weights,
            'strategy': 'Minimum Volatility',
            'expected_return': ret,
            'expected_risk': vol,
            'sharpe_ratio': sharpe,
            'success': True
        }
    
    def _max_sharpe(self, mu, S, **kwargs) -> Dict:
        """Maximum Sharpe ratio portfolio"""
        risk_free_rate = kwargs.get('risk_free_rate', 0.03) / TRADING_DAYS
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
        ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=risk_free_rate)
        
        return {
            'weights': np.array(list(weights.values())),
            'weights_dict': weights,
            'strategy': 'Maximum Sharpe Ratio',
            'expected_return': ret,
            'expected_risk': vol,
            'sharpe_ratio': sharpe,
            'success': True
        }
    
    def _max_quadratic_utility(self, mu, S, **kwargs) -> Dict:
        """Maximum quadratic utility portfolio"""
        risk_aversion = kwargs.get('risk_aversion', 1.0)
        ef = EfficientFrontier(mu, S)
        weights = ef.max_quadratic_utility(risk_aversion=risk_aversion)
        ret, vol, sharpe = ef.portfolio_performance()
        
        return {
            'weights': np.array(list(weights.values())),
            'weights_dict': weights,
            'strategy': 'Maximum Quadratic Utility',
            'expected_return': ret,
            'expected_risk': vol,
            'sharpe_ratio': sharpe,
            'success': True
        }
    
    def _efficient_risk(self, mu, S, **kwargs) -> Dict:
        """Efficient risk portfolio"""
        target_risk = kwargs.get('target_risk', 0.15) / np.sqrt(TRADING_DAYS)
        ef = EfficientFrontier(mu, S)
        weights = ef.efficient_risk(target_volatility=target_risk)
        ret, vol, sharpe = ef.portfolio_performance()
        
        return {
            'weights': np.array(list(weights.values())),
            'weights_dict': weights,
            'strategy': 'Efficient Risk',
            'expected_return': ret,
            'expected_risk': vol,
            'sharpe_ratio': sharpe,
            'success': True
        }
    
    def _efficient_return(self, mu, S, **kwargs) -> Dict:
        """Efficient return portfolio"""
        target_return = kwargs.get('target_return', 0.10) / TRADING_DAYS
        ef = EfficientFrontier(mu, S)
        weights = ef.efficient_return(target_return=target_return)
        ret, vol, sharpe = ef.portfolio_performance()
        
        return {
            'weights': np.array(list(weights.values())),
            'weights_dict': weights,
            'strategy': 'Efficient Return',
            'expected_return': ret,
            'expected_risk': vol,
            'sharpe_ratio': sharpe,
            'success': True
        }
    
    def _max_diversification(self, mu, S, **kwargs) -> Dict:
        """Maximum diversification portfolio"""
        # Implement diversification ratio maximization
        weights = self._risk_parity_weights(S)
        
        # Calculate metrics
        portfolio_return = mu.dot(weights)
        portfolio_risk = np.sqrt(weights @ S.dot(weights))
        sharpe = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        return {
            'weights': weights,
            'weights_dict': dict(zip(mu.index, weights)),
            'strategy': 'Maximum Diversification',
            'expected_return': portfolio_return,
            'expected_risk': portfolio_risk,
            'sharpe_ratio': sharpe,
            'success': True
        }
    
    def _min_cvar(self, returns: pd.DataFrame, **kwargs) -> Dict:
        """Minimum CVaR portfolio"""
        if not PYPFOPT_AVAILABLE:
            return self._equal_weight(returns)
        
        try:
            mu = expected_returns.mean_historical_return(returns)
            ef_cvar = EfficientCVaR(mu, returns)
            target_return = kwargs.get('target_return', mu.mean())
            ef_cvar.efficient_return(target_return=target_return)
            weights = ef_cvar.clean_weights()
            
            # Calculate portfolio performance
            portfolio_returns = returns.dot(list(weights.values()))
            ret = portfolio_returns.mean() * TRADING_DAYS
            vol = portfolio_returns.std() * np.sqrt(TRADING_DAYS)
            sharpe = ret / vol if vol > 0 else 0
            
            return {
                'weights': np.array(list(weights.values())),
                'weights_dict': weights,
                'strategy': 'Minimum CVaR',
                'expected_return': ret,
                'expected_risk': vol,
                'sharpe_ratio': sharpe,
                'success': True
            }
        except:
            return self._equal_weight(returns)
    
    def _hierarchical_risk_parity(self, returns: pd.DataFrame, **kwargs) -> Dict:
        """Hierarchical Risk Parity portfolio"""
        if not PYPFOPT_AVAILABLE:
            return self._equal_weight(returns)
        
        try:
            hrp = HRPOpt(returns)
            weights = hrp.optimize()
            
            # Calculate portfolio performance
            portfolio_returns = returns.dot(list(weights.values()))
            ret = portfolio_returns.mean() * TRADING_DAYS
            vol = portfolio_returns.std() * np.sqrt(TRADING_DAYS)
            sharpe = ret / vol if vol > 0 else 0
            
            return {
                'weights': np.array(list(weights.values())),
                'weights_dict': weights,
                'strategy': 'Hierarchical Risk Parity',
                'expected_return': ret,
                'expected_risk': vol,
                'sharpe_ratio': sharpe,
                'success': True
            }
        except:
            return self._equal_weight(returns)
    
    def _risk_parity(self, returns: pd.DataFrame, **kwargs) -> Dict:
        """Risk parity portfolio"""
        # Simple inverse volatility weighting
        volatilities = returns.std()
        inv_vol = 1 / volatilities
        weights = inv_vol / inv_vol.sum()
        
        # Calculate portfolio performance
        portfolio_returns = returns.dot(weights)
        ret = portfolio_returns.mean() * TRADING_DAYS
        vol = portfolio_returns.std() * np.sqrt(TRADING_DAYS)
        sharpe = ret / vol if vol > 0 else 0
        
        return {
            'weights': weights.values,
            'weights_dict': dict(zip(returns.columns, weights.values)),
            'strategy': 'Risk Parity',
            'expected_return': ret,
            'expected_risk': vol,
            'sharpe_ratio': sharpe,
            'success': True
        }
    
    def _risk_parity_weights(self, covariance_matrix: pd.DataFrame) -> np.ndarray:
        """Calculate risk parity weights using optimization"""
        n = len(covariance_matrix)
        
        def objective(weights):
            portfolio_risk = np.sqrt(weights @ covariance_matrix.values @ weights)
            risk_contributions = weights * (covariance_matrix.values @ weights) / portfolio_risk
            target_contributions = portfolio_risk / n
            return np.sum((risk_contributions - target_contributions) ** 2)
        
        # Constraints: sum of weights = 1, weights >= 0
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = [(0, 1) for _ in range(n)]
        
        # Initial guess (equal weights)
        x0 = np.ones(n) / n
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x
    
    def _mean_variance(self, mu, S, **kwargs) -> Dict:
        """Mean-variance optimal portfolio"""
        risk_free_rate = kwargs.get('risk_free_rate', 0.03) / TRADING_DAYS
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
        ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=risk_free_rate)
        
        return {
            'weights': np.array(list(weights.values())),
            'weights_dict': weights,
            'strategy': 'Mean-Variance Optimal',
            'expected_return': ret,
            'expected_risk': vol,
            'sharpe_ratio': sharpe,
            'success': True
        }
    
    def _black_litterman(self, mu, S, returns: pd.DataFrame, **kwargs) -> Dict:
        """Black-Litterman portfolio"""
        try:
            # Market implied returns (using market cap weights as proxy)
            market_caps = kwargs.get('market_caps', None)
            if market_caps is None:
                # Use inverse volatility as proxy for market cap
                volatilities = returns.std()
                market_caps = 1 / volatilities
                market_caps = market_caps / market_caps.sum()
            else:
                market_caps = np.array(market_caps)
                market_caps = market_caps / market_caps.sum()
            
            # Risk aversion parameter
            delta = kwargs.get('delta', 2.5)
            
            # Implied equilibrium returns
            pi = delta * S @ market_caps
            
            # Create view matrix (simplified - bullish on all assets)
            P = np.eye(len(mu))
            Q = pi * 1.1  # 10% higher expected returns
            
            # Uncertainty of views
            tau = 0.05
            omega = np.diag(np.diag(P @ (tau * S) @ P.T))
            
            # Black-Litterman formula
            try:
                M = tau * S
                pi_bl = pi + M @ P.T @ np.linalg.inv(P @ M @ P.T + omega) @ (Q - P @ pi)
                S_bl = S + M - M @ P.T @ np.linalg.inv(P @ M @ P.T + omega) @ P @ M
            except np.linalg.LinAlgError:
                # Fallback if matrix inversion fails
                pi_bl = pi
                S_bl = S
            
            # Optimize with BL returns
            ef = EfficientFrontier(pi_bl, S_bl)
            weights = ef.max_sharpe(risk_free_rate=kwargs.get('risk_free_rate', 0.03)/TRADING_DAYS)
            ret, vol, sharpe = ef.portfolio_performance()
            
            return {
                'weights': np.array(list(weights.values())),
                'weights_dict': weights,
                'strategy': 'Black-Litterman',
                'expected_return': ret,
                'expected_risk': vol,
                'sharpe_ratio': sharpe,
                'success': True
            }
        except:
            return self._max_sharpe(mu, S, **kwargs)
    
    def calculate_efficient_frontier(self, returns: pd.DataFrame, n_points: int = 50) -> pd.DataFrame:
        """Calculate efficient frontier points"""
        if not PYPFOPT_AVAILABLE or len(returns.columns) < 2:
            return pd.DataFrame()
        
        try:
            mu = expected_returns.mean_historical_return(returns)
            S = risk_models.sample_cov(returns)
            
            ef = EfficientFrontier(mu, S)
            
            # Get min and max return portfolios
            ef.min_volatility()
            min_ret, min_vol, _ = ef.portfolio_performance()
            
            target_returns = np.linspace(min_ret, mu.max() * 1.5, n_points)
            
            frontier_points = []
            
            for target in target_returns:
                try:
                    ef.efficient_return(target_return=target/TRADING_DAYS)
                    ret, vol, sharpe = ef.portfolio_performance()
                    frontier_points.append({
                        'return': ret,
                        'risk': vol,
                        'sharpe': sharpe
                    })
                except:
                    continue
            
            return pd.DataFrame(frontier_points)
            
        except Exception as e:
            return pd.DataFrame()

# =============================================================
# DATA LOADING UTILITIES (FIXED)
# =============================================================
@st.cache_data(show_spinner=False, ttl=3600, max_entries=20)
def load_market_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Load market data with proper error handling"""
    
    if not tickers:
        return pd.DataFrame()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    prices_dict = {}
    successful_tickers = []
    
    # Ensure dates are strings
    if isinstance(start_date, (datetime, pd.Timestamp)):
        start_date = start_date.strftime('%Y-%m-%d')
    if isinstance(end_date, (datetime, pd.Timestamp)):
        end_date = end_date.strftime('%Y-%m-%d')
    
    total = len(tickers)
    
    for i, ticker in enumerate(tickers):
        try:
            # Download data
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True
            )
            
            if not data.empty and 'Close' in data.columns:
                prices_dict[ticker] = data['Close']
                successful_tickers.append(ticker)
            
        except Exception as e:
            st.debug(f"Failed to load {ticker}: {str(e)[:100]}")
        
        # Update progress
        progress = (i + 1) / total
        progress_bar.progress(progress)
        status_text.text(f"Loading {i + 1}/{total}...")
    
    progress_bar.empty()
    status_text.empty()
    
    if not prices_dict:
        st.error("❌ No data loaded. Check ticker symbols and internet connection.")
        return pd.DataFrame()
    
    prices_df = pd.DataFrame(prices_dict)
    
    # Basic cleaning
    prices_df = prices_df.ffill().bfill()
    prices_df = prices_df.dropna(axis=1, how='all')
    
    if len(prices_df.columns) < 2:
        st.error("❌ Need at least 2 assets with valid data.")
        return pd.DataFrame()
    
    st.success(f"✅ Loaded {len(successful_tickers)}/{total} assets")
    
    return prices_df

def generate_demo_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Generate realistic demo data"""
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    if len(dates) < 100:
        dates = pd.date_range(end=pd.Timestamp.today(), periods=252, freq='B')
    
    n_assets = len(tickers)
    n_days = len(dates)
    
    # Create correlated returns
    np.random.seed(42)
    base_corr = 0.3
    corr_matrix = np.eye(n_assets) * (1 - base_corr) + base_corr
    
    # Generate correlated returns
    L = np.linalg.cholesky(corr_matrix)
    uncorrelated = np.random.randn(n_assets, n_days)
    correlated = L @ uncorrelated
    
    # Convert to prices
    prices_dict = {}
    for idx, ticker in enumerate(tickers):
        # Add trend and volatility
        daily_return = 0.0003 + 0.01 * correlated[idx]
        cumulative_return = np.exp(np.cumsum(daily_return))
        prices = 100 * cumulative_return
        prices_dict[ticker] = pd.Series(prices, index=dates)
    
    return pd.DataFrame(prices_dict)

# =============================================================
# STREAMLIT APPLICATION
# =============================================================
def main():
    st.title(APP_TITLE)
    
    # Initialize session state
    if 'selected_tickers' not in st.session_state:
        st.session_state.selected_tickers = ["SPY", "TLT", "GLD", "AAPL", "MSFT"]
    if 'strategy' not in st.session_state:
        st.session_state.strategy = "Maximum Sharpe Ratio"
    if 'use_demo_data' not in st.session_state:
        st.session_state.use_demo_data = False
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Asset Selection
        st.subheader("🌍 Asset Selection")
        category_filter = st.multiselect(
            "Filter by Category",
            list(GLOBAL_ASSET_UNIVERSE.keys()),
            default=["US_Indices", "US_Tech_Stocks", "Bonds", "Commodities"]
        )
        
        # Get filtered tickers
        filtered_tickers = []
        if category_filter:
            for category in category_filter:
                filtered_tickers.extend(GLOBAL_ASSET_UNIVERSE[category])
        else:
            filtered_tickers = ALL_TICKERS
        
        filtered_tickers = list(dict.fromkeys(filtered_tickers))
        
        # Ticker selection
        selected_tickers = st.multiselect(
            "Select Assets (3-15 recommended)",
            filtered_tickers,
            default=st.session_state.selected_tickers
        )
        
        if len(selected_tickers) < 2:
            st.error("Select at least 2 assets")
        
        st.session_state.selected_tickers = selected_tickers
        
        # Benchmark
        benchmark = st.selectbox(
            "📊 Benchmark",
            ["SPY", "QQQ", "VTI", "IWM"] + [t for t in ALL_TICKERS if t not in selected_tickers],
            index=0
        )
        
        # Date Range
        st.subheader("📅 Date Range")
        date_preset = st.selectbox(
            "Time Period",
            ["1 Year", "3 Years", "5 Years", "10 Years", "Max", "Custom"],
            index=1
        )
        
        end_date = pd.Timestamp.today()
        if date_preset == "1 Year":
            start_date = end_date - pd.DateOffset(years=1)
        elif date_preset == "3 Years":
            start_date = end_date - pd.DateOffset(years=3)
        elif date_preset == "5 Years":
            start_date = end_date - pd.DateOffset(years=5)
        elif date_preset == "10 Years":
            start_date = end_date - pd.DateOffset(years=10)
        elif date_preset == "Max":
            start_date = end_date - pd.DateOffset(years=20)
        else:
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=end_date - pd.DateOffset(years=3))
            with col2:
                end_date = st.date_input("End Date", value=end_date)
            start_date = pd.Timestamp(start_date)
            end_date = pd.Timestamp(end_date)
        
        # Strategy Selection
        st.subheader("🎯 Optimization Strategy")
        strategy = st.selectbox(
            "Portfolio Strategy",
            [s.value for s in PortfolioOptimizerEnhanced.Strategy],
            index=1
        )
        st.session_state.strategy = strategy
        
        # Risk Parameters
        with st.expander("⚙️ Risk Parameters", expanded=False):
            rf_rate = st.number_input(
                "Risk-Free Rate (%)",
                value=3.0,
                min_value=0.0,
                max_value=20.0,
                step=0.1
            ) / 100
            
            confidence_level = st.slider(
                "VaR Confidence Level",
                min_value=0.90,
                max_value=0.99,
                value=0.95,
                step=0.01
            )
            
            risk_aversion = st.slider(
                "Risk Aversion",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.1
            )
        
        # Data Options
        with st.expander("📊 Data Options", expanded=False):
            use_demo = st.checkbox(
                "Use Demo Data",
                value=st.session_state.use_demo_data,
                help="Use synthetic data for testing"
            )
            st.session_state.use_demo_data = use_demo
        
        # Run Analysis Button
        st.markdown("---")
        run_analysis = st.button(
            "🚀 Run Comprehensive Analysis",
            type="primary",
            use_container_width=True,
            disabled=len(selected_tickers) < 2
        )
    
    # Main Content
    if not run_analysis or len(selected_tickers) < 2:
        # Welcome screen
        st.info("👈 Configure portfolio in sidebar and click 'Run Comprehensive Analysis'")
        
        # Show system status
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Assets", len(ALL_TICKERS))
        with col2:
            st.metric("Categories", len(GLOBAL_ASSET_UNIVERSE))
        with col3:
            st.metric("Available Strategies", len(PortfolioOptimizerEnhanced.Strategy))
        
        return
    
    # Load Data
    st.header("📊 Loading Market Data")
    
    try:
        # Combine tickers with benchmark
        all_tickers = list(set(selected_tickers + [benchmark]))
        
        if st.session_state.use_demo_data:
            prices = generate_demo_data(all_tickers, start_date, end_date)
            st.info("📊 Using demo data for testing")
        else:
            with st.spinner("Downloading market data..."):
                prices = load_market_data(all_tickers, start_date, end_date)
            
            if prices.empty or len(prices.columns) < 2:
                st.error("Failed to load data. Try demo mode or different assets.")
                return
        
        # Update selected tickers based on what loaded
        loaded_tickers = [t for t in selected_tickers if t in prices.columns]
        if benchmark not in prices.columns:
            benchmark = prices.columns[0] if len(prices.columns) > 0 else "SPY"
            st.warning(f"Benchmark not found, using {benchmark}")
        
        if len(loaded_tickers) < 2:
            st.error("Need at least 2 assets with data")
            return
        
        st.session_state.selected_tickers = loaded_tickers
        st.session_state.benchmark = benchmark
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        portfolio_returns = returns[loaded_tickers]
        benchmark_returns = returns[benchmark]
        
        # Store in session state
        st.session_state.prices = prices
        st.session_state.returns = returns
        st.session_state.portfolio_returns = portfolio_returns
        st.session_state.benchmark_returns = benchmark_returns
        
        st.success(f"✅ Data loaded: {len(loaded_tickers)} assets, {len(returns)} trading days")
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return
    
    # Create tabs for different analyses
    tab_titles = [
        "📊 Portfolio Overview",
        "📈 Quantitative Metrics",
        "⚖️ Risk Analytics",
        "🎯 Portfolio Optimization",
        "📉 VaR Analysis",
        "🌋 Stress Testing",
        "📊 Volatility Modeling",
        "📋 Reports"
    ]
    
    tabs = st.tabs(tab_titles)
    
    # Tab 1: Portfolio Overview
    with tabs[0]:
        st.header("📊 Portfolio Overview")
        
        if 'portfolio_returns' not in st.session_state:
            st.warning("Run analysis first")
            return
        
        # Calculate equal weight portfolio for overview
        n_assets = len(st.session_state.selected_tickers)
        equal_weights = np.ones(n_assets) / n_assets
        
        # Portfolio returns
        portfolio_returns = st.session_state.portfolio_returns.dot(equal_weights)
        benchmark_returns = st.session_state.benchmark_returns
        
        # Performance chart
        portfolio_cumulative = (1 + portfolio_returns).cumprod()
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=portfolio_cumulative.index,
            y=portfolio_cumulative.values,
            name="Portfolio",
            line=dict(color='#1a5fb4', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=benchmark_cumulative.index,
            y=benchmark_cumulative.values,
            name=st.session_state.benchmark,
            line=dict(color='#26a269', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Portfolio vs Benchmark Performance",
            height=500,
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_return = portfolio_cumulative.iloc[-1] - 1
            st.metric("Total Return", f"{total_return:.2%}")
        
        with col2:
            annual_return = (1 + total_return) ** (TRADING_DAYS/len(portfolio_returns)) - 1
            st.metric("Annualized Return", f"{annual_return:.2%}")
        
        with col3:
            annual_vol = portfolio_returns.std() * np.sqrt(TRADING_DAYS)
            st.metric("Annual Volatility", f"{annual_vol:.2%}")
        
        with col4:
            sharpe = (annual_return - rf_rate) / annual_vol if annual_vol > 0 else 0
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        
        # Asset weights and correlations
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("📊 Asset Allocation")
            weights_df = pd.DataFrame({
                'Asset': st.session_state.selected_tickers,
                'Weight': equal_weights
            }).sort_values('Weight', ascending=False)
            
            # Pie chart
            fig = px.pie(
                weights_df,
                values='Weight',
                names='Asset',
                title='Portfolio Allocation',
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_right:
            st.subheader("🔗 Correlation Matrix")
            correlation = st.session_state.portfolio_returns.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation.values,
                x=correlation.columns,
                y=correlation.index,
                colorscale='RdBu',
                zmid=0
            ))
            
            fig.update_layout(
                height=400,
                title="Asset Correlations"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Quantitative Metrics
    with tabs[1]:
        st.header("📈 Comprehensive Quantitative Metrics")
        
        if 'portfolio_returns' not in st.session_state:
            st.warning("Run analysis first")
            return
        
        portfolio_returns = st.session_state.portfolio_returns.dot(equal_weights)
        analytics = QuantitativeAnalytics()
        
        # Calculate all metrics
        with st.spinner("Calculating quantitative metrics..."):
            metrics = analytics.calculate_all_metrics(
                portfolio_returns,
                st.session_state.benchmark_returns,
                rf_rate
            )
        
        # Display metrics in categories
        st.subheader("📊 Return Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        return_metrics = [
            ('Total Return', metrics.get('total_return', 0), '.2%'),
            ('Annualized Return', metrics.get('annualized_return', 0), '.2%'),
            ('Arithmetic Mean', metrics.get('arithmetic_mean', 0), '.2%'),
            ('Geometric Mean', metrics.get('geometric_mean', 0), '.2%'),
            ('Max Gain', metrics.get('max_gain', 0), '.4f'),
            ('Max Loss', metrics.get('max_loss', 0), '.4f'),
            ('Win Rate', metrics.get('win_rate', 0), '.2%'),
            ('Profit Factor', metrics.get('profit_factor', 0), '.2f')
        ]
        
        cols = [col1, col2, col3, col4]
        for idx, (name, value, fmt) in enumerate(return_metrics):
            with cols[idx % 4]:
                if not pd.isna(value):
                    if '%' in fmt:
                        st.metric(name, f"{value:{fmt}}")
                    else:
                        st.metric(name, f"{value:{fmt}}")
        
        st.subheader("⚖️ Risk Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        risk_metrics = [
            ('Annual Volatility', metrics.get('annualized_volatility', 0), '.2%'),
            ('Downside Deviation', metrics.get('downside_deviation', 0), '.2%'),
            ('Max Drawdown', metrics.get('max_drawdown', 0), '.2%'),
            ('Avg Drawdown', metrics.get('avg_drawdown', 0), '.2%'),
            ('Ulcer Index', metrics.get('ulcer_index', 0), '.4f'),
            ('Pain Index', metrics.get('pain_index', 0), '.2%'),
            ('VaR (95%)', metrics.get('var_historical', 0), '.2%'),
            ('CVaR (95%)', metrics.get('cvar_95', 0), '.2%')
        ]
        
        cols = [col1, col2, col3, col4]
        for idx, (name, value, fmt) in enumerate(risk_metrics):
            with cols[idx % 4]:
                if not pd.isna(value):
                    st.metric(name, f"{value:{fmt}}")
        
        st.subheader("📈 Risk-Adjusted Ratios")
        col1, col2, col3, col4 = st.columns(4)
        
        ratio_metrics = [
            ('Sharpe Ratio', metrics.get('sharpe_ratio', 0), '.2f'),
            ('Sortino Ratio', metrics.get('sortino_ratio', 0), '.2f'),
            ('Calmar Ratio', metrics.get('calmar_ratio', 0), '.2f'),
            ('Omega Ratio', metrics.get('omega_ratio', 0), '.2f'),
            ('Burke Ratio', metrics.get('burke_ratio', 0), '.2f'),
            ('Treynor Ratio', metrics.get('treynor_ratio', 0), '.2f'),
            ('Information Ratio', metrics.get('information_ratio', 0), '.2f'),
            ('Martin Ratio', metrics.get('martin_ratio', 0), '.2f')
        ]
        
        cols = [col1, col2, col3, col4]
        for idx, (name, value, fmt) in enumerate(ratio_metrics):
            with cols[idx % 4]:
                if not pd.isna(value):
                    st.metric(name, f"{value:{fmt}}")
        
        # Statistical Tests
        st.subheader("📊 Statistical Analysis")
        
        if 'jarque_bera' in metrics:
            jb_test = metrics['jarque_bera']
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Jarque-Bera Test",
                    "Normal" if jb_test.get('is_normal', False) else "Not Normal",
                    f"p={jb_test.get('pvalue', 0):.4f}"
                )
            
            with col2:
                st.metric("Skewness", f"{metrics.get('skewness', 0):.4f}")
            
            with col3:
                st.metric("Excess Kurtosis", f"{metrics.get('excess_kurtosis', 0):.4f}")
    
    # Tab 3: Risk Analytics
    with tabs[2]:
        st.header("⚖️ Advanced Risk Analytics")
        
        if 'portfolio_returns' not in st.session_state:
            st.warning("Run analysis first")
            return
        
        portfolio_returns = st.session_state.portfolio_returns.dot(equal_weights)
        analytics = QuantitativeAnalytics()
        
        # Risk metrics comparison
        st.subheader("📊 Risk Metrics Comparison")
        
        # Calculate rolling metrics
        window = st.slider("Rolling Window (days)", 20, 252, 63)
        
        rolling_vol = portfolio_returns.rolling(window=window).std() * np.sqrt(TRADING_DAYS)
        rolling_sharpe = (portfolio_returns.rolling(window=window).mean() * TRADING_DAYS - rf_rate) / rolling_vol
        rolling_sortino = portfolio_returns.rolling(window=window).apply(
            lambda x: QuantitativeAnalytics().calculate_all_metrics(x)['sortino_ratio']
        )
        
        # Rolling metrics chart
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Rolling Volatility", "Rolling Sharpe Ratio", "Rolling Sortino Ratio"),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                name="Volatility",
                line=dict(color='#1a5fb4')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                name="Sharpe",
                line=dict(color='#26a269')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=rolling_sortino.index,
                y=rolling_sortino.values,
                name="Sortino",
                line=dict(color='#f5a623')
            ),
            row=3, col=1
        )
        
        fig.update_layout(height=800, showlegend=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown analysis
        st.subheader("📉 Drawdown Analysis")
        
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            fill='tozeroy',
            line=dict(color='#c01c28'),
            name="Drawdown"
        ))
        
        fig.update_layout(
            title="Portfolio Drawdown",
            yaxis_title="Drawdown (%)",
            height=400,
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Max Drawdown", f"{drawdown.min()*100:.2f}%")
        
        with col2:
            st.metric("Avg Drawdown", f"{drawdown.mean()*100:.2f}%")
        
        with col3:
            duration = QuantitativeAnalytics._calculate_drawdown_duration(drawdown)
            st.metric("Max Duration", f"{duration} days")
        
        with col4:
            recovery = QuantitativeAnalytics._calculate_drawdown_duration(-drawdown)
            st.metric("Recovery Period", f"{recovery} days")
    
    # Tab 4: Portfolio Optimization
    with tabs[3]:
        st.header("🎯 Portfolio Optimization")
        
        if 'portfolio_returns' not in st.session_state:
            st.warning("Run analysis first")
            return
        
        returns = st.session_state.portfolio_returns
        optimizer = PortfolioOptimizerEnhanced()
        
        # Strategy configuration
        col1, col2 = st.columns(2)
        
        with col1:
            strategy = st.selectbox(
                "Optimization Strategy",
                [s.value for s in PortfolioOptimizerEnhanced.Strategy],
                index=1
            )
        
        with col2:
            if strategy in ["Efficient Risk", "Efficient Return", "Minimum CVaR"]:
                target_value = st.number_input(
                    "Target Value (%)",
                    value=10.0,
                    min_value=0.0,
                    max_value=50.0,
                    step=0.5
                ) / 100
        
        # Optimization parameters
        with st.expander("⚙️ Optimization Parameters", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                allow_short = st.checkbox("Allow Short Selling", value=False)
            
            with col2:
                max_weight = st.slider("Maximum Weight (%)", 10, 100, 100) / 100
        
        # Run optimization
        if st.button("🚀 Optimize Portfolio", type="primary"):
            with st.spinner("Optimizing portfolio..."):
                kwargs = {
                    'risk_free_rate': rf_rate,
                    'risk_aversion': risk_aversion
                }
                
                if strategy in ["Efficient Risk", "Efficient Return", "Minimum CVaR"]:
                    if "Risk" in strategy:
                        kwargs['target_risk'] = target_value
                    else:
                        kwargs['target_return'] = target_value
                
                result = optimizer.optimize(
                    returns,
                    PortfolioOptimizerEnhanced.Strategy(strategy),
                    **kwargs
                )
                
                if result['success']:
                    st.session_state.optimization_result = result
                    st.success(f"✅ Optimization complete using {result['strategy']}")
                else:
                    st.error("Optimization failed")
        
        # Display results if available
        if 'optimization_result' in st.session_state:
            result = st.session_state.optimization_result
            
            st.subheader("📊 Optimization Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Strategy", result['strategy'])
            
            with col2:
                st.metric("Expected Return", f"{result['expected_return']:.2%}")
            
            with col3:
                st.metric("Expected Risk", f"{result['expected_risk']:.2%}")
            
            with col4:
                st.metric("Sharpe Ratio", f"{result.get('sharpe_ratio', 0):.2f}")
            
            # Display weights
            st.subheader("📈 Optimized Weights")
            
            weights_df = pd.DataFrame({
                'Asset': list(result['weights_dict'].keys()),
                'Weight': list(result['weights_dict'].values())
            }).sort_values('Weight', ascending=False)
            
            # Bar chart
            fig = px.bar(
                weights_df,
                x='Asset',
                y='Weight',
                title='Portfolio Weights',
                color='Weight',
                color_continuous_scale='Blues'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Efficient frontier
            st.subheader("📈 Efficient Frontier")
            
            frontier = optimizer.calculate_efficient_frontier(returns)
            
            if not frontier.empty:
                fig = go.Figure()
                
                # Efficient frontier
                fig.add_trace(go.Scatter(
                    x=frontier['risk'],
                    y=frontier['return'],
                    mode='lines',
                    name='Efficient Frontier',
                    line=dict(color='#1a5fb4', width=3)
                ))
                
                # Individual assets
                asset_returns = returns.mean() * TRADING_DAYS
                asset_risks = returns.std() * np.sqrt(TRADING_DAYS)
                
                fig.add_trace(go.Scatter(
                    x=asset_risks,
                    y=asset_returns,
                    mode='markers+text',
                    name='Assets',
                    marker=dict(size=10, color='#26a269'),
                    text=returns.columns
                ))
                
                # Optimized portfolio
                fig.add_trace(go.Scatter(
                    x=[result['expected_risk']],
                    y=[result['expected_return']],
                    mode='markers',
                    name='Optimized',
                    marker=dict(size=15, color='#f5a623', symbol='star')
                ))
                
                fig.update_layout(
                    title="Efficient Frontier",
                    xaxis_title="Risk (Annualized)",
                    yaxis_title="Return (Annualized)",
                    height=500,
                    template="plotly_dark"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: VaR Analysis
    with tabs[4]:
        st.header("📉 Value at Risk Analysis")
        
        if 'portfolio_returns' not in st.session_state:
            st.warning("Run analysis first")
            return
        
        portfolio_returns = st.session_state.portfolio_returns.dot(equal_weights)
        analytics = QuantitativeAnalytics()
        
        # VaR Configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            confidence = st.slider(
                "Confidence Level",
                0.90, 0.99, 0.95, 0.01,
                key="var_confidence"
            )
        
        with col2:
            horizon = st.selectbox(
                "Time Horizon",
                ["1 Day", "5 Days", "10 Days", "1 Month", "3 Months"],
                index=0
            )
            horizon_map = {"1 Day": 1, "5 Days": 5, "10 Days": 10, "1 Month": 21, "3 Months": 63}
            horizon_days = horizon_map[horizon]
        
        with col3:
            var_method = st.selectbox(
                "VaR Method",
                ["Historical", "Parametric Normal", "Parametric t", "Modified", "EWMA", "Monte Carlo", "GARCH"],
                index=0
            )
        
        # Calculate VaR
        if st.button("📊 Calculate VaR", type="primary"):
            with st.spinner("Calculating Value at Risk..."):
                # Aggregate returns for horizon
                if horizon_days > 1:
                    horizon_returns = portfolio_returns.rolling(horizon_days).apply(
                        lambda x: np.prod(1 + x) - 1, raw=True
                    ).dropna()
                else:
                    horizon_returns = portfolio_returns
                
                # Calculate VaR based on method
                if var_method == "Historical":
                    var = np.percentile(horizon_returns, (1 - confidence) * 100)
                    method_name = "Historical Simulation"
                
                elif var_method == "Parametric Normal":
                    mu, sigma = horizon_returns.mean(), horizon_returns.std()
                    z = norm.ppf(1 - confidence)
                    var = mu + z * sigma
                    method_name = "Parametric (Normal)"
                
                elif var_method == "Parametric t":
                    df, mu_t, sigma_t = t.fit(horizon_returns)
                    t_score = t.ppf(1 - confidence, df)
                    var = mu_t + t_score * sigma_t
                    method_name = "Parametric (Student's t)"
                
                elif var_method == "Modified":
                    mu, sigma = horizon_returns.mean(), horizon_returns.std()
                    skew_val = horizon_returns.skew()
                    kurt_val = horizon_returns.kurtosis()
                    z = norm.ppf(1 - confidence)
                    z_cf = z + (z**2 - 1) * skew_val/6 + (z**3 - 3*z) * (kurt_val-3)/24 - (2*z**3 - 5*z) * skew_val**2/36
                    var = mu + z_cf * sigma
                    method_name = "Modified (Cornish-Fisher)"
                
                elif var_method == "EWMA":
                    var = analytics._calculate_ewma_var(horizon_returns, confidence)
                    method_name = "EWMA"
                
                elif var_method == "Monte Carlo":
                    var = analytics._calculate_monte_carlo_var(horizon_returns, confidence)
                    method_name = "Monte Carlo"
                
                elif var_method == "GARCH":
                    var = analytics._calculate_garch_var(horizon_returns, confidence)
                    method_name = "GARCH"
                
                # Calculate CVaR
                cvar = analytics._calculate_cvar(horizon_returns, confidence)
                
                # Store results
                st.session_state.var_result = {
                    'var': var,
                    'cvar': cvar,
                    'method': method_name,
                    'confidence': confidence,
                    'horizon': horizon,
                    'success': True
                }
        
        # Display results
        if 'var_result' in st.session_state:
            result = st.session_state.var_result
            
            st.subheader("📊 VaR Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    f"VaR ({result['horizon']})",
                    f"{result['var']:.4f}",
                    f"{result['confidence']*100:.0f}% Confidence"
                )
            
            with col2:
                st.metric(
                    f"CVaR ({result['horizon']})",
                    f"{result['cvar']:.4f}",
                    f"Expected Shortfall"
                )
            
            with col3:
                st.metric("Method", result['method'])
            
            with col4:
                probability = 1 - result['confidence']
                st.metric(
                    "Probability",
                    f"{probability:.1%}",
                    "Loss exceeds VaR"
                )
            
            # Distribution plot with VaR
            st.subheader("📈 Return Distribution with VaR")
            
            if horizon_days > 1:
                returns_for_plot = horizon_returns
            else:
                returns_for_plot = portfolio_returns
            
            fig = go.Figure()
            
            # Histogram
            fig.add_trace(go.Histogram(
                x=returns_for_plot,
                nbinsx=50,
                name="Returns",
                marker_color='#1a5fb4',
                opacity=0.7,
                histnorm='probability density'
            ))
            
            # VaR line
            fig.add_vline(
                x=result['var'],
                line_dash="dash",
                line_color="#f5a623",
                annotation_text=f"VaR: {result['var']:.4f}",
                annotation_position="top left"
            )
            
            # CVaR line
            fig.add_vline(
                x=result['cvar'],
                line_dash="dot",
                line_color="#c01c28",
                annotation_text=f"CVaR: {result['cvar']:.4f}",
                annotation_position="top right"
            )
            
            fig.update_layout(
                title=f"Return Distribution with Risk Measures ({result['method']})",
                xaxis_title="Return",
                yaxis_title="Density",
                height=500,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # VaR Comparison Table
            st.subheader("📋 VaR Method Comparison")
            
            # Calculate all VaR methods
            var_methods = ["Historical", "Parametric Normal", "Parametric t", "Modified", "EWMA", "Monte Carlo"]
            if ARCH_AVAILABLE:
                var_methods.append("GARCH")
            
            comparison_results = []
            
            for method in var_methods:
                try:
                    if method == "Historical":
                        var_val = np.percentile(returns_for_plot, (1 - confidence) * 100)
                    elif method == "Parametric Normal":
                        mu, sigma = returns_for_plot.mean(), returns_for_plot.std()
                        z = norm.ppf(1 - confidence)
                        var_val = mu + z * sigma
                    elif method == "Parametric t":
                        df, mu_t, sigma_t = t.fit(returns_for_plot)
                        t_score = t.ppf(1 - confidence, df)
                        var_val = mu_t + t_score * sigma_t
                    elif method == "Modified":
                        mu, sigma = returns_for_plot.mean(), returns_for_plot.std()
                        skew_val = returns_for_plot.skew()
                        kurt_val = returns_for_plot.kurtosis()
                        z = norm.ppf(1 - confidence)
                        z_cf = z + (z**2 - 1) * skew_val/6 + (z**3 - 3*z) * (kurt_val-3)/24 - (2*z**3 - 5*z) * skew_val**2/36
                        var_val = mu + z_cf * sigma
                    elif method == "EWMA":
                        var_val = analytics._calculate_ewma_var(returns_for_plot, confidence)
                    elif method == "Monte Carlo":
                        var_val = analytics._calculate_monte_carlo_var(returns_for_plot, confidence)
                    elif method == "GARCH":
                        var_val = analytics._calculate_garch_var(returns_for_plot, confidence)
                    
                    cvar_val = analytics._calculate_cvar(returns_for_plot, confidence)
                    
                    comparison_results.append({
                        'Method': method,
                        'VaR': var_val,
                        'CVaR': cvar_val,
                        'Difference': abs(var_val - cvar_val)
                    })
                    
                except Exception as e:
                    comparison_results.append({
                        'Method': method,
                        'VaR': np.nan,
                        'CVaR': np.nan,
                        'Difference': np.nan
                    })
            
            comparison_df = pd.DataFrame(comparison_results)
            
            # Display comparison table
            st.dataframe(
                comparison_df.style.format({
                    'VaR': '{:.6f}',
                    'CVaR': '{:.6f}',
                    'Difference': '{:.6f}'
                }).background_gradient(
                    subset=['VaR', 'CVaR'],
                    cmap='Reds'
                ),
                use_container_width=True,
                height=400
            )
    
    # Tab 6: Stress Testing
    with tabs[5]:
        st.header("🌋 Historical Stress Testing")
        
        if 'portfolio_returns' not in st.session_state:
            st.warning("Run analysis first")
            return
        
        returns = st.session_state.portfolio_returns
        analytics = QuantitativeAnalytics()
        
        # Stress testing configuration
        st.subheader("⚙️ Stress Test Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_scenarios = st.multiselect(
                "Select Historical Scenarios",
                list(HISTORICAL_STRESS_SCENARIOS.keys()),
                default=list(HISTORICAL_STRESS_SCENARIOS.keys())[:5]
            )
        
        with col2:
            shock_multiplier = st.slider(
                "Shock Multiplier",
                0.5, 2.0, 1.0, 0.1,
                help="Multiply historical shocks by this factor"
            )
        
        # Run stress test
        if st.button("🌋 Run Stress Test", type="primary"):
            with st.spinner("Running stress tests..."):
                # Prepare stress scenarios
                stress_scenarios = {}
                for scenario in selected_scenarios:
                    if scenario in HISTORICAL_STRESS_SCENARIOS:
                        scenario_data = HISTORICAL_STRESS_SCENARIOS[scenario].copy()
                        scenario_data['shock'] = scenario_data['shock'] * shock_multiplier
                        stress_scenarios[scenario] = scenario_data
                
                # Perform stress test
                results = analytics.perform_stress_test(returns, stress_scenarios)
                st.session_state.stress_results = results
        
        # Display results
        if 'stress_results' in st.session_state:
            results = st.session_state.stress_results
            
            st.subheader("📊 Stress Test Results")
            
            # Display table
            st.dataframe(
                results.style.format({
                    'Portfolio Impact': '{:.2%}',
                    'Max Loss': '{:.4f}',
                    'Avg Loss': '{:.4f}'
                }).background_gradient(
                    subset=['Portfolio Impact'],
                    cmap='Reds_r'
                ),
                use_container_width=True,
                height=400
            )
            
            # Visualization
            st.subheader("📈 Stress Test Impact Comparison")
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=results['Scenario'],
                y=results['Portfolio Impact'] * 100,
                name='Portfolio Impact',
                marker_color=results['Portfolio Impact'].apply(
                    lambda x: '#c01c28' if x < 0 else '#26a269'
                )
            ))
            
            fig.update_layout(
                title="Portfolio Impact by Stress Scenario",
                xaxis_title="Scenario",
                yaxis_title="Impact (%)",
                xaxis_tickangle=45,
                height=500,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Scenario details
            st.subheader("📋 Historical Scenario Details")
            
            for scenario in selected_scenarios:
                if scenario in HISTORICAL_STRESS_SCENARIOS:
                    data = HISTORICAL_STRESS_SCENARIOS[scenario]
                    
                    with st.expander(f"{scenario}", expanded=False):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Market Drop", f"{data['market_drop']}%")
                        
                        with col2:
                            st.metric("Duration", data['duration'])
                        
                        with col3:
                            st.metric("Period", data['period'])
                        
                        with col4:
                            st.metric("Applied Shock", f"{data['shock']*shock_multiplier:.1%}")
    
    # Tab 7: Volatility Modeling
    with tabs[6]:
        st.header("📊 Advanced Volatility Modeling")
        
        if 'portfolio_returns' not in st.session_state:
            st.warning("Run analysis first")
            return
        
        returns = st.session_state.portfolio_returns
        analytics = QuantitativeAnalytics()
        
        # Volatility model selection
        st.subheader("⚙️ Volatility Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Volatility Model",
                ["EWMA", "GARCH"],
                index=0
            )
        
        with col2:
            selected_assets = st.multiselect(
                "Select Assets to Model",
                returns.columns.tolist(),
                default=returns.columns.tolist()[:3]
            )
        
        # Model parameters
        with st.expander("📊 Model Parameters", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                if model_type == "EWMA":
                    lambda_param = st.slider(
                        "Decay Factor (λ)",
                        0.85, 0.99, 0.94, 0.01
                    )
                else:  # GARCH
                    p_order = st.slider("GARCH(p)", 1, 3, 1)
            
            with col2:
                if model_type == "GARCH":
                    q_order = st.slider("GARCH(q)", 1, 3, 1)
        
        # Calculate volatility
        if st.button("📈 Calculate Volatility", type="primary"):
            with st.spinner(f"Calculating {model_type} volatility..."):
                if model_type == "EWMA":
                    vol_results = analytics.calculate_ewma_volatility(
                        returns[selected_assets],
                        lambda_param=lambda_param
                    )
                else:  # GARCH
                    if not ARCH_AVAILABLE:
                        st.error("ARCH package not installed. Install with: pip install arch")
                        return
                    
                    vol_results = analytics.calculate_garch_volatility(
                        returns[selected_assets],
                        p=p_order,
                        q=q_order
                    )
                
                st.session_state.volatility_results = {
                    'data': vol_results,
                    'model': model_type,
                    'assets': selected_assets
                }
        
        # Display results
        if 'volatility_results' in st.session_state:
            results = st.session_state.volatility_results
            vol_data = results['data']
            
            if not vol_data.empty:
                st.subheader(f"📈 {results['model']} Volatility")
                
                # Volatility chart
                fig = go.Figure()
                
                for asset in results['assets']:
                    if asset in vol_data.columns:
                        fig.add_trace(go.Scatter(
                            x=vol_data.index,
                            y=vol_data[asset].values * 100,  # Convert to percentage
                            name=asset,
                            mode='lines'
                        ))
                
                fig.update_layout(
                    title=f"{results['model']} Volatility Over Time",
                    xaxis_title="Date",
                    yaxis_title="Volatility (%)",
                    height=500,
                    template="plotly_dark",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Volatility statistics
                st.subheader("📊 Volatility Statistics")
                
                stats_df = pd.DataFrame({
                    'Asset': vol_data.columns,
                    'Mean Vol': vol_data.mean().values * 100,
                    'Std Vol': vol_data.std().values * 100,
                    'Min Vol': vol_data.min().values * 100,
                    'Max Vol': vol_data.max().values * 100,
                    'Latest Vol': vol_data.iloc[-1].values * 100
                })
                
                st.dataframe(
                    stats_df.style.format({
                        'Mean Vol': '{:.2f}%',
                        'Std Vol': '{:.2f}%',
                        'Min Vol': '{:.2f}%',
                        'Max Vol': '{:.2f}%',
                        'Latest Vol': '{:.2f}%'
                    }).background_gradient(
                        subset=['Mean Vol', 'Latest Vol'],
                        cmap='Reds'
                    ),
                    use_container_width=True,
                    height=300
                )
                
                # Compare with simple volatility
                st.subheader("📊 Comparison with Simple Volatility")
                
                simple_vol = returns[results['assets']].rolling(window=20).std() * np.sqrt(TRADING_DAYS) * 100
                
                # Select one asset for comparison
                if len(results['assets']) > 0:
                    compare_asset = st.selectbox(
                        "Select asset for detailed comparison",
                        results['assets'],
                        index=0
                    )
                    
                    if compare_asset in vol_data.columns and compare_asset in simple_vol.columns:
                        fig = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=(
                                f"{results['model']} vs Simple Volatility - {compare_asset}",
                                "Difference"
                            ),
                            vertical_spacing=0.15
                        )
                        
                        # Volatility comparison
                        fig.add_trace(
                            go.Scatter(
                                x=vol_data.index,
                                y=vol_data[compare_asset].values * 100,
                                name=f"{results['model']}",
                                line=dict(color='#1a5fb4')
                            ),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=simple_vol.index,
                                y=simple_vol[compare_asset].values,
                                name="Simple (20-day)",
                                line=dict(color='#26a269', dash='dash')
                            ),
                            row=1, col=1
                        )
                        
                        # Difference
                        aligned_vol = vol_data[compare_asset].reindex(simple_vol.index).dropna() * 100
                        aligned_simple = simple_vol[compare_asset].reindex(aligned_vol.index)
                        difference = aligned_vol - aligned_simple
                        
                        fig.add_trace(
                            go.Scatter(
                                x=difference.index,
                                y=difference.values,
                                name="Difference",
                                line=dict(color='#f5a623'),
                                fill='tozeroy'
                            ),
                            row=2, col=1
                        )
                        
                        fig.update_layout(
                            height=600,
                            template="plotly_dark",
                            showlegend=True
                        )
                        
                        fig.update_yaxes(title_text="Volatility (%)", row=1, col=1)
                        fig.update_yaxes(title_text="Difference (%)", row=2, col=1)
                        
                        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 8: Reports
    with tabs[7]:
        st.header("📋 Comprehensive Reports")
        
        if 'portfolio_returns' not in st.session_state:
            st.warning("Run analysis first")
            return
        
        # Report generation
        st.subheader("📄 Generate Report")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            report_type = st.selectbox(
                "Report Type",
                ["Summary Report", "Risk Report", "Performance Report", "Full Analysis"],
                index=0
            )
        
        with col2:
            include_charts = st.checkbox("Include Charts", value=True)
        
        with col3:
            format_type = st.selectbox(
                "Format",
                ["HTML", "Markdown", "JSON"],
                index=0
            )
        
        if st.button("📋 Generate Report", type="primary"):
            with st.spinner("Generating report..."):
                # Collect all analysis results
                report_data = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'portfolio': {
                        'assets': st.session_state.selected_tickers,
                        'benchmark': st.session_state.benchmark,
                        'strategy': st.session_state.strategy
                    },
                    'metrics': {}
                }
                
                # Add metrics if available
                if 'optimization_result' in st.session_state:
                    report_data['optimization'] = st.session_state.optimization_result
                
                if 'var_result' in st.session_state:
                    report_data['var_analysis'] = st.session_state.var_result
                
                if 'stress_results' in st.session_state:
                    report_data['stress_testing'] = st.session_state.stress_results.to_dict('records')
                
                # Generate report based on format
                if format_type == "JSON":
                    report_content = json.dumps(report_data, indent=2, default=str)
                    file_name = f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    mime_type = "application/json"
                
                elif format_type == "HTML":
                    # Simple HTML report
                    html_content = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Portfolio Analysis Report</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 40px; }}
                            .header {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; }}
                            .section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                            .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
                            .metric-value {{ font-size: 24px; font-weight: bold; color: #1a5fb4; }}
                            .metric-label {{ font-size: 12px; color: #666; }}
                        </style>
                    </head>
                    <body>
                        <div class="header">
                            <h1>Portfolio Analysis Report</h1>
                            <p>Generated: {report_data['timestamp']}</p>
                        </div>
                        
                        <div class="section">
                            <h2>Portfolio Configuration</h2>
                            <p><strong>Assets:</strong> {', '.join(report_data['portfolio']['assets'])}</p>
                            <p><strong>Benchmark:</strong> {report_data['portfolio']['benchmark']}</p>
                            <p><strong>Strategy:</strong> {report_data['portfolio']['strategy']}</p>
                        </div>
                        
                        <div class="section">
                            <h2>Key Metrics</h2>
                            <!-- Add metrics here -->
                        </div>
                        
                        <div class="footer">
                            <p>Generated by Apollo/ENIGMA Quantitative Portfolio Terminal v6.0</p>
                        </div>
                    </body>
                    </html>
                    """
                    report_content = html_content
                    file_name = f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                    mime_type = "text/html"
                
                else:  # Markdown
                    markdown_content = f"""
# Portfolio Analysis Report

**Generated:** {report_data['timestamp']}

## Portfolio Configuration
- **Assets:** {', '.join(report_data['portfolio']['assets'])}
- **Benchmark:** {report_data['portfolio']['benchmark']}
- **Strategy:** {report_data['portfolio']['strategy']}

## Analysis Results

### Optimization Results
{json.dumps(report_data.get('optimization', {}), indent=2)}

### Risk Analysis
{json.dumps(report_data.get('var_analysis', {}), indent=2)}

---

*Report generated by Apollo/ENIGMA Quantitative Portfolio Terminal v6.0*
"""
                    report_content = markdown_content
                    file_name = f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                    mime_type = "text/markdown"
                
                # Provide download button
                st.download_button(
                    label="📥 Download Report",
                    data=report_content,
                    file_name=file_name,
                    mime=mime_type
                )
                
                st.success("✅ Report generated successfully!")

# =============================================================
# RUN APPLICATION
# =============================================================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.code(traceback.format_exc())
