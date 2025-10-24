"""
## 1. FUNDAMENTAL ANALYSIS
   - Valuation Ratios (P/E, P/B, P/S, PEG, EV/EBITDA)
   - Profitability Metrics (ROE, ROA, ROIC, Margins)
   - Growth Indicators (Revenue Growth, EPS Growth)

## 2. TECHNICAL ANALYSIS
   - Trend Indicators (MA, MACD, ADX)
   - Momentum Indicators (RSI, Stochastic, ROC, CCI)
   - Volume Indicators (OBV, VWAP, MFI)
   - Volatility Indicators (Bollinger Bands, ATR)
   - Support/Resistance (Fibonacci, Pivot Points)
   - Market Breadth (A/D Line, Put-Call Ratio)

## 3. FINANCIAL STRENGTH
   - Liquidity Ratios (Current, Quick, Cash Ratios)
   - Solvency Ratios (D/E, D/A, Interest Coverage)
   - Efficiency Ratios (Asset Turnover, Inventory Turnover)
   - Cash Flow Indicators (OCF Ratio, Free Cash Flow)
   - Financial Health Scores (Altman Z-Score, Piotroski F-Score, Beneish M-Score)

## 4. RISK MANAGEMENT
   - Volatility Metrics (Annualized Vol, Rolling Vol)
   - Downside Risk (Downside Deviation, Sortino Ratio)
   - Risk-Adjusted Returns (Sharpe, Treynor, Jensen's Alpha)
   - Drawdown Analysis (Max Drawdown, Recovery Time)
   - Value at Risk (VaR, CVaR, Parametric VaR)

## 5. PORTFOLIO MANAGEMENT
   - Diversification Metrics (Herfindahl Index, ENS, Diversification Ratio)
   - Portfolio Performance (Returns, Volatility, Sharpe)
   - Asset Allocation (Minimum Variance, Maximum Sharpe, Risk Parity)
   - Rebalancing Signals (Drift Detection, Turnover Rate)
   - Portfolio Risk (Contribution to Risk, Portfolio VaR)
"""

from core_services.ml_models.indicators.financial_strength import FinancialStrengthIndicators
from core_services.ml_models.indicators.fundamental import FundamentalIndicators
from core_services.ml_models.indicators.market_breadth import MarketBreadthIndicators
from core_services.ml_models.indicators.momentum import MomentumIndicators
from core_services.ml_models.indicators.portfolio_management import PortfolioManagementIndicators
from core_services.ml_models.indicators.risk_management import RiskManagementIndicators
from core_services.ml_models.indicators.support_resistance import SupportResistanceIndicators
from core_services.ml_models.indicators.trend import TrendIndicators
from core_services.ml_models.indicators.volatility import VolatilityIndicators
from core_services.ml_models.indicators.volume import VolumeIndicators

__all__ = [
    # Fundamental Analysis
    "FundamentalIndicators",
    # Technical Analysis
    "TrendIndicators",
    "MomentumIndicators",
    "VolumeIndicators",
    "VolatilityIndicators",
    "SupportResistanceIndicators",
    "MarketBreadthIndicators",
    # Financial Strength
    "FinancialStrengthIndicators",
    # Risk Management
    "RiskManagementIndicators",
    # Portfolio Management
    "PortfolioManagementIndicators",
]

__version__ = "1.0.0"
