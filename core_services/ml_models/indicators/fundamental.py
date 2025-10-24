"""
Fundamental Analysis Indicators for Stock Analysis
==================================================

Comprehensive fundamental analysis indicators including:
- Valuation Ratios (P/E, P/B, P/S, PEG, EV/EBITDA)
- Per-Share Metrics (EPS, BVPS, DPS, Dividend Yield)
- Profitability Ratios (ROE, ROA, ROIC, Margins)
- Leverage/Solvency (D/E, D/A, Interest Coverage)
- Growth & Risk Metrics (Revenue Growth, EPS Growth, Beta)
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd


class FundamentalIndicators:
    """Calculate fundamental analysis indicators for stocks."""

    @staticmethod
    def price_to_earnings(price: float, eps: float) -> Optional[float]:
        """Calculate Price-to-Earnings (P/E) ratio."""
        if eps == 0:
            return None
        return price / eps

    @staticmethod
    def price_to_book(price: float, bvps: float) -> Optional[float]:
        """Calculate Price-to-Book (P/B) ratio."""
        if bvps == 0:
            return None
        return price / bvps

    @staticmethod
    def price_to_sales(market_cap: float, revenue: float) -> Optional[float]:
        """Calculate Price-to-Sales (P/S) ratio."""
        if revenue == 0:
            return None
        return market_cap / revenue

    @staticmethod
    def peg_ratio(pe_ratio: float, eps_growth_rate: float) -> Optional[float]:
        """Calculate PEG (P/E to Growth) ratio."""
        if eps_growth_rate == 0 or pe_ratio is None:
            return None
        return pe_ratio / eps_growth_rate

    @staticmethod
    def ev_to_ebitda(enterprise_value: float, ebitda: float) -> Optional[float]:
        """Calculate EV/EBITDA ratio."""
        if ebitda == 0:
            return None
        return enterprise_value / ebitda

    @staticmethod
    def earnings_per_share(net_income: float, shares_outstanding: float) -> Optional[float]:
        """Calculate Earnings Per Share (EPS)."""
        if shares_outstanding == 0:
            return None
        return net_income / shares_outstanding

    @staticmethod
    def book_value_per_share(total_equity: float, shares_outstanding: float) -> Optional[float]:
        """Calculate Book Value Per Share (BVPS)."""
        if shares_outstanding == 0:
            return None
        return total_equity / shares_outstanding

    @staticmethod
    def dividend_per_share(total_dividends: float, shares_outstanding: float) -> Optional[float]:
        """Calculate Dividends Per Share (DPS)."""
        if shares_outstanding == 0:
            return None
        return total_dividends / shares_outstanding

    @staticmethod
    def dividend_yield(annual_dividend: float, price: float) -> Optional[float]:
        """Calculate Dividend Yield (%)."""
        if price == 0:
            return None
        return (annual_dividend / price) * 100

    @staticmethod
    def return_on_equity(net_income: float, shareholders_equity: float) -> Optional[float]:
        """Calculate Return on Equity (ROE) %."""
        if shareholders_equity == 0:
            return None
        return (net_income / shareholders_equity) * 100

    @staticmethod
    def return_on_assets(net_income: float, total_assets: float) -> Optional[float]:
        """Calculate Return on Assets (ROA) %."""
        if total_assets == 0:
            return None
        return (net_income / total_assets) * 100

    @staticmethod
    def return_on_invested_capital(nopat: float, invested_capital: float) -> Optional[float]:
        """Calculate Return on Invested Capital (ROIC) %."""
        if invested_capital == 0:
            return None
        return (nopat / invested_capital) * 100

    @staticmethod
    def net_profit_margin(net_income: float, revenue: float) -> Optional[float]:
        """Calculate Net Profit Margin (%)."""
        if revenue == 0:
            return None
        return (net_income / revenue) * 100

    @staticmethod
    def operating_margin(operating_income: float, revenue: float) -> Optional[float]:
        """Calculate Operating Margin (%)."""
        if revenue == 0:
            return None
        return (operating_income / revenue) * 100

    @staticmethod
    def gross_margin(gross_profit: float, revenue: float) -> Optional[float]:
        """Calculate Gross Margin (%)."""
        if revenue == 0:
            return None
        return (gross_profit / revenue) * 100

    @staticmethod
    def debt_to_equity(total_debt: float, shareholders_equity: float) -> Optional[float]:
        """Calculate Debt-to-Equity (D/E) ratio."""
        if shareholders_equity == 0:
            return None
        return total_debt / shareholders_equity

    @staticmethod
    def debt_to_assets(total_debt: float, total_assets: float) -> Optional[float]:
        """Calculate Debt-to-Assets (D/A) ratio."""
        if total_assets == 0:
            return None
        return total_debt / total_assets

    @staticmethod
    def interest_coverage(ebit: float, interest_expense: float) -> Optional[float]:
        """Calculate Interest Coverage ratio."""
        if interest_expense == 0:
            return None
        return ebit / interest_expense

    @staticmethod
    def revenue_growth(current_revenue: float, previous_revenue: float) -> Optional[float]:
        """Calculate Revenue Growth (%)."""
        if previous_revenue == 0:
            return None
        return ((current_revenue - previous_revenue) / previous_revenue) * 100

    @staticmethod
    def eps_growth(current_eps: float, previous_eps: float) -> Optional[float]:
        """Calculate EPS Growth (%)."""
        if previous_eps == 0:
            return None
        return ((current_eps - previous_eps) / previous_eps) * 100

    @staticmethod
    def beta(stock_returns: pd.Series, market_returns: pd.Series) -> Optional[float]:
        """Calculate Beta (systematic risk measure)."""
        if len(stock_returns) != len(market_returns):
            return None

        covariance = np.cov(stock_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)

        if market_variance == 0:
            return None

        return covariance / market_variance

    @classmethod
    def calculate_all_ratios(
        cls,
        price: float,
        financial_data: Dict[str, float],
        stock_returns: Optional[pd.Series] = None,
        market_returns: Optional[pd.Series] = None,
    ) -> Dict[str, Optional[float]]:
        """Calculate all fundamental indicators at once."""
        results: Dict[str, Optional[float]] = {}

        # Valuation Ratios
        results["pe_ratio"] = cls.price_to_earnings(price, financial_data.get("eps", 0))
        results["pb_ratio"] = cls.price_to_book(price, financial_data.get("bvps", 0))
        results["ps_ratio"] = cls.price_to_sales(
            financial_data.get("market_cap", 0),
            financial_data.get("revenue", 0)
        )
        results["peg_ratio"] = cls.peg_ratio(
            results["pe_ratio"] or 0,
            financial_data.get("eps_growth", 0)
        )
        results["ev_ebitda"] = cls.ev_to_ebitda(
            financial_data.get("enterprise_value", 0),
            financial_data.get("ebitda", 0)
        )

        # Per-Share Metrics
        results["eps"] = financial_data.get("eps")
        results["bvps"] = financial_data.get("bvps")
        results["dps"] = cls.dividend_per_share(
            financial_data.get("total_dividends", 0),
            financial_data.get("shares_outstanding", 0)
        )
        results["dividend_yield"] = cls.dividend_yield(financial_data.get("annual_dividend", 0), price)

        # Profitability Ratios
        results["roe"] = cls.return_on_equity(
            financial_data.get("net_income", 0),
            financial_data.get("shareholders_equity", 0)
        )
        results["roa"] = cls.return_on_assets(
            financial_data.get("net_income", 0),
            financial_data.get("total_assets", 0)
        )
        results["roic"] = cls.return_on_invested_capital(
            financial_data.get("nopat", 0),
            financial_data.get("invested_capital", 0)
        )
        results["net_profit_margin"] = cls.net_profit_margin(
            financial_data.get("net_income", 0),
            financial_data.get("revenue", 0)
        )
        results["operating_margin"] = cls.operating_margin(
            financial_data.get("operating_income", 0),
            financial_data.get("revenue", 0)
        )
        results["gross_margin"] = cls.gross_margin(
            financial_data.get("gross_profit", 0),
            financial_data.get("revenue", 0)
        )

        # Leverage/Solvency
        results["debt_to_equity"] = cls.debt_to_equity(
            financial_data.get("total_debt", 0),
            financial_data.get("shareholders_equity", 0)
        )
        results["debt_to_assets"] = cls.debt_to_assets(
            financial_data.get("total_debt", 0),
            financial_data.get("total_assets", 0)
        )
        results["interest_coverage"] = cls.interest_coverage(
            financial_data.get("ebit", 0),
            financial_data.get("interest_expense", 0)
        )

        # Growth Metrics
        results["revenue_growth"] = cls.revenue_growth(
            financial_data.get("revenue", 0),
            financial_data.get("previous_revenue", 0)
        )
        results["eps_growth"] = financial_data.get("eps_growth")

        # Risk Metrics
        if stock_returns is not None and market_returns is not None:
            results["beta"] = cls.beta(stock_returns, market_returns)
        else:
            results["beta"] = None

        return results
