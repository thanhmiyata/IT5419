"""
Fama-French Factor Model
=========================

Multi-factor model for asset pricing.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression


class FamaFrenchModel:
    """Fama-French 3-factor and 5-factor models."""

    def __init__(self, model_type: str = '3-factor'):
        """
        Initialize Fama-French model.

        Args:
            model_type: '3-factor' or '5-factor'
        """
        self.model_type = model_type
        self.regression_results = {}

    def calculate_smb(
        self,
        returns: pd.DataFrame,
        market_caps: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate SMB (Small Minus Big) factor.

        Args:
            returns: Stock returns
            market_caps: Market capitalizations

        Returns:
            SMB factor returns
        """
        # Determine median market cap
        median_cap = market_caps.median(axis=1)

        smb_returns = []

        for date in returns.index:
            date_returns = returns.loc[date]
            date_caps = market_caps.loc[date]

            # Split into small and big
            small_stocks = date_returns[date_caps <= median_cap[date]]
            big_stocks = date_returns[date_caps > median_cap[date]]

            # SMB = average small - average big
            smb = small_stocks.mean() - big_stocks.mean()
            smb_returns.append(smb)

        return pd.Series(smb_returns, index=returns.index)

    def calculate_hml(
        self,
        returns: pd.DataFrame,
        book_to_market: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate HML (High Minus Low) factor.

        Args:
            returns: Stock returns
            book_to_market: Book-to-market ratios

        Returns:
            HML factor returns
        """
        hml_returns = []

        for date in returns.index:
            date_returns = returns.loc[date]
            date_btm = book_to_market.loc[date]

            # Split into high and low B/M
            top_30_pct = date_btm.quantile(0.7)
            bottom_30_pct = date_btm.quantile(0.3)

            high_btm = date_returns[date_btm >= top_30_pct]
            low_btm = date_returns[date_btm <= bottom_30_pct]

            # HML = average high - average low
            hml = high_btm.mean() - low_btm.mean()
            hml_returns.append(hml)

        return pd.Series(hml_returns, index=returns.index)

    def calculate_rmw(
        self,
        returns: pd.DataFrame,
        profitability: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate RMW (Robust Minus Weak) factor.

        Args:
            returns: Stock returns
            profitability: Operating profitability

        Returns:
            RMW factor returns
        """
        rmw_returns = []

        for date in returns.index:
            date_returns = returns.loc[date]
            date_profit = profitability.loc[date]

            # Split by profitability
            top_30_pct = date_profit.quantile(0.7)
            bottom_30_pct = date_profit.quantile(0.3)

            robust = date_returns[date_profit >= top_30_pct]
            weak = date_returns[date_profit <= bottom_30_pct]

            # RMW = average robust - average weak
            rmw = robust.mean() - weak.mean()
            rmw_returns.append(rmw)

        return pd.Series(rmw_returns, index=returns.index)

    def calculate_cma(
        self,
        returns: pd.DataFrame,
        investment: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate CMA (Conservative Minus Aggressive) factor.

        Args:
            returns: Stock returns
            investment: Asset growth

        Returns:
            CMA factor returns
        """
        cma_returns = []

        for date in returns.index:
            date_returns = returns.loc[date]
            date_inv = investment.loc[date]

            # Split by investment
            top_30_pct = date_inv.quantile(0.7)
            bottom_30_pct = date_inv.quantile(0.3)

            conservative = date_returns[date_inv <= bottom_30_pct]
            aggressive = date_returns[date_inv >= top_30_pct]

            # CMA = average conservative - average aggressive
            cma = conservative.mean() - aggressive.mean()
            cma_returns.append(cma)

        return pd.Series(cma_returns, index=returns.index)

    def run_3factor_regression(
        self,
        stock_returns: pd.Series,
        market_returns: pd.Series,
        smb: pd.Series,
        hml: pd.Series,
        risk_free_rate: pd.Series
    ) -> dict:
        """
        Run 3-factor regression.

        Args:
            stock_returns: Stock excess returns
            market_returns: Market excess returns
            smb: SMB factor
            hml: HML factor
            risk_free_rate: Risk-free rate

        Returns:
            Dictionary with regression results
        """
        # Excess returns
        stock_excess = stock_returns - risk_free_rate
        market_excess = market_returns - risk_free_rate

        # Prepare data
        X = pd.DataFrame({
            'MKT': market_excess,
            'SMB': smb,
            'HML': hml
        }).dropna()

        y = stock_excess.loc[X.index]

        # Run regression
        model = LinearRegression()
        model.fit(X, y)

        # Calculate R-squared
        r_squared = model.score(X, y)

        return {
            'alpha': model.intercept_,
            'beta_market': model.coef_[0],
            'beta_smb': model.coef_[1],
            'beta_hml': model.coef_[2],
            'r_squared': r_squared
        }

    def run_5factor_regression(
        self,
        stock_returns: pd.Series,
        market_returns: pd.Series,
        smb: pd.Series,
        hml: pd.Series,
        rmw: pd.Series,
        cma: pd.Series,
        risk_free_rate: pd.Series
    ) -> dict:
        """
        Run 5-factor regression.

        Returns:
            Dictionary with regression results
        """
        # Excess returns
        stock_excess = stock_returns - risk_free_rate
        market_excess = market_returns - risk_free_rate

        # Prepare data
        X = pd.DataFrame({
            'MKT': market_excess,
            'SMB': smb,
            'HML': hml,
            'RMW': rmw,
            'CMA': cma
        }).dropna()

        y = stock_excess.loc[X.index]

        # Run regression
        model = LinearRegression()
        model.fit(X, y)

        r_squared = model.score(X, y)

        return {
            'alpha': model.intercept_,
            'beta_market': model.coef_[0],
            'beta_smb': model.coef_[1],
            'beta_hml': model.coef_[2],
            'beta_rmw': model.coef_[3],
            'beta_cma': model.coef_[4],
            'r_squared': r_squared
        }
