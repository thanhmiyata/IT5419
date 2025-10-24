"""
Portfolio Management Indicators
================================

Portfolio construction, optimization, and management indicators:
- Diversification Metrics
- Portfolio Performance
- Asset Allocation
- Rebalancing Signals
- Portfolio Risk Metrics
"""

from typing import Dict, List

import numpy as np
import pandas as pd


class PortfolioManagementIndicators:
    """Calculate portfolio management and optimization indicators."""

    # Diversification Metrics
    @staticmethod
    def portfolio_weights(positions: pd.Series) -> pd.Series:
        """
        Calculate portfolio weights from positions.

        Args:
            positions: Series of position values

        Returns:
            Series of portfolio weights
        """
        total_value = positions.sum()
        if total_value == 0:
            return pd.Series(0, index=positions.index)
        return positions / total_value

    @staticmethod
    def herfindahl_index(weights: pd.Series) -> float:
        """
        Calculate Herfindahl Index (concentration measure).

        Args:
            weights: Series of portfolio weights

        Returns:
            Herfindahl index (0=diversified, 1=concentrated)
        """
        return (weights ** 2).sum()

    @staticmethod
    def effective_number_of_stocks(weights: pd.Series) -> float:
        """
        Calculate Effective Number of Stocks (ENS).

        Args:
            weights: Series of portfolio weights

        Returns:
            ENS (higher = more diversified)
        """
        hhi = (weights ** 2).sum()
        if hhi == 0:
            return 0.0
        return 1 / hhi

    @staticmethod
    def diversification_ratio(weights: pd.Series, volatilities: pd.Series, portfolio_vol: float) -> float:
        """
        Calculate Diversification Ratio.

        Args:
            weights: Portfolio weights
            volatilities: Individual asset volatilities
            portfolio_vol: Portfolio volatility

        Returns:
            Diversification ratio (> 1 indicates diversification benefit)
        """
        if portfolio_vol == 0:
            return 0.0
        weighted_vol = (weights * volatilities).sum()
        return weighted_vol / portfolio_vol

    # Portfolio Performance
    @staticmethod
    def portfolio_return(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
        """
        Calculate portfolio returns.

        Args:
            returns: DataFrame of asset returns
            weights: Series of portfolio weights

        Returns:
            Series of portfolio returns
        """
        return (returns * weights).sum(axis=1)

    @staticmethod
    def cumulative_return(returns: pd.Series) -> float:
        """
        Calculate cumulative return.

        Args:
            returns: Series of returns

        Returns:
            Cumulative return
        """
        return (1 + returns).prod() - 1

    @staticmethod
    def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate annualized return.

        Args:
            returns: Series of returns
            periods_per_year: Trading periods per year

        Returns:
            Annualized return
        """
        cum_return = (1 + returns).prod()
        num_periods = len(returns)
        if num_periods == 0:
            return 0.0
        return cum_return ** (periods_per_year / num_periods) - 1

    @staticmethod
    def portfolio_volatility(returns: pd.DataFrame, weights: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate portfolio volatility.

        Args:
            returns: DataFrame of asset returns
            weights: Portfolio weights
            periods_per_year: Trading periods per year

        Returns:
            Annualized portfolio volatility
        """
        cov_matrix = returns.cov()
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        return np.sqrt(portfolio_variance * periods_per_year)

    # Asset Allocation
    @staticmethod
    def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix.

        Args:
            returns: DataFrame of asset returns

        Returns:
            Correlation matrix
        """
        return returns.corr()

    @staticmethod
    def covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate covariance matrix.

        Args:
            returns: DataFrame of asset returns

        Returns:
            Covariance matrix
        """
        return returns.cov()

    @staticmethod
    def minimum_variance_portfolio(cov_matrix: pd.DataFrame) -> pd.Series:
        """
        Calculate minimum variance portfolio weights.

        Args:
            cov_matrix: Covariance matrix

        Returns:
            Series of optimal weights
        """
        inv_cov = np.linalg.inv(cov_matrix)
        ones = np.ones(len(cov_matrix))
        weights = np.dot(inv_cov, ones) / np.dot(ones, np.dot(inv_cov, ones))
        return pd.Series(weights, index=cov_matrix.columns)

    @staticmethod
    def maximum_sharpe_ratio_weights(
        returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float = 0.02
    ) -> pd.Series:
        """
        Calculate maximum Sharpe ratio portfolio weights (simplified).

        Args:
            returns: Expected returns
            cov_matrix: Covariance matrix
            risk_free_rate: Risk-free rate

        Returns:
            Series of optimal weights
        """
        excess_returns = returns - risk_free_rate
        inv_cov = np.linalg.inv(cov_matrix)
        weights = np.dot(inv_cov, excess_returns)
        weights = weights / weights.sum()
        return pd.Series(weights, index=cov_matrix.columns)

    # Rebalancing Signals
    @staticmethod
    def drift_from_target(current_weights: pd.Series, target_weights: pd.Series) -> pd.Series:
        """
        Calculate drift from target weights.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights

        Returns:
            Series of weight differences
        """
        return current_weights - target_weights

    @staticmethod
    def rebalance_needed(
        current_weights: pd.Series,
        target_weights: pd.Series,
        threshold: float = 0.05
    ) -> bool:
        """
        Check if rebalancing is needed.

        Args:
            current_weights: Current weights
            target_weights: Target weights
            threshold: Maximum allowed drift

        Returns:
            True if rebalancing needed
        """
        drift = abs(current_weights - target_weights)
        return (drift > threshold).any()

    @staticmethod
    def turnover_rate(trades: pd.Series, portfolio_value: float) -> float:
        """
        Calculate portfolio turnover rate.

        Args:
            trades: Series of trade values (absolute)
            portfolio_value: Total portfolio value

        Returns:
            Turnover rate
        """
        if portfolio_value == 0:
            return 0.0
        return trades.abs().sum() / (2 * portfolio_value)

    # Portfolio Risk Metrics
    @staticmethod
    def contribution_to_risk(weights: pd.Series, cov_matrix: pd.DataFrame) -> pd.Series:
        """
        Calculate each asset's contribution to portfolio risk.

        Args:
            weights: Portfolio weights
            cov_matrix: Covariance matrix

        Returns:
            Series of risk contributions
        """
        portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
        marginal_contrib = np.dot(cov_matrix, weights)
        contrib_to_risk = weights * marginal_contrib

        if portfolio_var == 0:
            return pd.Series(0, index=cov_matrix.index)

        return pd.Series(contrib_to_risk / np.sqrt(portfolio_var), index=cov_matrix.index)

    @staticmethod
    def value_at_risk_portfolio(
        returns: pd.DataFrame,
        weights: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate portfolio Value at Risk.

        Args:
            returns: DataFrame of asset returns
            weights: Portfolio weights
            confidence_level: Confidence level

        Returns:
            Portfolio VaR
        """
        portfolio_returns = (returns * weights).sum(axis=1)
        return portfolio_returns.quantile(1 - confidence_level)

    @staticmethod
    def conditional_drawdown_risk(returns: pd.Series, percentile: int = 5) -> float:
        """
        Calculate Conditional Drawdown at Risk (CDaR).

        Args:
            returns: Portfolio returns
            percentile: Percentile for worst drawdowns

        Returns:
            Average of worst drawdowns
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        threshold = drawdown.quantile(percentile / 100)
        worst_drawdowns = drawdown[drawdown <= threshold]

        return worst_drawdowns.mean()

    @staticmethod
    def risk_parity_weights(volatilities: pd.Series) -> pd.Series:
        """
        Calculate risk parity weights (equal risk contribution).

        Args:
            volatilities: Series of asset volatilities

        Returns:
            Series of risk parity weights
        """
        inv_vol = 1 / volatilities
        weights = inv_vol / inv_vol.sum()
        return weights

    @staticmethod
    def equal_weight_portfolio(num_assets: int, asset_names: List[str]) -> pd.Series:
        """
        Calculate equal weight portfolio.

        Args:
            num_assets: Number of assets
            asset_names: List of asset names

        Returns:
            Series of equal weights
        """
        weight = 1.0 / num_assets
        return pd.Series(weight, index=asset_names)

    @classmethod
    def portfolio_summary(
        cls,
        returns: pd.DataFrame,
        weights: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> Dict[str, float]:
        """
        Calculate comprehensive portfolio summary statistics.

        Returns:
            Dictionary of portfolio metrics
        """
        portfolio_ret = cls.portfolio_return(returns, weights)
        portfolio_vol = cls.portfolio_volatility(returns, weights, periods_per_year)

        summary = {
            "annualized_return": cls.annualized_return(portfolio_ret, periods_per_year),
            "annualized_volatility": portfolio_vol,
            "sharpe_ratio": (cls.annualized_return(portfolio_ret, periods_per_year) - risk_free_rate) / portfolio_vol
            if portfolio_vol > 0 else 0.0,
            "cumulative_return": cls.cumulative_return(portfolio_ret),
            "max_drawdown": (1 + portfolio_ret).cumprod().div((1 + portfolio_ret).cumprod().cummax()).min() - 1,
            "var_95": cls.value_at_risk_portfolio(returns, weights, 0.95),
            "herfindahl_index": cls.herfindahl_index(weights),
            "effective_stocks": cls.effective_number_of_stocks(weights),
        }

        return summary
