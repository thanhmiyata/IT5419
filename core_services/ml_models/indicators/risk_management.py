"""
Risk Management Indicators
==========================

Risk measurement and management indicators for investment analysis:
- Volatility Metrics
- Downside Risk Measures
- Risk-Adjusted Returns
- Drawdown Analysis
- Value at Risk (VaR)
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd


class RiskManagementIndicators:
    """Calculate risk management and measurement indicators."""

    # Volatility Metrics
    @staticmethod
    def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate annualized volatility (standard deviation).

        Args:
            returns: Series of returns
            periods_per_year: Trading periods per year (252 for daily, 12 for monthly)

        Returns:
            Annualized volatility
        """
        return returns.std() * np.sqrt(periods_per_year)

    @staticmethod
    def rolling_volatility(returns: pd.Series, window: int = 20, periods_per_year: int = 252) -> pd.Series:
        """
        Calculate rolling volatility.

        Args:
            returns: Series of returns
            window: Rolling window size
            periods_per_year: Trading periods per year

        Returns:
            Series of rolling volatility
        """
        return returns.rolling(window=window).std() * np.sqrt(periods_per_year)

    # Downside Risk Measures
    @staticmethod
    def downside_deviation(returns: pd.Series, target_return: float = 0.0, periods_per_year: int = 252) -> float:
        """
        Calculate downside deviation (semi-deviation).

        Args:
            returns: Series of returns
            target_return: Minimum acceptable return (MAR)
            periods_per_year: Trading periods per year

        Returns:
            Annualized downside deviation
        """
        downside_returns = returns[returns < target_return]
        downside_diff = downside_returns - target_return
        return np.sqrt((downside_diff ** 2).mean()) * np.sqrt(periods_per_year)

    @staticmethod
    def sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        target_return: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sortino Ratio (risk-adjusted return using downside deviation).

        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate
            target_return: Minimum acceptable return
            periods_per_year: Trading periods per year

        Returns:
            Sortino ratio (higher is better)
        """
        excess_return = (returns.mean() - risk_free_rate) * periods_per_year
        downside_dev = RiskManagementIndicators.downside_deviation(returns, target_return, periods_per_year)

        if downside_dev == 0:
            return 0.0

        return excess_return / downside_dev

    # Risk-Adjusted Returns
    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe Ratio (risk-adjusted return).

        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate (annual)
            periods_per_year: Trading periods per year

        Returns:
            Sharpe ratio (higher is better)
        """
        excess_returns = returns - (risk_free_rate / periods_per_year)
        if excess_returns.std() == 0:
            return 0.0

        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)

    @staticmethod
    def information_ratio(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate Information Ratio (alpha per unit of tracking error).

        Args:
            portfolio_returns: Portfolio returns series
            benchmark_returns: Benchmark returns series

        Returns:
            Information ratio
        """
        active_returns = portfolio_returns - benchmark_returns
        tracking_error = active_returns.std()

        if tracking_error == 0:
            return 0.0

        return active_returns.mean() / tracking_error

    @staticmethod
    def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate Calmar Ratio (annual return / max drawdown).

        Args:
            returns: Series of returns
            periods_per_year: Trading periods per year

        Returns:
            Calmar ratio
        """
        annual_return = returns.mean() * periods_per_year
        max_dd = RiskManagementIndicators.maximum_drawdown(returns)

        if max_dd == 0:
            return 0.0

        return annual_return / abs(max_dd)

    @staticmethod
    def omega_ratio(returns: pd.Series, target_return: float = 0.0) -> float:
        """
        Calculate Omega Ratio (probability-weighted ratio of gains vs losses).

        Args:
            returns: Series of returns
            target_return: Threshold return

        Returns:
            Omega ratio (> 1 is good)
        """
        gains = returns[returns > target_return] - target_return
        losses = target_return - returns[returns < target_return]

        if losses.sum() == 0:
            return float("inf") if gains.sum() > 0 else 0.0

        return gains.sum() / losses.sum()

    # Drawdown Analysis
    @staticmethod
    def maximum_drawdown(returns: pd.Series) -> float:
        """
        Calculate Maximum Drawdown (MDD).

        Args:
            returns: Series of returns

        Returns:
            Maximum drawdown (negative value)
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    @staticmethod
    def drawdown_series(returns: pd.Series) -> pd.Series:
        """
        Calculate drawdown series.

        Args:
            returns: Series of returns

        Returns:
            Series of drawdown values
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown

    @staticmethod
    def average_drawdown(returns: pd.Series) -> float:
        """
        Calculate Average Drawdown.

        Args:
            returns: Series of returns

        Returns:
            Average drawdown
        """
        drawdown = RiskManagementIndicators.drawdown_series(returns)
        return drawdown[drawdown < 0].mean()

    @staticmethod
    def recovery_time(returns: pd.Series) -> Optional[int]:
        """
        Calculate time to recover from maximum drawdown.

        Args:
            returns: Series of returns

        Returns:
            Number of periods to recover, or None if not recovered
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        max_dd_idx = drawdown.idxmin()
        post_dd = cumulative[max_dd_idx:]

        recovery_idx = post_dd[post_dd >= running_max[max_dd_idx]].index

        if len(recovery_idx) == 0:
            return None

        return len(post_dd[:recovery_idx[0]])

    # Value at Risk
    @staticmethod
    def value_at_risk(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR) using historical method.

        Args:
            returns: Series of returns
            confidence_level: Confidence level (0.95 = 95%)

        Returns:
            VaR (negative value represents potential loss)
        """
        return returns.quantile(1 - confidence_level)

    @staticmethod
    def conditional_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR / Expected Shortfall).

        Args:
            returns: Series of returns
            confidence_level: Confidence level

        Returns:
            CVaR (average loss beyond VaR)
        """
        var = RiskManagementIndicators.value_at_risk(returns, confidence_level)
        return returns[returns <= var].mean()

    @staticmethod
    def parametric_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate parametric VaR (assumes normal distribution).

        Args:
            returns: Series of returns
            confidence_level: Confidence level

        Returns:
            Parametric VaR
        """
        from scipy import stats
        z_score = stats.norm.ppf(1 - confidence_level)
        return returns.mean() + z_score * returns.std()

    # Risk Metrics
    @staticmethod
    def tracking_error(portfolio_returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int = 252
                       ) -> float:
        """
        Calculate Tracking Error (annualized).

        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            periods_per_year: Trading periods per year

        Returns:
            Annualized tracking error
        """
        active_returns = portfolio_returns - benchmark_returns
        return active_returns.std() * np.sqrt(periods_per_year)

    @staticmethod
    def beta_coefficient(stock_returns: pd.Series, market_returns: pd.Series) -> float:
        """
        Calculate Beta (systematic risk).

        Args:
            stock_returns: Stock returns
            market_returns: Market returns

        Returns:
            Beta coefficient
        """
        covariance = np.cov(stock_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)

        if market_variance == 0:
            return 0.0

        return covariance / market_variance

    @staticmethod
    def treynor_ratio(
        returns: pd.Series,
        market_returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Treynor Ratio (excess return per unit of systematic risk).

        Args:
            returns: Portfolio returns
            market_returns: Market returns
            risk_free_rate: Risk-free rate
            periods_per_year: Trading periods per year

        Returns:
            Treynor ratio
        """
        beta = RiskManagementIndicators.beta_coefficient(returns, market_returns)

        if beta == 0:
            return 0.0

        excess_return = (returns.mean() - risk_free_rate) * periods_per_year
        return excess_return / beta

    @staticmethod
    def jensens_alpha(
        returns: pd.Series,
        market_returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Jensen's Alpha (excess return over CAPM prediction).

        Args:
            returns: Portfolio returns
            market_returns: Market returns
            risk_free_rate: Risk-free rate
            periods_per_year: Trading periods per year

        Returns:
            Jensen's alpha (annualized)
        """
        beta = RiskManagementIndicators.beta_coefficient(returns, market_returns)

        portfolio_return = returns.mean() * periods_per_year
        market_return = market_returns.mean() * periods_per_year

        expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
        alpha = portfolio_return - expected_return

        return alpha

    @staticmethod
    def tail_ratio(returns: pd.Series, percentile: int = 5) -> float:
        """
        Calculate Tail Ratio (right tail / left tail).

        Args:
            returns: Series of returns
            percentile: Percentile for tail (5 = 5%)

        Returns:
            Tail ratio (> 1 means right tail is larger)
        """
        right_tail = returns.quantile(1 - percentile / 100)
        left_tail = abs(returns.quantile(percentile / 100))

        if left_tail == 0:
            return float("inf") if right_tail > 0 else 0.0

        return right_tail / left_tail

    @staticmethod
    def stability_of_timeseries(returns: pd.Series) -> float:
        """
        Calculate stability (R-squared of linear regression vs time).

        Args:
            returns: Series of returns

        Returns:
            R-squared value (0-1, higher is more stable)
        """
        cumulative = (1 + returns).cumprod()
        x = np.arange(len(cumulative))
        y = cumulative.values

        coefficients = np.polyfit(x, y, 1)
        fitted = np.polyval(coefficients, x)

        ss_res = np.sum((y - fitted) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 0.0

        r_squared = 1 - (ss_res / ss_tot)
        return r_squared

    @classmethod
    def calculate_all_risk_metrics(
        cls,
        returns: pd.Series,
        market_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> Tuple[dict, dict]:
        """
        Calculate all risk metrics at once.

        Returns:
            Tuple of (basic_metrics, advanced_metrics)
        """
        basic_metrics = {
            "annualized_volatility": cls.annualized_volatility(returns, periods_per_year),
            "sharpe_ratio": cls.sharpe_ratio(returns, risk_free_rate, periods_per_year),
            "sortino_ratio": cls.sortino_ratio(returns, risk_free_rate, 0.0, periods_per_year),
            "maximum_drawdown": cls.maximum_drawdown(returns),
            "var_95": cls.value_at_risk(returns, 0.95),
            "cvar_95": cls.conditional_var(returns, 0.95),
        }

        advanced_metrics = {
            "downside_deviation": cls.downside_deviation(returns, 0.0, periods_per_year),
            "calmar_ratio": cls.calmar_ratio(returns, periods_per_year),
            "omega_ratio": cls.omega_ratio(returns, 0.0),
            "tail_ratio": cls.tail_ratio(returns),
            "stability": cls.stability_of_timeseries(returns),
        }

        if market_returns is not None:
            advanced_metrics["beta"] = cls.beta_coefficient(returns, market_returns)
            advanced_metrics["treynor_ratio"] = cls.treynor_ratio(
                returns, market_returns, risk_free_rate, periods_per_year
            )
            advanced_metrics["jensens_alpha"] = cls.jensens_alpha(
                returns, market_returns, risk_free_rate, periods_per_year
            )
            advanced_metrics["information_ratio"] = cls.information_ratio(returns, market_returns)
            advanced_metrics["tracking_error"] = cls.tracking_error(returns, market_returns, periods_per_year)

        return basic_metrics, advanced_metrics
