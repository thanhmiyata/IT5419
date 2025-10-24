"""
Trading and Portfolio Performance Metrics
==========================================

Comprehensive metrics for evaluating trading strategies and models.
"""

import numpy as np
import pandas as pd


class PerformanceMetrics:
    """Calculate performance metrics for trading strategies."""

    @staticmethod
    def total_return(returns: pd.Series) -> float:
        """
        Calculate total return.

        Args:
            returns: Series of returns

        Returns:
            Total return
        """
        return (1 + returns).prod() - 1

    @staticmethod
    def annualized_return(
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate annualized return.

        Args:
            returns: Series of returns
            periods_per_year: Trading periods per year (252 for daily)

        Returns:
            Annualized return
        """
        total_ret = (1 + returns).prod()
        n_periods = len(returns)
        years = n_periods / periods_per_year

        if years == 0:
            return 0.0

        return total_ret ** (1 / years) - 1

    @staticmethod
    def annualized_volatility(
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate annualized volatility.

        Args:
            returns: Series of returns
            periods_per_year: Trading periods per year

        Returns:
            Annualized volatility
        """
        return returns.std() * np.sqrt(periods_per_year)

    @staticmethod
    def sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year

        Returns:
            Sharpe ratio
        """
        excess_returns = returns - risk_free_rate / periods_per_year

        if excess_returns.std() == 0:
            return 0.0

        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)

    @staticmethod
    def sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sortino ratio (downside deviation).

        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year

        Returns:
            Sortino ratio
        """
        excess_returns = returns - risk_free_rate / periods_per_year
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        downside_std = downside_returns.std()
        return (excess_returns.mean() / downside_std) * np.sqrt(periods_per_year)

    @staticmethod
    def calmar_ratio(
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Calmar ratio (return / max drawdown).

        Args:
            returns: Series of returns
            periods_per_year: Trading periods per year

        Returns:
            Calmar ratio
        """
        annual_ret = PerformanceMetrics.annualized_return(returns, periods_per_year)
        max_dd = PerformanceMetrics.maximum_drawdown(returns)

        if max_dd == 0:
            return 0.0

        return annual_ret / abs(max_dd)

    @staticmethod
    def maximum_drawdown(returns: pd.Series) -> float:
        """
        Calculate maximum drawdown.

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
    def drawdown_duration(returns: pd.Series) -> dict:
        """
        Calculate drawdown statistics.

        Args:
            returns: Series of returns

        Returns:
            Dictionary with drawdown statistics
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        # Find drawdown periods
        in_drawdown = drawdown < 0

        if not in_drawdown.any():
            return {
                'max_drawdown': 0.0,
                'avg_drawdown': 0.0,
                'max_duration': 0,
                'avg_duration': 0
            }

        # Calculate durations
        drawdown_starts = in_drawdown & ~in_drawdown.shift(1, fill_value=False)
        drawdown_ends = ~in_drawdown & in_drawdown.shift(1, fill_value=False)

        durations = []
        magnitudes = []

        start_idx = None
        for idx, is_start in drawdown_starts.items():
            if is_start:
                start_idx = idx
            elif start_idx is not None and drawdown_ends[idx]:
                duration = (idx - start_idx).days if hasattr(idx, 'days') else (
                    returns.index.get_loc(idx) - returns.index.get_loc(start_idx)
                )
                durations.append(duration)
                magnitudes.append(drawdown[start_idx:idx].min())
                start_idx = None

        return {
            'max_drawdown': drawdown.min(),
            'avg_drawdown': np.mean(magnitudes) if magnitudes else 0.0,
            'max_duration': max(durations) if durations else 0,
            'avg_duration': np.mean(durations) if durations else 0
        }

    @staticmethod
    def win_rate(returns: pd.Series) -> float:
        """
        Calculate win rate.

        Args:
            returns: Series of returns

        Returns:
            Win rate (0 to 1)
        """
        if len(returns) == 0:
            return 0.0

        return (returns > 0).sum() / len(returns)

    @staticmethod
    def profit_factor(returns: pd.Series) -> float:
        """
        Calculate profit factor (gross profit / gross loss).

        Args:
            returns: Series of returns

        Returns:
            Profit factor
        """
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())

        if losses == 0:
            return np.inf if gains > 0 else 0.0

        return gains / losses

    @staticmethod
    def payoff_ratio(returns: pd.Series) -> float:
        """
        Calculate payoff ratio (avg win / avg loss).

        Args:
            returns: Series of returns

        Returns:
            Payoff ratio
        """
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        if len(losses) == 0 or losses.mean() == 0:
            return 0.0

        if len(wins) == 0:
            return 0.0

        return abs(wins.mean() / losses.mean())

    @staticmethod
    def value_at_risk(
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk (VaR).

        Args:
            returns: Series of returns
            confidence_level: Confidence level (e.g., 0.95)

        Returns:
            VaR value
        """
        return np.percentile(returns, (1 - confidence_level) * 100)

    @staticmethod
    def conditional_value_at_risk(
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall).

        Args:
            returns: Series of returns
            confidence_level: Confidence level

        Returns:
            CVaR value
        """
        var = PerformanceMetrics.value_at_risk(returns, confidence_level)
        return returns[returns <= var].mean()

    @staticmethod
    def information_ratio(
        returns: pd.Series,
        benchmark_returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Information Ratio.

        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            periods_per_year: Trading periods per year

        Returns:
            Information ratio
        """
        excess_returns = returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(periods_per_year)

        if tracking_error == 0:
            return 0.0

        return (excess_returns.mean() * periods_per_year) / tracking_error

    @staticmethod
    def beta(
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """
        Calculate beta (market sensitivity).

        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns

        Returns:
            Beta coefficient
        """
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = benchmark_returns.var()

        if benchmark_variance == 0:
            return 0.0

        return covariance / benchmark_variance

    @staticmethod
    def alpha(
        returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Jensen's alpha.

        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year

        Returns:
            Alpha
        """
        beta_coef = PerformanceMetrics.beta(returns, benchmark_returns)

        rf_per_period = risk_free_rate / periods_per_year

        strategy_return = returns.mean()
        benchmark_return = benchmark_returns.mean()

        alpha_val = strategy_return - (rf_per_period + beta_coef * (benchmark_return - rf_per_period))

        return alpha_val * periods_per_year

    @staticmethod
    def omega_ratio(
        returns: pd.Series,
        threshold: float = 0.0
    ) -> float:
        """
        Calculate Omega ratio.

        Args:
            returns: Series of returns
            threshold: Threshold return

        Returns:
            Omega ratio
        """
        excess = returns - threshold
        gains = excess[excess > 0].sum()
        losses = abs(excess[excess < 0].sum())

        if losses == 0:
            return np.inf if gains > 0 else 0.0

        return gains / losses

    @staticmethod
    def tail_ratio(returns: pd.Series) -> float:
        """
        Calculate tail ratio (95th percentile / 5th percentile).

        Args:
            returns: Series of returns

        Returns:
            Tail ratio
        """
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)

        if p5 == 0:
            return 0.0

        return abs(p95 / p5)

    @staticmethod
    def calculate_all_metrics(
        returns: pd.Series,
        benchmark_returns: pd.Series = None,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> dict:
        """
        Calculate all performance metrics.

        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns (optional)
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year

        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'total_return': PerformanceMetrics.total_return(returns),
            'annualized_return': PerformanceMetrics.annualized_return(returns, periods_per_year),
            'annualized_volatility': PerformanceMetrics.annualized_volatility(returns, periods_per_year),
            'sharpe_ratio': PerformanceMetrics.sharpe_ratio(returns, risk_free_rate, periods_per_year),
            'sortino_ratio': PerformanceMetrics.sortino_ratio(returns, risk_free_rate, periods_per_year),
            'calmar_ratio': PerformanceMetrics.calmar_ratio(returns, periods_per_year),
            'maximum_drawdown': PerformanceMetrics.maximum_drawdown(returns),
            'win_rate': PerformanceMetrics.win_rate(returns),
            'profit_factor': PerformanceMetrics.profit_factor(returns),
            'payoff_ratio': PerformanceMetrics.payoff_ratio(returns),
            'var_95': PerformanceMetrics.value_at_risk(returns, 0.95),
            'cvar_95': PerformanceMetrics.conditional_value_at_risk(returns, 0.95),
            'omega_ratio': PerformanceMetrics.omega_ratio(returns),
            'tail_ratio': PerformanceMetrics.tail_ratio(returns)
        }

        # Add drawdown statistics
        dd_stats = PerformanceMetrics.drawdown_duration(returns)
        metrics.update({
            f'drawdown_{k}': v for k, v in dd_stats.items()
        })

        # Add benchmark-relative metrics if provided
        if benchmark_returns is not None:
            metrics.update({
                'beta': PerformanceMetrics.beta(returns, benchmark_returns),
                'alpha': PerformanceMetrics.alpha(returns, benchmark_returns, risk_free_rate, periods_per_year),
                'information_ratio': PerformanceMetrics.information_ratio(returns, benchmark_returns, periods_per_year)
            })

        return metrics


class ForecastMetrics:
    """Metrics for evaluating forecast accuracy."""

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error."""
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return np.sqrt(ForecastMetrics.mse(y_true, y_pred))

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error."""
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared score."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_tot == 0:
            return 0.0

        return 1 - (ss_res / ss_tot)

    @staticmethod
    def directional_accuracy(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate directional accuracy (% of correct direction predictions).

        Args:
            y_true: Actual values
            y_pred: Predicted values

        Returns:
            Directional accuracy (0 to 1)
        """
        if len(y_true) < 2:
            return 0.0

        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))

        return np.mean(true_direction == pred_direction)

    @staticmethod
    def hit_ratio(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.0
    ) -> float:
        """
        Calculate hit ratio (% of predictions within threshold).

        Args:
            y_true: Actual values
            y_pred: Predicted values
            threshold: Error threshold

        Returns:
            Hit ratio (0 to 1)
        """
        errors = np.abs(y_true - y_pred)
        return np.mean(errors <= threshold)

    @staticmethod
    def calculate_all_forecast_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> dict:
        """
        Calculate all forecast metrics.

        Args:
            y_true: Actual values
            y_pred: Predicted values

        Returns:
            Dictionary with all metrics
        """
        return {
            'mse': ForecastMetrics.mse(y_true, y_pred),
            'rmse': ForecastMetrics.rmse(y_true, y_pred),
            'mae': ForecastMetrics.mae(y_true, y_pred),
            'mape': ForecastMetrics.mape(y_true, y_pred),
            'r2_score': ForecastMetrics.r2_score(y_true, y_pred),
            'directional_accuracy': ForecastMetrics.directional_accuracy(y_true, y_pred)
        }


class ClassificationMetrics:
    """Metrics for regime classification and signal prediction."""

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Classification accuracy."""
        return np.mean(y_true == y_pred)

    @staticmethod
    def precision(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        positive_class: int = 1
    ) -> float:
        """Precision score."""
        tp = np.sum((y_true == positive_class) & (y_pred == positive_class))
        fp = np.sum((y_true != positive_class) & (y_pred == positive_class))

        if tp + fp == 0:
            return 0.0

        return tp / (tp + fp)

    @staticmethod
    def recall(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        positive_class: int = 1
    ) -> float:
        """Recall score."""
        tp = np.sum((y_true == positive_class) & (y_pred == positive_class))
        fn = np.sum((y_true == positive_class) & (y_pred != positive_class))

        if tp + fn == 0:
            return 0.0

        return tp / (tp + fn)

    @staticmethod
    def f1_score(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        positive_class: int = 1
    ) -> float:
        """F1 score."""
        prec = ClassificationMetrics.precision(y_true, y_pred, positive_class)
        rec = ClassificationMetrics.recall(y_true, y_pred, positive_class)

        if prec + rec == 0:
            return 0.0

        return 2 * (prec * rec) / (prec + rec)

    @staticmethod
    def confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """Calculate confusion matrix."""
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(classes)

        cm = np.zeros((n_classes, n_classes), dtype=int)

        for i, true_class in enumerate(classes):
            for j, pred_class in enumerate(classes):
                cm[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))

        return cm
