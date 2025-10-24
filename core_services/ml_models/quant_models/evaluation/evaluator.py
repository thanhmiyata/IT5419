"""
Model Evaluation Framework
===========================

Comprehensive evaluation of trading models and strategies.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core_services.ml_models.quant_models.evaluation.metrics import (ClassificationMetrics, ForecastMetrics,
                                                                     PerformanceMetrics)


class ModelEvaluator:
    """Evaluate trading models and strategies."""

    def __init__(self):
        """Initialize evaluator."""
        self.results = {}

    def evaluate_forecast_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = 'model'
    ) -> Dict[str, float]:
        """
        Evaluate forecast model.

        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of model

        Returns:
            Dictionary with forecast metrics
        """
        metrics = ForecastMetrics.calculate_all_forecast_metrics(y_true, y_pred)

        self.results[model_name] = {
            'type': 'forecast',
            'metrics': metrics
        }

        return metrics

    def evaluate_classification_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = 'model'
    ) -> Dict[str, float]:
        """
        Evaluate classification model.

        Args:
            y_true: Actual labels
            y_pred: Predicted labels
            model_name: Name of model

        Returns:
            Dictionary with classification metrics
        """
        metrics = {
            'accuracy': ClassificationMetrics.accuracy(y_true, y_pred),
            'precision': ClassificationMetrics.precision(y_true, y_pred),
            'recall': ClassificationMetrics.recall(y_true, y_pred),
            'f1_score': ClassificationMetrics.f1_score(y_true, y_pred)
        }

        self.results[model_name] = {
            'type': 'classification',
            'metrics': metrics,
            'confusion_matrix': ClassificationMetrics.confusion_matrix(y_true, y_pred)
        }

        return metrics

    def evaluate_trading_strategy(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
        strategy_name: str = 'strategy'
    ) -> Dict[str, float]:
        """
        Evaluate trading strategy.

        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns (optional)
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
            strategy_name: Name of strategy

        Returns:
            Dictionary with performance metrics
        """
        metrics = PerformanceMetrics.calculate_all_metrics(
            returns,
            benchmark_returns,
            risk_free_rate,
            periods_per_year
        )

        self.results[strategy_name] = {
            'type': 'trading_strategy',
            'metrics': metrics
        }

        return metrics

    def compare_models(
        self,
        model_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple models.

        Args:
            model_names: List of model names to compare (None = all)

        Returns:
            DataFrame comparing models
        """
        if model_names is None:
            model_names = list(self.results.keys())

        comparison = []

        for name in model_names:
            if name not in self.results:
                continue

            result = self.results[name]
            row = {'model': name, 'type': result['type']}
            row.update(result['metrics'])
            comparison.append(row)

        return pd.DataFrame(comparison)

    def get_best_model(
        self,
        metric: str,
        mode: str = 'max'
    ) -> str:
        """
        Get best model based on metric.

        Args:
            metric: Metric to compare
            mode: 'max' or 'min'

        Returns:
            Name of best model
        """
        comparison = self.compare_models()

        if metric not in comparison.columns:
            raise ValueError(f"Metric {metric} not found in results")

        if mode == 'max':
            best_idx = comparison[metric].idxmax()
        else:
            best_idx = comparison[metric].idxmin()

        return comparison.loc[best_idx, 'model']

    def summary_report(self) -> str:
        """
        Generate summary report.

        Returns:
            String with formatted report
        """
        report = ["=" * 60]
        report.append("MODEL EVALUATION SUMMARY")
        report.append("=" * 60)
        report.append("")

        for name, result in self.results.items():
            report.append(f"\n{name} ({result['type']})")
            report.append("-" * 60)

            for metric, value in result['metrics'].items():
                if isinstance(value, (int, float)):
                    report.append(f"  {metric:30s}: {value:>12.6f}")
                else:
                    report.append(f"  {metric:30s}: {value}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)


class BacktestEvaluator:
    """Evaluate backtest results."""

    @staticmethod
    def evaluate_backtest(
        backtest_results: pd.DataFrame,
        initial_capital: float = 100000,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> Dict[str, Any]:
        """
        Evaluate backtest results.

        Args:
            backtest_results: DataFrame with backtest results
            initial_capital: Starting capital
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year

        Returns:
            Dictionary with comprehensive backtest metrics
        """
        # Extract returns
        if 'returns' in backtest_results.columns:
            returns = backtest_results['returns']
        elif 'strategy_returns' in backtest_results.columns:
            returns = backtest_results['strategy_returns']
        else:
            raise ValueError("No returns column found in backtest results")

        # Performance metrics
        perf_metrics = PerformanceMetrics.calculate_all_metrics(
            returns,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year
        )

        # Additional backtest-specific metrics
        portfolio_value = backtest_results.get('portfolio_value', initial_capital * (1 + returns).cumprod())

        final_value = portfolio_value.iloc[-1]

        # Trade statistics
        if 'signal' in backtest_results.columns:
            signals = backtest_results['signal']
            trades = signals.diff().abs()
            n_trades = int(trades.sum() / 2)  # Each trade has entry and exit

            # Long/short breakdown
            long_periods = (signals == 1).sum()
            short_periods = (signals == -1).sum()
            neutral_periods = (signals == 0).sum()
        else:
            n_trades = None
            long_periods = None
            short_periods = None
            neutral_periods = None

        # Time in market
        total_periods = len(returns)

        results = {
            **perf_metrics,
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_pnl': final_value - initial_capital,
            'n_trades': n_trades,
            'long_periods': long_periods,
            'short_periods': short_periods,
            'neutral_periods': neutral_periods,
            'total_periods': total_periods,
            'time_in_market': (long_periods + short_periods) / total_periods if long_periods else None
        }

        return results

    @staticmethod
    def plot_backtest_results(
        backtest_results: pd.DataFrame,
        show_signals: bool = True
    ):
        """
        Plot backtest results.

        Args:
            backtest_results: DataFrame with backtest results
            show_signals: Whether to show trading signals
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # Portfolio value
        if 'portfolio_value' in backtest_results.columns:
            axes[0].plot(backtest_results.index, backtest_results['portfolio_value'], label='Portfolio Value')
            axes[0].set_ylabel('Portfolio Value')
            axes[0].set_title('Portfolio Value Over Time')
            axes[0].legend()
            axes[0].grid(True)

        # Price with signals
        if 'price' in backtest_results.columns:
            axes[1].plot(backtest_results.index, backtest_results['price'], label='Price', alpha=0.7)

            if show_signals and 'signal' in backtest_results.columns:
                # Buy signals
                buy_signals = backtest_results[backtest_results['signal'] == 1]
                axes[1].scatter(buy_signals.index, buy_signals['price'], color='green',
                                marker='^', s=100, label='Buy', zorder=5)

                # Sell signals
                sell_signals = backtest_results[backtest_results['signal'] == -1]
                axes[1].scatter(sell_signals.index, sell_signals['price'], color='red',
                                marker='v', s=100, label='Sell', zorder=5)

            axes[1].set_ylabel('Price')
            axes[1].set_title('Price and Trading Signals')
            axes[1].legend()
            axes[1].grid(True)

        # Drawdown
        if 'portfolio_value' in backtest_results.columns:
            portfolio_value = backtest_results['portfolio_value']
            running_max = portfolio_value.cummax()
            drawdown = (portfolio_value - running_max) / running_max

            axes[2].fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
            axes[2].set_ylabel('Drawdown')
            axes[2].set_xlabel('Date')
            axes[2].set_title('Drawdown')
            axes[2].grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def monthly_returns_table(
        returns: pd.Series
    ) -> pd.DataFrame:
        """
        Create monthly returns table.

        Args:
            returns: Series of returns with datetime index

        Returns:
            DataFrame with monthly returns
        """
        # Resample to monthly
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

        # Create pivot table
        monthly_returns_df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values
        })

        pivot = monthly_returns_df.pivot(index='year', columns='month', values='return')

        # Rename columns to month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot.columns = [month_names[m - 1] for m in pivot.columns]

        # Add year total
        pivot['Year'] = pivot.sum(axis=1)

        return pivot * 100  # Convert to percentage


class StrategyComparator:
    """Compare multiple trading strategies."""

    def __init__(self):
        """Initialize comparator."""
        self.strategies = {}

    def add_strategy(
        self,
        name: str,
        returns: pd.Series,
        metadata: Optional[Dict] = None
    ):
        """
        Add strategy for comparison.

        Args:
            name: Strategy name
            returns: Strategy returns
            metadata: Optional metadata
        """
        self.strategies[name] = {
            'returns': returns,
            'metadata': metadata or {}
        }

    def compare(
        self,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> pd.DataFrame:
        """
        Compare all strategies.

        Args:
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year

        Returns:
            DataFrame comparing strategies
        """
        comparison = []

        for name, data in self.strategies.items():
            returns = data['returns']

            metrics = PerformanceMetrics.calculate_all_metrics(
                returns,
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year
            )

            row = {'strategy': name, **metrics, **data['metadata']}
            comparison.append(row)

        return pd.DataFrame(comparison)

    def plot_cumulative_returns(self):
        """Plot cumulative returns for all strategies."""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))

        for name, data in self.strategies.items():
            returns = data['returns']
            cumulative = (1 + returns).cumprod()
            plt.plot(cumulative.index, cumulative, label=name)

        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.title('Cumulative Returns Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_drawdowns(self):
        """Plot drawdowns for all strategies."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))

        for name, data in self.strategies.items():
            returns = data['returns']
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max

            ax.plot(drawdown.index, drawdown, label=name, alpha=0.7)

        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown')
        ax.set_title('Drawdown Comparison')
        ax.legend()
        ax.grid(True)
        plt.show()

    def rolling_sharpe(
        self,
        window: int = 252,
        risk_free_rate: float = 0.02
    ):
        """
        Plot rolling Sharpe ratios.

        Args:
            window: Rolling window size
            risk_free_rate: Annual risk-free rate
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))

        for name, data in self.strategies.items():
            returns = data['returns']

            rolling_sharpe = (
                (returns.rolling(window).mean() - risk_free_rate / 252)
                / returns.rolling(window).std()
                * np.sqrt(252)
            )

            plt.plot(rolling_sharpe.index, rolling_sharpe, label=name, alpha=0.7)

        plt.xlabel('Date')
        plt.ylabel('Sharpe Ratio')
        plt.title(f'Rolling Sharpe Ratio ({window}-day window)')
        plt.legend()
        plt.grid(True)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        plt.show()
