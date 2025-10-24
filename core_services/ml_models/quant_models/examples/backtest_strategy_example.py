"""
Example: Backtesting Trading Strategy
======================================

This example demonstrates how to backtest and evaluate trading strategies.
"""

import numpy as np
import pandas as pd

from core_services.ml_models.quant_models.evaluation.evaluator import BacktestEvaluator, StrategyComparator
from core_services.ml_models.quant_models.evaluation.metrics import PerformanceMetrics
from core_services.ml_models.quant_models.strategies.mean_reversion import MeanReversionStrategy
from core_services.ml_models.quant_models.strategies.momentum import MomentumStrategy
from core_services.utils.logger_utils import logger


def generate_sample_price_data(n_samples=1000, n_stocks=5):
    """Generate sample price data for multiple stocks."""
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')

    data = {}

    for i in range(n_stocks):
        # Different trend for each stock
        trend = np.linspace(100 + i * 10, 150 + i * 10, n_samples)
        seasonality = 10 * np.sin(np.arange(n_samples) * 2 * np.pi / 365)
        noise = np.random.normal(0, 3, n_samples)

        prices = trend + seasonality + noise
        data[f'Stock_{i + 1}'] = prices

    df = pd.DataFrame(data, index=dates)

    return df


def main():
    """Main backtesting function."""
    logger.info("=" * 80)
    logger.info("Trading Strategy Backtesting Example")
    logger.info("=" * 80)

    # 1. Generate data
    logger.info("\n1. Generating sample data...")
    prices = generate_sample_price_data(n_samples=1000, n_stocks=5)
    logger.info(f"   Price data shape: {prices.shape}")
    logger.info(f"   Stocks: {list(prices.columns)}")
    logger.info(f"   Date range: {prices.index[0]} to {prices.index[-1]}")

    # 2. Initialize strategies
    logger.info("\n2. Initializing strategies...")

    # Momentum strategy
    momentum_strategy = MomentumStrategy(
        lookback_period=20,
        holding_period=5,
        top_n=3
    )

    # Mean reversion strategy
    mean_reversion_strategy = MeanReversionStrategy(
        lookback_period=20,
        num_std=2.0
    )

    logger.info("   - Momentum Strategy (20-day lookback, top 3 stocks)")
    logger.info("   - Mean Reversion Strategy (Bollinger Bands)")

    # 3. Generate signals
    logger.info("\n3. Generating trading signals...")

    momentum_signals = momentum_strategy.generate_signals(prices, method='simple')
    mean_reversion_signals = mean_reversion_strategy.generate_signals(
        prices,
        method='bollinger'
    )

    logger.info(f"   Momentum signals generated: {momentum_signals.shape}")
    logger.info(f"   Mean reversion signals generated: {mean_reversion_signals.shape}")

    # 4. Backtest strategies
    logger.info("\n4. Backtesting strategies...")
    logger.info("-" * 80)

    initial_capital = 100000

    # Momentum backtest
    momentum_results = momentum_strategy.backtest(
        prices,
        momentum_signals,
        initial_capital=initial_capital
    )

    # Mean reversion backtest
    mean_reversion_results = mean_reversion_strategy.backtest(
        prices,
        mean_reversion_signals,
        initial_capital=initial_capital
    )

    # Buy and hold benchmark
    buy_hold_returns = prices.mean(axis=1).pct_change().fillna(0)
    buy_hold_cumulative = (1 + buy_hold_returns).cumprod()
    buy_hold_value = initial_capital * buy_hold_cumulative

    logger.info("   Backtesting completed!")
    logger.info("-" * 80)

    # 5. Evaluate performance
    logger.info("\n5. Evaluating strategies...")
    logger.info("-" * 80)

    evaluator = BacktestEvaluator()

    # Momentum metrics
    logger.info("\n   MOMENTUM STRATEGY")
    logger.info("   " + "-" * 76)
    momentum_metrics = evaluator.evaluate_backtest(
        pd.DataFrame({
            'returns': momentum_results['returns'],
            'portfolio_value': momentum_results['portfolio_value']
        }),
        initial_capital=initial_capital
    )

    for metric, value in list(momentum_metrics.items())[:10]:
        if isinstance(value, (int, float)) and value is not None:
            logger.info(f"   {metric:30s}: {value:>15,.2f}" if abs(value) > 1
                        else f"   {metric:30s}: {value:>15.6f}")

    # Mean reversion metrics
    logger.info("\n   MEAN REVERSION STRATEGY")
    logger.info("   " + "-" * 76)
    mr_metrics = evaluator.evaluate_backtest(
        pd.DataFrame({
            'returns': mean_reversion_results['returns'],
            'portfolio_value': mean_reversion_results['portfolio_value']
        }),
        initial_capital=initial_capital
    )

    for metric, value in list(mr_metrics.items())[:10]:
        if isinstance(value, (int, float)) and value is not None:
            logger.info(f"   {metric:30s}: {value:>15,.2f}" if abs(value) > 1
                        else f"   {metric:30s}: {value:>15.6f}")

    # Buy and hold metrics
    logger.info("\n   BUY & HOLD BENCHMARK")
    logger.info("   " + "-" * 76)
    bh_metrics = PerformanceMetrics.calculate_all_metrics(buy_hold_returns)

    for metric, value in list(bh_metrics.items())[:10]:
        if isinstance(value, (int, float)) and value is not None:
            logger.info(f"   {metric:30s}: {value:>15,.2f}" if abs(value) > 1
                        else f"   {metric:30s}: {value:>15.6f}")

    logger.info("-" * 80)

    # 6. Compare strategies
    logger.info("\n6. Strategy Comparison:")
    logger.info("-" * 80)

    comparator = StrategyComparator()
    comparator.add_strategy('Momentum', momentum_results['returns'])
    comparator.add_strategy('Mean Reversion', mean_reversion_results['returns'])
    comparator.add_strategy('Buy & Hold', buy_hold_returns)

    comparison = comparator.compare()

    logger.info("\n   Key Metrics Comparison:")
    logger.info("   " + "-" * 76)
    logger.info(f"   {'Strategy':<20} {'Return':>12} {'Sharpe':>12} {'Max DD':>12} {'Win Rate':>12}")
    logger.info("   " + "-" * 76)

    for _, row in comparison.iterrows():
        logger.info(f"   {row['strategy']:<20} "
                    f"{row['total_return']:>11.2%} "
                    f"{row['sharpe_ratio']:>12.4f} "
                    f"{row['maximum_drawdown']:>11.2%} "
                    f"{row['win_rate']:>11.2%}")

    logger.info("   " + "-" * 76)

    # 7. Summary
    logger.info("\n7. Summary:")
    logger.info("-" * 80)

    best_return = comparison.loc[comparison['total_return'].idxmax(), 'strategy']
    best_sharpe = comparison.loc[comparison['sharpe_ratio'].idxmax(), 'strategy']

    logger.info(f"   Best Total Return:  {best_return}")
    logger.info(f"   Best Sharpe Ratio:  {best_sharpe}")
    logger.info("\n   Final Values:")
    logger.info(f"   - Momentum:         ${momentum_results['portfolio_value'].iloc[-1]:>12,.2f}")
    logger.info(f"   - Mean Reversion:   ${mean_reversion_results['portfolio_value'].iloc[-1]:>12,.2f}")
    logger.info(f"   - Buy & Hold:       ${buy_hold_value.iloc[-1]:>12,.2f}")

    logger.info("\n" + "=" * 80)
    logger.info("Backtesting completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
