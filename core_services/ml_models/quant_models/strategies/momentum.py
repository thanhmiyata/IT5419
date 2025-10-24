"""
Momentum Trading Strategy
==========================

Trend-following strategies based on price momentum.
"""

from typing import Optional

import numpy as np
import pandas as pd


class MomentumStrategy:
    """Momentum-based trading strategy."""

    def __init__(
        self,
        lookback_period: int = 20,
        holding_period: int = 5,
        top_n: Optional[int] = None
    ):
        """
        Initialize momentum strategy.

        Args:
            lookback_period: Period to calculate momentum
            holding_period: How long to hold positions
            top_n: Number of top momentum stocks to hold (None = all)
        """
        self.lookback_period = lookback_period
        self.holding_period = holding_period
        self.top_n = top_n

    def calculate_momentum(
        self,
        prices: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate momentum for each asset.

        Args:
            prices: DataFrame with prices (columns = assets, index = dates)

        Returns:
            DataFrame with momentum scores
        """
        # Calculate returns over lookback period
        momentum = prices.pct_change(self.lookback_period)
        return momentum

    def generate_signals(
        self,
        prices: pd.DataFrame,
        method: str = 'simple'
    ) -> pd.DataFrame:
        """
        Generate trading signals based on momentum.

        Args:
            prices: Price data
            method: 'simple', 'ma_crossover', 'rsi'

        Returns:
            DataFrame with signals (1=long, -1=short, 0=neutral)
        """
        if method == 'simple':
            return self._simple_momentum_signals(prices)
        elif method == 'ma_crossover':
            return self._ma_crossover_signals(prices)
        elif method == 'rsi':
            return self._rsi_signals(prices)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _simple_momentum_signals(
        self,
        prices: pd.DataFrame
    ) -> pd.DataFrame:
        """Simple momentum: buy if positive, sell if negative."""
        momentum = self.calculate_momentum(prices)

        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)

        # Long if momentum is positive
        signals[momentum > 0] = 1

        # Short if momentum is negative
        signals[momentum < 0] = -1

        # Select top N if specified
        if self.top_n is not None:
            signals = self._select_top_n(momentum, signals)

        return signals

    def _ma_crossover_signals(
        self,
        prices: pd.DataFrame,
        fast_period: int = 10,
        slow_period: int = 50
    ) -> pd.DataFrame:
        """Moving average crossover strategy."""
        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)

        for col in prices.columns:
            fast_ma = prices[col].rolling(window=fast_period).mean()
            slow_ma = prices[col].rolling(window=slow_period).mean()

            # Long when fast > slow
            signals.loc[fast_ma > slow_ma, col] = 1

            # Short when fast < slow
            signals.loc[fast_ma < slow_ma, col] = -1

        return signals

    def _rsi_signals(
        self,
        prices: pd.DataFrame,
        period: int = 14,
        overbought: float = 70,
        oversold: float = 30
    ) -> pd.DataFrame:
        """RSI-based momentum strategy."""
        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)

        for col in prices.columns:
            # Calculate RSI
            delta = prices[col].diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # Long when oversold
            signals.loc[rsi < oversold, col] = 1

            # Short when overbought
            signals.loc[rsi > overbought, col] = -1

        return signals

    def _select_top_n(
        self,
        momentum: pd.DataFrame,
        signals: pd.DataFrame
    ) -> pd.DataFrame:
        """Select top N momentum stocks."""
        new_signals = pd.DataFrame(0, index=signals.index, columns=signals.columns)

        for date in momentum.index:
            # Get momentum values for this date
            mom_values = momentum.loc[date].dropna()

            if len(mom_values) == 0:
                continue

            # Sort by momentum
            sorted_stocks = mom_values.sort_values(ascending=False)

            # Select top N
            top_stocks = sorted_stocks.head(self.top_n).index

            # Set signals for top stocks
            new_signals.loc[date, top_stocks] = signals.loc[date, top_stocks]

        return new_signals

    def backtest(
        self,
        prices: pd.DataFrame,
        signals: pd.DataFrame,
        initial_capital: float = 100000
    ) -> pd.DataFrame:
        """
        Backtest the strategy.

        Args:
            prices: Price data
            signals: Trading signals
            initial_capital: Starting capital

        Returns:
            DataFrame with portfolio value over time
        """
        # Calculate returns
        returns = prices.pct_change()

        # Calculate strategy returns (signals shifted by 1 day)
        strategy_returns = (signals.shift(1) * returns).sum(axis=1)

        # Calculate cumulative returns
        cumulative_returns = (1 + strategy_returns).cumprod()

        # Portfolio value
        portfolio_value = initial_capital * cumulative_returns

        # Create result DataFrame
        results = pd.DataFrame({
            'portfolio_value': portfolio_value,
            'returns': strategy_returns,
            'cumulative_returns': cumulative_returns
        })

        return results

    def calculate_performance_metrics(
        self,
        backtest_results: pd.DataFrame,
        risk_free_rate: float = 0.02
    ) -> dict:
        """
        Calculate performance metrics.

        Args:
            backtest_results: Results from backtest()
            risk_free_rate: Annual risk-free rate

        Returns:
            Dictionary of performance metrics
        """
        returns = backtest_results['returns'].dropna()

        # Annualized return
        total_return = backtest_results['cumulative_returns'].iloc[-1] - 1
        n_years = len(returns) / 252  # Assuming daily data
        annual_return = (1 + total_return) ** (1 / n_years) - 1

        # Annualized volatility
        annual_vol = returns.std() * np.sqrt(252)

        # Sharpe ratio
        sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0

        # Maximum drawdown
        cumulative = backtest_results['cumulative_returns']
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        win_rate = (returns > 0).sum() / len(returns)

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(returns)
        }
