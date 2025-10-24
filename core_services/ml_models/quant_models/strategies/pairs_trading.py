"""
Pairs Trading Strategy
=======================

Statistical arbitrage strategy using cointegration and Kalman filter.
"""

import pandas as pd
from statsmodels.tsa.stattools import coint


class PairsTradingStrategy:
    """Pairs trading using cointegration."""

    def __init__(
        self,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        lookback_window: int = 20
    ):
        """
        Initialize pairs trading strategy.

        Args:
            entry_threshold: Z-score for entry
            exit_threshold: Z-score for exit
            lookback_window: Window for z-score calculation
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.lookback_window = lookback_window

    def test_cointegration(
        self,
        stock1: pd.Series,
        stock2: pd.Series
    ) -> dict:
        """
        Test if two stocks are cointegrated.

        Args:
            stock1: First stock price series
            stock2: Second stock price series

        Returns:
            Dictionary with test results
        """
        score, pvalue, _ = coint(stock1, stock2)

        return {
            'cointegration_score': score,
            'p_value': pvalue,
            'is_cointegrated': pvalue < 0.05
        }

    def calculate_hedge_ratio(
        self,
        stock1: pd.Series,
        stock2: pd.Series,
        method: str = 'ols'
    ) -> float:
        """
        Calculate hedge ratio between two stocks.

        Args:
            stock1: First stock (dependent)
            stock2: Second stock (independent)
            method: 'ols' or 'total_least_squares'

        Returns:
            Hedge ratio
        """
        if method == 'ols':
            # Ordinary least squares
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(stock2.values.reshape(-1, 1), stock1.values)
            return model.coef_[0]
        else:
            # Simple ratio
            return (stock1 / stock2).mean()

    def calculate_spread(
        self,
        stock1: pd.Series,
        stock2: pd.Series,
        hedge_ratio: float = None
    ) -> pd.Series:
        """
        Calculate spread between two stocks.

        Args:
            stock1: First stock prices
            stock2: Second stock prices
            hedge_ratio: Hedge ratio (calculated if None)

        Returns:
            Spread series
        """
        if hedge_ratio is None:
            hedge_ratio = self.calculate_hedge_ratio(stock1, stock2)

        spread = stock1 - hedge_ratio * stock2
        return spread

    def calculate_zscore(
        self,
        spread: pd.Series,
        window: int = None
    ) -> pd.Series:
        """
        Calculate rolling z-score of spread.

        Args:
            spread: Spread series
            window: Rolling window (uses lookback_window if None)

        Returns:
            Z-score series
        """
        if window is None:
            window = self.lookback_window

        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()

        zscore = (spread - rolling_mean) / rolling_std
        return zscore

    def generate_signals(
        self,
        stock1: pd.Series,
        stock2: pd.Series
    ) -> pd.DataFrame:
        """
        Generate pairs trading signals.

        Args:
            stock1: First stock prices
            stock2: Second stock prices

        Returns:
            DataFrame with spread, z-score, and signals
        """
        # Calculate hedge ratio
        hedge_ratio = self.calculate_hedge_ratio(stock1, stock2)

        # Calculate spread
        spread = self.calculate_spread(stock1, stock2, hedge_ratio)

        # Calculate z-score
        zscore = self.calculate_zscore(spread)

        # Generate signals
        signals = pd.Series(0, index=stock1.index)

        # Long spread when z-score < -entry_threshold
        # (buy stock1, sell stock2)
        signals[zscore < -self.entry_threshold] = 1

        # Short spread when z-score > entry_threshold
        # (sell stock1, buy stock2)
        signals[zscore > self.entry_threshold] = -1

        # Exit when abs(z-score) < exit_threshold
        signals[abs(zscore) < self.exit_threshold] = 0

        return pd.DataFrame({
            'stock1': stock1,
            'stock2': stock2,
            'hedge_ratio': hedge_ratio,
            'spread': spread,
            'zscore': zscore,
            'signal': signals
        })

    def backtest(
        self,
        stock1: pd.Series,
        stock2: pd.Series,
        initial_capital: float = 100000
    ) -> pd.DataFrame:
        """
        Backtest pairs trading strategy.

        Args:
            stock1: First stock prices
            stock2: Second stock prices
            initial_capital: Starting capital

        Returns:
            DataFrame with backtest results
        """
        # Generate signals
        signals_df = self.generate_signals(stock1, stock2)
        signals = signals_df['signal']
        hedge_ratio = signals_df['hedge_ratio'].iloc[0]

        # Calculate returns for each stock
        returns1 = stock1.pct_change()
        returns2 = stock2.pct_change()

        # Calculate strategy returns
        # Long spread: long stock1, short stock2
        # Short spread: short stock1, long stock2
        strategy_returns = signals.shift(1) * (
            returns1 - hedge_ratio * returns2
        )

        # Cumulative returns
        cumulative_returns = (1 + strategy_returns).cumprod()

        # Portfolio value
        portfolio_value = initial_capital * cumulative_returns

        return pd.DataFrame({
            'stock1': stock1,
            'stock2': stock2,
            'spread': signals_df['spread'],
            'zscore': signals_df['zscore'],
            'signal': signals,
            'returns': strategy_returns,
            'cumulative_returns': cumulative_returns,
            'portfolio_value': portfolio_value
        })

    def find_cointegrated_pairs(
        self,
        prices: pd.DataFrame,
        pvalue_threshold: float = 0.05
    ) -> list:
        """
        Find cointegrated pairs from a set of stocks.

        Args:
            prices: DataFrame with stock prices
            pvalue_threshold: P-value threshold for cointegration

        Returns:
            List of tuples (stock1, stock2, p_value)
        """
        n_stocks = len(prices.columns)
        pairs = []

        for i in range(n_stocks):
            for j in range(i + 1, n_stocks):
                stock1 = prices.columns[i]
                stock2 = prices.columns[j]

                result = self.test_cointegration(
                    prices[stock1],
                    prices[stock2]
                )

                if result['p_value'] < pvalue_threshold:
                    pairs.append((
                        stock1,
                        stock2,
                        result['p_value']
                    ))

        # Sort by p-value (best pairs first)
        pairs.sort(key=lambda x: x[2])

        return pairs
