"""
Momentum Factor
===============

Momentum-based factor for asset selection.
"""

import pandas as pd


class MomentumFactor:
    """Momentum factor calculations."""

    def __init__(self, lookback_periods: list = [1, 3, 6, 12]):
        """
        Initialize momentum factor.

        Args:
            lookback_periods: Months to calculate momentum
        """
        self.lookback_periods = lookback_periods

    def calculate_momentum(
        self,
        prices: pd.DataFrame,
        period: int = 12,
        skip: int = 1
    ) -> pd.DataFrame:
        """
        Calculate momentum (skip last month to avoid reversal).

        Args:
            prices: Price data
            period: Lookback period in months
            skip: Number of recent months to skip

        Returns:
            DataFrame with momentum scores
        """
        # Calculate returns from period ago to skip months ago
        momentum = prices.pct_change(period * 21).shift(skip * 21)

        return momentum

    def calculate_dual_momentum(
        self,
        asset_prices: pd.Series,
        benchmark_prices: pd.Series,
        period: int = 12
    ) -> pd.DataFrame:
        """
        Calculate dual momentum (absolute + relative).

        Args:
            asset_prices: Asset price series
            benchmark_prices: Benchmark price series
            period: Lookback period

        Returns:
            DataFrame with momentum signals
        """
        # Absolute momentum (trend)
        abs_momentum = asset_prices.pct_change(period * 21)

        # Relative momentum (vs benchmark)
        rel_momentum = abs_momentum - benchmark_prices.pct_change(period * 21)

        # Dual momentum signal
        signal = ((abs_momentum > 0) & (rel_momentum > 0)).astype(int)

        return pd.DataFrame({
            'absolute_momentum': abs_momentum,
            'relative_momentum': rel_momentum,
            'signal': signal
        })

    def rank_by_momentum(
        self,
        prices: pd.DataFrame,
        period: int = 12
    ) -> pd.DataFrame:
        """
        Rank assets by momentum.

        Args:
            prices: Price data for multiple assets
            period: Lookback period

        Returns:
            DataFrame with momentum ranks
        """
        momentum = self.calculate_momentum(prices, period)

        # Rank (1 = highest momentum)
        ranks = momentum.rank(axis=1, ascending=False)

        return ranks

    def create_long_short_portfolio(
        self,
        prices: pd.DataFrame,
        period: int = 12,
        top_pct: float = 0.3,
        bottom_pct: float = 0.3
    ) -> pd.DataFrame:
        """
        Create long-short momentum portfolio.

        Args:
            prices: Price data
            period: Lookback period
            top_pct: Percentage of top stocks to long
            bottom_pct: Percentage of bottom stocks to short

        Returns:
            DataFrame with portfolio weights
        """
        momentum = self.calculate_momentum(prices, period)
        weights = pd.DataFrame(0.0, index=momentum.index, columns=momentum.columns)

        for date in momentum.index:
            mom_values = momentum.loc[date].dropna()

            if len(mom_values) == 0:
                continue

            # Top and bottom
            n_long = int(len(mom_values) * top_pct)
            n_short = int(len(mom_values) * bottom_pct)

            sorted_stocks = mom_values.sort_values(ascending=False)

            # Long top stocks
            long_stocks = sorted_stocks.head(n_long)
            weights.loc[date, long_stocks.index] = 1.0 / n_long

            # Short bottom stocks
            short_stocks = sorted_stocks.tail(n_short)
            weights.loc[date, short_stocks.index] = -1.0 / n_short

        return weights
