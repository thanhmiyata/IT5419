"""
Multi-Factor Alpha Model
=========================

Combines multiple factors for stock selection.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class MultiFactorModel:
    """Multi-factor alpha model."""

    def __init__(self):
        """Initialize multi-factor model."""
        self.scaler = StandardScaler()
        self.model = None
        self.factor_weights = None

    def calculate_value_factor(
        self,
        prices: pd.DataFrame,
        book_values: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate value factor (book-to-market)."""
        return book_values / prices

    def calculate_quality_factor(
        self,
        roe: pd.DataFrame,
        debt_to_equity: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate quality factor."""
        # High ROE, low D/E = high quality
        quality = roe / (1 + debt_to_equity)
        return quality

    def calculate_size_factor(
        self,
        market_caps: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate size factor (log market cap)."""
        return np.log(market_caps)

    def calculate_volatility_factor(
        self,
        returns: pd.DataFrame,
        window: int = 60
    ) -> pd.DataFrame:
        """Calculate low volatility factor."""
        # Lower volatility = higher score
        volatility = returns.rolling(window=window).std()
        return 1 / volatility

    def combine_factors(
        self,
        factors_dict: dict,
        weights: dict = None
    ) -> pd.DataFrame:
        """
        Combine multiple factors into composite score.

        Args:
            factors_dict: Dictionary of factor DataFrames
            weights: Dictionary of factor weights (equal if None)

        Returns:
            Combined factor scores
        """
        if weights is None:
            # Equal weights
            weights = {name: 1.0 / len(factors_dict) for name in factors_dict.keys()}

        # Standardize each factor
        standardized_factors = {}

        for name, factor_df in factors_dict.items():
            # Z-score normalization
            mean = factor_df.mean(axis=1)
            std = factor_df.std(axis=1)

            standardized = factor_df.sub(mean, axis=0).div(std, axis=0)
            standardized_factors[name] = standardized

        # Combine with weights
        combined_score = pd.DataFrame(0.0, index=factor_df.index, columns=factor_df.columns)

        for name, factor_df in standardized_factors.items():
            combined_score += weights[name] * factor_df

        return combined_score

    def rank_stocks(
        self,
        factor_scores: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Rank stocks by factor scores.

        Args:
            factor_scores: Combined factor scores

        Returns:
            DataFrame with ranks (1 = best)
        """
        return factor_scores.rank(axis=1, ascending=False)

    def create_long_short_portfolio(
        self,
        factor_scores: pd.DataFrame,
        long_pct: float = 0.2,
        short_pct: float = 0.2
    ) -> pd.DataFrame:
        """
        Create long-short portfolio from factor scores.

        Args:
            factor_scores: Combined factor scores
            long_pct: Percentage to long
            short_pct: Percentage to short

        Returns:
            Portfolio weights
        """
        weights = pd.DataFrame(0.0, index=factor_scores.index, columns=factor_scores.columns)

        for date in factor_scores.index:
            scores = factor_scores.loc[date].dropna()

            if len(scores) == 0:
                continue

            n_long = int(len(scores) * long_pct)
            n_short = int(len(scores) * short_pct)

            # Sort by scores
            sorted_stocks = scores.sort_values(ascending=False)

            # Long top stocks
            long_stocks = sorted_stocks.head(n_long)
            if n_long > 0:
                weights.loc[date, long_stocks.index] = 1.0 / n_long

            # Short bottom stocks
            short_stocks = sorted_stocks.tail(n_short)
            if n_short > 0:
                weights.loc[date, short_stocks.index] = -1.0 / n_short

        return weights

    def backtest_factor(
        self,
        returns: pd.DataFrame,
        factor_scores: pd.DataFrame,
        long_pct: float = 0.2,
        short_pct: float = 0.2
    ) -> pd.DataFrame:
        """
        Backtest factor-based strategy.

        Args:
            returns: Stock returns
            factor_scores: Factor scores
            long_pct: Percentage to long
            short_pct: Percentage to short

        Returns:
            DataFrame with backtest results
        """
        # Create portfolio weights
        weights = self.create_long_short_portfolio(
            factor_scores,
            long_pct,
            short_pct
        )

        # Calculate portfolio returns
        portfolio_returns = (weights.shift(1) * returns).sum(axis=1)

        # Cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()

        return pd.DataFrame({
            'returns': portfolio_returns,
            'cumulative_returns': cumulative_returns
        })
