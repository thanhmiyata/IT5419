"""
Mean Reversion Trading Strategy
================================

Strategies based on mean reversion principle.
"""

import pandas as pd


class MeanReversionStrategy:
    """Mean reversion trading strategy."""

    def __init__(
        self,
        lookback_period: int = 20,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5
    ):
        """
        Initialize mean reversion strategy.

        Args:
            lookback_period: Period for calculating statistics
            entry_threshold: Z-score threshold for entry
            exit_threshold: Z-score threshold for exit
        """
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

    def calculate_zscore(
        self,
        prices: pd.Series,
        window: int = None
    ) -> pd.Series:
        """
        Calculate rolling z-score.

        Args:
            prices: Price series
            window: Rolling window (uses lookback_period if None)

        Returns:
            Z-score series
        """
        if window is None:
            window = self.lookback_period

        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()

        zscore = (prices - rolling_mean) / rolling_std

        return zscore

    def bollinger_bands_strategy(
        self,
        prices: pd.Series,
        window: int = 20,
        num_std: float = 2.0
    ) -> pd.DataFrame:
        """
        Bollinger Bands mean reversion strategy.

        Args:
            prices: Price series
            window: Window for moving average
            num_std: Number of standard deviations

        Returns:
            DataFrame with signals and bands
        """
        # Calculate Bollinger Bands
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()

        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)

        # Generate signals
        signals = pd.Series(0, index=prices.index)

        # Buy when price touches lower band
        signals[prices <= lower_band] = 1

        # Sell when price touches upper band
        signals[prices >= upper_band] = -1

        # Exit when price crosses middle band
        signals[(prices > sma) & (signals.shift(1) == 1)] = 0
        signals[(prices < sma) & (signals.shift(1) == -1)] = 0

        return pd.DataFrame({
            'price': prices,
            'sma': sma,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'signal': signals
        })

    def zscore_strategy(
        self,
        prices: pd.Series
    ) -> pd.DataFrame:
        """
        Z-score based mean reversion.

        Args:
            prices: Price series

        Returns:
            DataFrame with signals and z-scores
        """
        zscore = self.calculate_zscore(prices)

        signals = pd.Series(0, index=prices.index)

        # Long when z-score < -entry_threshold
        signals[zscore < -self.entry_threshold] = 1

        # Short when z-score > entry_threshold
        signals[zscore > self.entry_threshold] = -1

        # Exit when abs(z-score) < exit_threshold
        signals[abs(zscore) < self.exit_threshold] = 0

        return pd.DataFrame({
            'price': prices,
            'zscore': zscore,
            'signal': signals
        })

    def rsi_mean_reversion(
        self,
        prices: pd.Series,
        period: int = 14,
        oversold: float = 30,
        overbought: float = 70,
        neutral: float = 50
    ) -> pd.DataFrame:
        """
        RSI-based mean reversion strategy.

        Args:
            prices: Price series
            period: RSI period
            oversold: Oversold threshold
            overbought: Overbought threshold
            neutral: Neutral level for exit

        Returns:
            DataFrame with signals and RSI
        """
        # Calculate RSI
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Generate signals
        signals = pd.Series(0, index=prices.index)

        # Long when oversold
        signals[rsi < oversold] = 1

        # Short when overbought
        signals[rsi > overbought] = -1

        # Exit when RSI crosses neutral
        prev_rsi = rsi.shift(1)
        signals[(rsi > neutral) & (prev_rsi <= neutral) & (signals.shift(1) == 1)] = 0
        signals[(rsi < neutral) & (prev_rsi >= neutral) & (signals.shift(1) == -1)] = 0

        return pd.DataFrame({
            'price': prices,
            'rsi': rsi,
            'signal': signals
        })

    def statistical_arbitrage(
        self,
        stock1: pd.Series,
        stock2: pd.Series,
        hedge_ratio: float = None
    ) -> pd.DataFrame:
        """
        Statistical arbitrage between two stocks.

        Args:
            stock1: First stock price series
            stock2: Second stock price series
            hedge_ratio: Hedge ratio (calculated if None)

        Returns:
            DataFrame with spread and signals
        """
        # Calculate hedge ratio if not provided
        if hedge_ratio is None:
            hedge_ratio = (stock1 / stock2).mean()

        # Calculate spread
        spread = stock1 - hedge_ratio * stock2

        # Calculate z-score of spread
        zscore = self.calculate_zscore(spread)

        # Generate signals
        signals = pd.Series(0, index=stock1.index)

        # Long spread when z-score < -entry_threshold
        signals[zscore < -self.entry_threshold] = 1

        # Short spread when z-score > entry_threshold
        signals[zscore > self.entry_threshold] = -1

        # Exit when abs(z-score) < exit_threshold
        signals[abs(zscore) < self.exit_threshold] = 0

        return pd.DataFrame({
            'stock1': stock1,
            'stock2': stock2,
            'spread': spread,
            'zscore': zscore,
            'signal': signals
        })

    def backtest(
        self,
        prices: pd.Series,
        signals: pd.Series,
        initial_capital: float = 100000,
        position_size: float = 1.0
    ) -> pd.DataFrame:
        """
        Backtest mean reversion strategy.

        Args:
            prices: Price series
            signals: Trading signals
            initial_capital: Starting capital
            position_size: Fraction of capital per trade

        Returns:
            DataFrame with backtest results
        """
        # Calculate returns
        returns = prices.pct_change()

        # Position sizing
        positions = signals.shift(1) * position_size

        # Strategy returns
        strategy_returns = positions * returns

        # Cumulative returns
        cumulative_returns = (1 + strategy_returns).cumprod()

        # Portfolio value
        portfolio_value = initial_capital * cumulative_returns

        return pd.DataFrame({
            'price': prices,
            'signal': signals,
            'position': positions,
            'returns': strategy_returns,
            'cumulative_returns': cumulative_returns,
            'portfolio_value': portfolio_value
        })
