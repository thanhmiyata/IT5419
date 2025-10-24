"""
Breakout Trading Strategy
==========================

Breakout strategies for trend following.
"""

import numpy as np
import pandas as pd


class BreakoutStrategy:
    """Breakout-based trading strategy."""

    def __init__(
        self,
        lookback_period: int = 20,
        atr_period: int = 14,
        atr_multiplier: float = 2.0
    ):
        """
        Initialize breakout strategy.

        Args:
            lookback_period: Period for breakout calculation
            atr_period: Period for ATR calculation
            atr_multiplier: Multiplier for ATR-based stops
        """
        self.lookback_period = lookback_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    def calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> pd.Series:
        """
        Calculate Average True Range.

        Args:
            high: High prices
            low: Low prices
            close: Close prices

        Returns:
            ATR series
        """
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Average True Range
        atr = true_range.rolling(window=self.atr_period).mean()

        return atr

    def donchian_channel_breakout(
        self,
        prices: pd.Series,
        upper_period: int = None,
        lower_period: int = None
    ) -> pd.DataFrame:
        """
        Donchian Channel breakout strategy.

        Args:
            prices: Price series
            upper_period: Period for upper channel (default: lookback_period)
            lower_period: Period for lower channel (default: lookback_period)

        Returns:
            DataFrame with signals and channel levels
        """
        if upper_period is None:
            upper_period = self.lookback_period
        if lower_period is None:
            lower_period = self.lookback_period

        # Calculate channels
        upper_channel = prices.rolling(window=upper_period).max()
        lower_channel = prices.rolling(window=lower_period).min()
        middle_channel = (upper_channel + lower_channel) / 2

        # Generate signals
        signals = pd.Series(0, index=prices.index)

        # Long on upper breakout
        signals[prices > upper_channel.shift(1)] = 1

        # Short on lower breakout
        signals[prices < lower_channel.shift(1)] = -1

        return pd.DataFrame({
            'price': prices,
            'upper_channel': upper_channel,
            'lower_channel': lower_channel,
            'middle_channel': middle_channel,
            'signal': signals
        })

    def bollinger_breakout(
        self,
        prices: pd.Series,
        window: int = None,
        num_std: float = 2.0
    ) -> pd.DataFrame:
        """
        Bollinger Band breakout strategy.

        Args:
            prices: Price series
            window: Window for moving average
            num_std: Number of standard deviations

        Returns:
            DataFrame with signals and bands
        """
        if window is None:
            window = self.lookback_period

        # Calculate Bollinger Bands
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()

        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)

        # Generate signals
        signals = pd.Series(0, index=prices.index)

        # Long on upper breakout
        signals[prices > upper_band.shift(1)] = 1

        # Short on lower breakout
        signals[prices < lower_band.shift(1)] = -1

        return pd.DataFrame({
            'price': prices,
            'upper_band': upper_band,
            'middle_band': sma,
            'lower_band': lower_band,
            'signal': signals
        })

    def volatility_breakout(
        self,
        open_price: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k: float = 0.5
    ) -> pd.DataFrame:
        """
        Volatility breakout strategy (Korean market popular).

        Args:
            open_price: Opening prices
            high: High prices
            low: Low prices
            close: Close prices
            k: Range multiplier (0.5 = half of yesterday's range)

        Returns:
            DataFrame with signals and breakout levels
        """
        # Calculate yesterday's range
        prev_range = (high - low).shift(1)

        # Breakout level = today's open + k * yesterday's range
        breakout_level = open_price + k * prev_range

        # Generate signals
        signals = pd.Series(0, index=close.index)

        # Long when price breaks above level
        signals[close > breakout_level] = 1

        # Exit at end of day (or hold)
        # For simplicity, we'll hold positions

        return pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'breakout_level': breakout_level,
            'signal': signals
        })

    def support_resistance_breakout(
        self,
        prices: pd.Series,
        window: int = None
    ) -> pd.DataFrame:
        """
        Support/Resistance level breakout.

        Args:
            prices: Price series
            window: Window for identifying levels

        Returns:
            DataFrame with signals and S/R levels
        """
        if window is None:
            window = self.lookback_period

        # Identify support and resistance
        resistance = prices.rolling(window=window).max()
        support = prices.rolling(window=window).min()

        # Generate signals
        signals = pd.Series(0, index=prices.index)

        # Long on resistance breakout
        signals[prices > resistance.shift(1)] = 1

        # Short on support breakdown
        signals[prices < support.shift(1)] = -1

        return pd.DataFrame({
            'price': prices,
            'resistance': resistance,
            'support': support,
            'signal': signals
        })

    def turtle_trading_system(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        entry_period: int = 20,
        exit_period: int = 10
    ) -> pd.DataFrame:
        """
        Turtle Trading breakout system.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            entry_period: Period for entry breakout (20-day)
            exit_period: Period for exit (10-day)

        Returns:
            DataFrame with signals and levels
        """
        # Entry levels
        entry_high = high.rolling(window=entry_period).max()
        entry_low = low.rolling(window=entry_period).min()

        # Exit levels
        exit_high = high.rolling(window=exit_period).max()
        exit_low = low.rolling(window=exit_period).min()

        # Calculate ATR for position sizing
        atr = self.calculate_atr(high, low, close)

        # Generate signals
        signals = pd.Series(0, index=close.index)
        position = 0

        for i in range(1, len(close)):
            # Long entry
            if close.iloc[i] > entry_high.iloc[i - 1] and position <= 0:
                signals.iloc[i] = 1
                position = 1
            # Long exit
            elif close.iloc[i] < exit_low.iloc[i - 1] and position > 0:
                signals.iloc[i] = 0
                position = 0
            # Short entry
            elif close.iloc[i] < entry_low.iloc[i - 1] and position >= 0:
                signals.iloc[i] = -1
                position = -1
            # Short exit
            elif close.iloc[i] > exit_high.iloc[i - 1] and position < 0:
                signals.iloc[i] = 0
                position = 0
            else:
                signals.iloc[i] = position

        # Calculate stop loss (2 ATR)
        stop_loss = atr * self.atr_multiplier

        return pd.DataFrame({
            'close': close,
            'entry_high': entry_high,
            'entry_low': entry_low,
            'exit_high': exit_high,
            'exit_low': exit_low,
            'atr': atr,
            'stop_loss': stop_loss,
            'signal': signals
        })

    def range_breakout(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        consolidation_period: int = 10,
        breakout_threshold: float = 0.02
    ) -> pd.DataFrame:
        """
        Range breakout after consolidation.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            consolidation_period: Period to identify consolidation
            breakout_threshold: Minimum breakout percentage

        Returns:
            DataFrame with signals
        """
        # Calculate range
        range_size = high - low
        avg_range = range_size.rolling(window=consolidation_period).mean()

        # Identify consolidation (narrow range)
        is_consolidation = range_size < avg_range * 0.5

        # Calculate recent high/low
        recent_high = high.rolling(window=consolidation_period).max()
        recent_low = low.rolling(window=consolidation_period).min()

        # Generate signals
        signals = pd.Series(0, index=close.index)

        # Long breakout after consolidation
        long_breakout = (
            (close > recent_high.shift(1))
            & is_consolidation.shift(1)
            & ((close - recent_high.shift(1)) / recent_high.shift(1) > breakout_threshold)
        )
        signals[long_breakout] = 1

        # Short breakout after consolidation
        short_breakout = (
            (close < recent_low.shift(1))
            & is_consolidation.shift(1)
            & ((recent_low.shift(1) - close) / recent_low.shift(1) > breakout_threshold)
        )
        signals[short_breakout] = -1

        return pd.DataFrame({
            'close': close,
            'range': range_size,
            'avg_range': avg_range,
            'is_consolidation': is_consolidation,
            'recent_high': recent_high,
            'recent_low': recent_low,
            'signal': signals
        })

    def backtest(
        self,
        strategy_results: pd.DataFrame,
        initial_capital: float = 100000,
        commission: float = 0.001
    ) -> pd.DataFrame:
        """
        Backtest breakout strategy.

        Args:
            strategy_results: Results from strategy methods
            initial_capital: Starting capital
            commission: Commission rate per trade

        Returns:
            DataFrame with backtest results
        """
        prices = strategy_results['close'] if 'close' in strategy_results.columns else strategy_results['price']
        signals = strategy_results['signal']

        # Calculate returns
        returns = prices.pct_change()

        # Account for commission on trades
        trades = signals.diff().abs()
        commission_cost = trades * commission

        # Calculate strategy returns
        strategy_returns = signals.shift(1) * returns - commission_cost

        # Calculate cumulative returns
        cumulative_returns = (1 + strategy_returns).cumprod()

        # Portfolio value
        portfolio_value = initial_capital * cumulative_returns

        # Calculate drawdown
        running_max = portfolio_value.cummax()
        drawdown = (portfolio_value - running_max) / running_max

        return pd.DataFrame({
            'price': prices,
            'signal': signals,
            'returns': returns,
            'strategy_returns': strategy_returns,
            'portfolio_value': portfolio_value,
            'cumulative_returns': cumulative_returns,
            'drawdown': drawdown
        })

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
        strategy_returns = backtest_results['strategy_returns'].dropna()

        # Total return
        total_return = backtest_results['cumulative_returns'].iloc[-1] - 1

        # Annualized return
        n_years = len(strategy_returns) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1

        # Annualized volatility
        annual_vol = strategy_returns.std() * np.sqrt(252)

        # Sharpe ratio
        sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0

        # Maximum drawdown
        max_drawdown = backtest_results['drawdown'].min()

        # Win rate
        wins = strategy_returns[strategy_returns > 0]
        win_rate = len(wins) / len(strategy_returns) if len(strategy_returns) > 0 else 0

        # Profit factor
        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(strategy_returns[strategy_returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        # Number of trades
        signals = backtest_results['signal']
        num_trades = signals.diff().abs().sum() / 2

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': num_trades
        }
