"""
Volatility Indicators for Technical Analysis
============================================

Volatility indicators including:
- Bollinger Bands
- ATR (Average True Range)
- Standard Deviation
- Keltner Channels
- Donchian Channels
"""

from typing import Tuple

import pandas as pd


class VolatilityIndicators:
    """Calculate volatility-based technical indicators."""

    @staticmethod
    def bollinger_bands(
        data: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Args:
            data: Price data series
            period: Number of periods for moving average
            std_dev: Number of standard deviations

        Returns:
            Tuple of (Upper Band, Middle Band, Lower Band)
        """
        middle_band = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()

        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)

        return upper_band, middle_band, lower_band

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Number of periods for calculation

        Returns:
            Series containing ATR values
        """
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_result = tr.rolling(window=period).mean()

        return atr_result

    @staticmethod
    def standard_deviation(data: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Standard Deviation.

        Args:
            data: Price data series
            period: Number of periods for calculation

        Returns:
            Series containing standard deviation values
        """
        return data.rolling(window=period).std()

    @staticmethod
    def keltner_channels(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Keltner Channels.

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Period for EMA calculation
            atr_period: Period for ATR calculation
            multiplier: ATR multiplier

        Returns:
            Tuple of (Upper Channel, Middle Channel, Lower Channel)
        """
        middle_channel = close.ewm(span=period, adjust=False).mean()
        atr_val = VolatilityIndicators.atr(high, low, close, atr_period)

        upper_channel = middle_channel + (atr_val * multiplier)
        lower_channel = middle_channel - (atr_val * multiplier)

        return upper_channel, middle_channel, lower_channel

    @staticmethod
    def donchian_channels(
        high: pd.Series,
        low: pd.Series,
        period: int = 20
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Donchian Channels.

        Args:
            high: High price series
            low: Low price series
            period: Number of periods for calculation

        Returns:
            Tuple of (Upper Channel, Middle Channel, Lower Channel)
        """
        upper_channel = high.rolling(window=period).max()
        lower_channel = low.rolling(window=period).min()
        middle_channel = (upper_channel + lower_channel) / 2

        return upper_channel, middle_channel, lower_channel

    @staticmethod
    def bollinger_bandwidth(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
        """
        Calculate Bollinger Bandwidth.

        Args:
            data: Price data series
            period: Number of periods for moving average
            std_dev: Number of standard deviations

        Returns:
            Series containing bandwidth values
        """
        upper, middle, lower = VolatilityIndicators.bollinger_bands(data, period, std_dev)
        bandwidth = (upper - lower) / middle * 100
        return bandwidth

    @staticmethod
    def bollinger_percent_b(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
        """
        Calculate Bollinger %B.

        Args:
            data: Price data series
            period: Number of periods for moving average
            std_dev: Number of standard deviations

        Returns:
            Series containing %B values
        """
        upper, middle, lower = VolatilityIndicators.bollinger_bands(data, period, std_dev)
        percent_b = (data - lower) / (upper - lower)
        return percent_b

    @staticmethod
    def historical_volatility(data: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Historical Volatility (annualized).

        Args:
            data: Price data series
            period: Number of periods for calculation

        Returns:
            Series containing annualized volatility values (%)
        """
        log_returns = data.pct_change().apply(lambda x: 0 if x == 0 else x).rolling(window=2).apply(
            lambda x: 0 if len(x) < 2 else (x.iloc[1] / x.iloc[0]) if x.iloc[0] != 0 else 0
        )
        volatility = log_returns.rolling(window=period).std() * (252 ** 0.5) * 100
        return volatility

    @staticmethod
    def ulcer_index(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Ulcer Index.

        Args:
            data: Price data series
            period: Number of periods for calculation

        Returns:
            Series containing Ulcer Index values
        """
        max_price = data.rolling(window=period).max()
        percentage_drawdown = ((data - max_price) / max_price) * 100
        squared_avg = (percentage_drawdown ** 2).rolling(window=period).mean()
        ulcer = squared_avg ** 0.5
        return ulcer

    @staticmethod
    def mass_index(high: pd.Series, low: pd.Series, period: int = 25, ema_period: int = 9) -> pd.Series:
        """
        Calculate Mass Index.

        Args:
            high: High price series
            low: Low price series
            period: Period for summation
            ema_period: Period for EMA calculation

        Returns:
            Series containing Mass Index values
        """
        price_range = high - low
        ema1 = price_range.ewm(span=ema_period, adjust=False).mean()
        ema2 = ema1.ewm(span=ema_period, adjust=False).mean()
        mass = ema1 / ema2
        mass_index_result = mass.rolling(window=period).sum()
        return mass_index_result

    @staticmethod
    def chaikin_volatility(high: pd.Series, low: pd.Series, period: int = 10, roc_period: int = 10) -> pd.Series:
        """
        Calculate Chaikin Volatility.

        Args:
            high: High price series
            low: Low price series
            period: Period for EMA of high-low
            roc_period: Period for rate of change

        Returns:
            Series containing Chaikin Volatility values
        """
        hl_range = high - low
        ema_hl = hl_range.ewm(span=period, adjust=False).mean()
        chaikin_vol = ((ema_hl - ema_hl.shift(roc_period)) / ema_hl.shift(roc_period)) * 100
        return chaikin_vol
