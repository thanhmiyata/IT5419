"""
Momentum Indicators for Technical Analysis
==========================================

Momentum indicators including:
- RSI (Relative Strength Index)
- Stochastic Oscillator
- ROC (Rate of Change)
- CCI (Commodity Channel Index)
- Williams %R
- Momentum
"""

from typing import Tuple

import pandas as pd


class MomentumIndicators:
    """Calculate momentum-based technical indicators."""

    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        Args:
            data: Price data series
            period: Number of periods for calculation

        Returns:
            Series containing RSI values (0-100)
        """
        delta = data.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi_result = 100 - (100 / (1 + rs))

        return rsi_result

    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            k_period: Period for %K calculation
            d_period: Period for %D smoothing

        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()

        return k, d

    @staticmethod
    def roc(data: pd.Series, period: int = 12) -> pd.Series:
        """
        Calculate Rate of Change (ROC).

        Args:
            data: Price data series
            period: Number of periods for calculation

        Returns:
            Series containing ROC values (%)
        """
        roc_result = ((data - data.shift(period)) / data.shift(period)) * 100
        return roc_result

    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Commodity Channel Index (CCI).

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Number of periods for calculation

        Returns:
            Series containing CCI values
        """
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: abs(x - x.mean()).mean(),
            raw=True
        )

        cci_result = (typical_price - sma) / (0.015 * mean_deviation)
        return cci_result

    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Williams %R.

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Number of periods for calculation

        Returns:
            Series containing Williams %R values (-100 to 0)
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()

        williams_result = -100 * (highest_high - close) / (highest_high - lowest_low)
        return williams_result

    @staticmethod
    def momentum(data: pd.Series, period: int = 10) -> pd.Series:
        """
        Calculate Momentum indicator.

        Args:
            data: Price data series
            period: Number of periods for calculation

        Returns:
            Series containing momentum values
        """
        return data - data.shift(period)

    @staticmethod
    def trix(data: pd.Series, period: int = 15) -> pd.Series:
        """
        Calculate TRIX (Triple Exponential Average).

        Args:
            data: Price data series
            period: Number of periods for EMA calculation

        Returns:
            Series containing TRIX values
        """
        ema1 = data.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()

        trix_result = 100 * (ema3.diff() / ema3.shift(1))
        return trix_result

    @staticmethod
    def ultimate_oscillator(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period1: int = 7,
        period2: int = 14,
        period3: int = 28
    ) -> pd.Series:
        """
        Calculate Ultimate Oscillator.

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period1: First period
            period2: Second period
            period3: Third period

        Returns:
            Series containing Ultimate Oscillator values (0-100)
        """
        prev_close = close.shift(1)
        buying_pressure = close - pd.concat([low, prev_close], axis=1).min(axis=1)
        true_range = pd.concat([high, prev_close], axis=1).max(axis=1) - \
            pd.concat([low, prev_close], axis=1).min(axis=1)

        avg1 = buying_pressure.rolling(window=period1).sum() / true_range.rolling(window=period1).sum()
        avg2 = buying_pressure.rolling(window=period2).sum() / true_range.rolling(window=period2).sum()
        avg3 = buying_pressure.rolling(window=period3).sum() / true_range.rolling(window=period3).sum()

        uo = 100 * ((4 * avg1 + 2 * avg2 + avg3) / 7)
        return uo

    @staticmethod
    def awesome_oscillator(high: pd.Series, low: pd.Series) -> pd.Series:
        """
        Calculate Awesome Oscillator.

        Args:
            high: High price series
            low: Low price series

        Returns:
            Series containing Awesome Oscillator values
        """
        median_price = (high + low) / 2
        ao = median_price.rolling(window=5).mean() - median_price.rolling(window=34).mean()
        return ao

    @staticmethod
    def kst(
        data: pd.Series,
        roc1: int = 10,
        roc2: int = 15,
        roc3: int = 20,
        roc4: int = 30,
        sma1: int = 10,
        sma2: int = 10,
        sma3: int = 10,
        sma4: int = 15
    ) -> pd.Series:
        """
        Calculate Know Sure Thing (KST) oscillator.

        Args:
            data: Price data series
            roc1-4: ROC periods
            sma1-4: SMA smoothing periods

        Returns:
            Series containing KST values
        """
        rocma1 = ((data - data.shift(roc1)) / data.shift(roc1) * 100).rolling(window=sma1).mean()
        rocma2 = ((data - data.shift(roc2)) / data.shift(roc2) * 100).rolling(window=sma2).mean()
        rocma3 = ((data - data.shift(roc3)) / data.shift(roc3) * 100).rolling(window=sma3).mean()
        rocma4 = ((data - data.shift(roc4)) / data.shift(roc4) * 100).rolling(window=sma4).mean()

        kst_result = rocma1 + 2 * rocma2 + 3 * rocma3 + 4 * rocma4
        return kst_result
