"""
Trend Indicators for Technical Analysis
=======================================

Trend indicators including:
- Moving Averages (SMA, EMA, WMA)
- MACD (Moving Average Convergence Divergence)
- ADX (Average Directional Index)
"""

from typing import Tuple

import numpy as np
import pandas as pd


class TrendIndicators:
    """Calculate trend-based technical indicators."""

    @staticmethod
    def sma(data: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Simple Moving Average (SMA).

        Args:
            data: Price data series
            period: Number of periods for moving average

        Returns:
            Series containing SMA values
        """
        return data.rolling(window=period).mean()

    @staticmethod
    def ema(data: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA).

        Args:
            data: Price data series
            period: Number of periods for moving average

        Returns:
            Series containing EMA values
        """
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def wma(data: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Weighted Moving Average (WMA).

        Args:
            data: Price data series
            period: Number of periods for moving average

        Returns:
            Series containing WMA values
        """
        weights = np.arange(1, period + 1)
        return data.rolling(window=period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    @staticmethod
    def macd(
        data: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            data: Price data series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period

        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        fast_ema = data.ewm(span=fast_period, adjust=False).mean()
        slow_ema = data.ewm(span=slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index (ADX).

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Number of periods for calculation

        Returns:
            Series containing ADX values
        """
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        # Calculate Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = pd.Series(0.0, index=high.index)
        minus_dm = pd.Series(0.0, index=high.index)

        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

        # Calculate Directional Indicators
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx_result = dx.rolling(window=period).mean()

        return adx_result

    @staticmethod
    def dmi(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Directional Movement Index (DMI).

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Number of periods for calculation

        Returns:
            Tuple of (+DI, -DI)
        """
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        # Calculate Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = pd.Series(0.0, index=high.index)
        minus_dm = pd.Series(0.0, index=high.index)

        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

        # Calculate Directional Indicators
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        return plus_di, minus_di

    @staticmethod
    def parabolic_sar(
        high: pd.Series,
        low: pd.Series,
        acceleration: float = 0.02,
        maximum: float = 0.2
    ) -> pd.Series:
        """
        Calculate Parabolic SAR (Stop and Reverse).

        Args:
            high: High price series
            low: Low price series
            acceleration: Acceleration factor
            maximum: Maximum acceleration

        Returns:
            Series containing Parabolic SAR values
        """
        sar = pd.Series(index=high.index, dtype=float)
        trend = pd.Series(index=high.index, dtype=int)
        ep = pd.Series(index=high.index, dtype=float)
        af = pd.Series(index=high.index, dtype=float)

        # Initialize
        sar.iloc[0] = low.iloc[0]
        trend.iloc[0] = 1
        ep.iloc[0] = high.iloc[0]
        af.iloc[0] = acceleration

        for i in range(1, len(high)):
            sar.iloc[i] = sar.iloc[i - 1] + af.iloc[i - 1] * (ep.iloc[i - 1] - sar.iloc[i - 1])

            if trend.iloc[i - 1] == 1:
                if low.iloc[i] < sar.iloc[i]:
                    trend.iloc[i] = -1
                    sar.iloc[i] = ep.iloc[i - 1]
                    ep.iloc[i] = low.iloc[i]
                    af.iloc[i] = acceleration
                else:
                    trend.iloc[i] = 1
                    if high.iloc[i] > ep.iloc[i - 1]:
                        ep.iloc[i] = high.iloc[i]
                        af.iloc[i] = min(af.iloc[i - 1] + acceleration, maximum)
                    else:
                        ep.iloc[i] = ep.iloc[i - 1]
                        af.iloc[i] = af.iloc[i - 1]
            else:
                if high.iloc[i] > sar.iloc[i]:
                    trend.iloc[i] = 1
                    sar.iloc[i] = ep.iloc[i - 1]
                    ep.iloc[i] = high.iloc[i]
                    af.iloc[i] = acceleration
                else:
                    trend.iloc[i] = -1
                    if low.iloc[i] < ep.iloc[i - 1]:
                        ep.iloc[i] = low.iloc[i]
                        af.iloc[i] = min(af.iloc[i - 1] + acceleration, maximum)
                    else:
                        ep.iloc[i] = ep.iloc[i - 1]
                        af.iloc[i] = af.iloc[i - 1]

        return sar

    @staticmethod
    def trend_strength(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate trend strength indicator.

        Args:
            data: Price data series
            period: Number of periods for calculation

        Returns:
            Series containing trend strength values (0-100)
        """
        gains = data.diff()
        up = gains.where(gains > 0, 0.0)
        down = -gains.where(gains < 0, 0.0)

        avg_up = up.rolling(window=period).mean()
        avg_down = down.rolling(window=period).mean()

        total_movement = avg_up + avg_down
        trend_strength_val = (abs(avg_up - avg_down) / total_movement * 100).fillna(0)
        return trend_strength_val
