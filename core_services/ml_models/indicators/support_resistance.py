"""
Support/Resistance Indicators for Technical Analysis
====================================================

Support and resistance indicators including:
- Fibonacci Retracement
- Pivot Points (Standard, Fibonacci, Camarilla, Woodie)
- Support/Resistance Levels
"""

from typing import Dict, List, Tuple

import pandas as pd


class SupportResistanceIndicators:
    """Calculate support and resistance levels."""

    @staticmethod
    def fibonacci_retracement(high: float, low: float) -> Dict[str, float]:
        """
        Calculate Fibonacci Retracement levels.

        Args:
            high: Highest price in the range
            low: Lowest price in the range

        Returns:
            Dictionary with Fibonacci levels
        """
        diff = high - low

        levels = {
            "0.0%": high,
            "23.6%": high - (diff * 0.236),
            "38.2%": high - (diff * 0.382),
            "50.0%": high - (diff * 0.500),
            "61.8%": high - (diff * 0.618),
            "78.6%": high - (diff * 0.786),
            "100.0%": low,
        }

        return levels

    @staticmethod
    def fibonacci_extension(high: float, low: float) -> Dict[str, float]:
        """
        Calculate Fibonacci Extension levels.

        Args:
            high: Highest price in the range
            low: Lowest price in the range

        Returns:
            Dictionary with Fibonacci extension levels
        """
        diff = high - low

        levels = {
            "0.0%": high,
            "61.8%": high + (diff * 0.618),
            "100.0%": high + diff,
            "161.8%": high + (diff * 1.618),
            "261.8%": high + (diff * 2.618),
            "423.6%": high + (diff * 4.236),
        }

        return levels

    @staticmethod
    def pivot_points_standard(high: float, low: float, close: float) -> Dict[str, float]:
        """
        Calculate Standard Pivot Points.

        Args:
            high: Previous period high
            low: Previous period low
            close: Previous period close

        Returns:
            Dictionary with pivot levels
        """
        pivot = (high + low + close) / 3

        r1 = (2 * pivot) - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)

        s1 = (2 * pivot) - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)

        return {
            "P": pivot,
            "R1": r1,
            "R2": r2,
            "R3": r3,
            "S1": s1,
            "S2": s2,
            "S3": s3,
        }

    @staticmethod
    def pivot_points_fibonacci(high: float, low: float, close: float) -> Dict[str, float]:
        """
        Calculate Fibonacci Pivot Points.

        Args:
            high: Previous period high
            low: Previous period low
            close: Previous period close

        Returns:
            Dictionary with Fibonacci pivot levels
        """
        pivot = (high + low + close) / 3
        range_hl = high - low

        r1 = pivot + (range_hl * 0.382)
        r2 = pivot + (range_hl * 0.618)
        r3 = pivot + (range_hl * 1.000)

        s1 = pivot - (range_hl * 0.382)
        s2 = pivot - (range_hl * 0.618)
        s3 = pivot - (range_hl * 1.000)

        return {
            "P": pivot,
            "R1": r1,
            "R2": r2,
            "R3": r3,
            "S1": s1,
            "S2": s2,
            "S3": s3,
        }

    @staticmethod
    def pivot_points_camarilla(high: float, low: float, close: float) -> Dict[str, float]:
        """
        Calculate Camarilla Pivot Points.

        Args:
            high: Previous period high
            low: Previous period low
            close: Previous period close

        Returns:
            Dictionary with Camarilla pivot levels
        """
        range_hl = high - low

        r1 = close + (range_hl * 1.1 / 12)
        r2 = close + (range_hl * 1.1 / 6)
        r3 = close + (range_hl * 1.1 / 4)
        r4 = close + (range_hl * 1.1 / 2)

        s1 = close - (range_hl * 1.1 / 12)
        s2 = close - (range_hl * 1.1 / 6)
        s3 = close - (range_hl * 1.1 / 4)
        s4 = close - (range_hl * 1.1 / 2)

        return {
            "R4": r4,
            "R3": r3,
            "R2": r2,
            "R1": r1,
            "S1": s1,
            "S2": s2,
            "S3": s3,
            "S4": s4,
        }

    @staticmethod
    def pivot_points_woodie(high: float, low: float, close: float) -> Dict[str, float]:
        """
        Calculate Woodie's Pivot Points.

        Args:
            high: Previous period high
            low: Previous period low
            close: Previous period close

        Returns:
            Dictionary with Woodie pivot levels
        """
        pivot = (high + low + 2 * close) / 4

        r1 = (2 * pivot) - low
        r2 = pivot + (high - low)

        s1 = (2 * pivot) - high
        s2 = pivot - (high - low)

        return {
            "P": pivot,
            "R1": r1,
            "R2": r2,
            "S1": s1,
            "S2": s2,
        }

    @staticmethod
    def support_resistance_levels(
        data: pd.Series,
        window: int = 20,
        num_levels: int = 3
    ) -> Tuple[List[float], List[float]]:
        """
        Identify support and resistance levels from price data.

        Args:
            data: Price data series
            window: Window for local extrema detection
            num_levels: Number of support/resistance levels to return

        Returns:
            Tuple of (support_levels, resistance_levels)
        """
        # Find local maxima (resistance)
        resistance = []
        for i in range(window, len(data) - window):
            if data.iloc[i] == data.iloc[i - window:i + window + 1].max():
                resistance.append(data.iloc[i])

        # Find local minima (support)
        support = []
        for i in range(window, len(data) - window):
            if data.iloc[i] == data.iloc[i - window:i + window + 1].min():
                support.append(data.iloc[i])

        # Sort and get top levels
        resistance_sorted = sorted(set(resistance), reverse=True)[:num_levels]
        support_sorted = sorted(set(support))[:num_levels]

        return support_sorted, resistance_sorted

    @staticmethod
    def demark_pivot_points(high: float, low: float, close: float, open_price: float) -> Dict[str, float]:
        """
        Calculate DeMark Pivot Points.

        Args:
            high: Previous period high
            low: Previous period low
            close: Previous period close
            open_price: Previous period open

        Returns:
            Dictionary with DeMark pivot levels
        """
        if close < open_price:
            x = high + (2 * low) + close
        elif close > open_price:
            x = (2 * high) + low + close
        else:
            x = high + low + (2 * close)

        pivot = x / 4
        r1 = x / 2 - low
        s1 = x / 2 - high

        return {
            "P": pivot,
            "R1": r1,
            "S1": s1,
        }

    @staticmethod
    def round_numbers(price: float, round_to: int = 1000) -> List[float]:
        """
        Calculate nearby round number levels (psychological levels).

        Args:
            price: Current price
            round_to: Rounding level (e.g., 1000, 500, 100)

        Returns:
            List of nearby round numbers
        """
        lower = (price // round_to) * round_to
        upper = lower + round_to

        return [
            lower - round_to,
            lower,
            upper,
            upper + round_to,
        ]

    @staticmethod
    def swing_highs_lows(
        high: pd.Series,
        low: pd.Series,
        left_bars: int = 5,
        right_bars: int = 5
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Identify swing highs and swing lows.

        Args:
            high: High price series
            low: Low price series
            left_bars: Number of bars to the left
            right_bars: Number of bars to the right

        Returns:
            Tuple of (swing_highs, swing_lows) as boolean Series
        """
        swing_highs = pd.Series(False, index=high.index)
        swing_lows = pd.Series(False, index=low.index)

        for i in range(left_bars, len(high) - right_bars):
            # Check swing high
            is_swing_high = True
            for j in range(i - left_bars, i + right_bars + 1):
                if j != i and high.iloc[j] >= high.iloc[i]:
                    is_swing_high = False
                    break
            swing_highs.iloc[i] = is_swing_high

            # Check swing low
            is_swing_low = True
            for j in range(i - left_bars, i + right_bars + 1):
                if j != i and low.iloc[j] <= low.iloc[i]:
                    is_swing_low = False
                    break
            swing_lows.iloc[i] = is_swing_low

        return swing_highs, swing_lows
