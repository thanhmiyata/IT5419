"""
Market Breadth Indicators for Technical Analysis
================================================

Market breadth indicators including:
- Advance-Decline Line
- Advance-Decline Ratio
- Put-Call Ratio
- McClellan Oscillator
- Arms Index (TRIN)
"""

import pandas as pd


class MarketBreadthIndicators:
    """Calculate market breadth indicators."""

    @staticmethod
    def advance_decline_line(advances: pd.Series, declines: pd.Series) -> pd.Series:
        """
        Calculate Advance-Decline Line.

        Args:
            advances: Number of advancing stocks
            declines: Number of declining stocks

        Returns:
            Series containing cumulative A/D line
        """
        net_advances = advances - declines
        ad_line = net_advances.cumsum()
        return ad_line

    @staticmethod
    def advance_decline_ratio(advances: pd.Series, declines: pd.Series) -> pd.Series:
        """
        Calculate Advance-Decline Ratio.

        Args:
            advances: Number of advancing stocks
            declines: Number of declining stocks

        Returns:
            Series containing A/D ratio
        """
        ad_ratio = advances / declines
        return ad_ratio.replace([float("inf"), -float("inf")], 0)

    @staticmethod
    def put_call_ratio(put_volume: pd.Series, call_volume: pd.Series) -> pd.Series:
        """
        Calculate Put-Call Ratio.

        Args:
            put_volume: Put option volume
            call_volume: Call option volume

        Returns:
            Series containing put-call ratio
        """
        pc_ratio = put_volume / call_volume
        return pc_ratio.replace([float("inf"), -float("inf")], 0)

    @staticmethod
    def mcclellan_oscillator(advances: pd.Series, declines: pd.Series) -> pd.Series:
        """
        Calculate McClellan Oscillator.

        Args:
            advances: Number of advancing stocks
            declines: Number of declining stocks

        Returns:
            Series containing McClellan Oscillator values
        """
        net_advances = advances - declines
        ema_fast = net_advances.ewm(span=19, adjust=False).mean()
        ema_slow = net_advances.ewm(span=39, adjust=False).mean()
        mcclellan = ema_fast - ema_slow
        return mcclellan

    @staticmethod
    def mcclellan_summation_index(advances: pd.Series, declines: pd.Series) -> pd.Series:
        """
        Calculate McClellan Summation Index.

        Args:
            advances: Number of advancing stocks
            declines: Number of declining stocks

        Returns:
            Series containing McClellan Summation Index
        """
        mcclellan = MarketBreadthIndicators.mcclellan_oscillator(advances, declines)
        summation_index = mcclellan.cumsum()
        return summation_index

    @staticmethod
    def arms_index(
        advances: pd.Series,
        declines: pd.Series,
        advancing_volume: pd.Series,
        declining_volume: pd.Series
    ) -> pd.Series:
        """
        Calculate Arms Index (TRIN - Trading Index).

        Args:
            advances: Number of advancing stocks
            declines: Number of declining stocks
            advancing_volume: Volume of advancing stocks
            declining_volume: Volume of declining stocks

        Returns:
            Series containing TRIN values
        """
        ad_ratio = advances / declines
        volume_ratio = advancing_volume / declining_volume
        trin = ad_ratio / volume_ratio
        return trin.replace([float("inf"), -float("inf")], 0)

    @staticmethod
    def new_highs_lows(new_highs: pd.Series, new_lows: pd.Series) -> pd.Series:
        """
        Calculate New Highs - New Lows indicator.

        Args:
            new_highs: Number of stocks making new highs
            new_lows: Number of stocks making new lows

        Returns:
            Series containing net new highs
        """
        return new_highs - new_lows

    @staticmethod
    def breadth_thrust(advances: pd.Series, declines: pd.Series, period: int = 10) -> pd.Series:
        """
        Calculate Breadth Thrust indicator.

        Args:
            advances: Number of advancing stocks
            declines: Number of declining stocks
            period: Number of periods for EMA

        Returns:
            Series containing breadth thrust values
        """
        total_issues = advances + declines
        advancing_ratio = advances / total_issues
        breadth_thrust_val = advancing_ratio.ewm(span=period, adjust=False).mean()
        return breadth_thrust_val

    @staticmethod
    def absolute_breadth_index(advances: pd.Series, declines: pd.Series) -> pd.Series:
        """
        Calculate Absolute Breadth Index (ABI).

        Args:
            advances: Number of advancing stocks
            declines: Number of declining stocks

        Returns:
            Series containing ABI values
        """
        abi = abs(advances - declines)
        return abi

    @staticmethod
    def stocks_above_ma(prices_df: pd.DataFrame, ma_period: int = 200) -> pd.Series:
        """
        Calculate percentage of stocks above their moving average.

        Args:
            prices_df: DataFrame with stock prices (columns are stocks)
            ma_period: Moving average period

        Returns:
            Series containing percentage of stocks above MA
        """
        ma = prices_df.rolling(window=ma_period).mean()
        above_ma = (prices_df > ma).sum(axis=1)
        total_stocks = prices_df.shape[1]
        percentage = (above_ma / total_stocks) * 100
        return percentage

    @staticmethod
    def bullish_percent_index(
        stocks_point_figure_signals: pd.DataFrame,
    ) -> pd.Series:
        """
        Calculate Bullish Percent Index.

        Args:
            stocks_point_figure_signals: DataFrame of bullish signals (1) or bearish (0)

        Returns:
            Series containing bullish percent values
        """
        bullish_count = stocks_point_figure_signals.sum(axis=1)
        total_stocks = stocks_point_figure_signals.shape[1]
        bullish_percent = (bullish_count / total_stocks) * 100
        return bullish_percent

    @staticmethod
    def up_down_volume_ratio(up_volume: pd.Series, down_volume: pd.Series, period: int = 10) -> pd.Series:
        """
        Calculate Up/Down Volume Ratio.

        Args:
            up_volume: Volume on up days
            down_volume: Volume on down days
            period: Period for moving average

        Returns:
            Series containing volume ratio
        """
        ratio = up_volume / down_volume
        ratio = ratio.replace([float("inf"), -float("inf")], 0)
        smoothed_ratio = ratio.rolling(window=period).mean()
        return smoothed_ratio

    @staticmethod
    def vix_sentiment(vix: pd.Series, lower_threshold: float = 20.0, upper_threshold: float = 30.0) -> pd.Series:
        """
        Calculate VIX-based sentiment indicator.

        Args:
            vix: VIX index values
            lower_threshold: Lower threshold for fear
            upper_threshold: Upper threshold for extreme fear

        Returns:
            Series with sentiment values (0=calm, 1=fear, 2=extreme fear)
        """
        sentiment = pd.Series(0, index=vix.index)
        sentiment[vix > lower_threshold] = 1
        sentiment[vix > upper_threshold] = 2
        return sentiment

    @staticmethod
    def high_low_index(new_highs: pd.Series, new_lows: pd.Series, period: int = 50) -> pd.Series:
        """
        Calculate High-Low Index.

        Args:
            new_highs: Number of stocks making new highs
            new_lows: Number of stocks making new lows
            period: Period for moving average

        Returns:
            Series containing High-Low Index
        """
        total_extremes = new_highs + new_lows
        hl_index = (new_highs / total_extremes * 100).rolling(window=period).mean()
        return hl_index.replace([float("inf"), -float("inf")], 0)

    @staticmethod
    def market_momentum(market_index: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate overall market momentum.

        Args:
            market_index: Market index values (e.g., VN-Index)
            period: Period for ROC calculation

        Returns:
            Series containing market momentum (%)
        """
        momentum = ((market_index - market_index.shift(period)) / market_index.shift(period)) * 100
        return momentum
