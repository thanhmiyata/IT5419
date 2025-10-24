"""
Volume Indicators for Technical Analysis
========================================

Volume indicators including:
- OBV (On-Balance Volume)
- VWAP (Volume Weighted Average Price)
- MFI (Money Flow Index)
- AD (Accumulation/Distribution)
- CMF (Chaikin Money Flow)
"""

import pandas as pd


class VolumeIndicators:
    """Calculate volume-based technical indicators."""

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV).

        Args:
            close: Close price series
            volume: Volume series

        Returns:
            Series containing OBV values
        """
        obv_result = pd.Series(0.0, index=close.index)
        obv_result.iloc[0] = volume.iloc[0]

        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i - 1]:
                obv_result.iloc[i] = obv_result.iloc[i - 1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i - 1]:
                obv_result.iloc[i] = obv_result.iloc[i - 1] - volume.iloc[i]
            else:
                obv_result.iloc[i] = obv_result.iloc[i - 1]

        return obv_result

    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            volume: Volume series

        Returns:
            Series containing VWAP values
        """
        typical_price = (high + low + close) / 3
        vwap_result = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap_result

    @staticmethod
    def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Money Flow Index (MFI).

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            volume: Volume series
            period: Number of periods for calculation

        Returns:
            Series containing MFI values (0-100)
        """
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume

        positive_flow = pd.Series(0.0, index=close.index)
        negative_flow = pd.Series(0.0, index=close.index)

        for i in range(1, len(typical_price)):
            if typical_price.iloc[i] > typical_price.iloc[i - 1]:
                positive_flow.iloc[i] = money_flow.iloc[i]
            elif typical_price.iloc[i] < typical_price.iloc[i - 1]:
                negative_flow.iloc[i] = money_flow.iloc[i]

        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()

        mfi_result = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi_result

    @staticmethod
    def ad(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate Accumulation/Distribution (A/D).

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            volume: Volume series

        Returns:
            Series containing A/D values
        """
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.fillna(0)
        mfv = mfm * volume
        ad_result = mfv.cumsum()
        return ad_result

    @staticmethod
    def cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Chaikin Money Flow (CMF).

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            volume: Volume series
            period: Number of periods for calculation

        Returns:
            Series containing CMF values (-1 to 1)
        """
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.fillna(0)
        mfv = mfm * volume

        cmf_result = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
        return cmf_result

    @staticmethod
    def vwma(close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Volume Weighted Moving Average (VWMA).

        Args:
            close: Close price series
            volume: Volume series
            period: Number of periods for calculation

        Returns:
            Series containing VWMA values
        """
        vwma_result = (close * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
        return vwma_result

    @staticmethod
    def pvt(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate Price Volume Trend (PVT).

        Args:
            close: Close price series
            volume: Volume series

        Returns:
            Series containing PVT values
        """
        price_change = close.pct_change()
        pvt_result = (price_change * volume).cumsum()
        return pvt_result

    @staticmethod
    def nvi(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate Negative Volume Index (NVI).

        Args:
            close: Close price series
            volume: Volume series

        Returns:
            Series containing NVI values
        """
        nvi_result = pd.Series(1000.0, index=close.index)

        for i in range(1, len(close)):
            if volume.iloc[i] < volume.iloc[i - 1]:
                nvi_result.iloc[i] = nvi_result.iloc[i - 1] * (1 + close.pct_change().iloc[i])
            else:
                nvi_result.iloc[i] = nvi_result.iloc[i - 1]

        return nvi_result

    @staticmethod
    def pvi(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate Positive Volume Index (PVI).

        Args:
            close: Close price series
            volume: Volume series

        Returns:
            Series containing PVI values
        """
        pvi_result = pd.Series(1000.0, index=close.index)

        for i in range(1, len(close)):
            if volume.iloc[i] > volume.iloc[i - 1]:
                pvi_result.iloc[i] = pvi_result.iloc[i - 1] * (1 + close.pct_change().iloc[i])
            else:
                pvi_result.iloc[i] = pvi_result.iloc[i - 1]

        return pvi_result

    @staticmethod
    def eom(high: pd.Series, low: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Ease of Movement (EOM).

        Args:
            high: High price series
            low: Low price series
            volume: Volume series
            period: Number of periods for smoothing

        Returns:
            Series containing EOM values
        """
        distance = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
        box_ratio = (volume / 1000000) / (high - low)
        eom_raw = distance / box_ratio
        eom_result = eom_raw.rolling(window=period).mean()
        return eom_result
