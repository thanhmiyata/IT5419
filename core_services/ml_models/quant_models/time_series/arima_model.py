"""
ARIMA Model for Time Series Forecasting
========================================

AutoRegressive Integrated Moving Average model for price prediction.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


class ARIMAForecaster:
    """ARIMA model for stock price forecasting."""

    def __init__(self, order: Tuple[int, int, int] = (5, 1, 0)):
        """
        Initialize ARIMA forecaster.

        Args:
            order: ARIMA order (p, d, q)
                p: AR order (autoregressive)
                d: differencing order
                q: MA order (moving average)
        """
        self.order = order
        self.model = None
        self.fitted_model = None
        self.history = None

    def check_stationarity(self, data: pd.Series) -> Dict[str, float]:
        """
        Check if series is stationary using Augmented Dickey-Fuller test.

        Args:
            data: Time series data

        Returns:
            Dictionary with test results
        """
        result = adfuller(data.dropna())
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05  # p-value < 0.05 means stationary
        }

    def fit(self, data: pd.Series) -> 'ARIMAForecaster':
        """
        Fit ARIMA model to historical data.

        Args:
            data: Historical price series

        Returns:
            Self for method chaining
        """
        self.history = data.copy()

        # Fit ARIMA model
        self.model = ARIMA(data, order=self.order)
        self.fitted_model = self.model.fit()

        return self

    def predict(self, steps: int = 1) -> pd.Series:
        """
        Forecast future values.

        Args:
            steps: Number of steps ahead to forecast

        Returns:
            Forecasted values
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")

        forecast = self.fitted_model.forecast(steps=steps)
        return forecast

    def predict_with_confidence(
        self,
        steps: int = 1,
        alpha: float = 0.05
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Forecast with confidence intervals.

        Args:
            steps: Number of steps ahead
            alpha: Significance level (0.05 for 95% CI)

        Returns:
            Tuple of (forecast, confidence_intervals)
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")

        forecast_result = self.fitted_model.get_forecast(steps=steps)
        forecast = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=alpha)

        return forecast, conf_int

    def get_residuals(self) -> pd.Series:
        """Get model residuals for diagnostics."""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        return self.fitted_model.resid

    def get_aic(self) -> float:
        """Get Akaike Information Criterion."""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        return self.fitted_model.aic

    def get_bic(self) -> float:
        """Get Bayesian Information Criterion."""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        return self.fitted_model.bic

    def summary(self) -> str:
        """Get model summary."""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        return str(self.fitted_model.summary())

    @staticmethod
    def auto_select_order(
        data: pd.Series,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5
    ) -> Tuple[int, int, int]:
        """
        Automatically select best ARIMA order using AIC.

        Args:
            data: Time series data
            max_p: Maximum AR order to try
            max_d: Maximum differencing order
            max_q: Maximum MA order to try

        Returns:
            Best (p, d, q) order
        """
        best_aic = np.inf
        best_order = None
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except Exception:
                        continue

        return best_order if best_order else (1, 1, 1)

    def rolling_forecast(
        self,
        data: pd.Series,
        train_size: int,
        steps: int = 1
    ) -> pd.Series:
        """
        Perform rolling window forecast.

        Args:
            data: Full time series
            train_size: Initial training window size
            steps: Steps ahead to forecast

        Returns:
            Series of forecasts
        """
        forecasts: List[float] = []
        history = data[:train_size].tolist()
        for i in range(train_size, len(data)):
            # Fit model on history
            model = ARIMA(history, order=self.order)
            fitted = model.fit()

            # Forecast
            forecast = fitted.forecast(steps=steps)[0]
            forecasts.append(forecast)

            # Update history
            history.append(data[i])
        return pd.Series(forecasts, index=data.index[train_size:])
