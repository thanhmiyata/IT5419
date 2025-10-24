"""
Time-Series Models for Stock Market Analysis
=============================================

Classical statistical time-series models:
- ARIMA (AutoRegressive Integrated Moving Average)
- GARCH (Generalized AutoRegressive Conditional Heteroskedasticity)
- Kalman Filter
"""

from core_services.ml_models.quant_models.time_series.arima_model import ARIMAForecaster
from core_services.ml_models.quant_models.time_series.garch_model import GARCHVolatilityModel
from core_services.ml_models.quant_models.time_series.kalman_filter import KalmanFilterModel

__all__ = [
    "ARIMAForecaster",
    "GARCHVolatilityModel",
    "KalmanFilterModel",
]

__version__ = "1.0.0"
