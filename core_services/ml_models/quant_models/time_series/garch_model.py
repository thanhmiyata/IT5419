"""
GARCH Model for Volatility Forecasting
=======================================

Generalized AutoRegressive Conditional Heteroskedasticity model.
"""

from typing import Dict

import numpy as np
import pandas as pd
from arch import arch_model


class GARCHVolatilityModel:
    """GARCH model for volatility forecasting."""

    def __init__(
        self,
        vol: str = 'GARCH',
        p: int = 1,
        q: int = 1,
        mean: str = 'Constant',
        dist: str = 'normal'
    ):
        """
        Initialize GARCH model.

        Args:
            vol: Volatility model type ('GARCH', 'EGARCH', 'GJR')
            p: GARCH order (lag order of volatility)
            q: ARCH order (lag order of squared returns)
            mean: Mean model ('Constant', 'Zero', 'AR')
            dist: Error distribution ('normal', 't', 'ged')
        """
        self.vol = vol
        self.p = p
        self.q = q
        self.mean = mean
        self.dist = dist
        self.model = None
        self.fitted_model = None

    def fit(self, returns: pd.Series) -> 'GARCHVolatilityModel':
        """
        Fit GARCH model to return series.

        Args:
            returns: Return series (usually percentage returns)

        Returns:
            Self for method chaining
        """
        # Create GARCH model
        self.model = arch_model(
            returns,
            vol=self.vol,
            p=self.p,
            q=self.q,
            mean=self.mean,
            dist=self.dist
        )

        # Fit model
        self.fitted_model = self.model.fit(disp='off')

        return self

    def forecast_volatility(
        self,
        horizon: int = 1
    ) -> pd.DataFrame:
        """
        Forecast conditional volatility.

        Args:
            horizon: Forecast horizon (number of periods)

        Returns:
            DataFrame with volatility forecasts
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")

        forecast = self.fitted_model.forecast(horizon=horizon)
        return forecast.variance

    def get_conditional_volatility(self) -> pd.Series:
        """
        Get fitted conditional volatility (in-sample).

        Returns:
            Series of conditional volatilities
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")

        return self.fitted_model.conditional_volatility

    def get_standardized_residuals(self) -> pd.Series:
        """Get standardized residuals for diagnostics."""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")

        return self.fitted_model.std_resid

    def get_parameters(self) -> Dict[str, float]:
        """
        Get estimated model parameters.

        Returns:
            Dictionary of parameter estimates
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")

        return self.fitted_model.params.to_dict()

    def summary(self) -> str:
        """Get model summary."""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")

        return str(self.fitted_model.summary())

    def calculate_var(
        self,
        horizon: int = 1,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk using GARCH forecast.

        Args:
            horizon: Forecast horizon
            confidence: Confidence level (0.95 for 95%)

        Returns:
            VaR estimate
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")

        # Get volatility forecast
        vol_forecast = self.forecast_volatility(horizon=horizon)
        vol = np.sqrt(vol_forecast.iloc[-1, 0])

        # Calculate VaR (assuming normal distribution)
        from scipy import stats
        z_score = stats.norm.ppf(1 - confidence)
        var = z_score * vol

        return var

    def calculate_es(
        self,
        horizon: int = 1,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Expected Shortfall (CVaR) using GARCH forecast.

        Args:
            horizon: Forecast horizon
            confidence: Confidence level

        Returns:
            Expected Shortfall estimate
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")

        # Get volatility forecast
        vol_forecast = self.forecast_volatility(horizon=horizon)
        vol = np.sqrt(vol_forecast.iloc[-1, 0])

        # Calculate ES (assuming normal distribution)
        from scipy import stats
        z_score = stats.norm.ppf(1 - confidence)
        es = vol * stats.norm.pdf(z_score) / (1 - confidence)

        return -es

    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """
        Calculate log returns from prices.

        Args:
            prices: Price series

        Returns:
            Log return series
        """
        return np.log(prices / prices.shift(1)).dropna() * 100

    def rolling_forecast(
        self,
        returns: pd.Series,
        train_size: int,
        refit_every: int = 20
    ) -> pd.Series:
        """
        Perform rolling window volatility forecast.

        Args:
            returns: Full return series
            train_size: Initial training window size
            refit_every: Refit model every N observations

        Returns:
            Series of volatility forecasts
        """
        forecasts = []
        fitted = None

        for i in range(train_size, len(returns)):
            # Refit model periodically
            if i == train_size or (i - train_size) % refit_every == 0:
                train_data = returns[i - train_size:i]
                model = arch_model(
                    train_data,
                    vol=self.vol,
                    p=self.p,
                    q=self.q,
                    mean=self.mean,
                    dist=self.dist
                )
                fitted = model.fit(disp='off')

            # Forecast next period volatility
            forecast = fitted.forecast(horizon=1)
            vol = np.sqrt(forecast.variance.iloc[-1, 0])
            forecasts.append(vol)

        return pd.Series(forecasts, index=returns.index[train_size:])
