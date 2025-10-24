"""
Kalman Filter for State Estimation
===================================

Kalman filter for pairs trading and dynamic beta estimation.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd


class KalmanFilterModel:
    """Kalman Filter for time-varying parameter estimation."""

    def __init__(
        self,
        transition_covariance: float = 0.01,
        observation_covariance: float = 1.0
    ):
        """
        Initialize Kalman Filter.

        Args:
            transition_covariance: Process noise covariance (Q)
            observation_covariance: Measurement noise covariance (R)
        """
        self.transition_cov = transition_covariance
        self.observation_cov = observation_covariance

        # State estimates
        self.state_mean = None
        self.state_covariance = None

        # History
        self.state_means = []
        self.state_covariances = []

    def initialize(self, initial_state: np.ndarray, initial_cov: float = 1.0):
        """
        Initialize filter state.

        Args:
            initial_state: Initial state estimate
            initial_cov: Initial state covariance
        """
        self.state_mean = initial_state
        self.state_covariance = np.eye(len(initial_state)) * initial_cov

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step (time update).

        Returns:
            Tuple of (predicted_state, predicted_covariance)
        """
        # State transition (assume identity for now - random walk)
        predicted_state = self.state_mean

        # Covariance prediction
        predicted_cov = self.state_covariance + \
            np.eye(len(self.state_mean)) * self.transition_cov

        return predicted_state, predicted_cov

    def update(
        self,
        observation: float,
        observation_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step (measurement update).

        Args:
            observation: Observed value
            observation_matrix: Observation matrix (H)

        Returns:
            Tuple of (updated_state, updated_covariance)
        """
        # Predict
        pred_state, pred_cov = self.predict()

        # Innovation (measurement residual)
        innovation = observation - np.dot(observation_matrix, pred_state)

        # Innovation covariance
        innovation_cov = np.dot(
            np.dot(observation_matrix, pred_cov),
            observation_matrix.T
        ) + self.observation_cov

        # Kalman gain
        kalman_gain = np.dot(
            np.dot(pred_cov, observation_matrix.T),
            1.0 / innovation_cov
        )

        # Update state
        self.state_mean = pred_state + kalman_gain * innovation

        # Update covariance
        self.state_covariance = pred_cov - np.outer(
            kalman_gain,
            np.dot(observation_matrix, pred_cov)
        )

        # Store history
        self.state_means.append(self.state_mean.copy())
        self.state_covariances.append(self.state_covariance.copy())

        return self.state_mean, self.state_covariance

    def fit_linear_regression(
        self,
        y: pd.Series,
        X: pd.Series
    ) -> pd.DataFrame:
        """
        Estimate time-varying linear regression: y = beta * X + alpha.

        Args:
            y: Dependent variable (e.g., stock A returns)
            X: Independent variable (e.g., stock B returns)

        Returns:
            DataFrame with time-varying coefficients (beta, alpha)
        """
        # Align series
        data = pd.DataFrame({'y': y, 'X': X}).dropna()

        # Initialize state [beta, alpha]
        self.initialize(initial_state=np.array([1.0, 0.0]))

        betas = []
        alphas = []

        for idx in range(len(data)):
            # Observation matrix: [X_t, 1]
            obs_matrix = np.array([data['X'].iloc[idx], 1.0])

            # Update with observation
            self.update(
                observation=data['y'].iloc[idx],
                observation_matrix=obs_matrix
            )

            betas.append(self.state_mean[0])
            alphas.append(self.state_mean[1])

        return pd.DataFrame({
            'beta': betas,
            'alpha': alphas
        }, index=data.index)

    def get_hedge_ratio(
        self,
        stock1: pd.Series,
        stock2: pd.Series
    ) -> pd.Series:
        """
        Calculate dynamic hedge ratio for pairs trading.

        Args:
            stock1: First stock price series
            stock2: Second stock price series

        Returns:
            Series of hedge ratios (beta)
        """
        coefficients = self.fit_linear_regression(stock1, stock2)
        return coefficients['beta']

    def calculate_spread(
        self,
        stock1: pd.Series,
        stock2: pd.Series,
        hedge_ratio: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Calculate spread for pairs trading.

        Args:
            stock1: First stock price series
            stock2: Second stock price series
            hedge_ratio: Optional pre-calculated hedge ratio

        Returns:
            Spread series
        """
        if hedge_ratio is None:
            hedge_ratio = self.get_hedge_ratio(stock1, stock2)

        # Align data
        data = pd.DataFrame({
            's1': stock1,
            's2': stock2,
            'hr': hedge_ratio
        }).dropna()

        # Calculate spread: stock1 - hedge_ratio * stock2
        spread = data['s1'] - data['hr'] * data['s2']

        return spread

    @staticmethod
    def calculate_z_score(spread: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate rolling z-score of spread.

        Args:
            spread: Spread series
            window: Rolling window size

        Returns:
            Z-score series
        """
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()

        z_score = (spread - rolling_mean) / rolling_std

        return z_score

    def generate_trading_signals(
        self,
        stock1: pd.Series,
        stock2: pd.Series,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Generate pairs trading signals.

        Args:
            stock1: First stock price series
            stock2: Second stock price series
            entry_threshold: Z-score threshold for entry
            exit_threshold: Z-score threshold for exit
            window: Rolling window for z-score calculation

        Returns:
            DataFrame with signals and spread info
        """
        # Calculate spread and z-score
        spread = self.calculate_spread(stock1, stock2)
        z_score = self.calculate_z_score(spread, window=window)

        # Generate signals
        signals = pd.DataFrame(index=z_score.index)
        signals['spread'] = spread
        signals['z_score'] = z_score
        signals['signal'] = 0

        # Long spread when z-score < -entry_threshold
        signals.loc[z_score < -entry_threshold, 'signal'] = 1

        # Short spread when z-score > entry_threshold
        signals.loc[z_score > entry_threshold, 'signal'] = -1

        # Exit when z-score crosses exit_threshold
        signals.loc[abs(z_score) < exit_threshold, 'signal'] = 0

        return signals
