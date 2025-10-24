"""
Risk Parity Portfolio Optimization
===================================

Equal risk contribution portfolio.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class RiskParityOptimizer:
    """Risk parity portfolio optimization."""

    def __init__(self):
        """Initialize risk parity optimizer."""
        self.weights = None
        self.cov_matrix = None

    def calculate_risk_contribution(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate risk contribution of each asset.

        Args:
            weights: Asset weights
            cov_matrix: Covariance matrix

        Returns:
            Array of risk contributions
        """
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        # Marginal contribution to risk
        marginal_contrib = np.dot(cov_matrix, weights)

        # Risk contribution
        risk_contrib = weights * marginal_contrib / portfolio_vol

        return risk_contrib

    def _risk_parity_objective(self, weights: np.ndarray) -> float:
        """
        Objective function for risk parity.

        Minimizes the sum of squared differences between risk contributions.
        """
        risk_contrib = self.calculate_risk_contribution(weights, self.cov_matrix)

        # Target: equal risk contribution
        target_risk = np.ones(len(weights)) / len(weights)

        # Sum of squared differences
        diff = risk_contrib - target_risk * risk_contrib.sum()

        return np.sum(diff ** 2)

    def optimize(
        self,
        cov_matrix: pd.DataFrame
    ) -> pd.Series:
        """
        Find risk parity portfolio weights.

        Args:
            cov_matrix: Covariance matrix

        Returns:
            Series of optimal weights
        """
        self.cov_matrix = cov_matrix.values
        n_assets = len(cov_matrix)

        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

        # Bounds: 0 <= weight <= 1
        bounds = tuple((0, 1) for _ in range(n_assets))

        # Initial guess: equal weights
        initial_weights = np.array([1.0 / n_assets] * n_assets)

        # Optimize
        result = minimize(
            self._risk_parity_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        self.weights = result.x

        return pd.Series(self.weights, index=cov_matrix.columns)

    def inverse_volatility_weights(
        self,
        volatilities: pd.Series
    ) -> pd.Series:
        """
        Simple risk parity using inverse volatility.

        Args:
            volatilities: Asset volatilities

        Returns:
            Series of weights
        """
        inv_vol = 1.0 / volatilities
        weights = inv_vol / inv_vol.sum()

        return weights

    def get_risk_contributions(
        self,
        weights: pd.Series,
        cov_matrix: pd.DataFrame
    ) -> pd.Series:
        """
        Get risk contribution for given weights.

        Args:
            weights: Asset weights
            cov_matrix: Covariance matrix

        Returns:
            Series of risk contributions
        """
        risk_contrib = self.calculate_risk_contribution(
            weights.values,
            cov_matrix.values
        )

        return pd.Series(risk_contrib, index=weights.index)
