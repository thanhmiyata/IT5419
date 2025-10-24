"""
Black-Litterman Model
=====================

Combines market equilibrium with investor views.
"""

import numpy as np
import pandas as pd


class BlackLittermanModel:
    """Black-Litterman portfolio optimization."""

    def __init__(
        self,
        risk_aversion: float = 2.5,
        tau: float = 0.05
    ):
        """
        Initialize Black-Litterman model.

        Args:
            risk_aversion: Market risk aversion parameter
            tau: Uncertainty in prior (typically 0.01-0.05)
        """
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.posterior_returns = None
        self.posterior_cov = None

    def calculate_equilibrium_returns(
        self,
        market_weights: pd.Series,
        cov_matrix: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate implied equilibrium returns (reverse optimization).

        Args:
            market_weights: Market capitalization weights
            cov_matrix: Covariance matrix

        Returns:
            Series of implied equilibrium returns
        """
        # Pi = delta * Sigma * w_mkt
        equilibrium_returns = self.risk_aversion * np.dot(
            cov_matrix.values,
            market_weights.values
        )

        return pd.Series(equilibrium_returns, index=market_weights.index)

    def calculate_posterior(
        self,
        equilibrium_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        views_matrix: np.ndarray,
        views_returns: np.ndarray,
        views_uncertainty: np.ndarray
    ) -> tuple:
        """
        Calculate Black-Litterman posterior returns.

        Args:
            equilibrium_returns: Market equilibrium returns
            cov_matrix: Covariance matrix
            views_matrix: P matrix (views on assets)
            views_returns: Q vector (expected returns from views)
            views_uncertainty: Omega matrix (uncertainty in views)

        Returns:
            Tuple of (posterior_returns, posterior_covariance)
        """
        # Convert to numpy
        pi = equilibrium_returns.values
        sigma = cov_matrix.values

        # tau * Sigma
        tau_sigma = self.tau * sigma

        # [(tau * Sigma)^-1 + P^T * Omega^-1 * P]^-1
        precision = np.linalg.inv(tau_sigma) + \
            np.dot(np.dot(views_matrix.T, np.linalg.inv(views_uncertainty)), views_matrix)

        posterior_cov = np.linalg.inv(precision)

        # E[R] = [(tau * Sigma)^-1 + P^T * Omega^-1 * P]^-1 *
        #        [(tau * Sigma)^-1 * Pi + P^T * Omega^-1 * Q]
        posterior_returns = np.dot(
            posterior_cov,
            np.dot(np.linalg.inv(tau_sigma), pi)
            + np.dot(np.dot(views_matrix.T, np.linalg.inv(views_uncertainty)),
                     views_returns)
        )

        self.posterior_returns = pd.Series(
            posterior_returns, index=equilibrium_returns.index)
        self.posterior_cov = pd.DataFrame(posterior_cov, index=cov_matrix.index, columns=cov_matrix.columns)
        return self.posterior_returns, self.posterior_cov

    def optimize_portfolio(
        self,
        posterior_returns: pd.Series,
        posterior_cov: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate optimal portfolio weights.

        Args:
            posterior_returns: Posterior expected returns
            posterior_cov: Posterior covariance matrix

        Returns:
            Series of optimal weights
        """
        # w = (delta * Sigma)^-1 * E[R]
        weights = np.dot(
            np.linalg.inv(self.risk_aversion * posterior_cov.values),
            posterior_returns.values
        )

        # Normalize to sum to 1
        weights = weights / weights.sum()

        return pd.Series(weights, index=posterior_returns.index)

    def create_view(
        self,
        asset_names: list,
        view_assets: list,
        view_weights: list
    ) -> np.ndarray:
        """
        Create a view (row of P matrix).

        Args:
            asset_names: All asset names
            view_assets: Assets in this view
            view_weights: Weights for each asset in view

        Returns:
            Array representing the view
        """
        view = np.zeros(len(asset_names))

        for asset, weight in zip(view_assets, view_weights):
            idx = asset_names.index(asset)
            view[idx] = weight

        return view
