"""
Markowitz Mean-Variance Optimization
=====================================

Modern Portfolio Theory optimization.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class MarkowitzOptimizer:
    """Markowitz mean-variance portfolio optimization."""

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize Markowitz optimizer.

        Args:
            risk_free_rate: Annual risk-free rate
        """
        self.risk_free_rate = risk_free_rate
        self.weights = None
        self.expected_returns = None
        self.cov_matrix = None

    def calculate_portfolio_stats(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray
    ) -> tuple:
        """
        Calculate portfolio return and volatility.

        Args:
            weights: Asset weights
            expected_returns: Expected returns vector
            cov_matrix: Covariance matrix

        Returns:
            Tuple of (return, volatility)
        """
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        return portfolio_return, portfolio_vol

    def sharpe_ratio(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray
    ) -> float:
        """Calculate portfolio Sharpe ratio."""
        port_return, port_vol = self.calculate_portfolio_stats(
            weights, expected_returns, cov_matrix
        )

        sharpe = (port_return - self.risk_free_rate) / port_vol
        return sharpe

    def _negative_sharpe(self, weights: np.ndarray) -> float:
        """Negative Sharpe for minimization."""
        return -self.sharpe_ratio(weights, self.expected_returns, self.cov_matrix)

    def _portfolio_variance(self, weights: np.ndarray) -> float:
        """Portfolio variance for minimization."""
        return np.dot(weights.T, np.dot(self.cov_matrix, weights))

    def maximum_sharpe_portfolio(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame
    ) -> pd.Series:
        """
        Find maximum Sharpe ratio portfolio.

        Args:
            expected_returns: Expected returns for each asset
            cov_matrix: Covariance matrix

        Returns:
            Series of optimal weights
        """
        self.expected_returns = expected_returns.values
        self.cov_matrix = cov_matrix.values

        n_assets = len(expected_returns)

        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

        # Bounds: 0 <= weight <= 1 (no short selling)
        bounds = tuple((0, 1) for _ in range(n_assets))

        # Initial guess: equal weights
        initial_weights = np.array([1.0 / n_assets] * n_assets)

        # Optimize
        result = minimize(
            self._negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        self.weights = result.x

        return pd.Series(self.weights, index=expected_returns.index)

    def minimum_variance_portfolio(
        self,
        cov_matrix: pd.DataFrame
    ) -> pd.Series:
        """
        Find minimum variance portfolio.

        Args:
            cov_matrix: Covariance matrix

        Returns:
            Series of optimal weights
        """
        self.cov_matrix = cov_matrix.values
        n_assets = len(cov_matrix)

        # Constraints
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = np.array([1.0 / n_assets] * n_assets)

        # Optimize
        result = minimize(
            self._portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        self.weights = result.x

        return pd.Series(self.weights, index=cov_matrix.columns)

    def efficient_frontier(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        num_portfolios: int = 100
    ) -> pd.DataFrame:
        """
        Generate efficient frontier.

        Args:
            expected_returns: Expected returns
            cov_matrix: Covariance matrix
            num_portfolios: Number of portfolios to generate

        Returns:
            DataFrame with frontier portfolios
        """
        self.expected_returns = expected_returns.values
        self.cov_matrix = cov_matrix.values

        n_assets = len(expected_returns)
        results = []

        # Target returns from min to max
        min_return = expected_returns.min()
        max_return = expected_returns.max()
        target_returns = np.linspace(min_return, max_return, num_portfolios)

        for target_return in target_returns:
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.dot(x, self.expected_returns) - target_return}
            ]

            bounds = tuple((0, 1) for _ in range(n_assets))
            initial_weights = np.array([1.0 / n_assets] * n_assets)

            # Optimize for minimum variance at target return
            result = minimize(
                self._portfolio_variance,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            if result.success:
                weights = result.x
                portfolio_return, portfolio_vol = self.calculate_portfolio_stats(
                    weights, self.expected_returns, self.cov_matrix
                )

                sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol

                results.append({
                    'return': portfolio_return,
                    'volatility': portfolio_vol,
                    'sharpe': sharpe,
                    'weights': weights
                })

        return pd.DataFrame(results)

    def target_return_portfolio(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        target_return: float
    ) -> pd.Series:
        """
        Find minimum variance portfolio for target return.

        Args:
            expected_returns: Expected returns
            cov_matrix: Covariance matrix
            target_return: Desired portfolio return

        Returns:
            Series of optimal weights
        """
        self.expected_returns = expected_returns.values
        self.cov_matrix = cov_matrix.values

        n_assets = len(expected_returns)

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.dot(x, self.expected_returns) - target_return}
        ]

        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = np.array([1.0 / n_assets] * n_assets)

        # Optimize
        result = minimize(
            self._portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        self.weights = result.x

        return pd.Series(self.weights, index=expected_returns.index)
