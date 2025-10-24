"""
Kelly Criterion
===============

Optimal position sizing for trading strategies.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class KellyCriterion:
    """Kelly Criterion for position sizing."""

    def __init__(self):
        """Initialize Kelly Criterion calculator."""
        pass

    def calculate_kelly_fraction(
        self,
        win_probability: float,
        win_loss_ratio: float
    ) -> float:
        """
        Calculate Kelly fraction for binary outcomes.

        Args:
            win_probability: Probability of winning (0 to 1)
            win_loss_ratio: Average win / Average loss

        Returns:
            Kelly fraction (percentage of capital to bet)

        Formula:
            f* = p - (1-p)/b
            where p = win probability, b = win/loss ratio
        """
        if win_probability <= 0 or win_probability >= 1:
            raise ValueError("Win probability must be between 0 and 1")

        if win_loss_ratio <= 0:
            raise ValueError("Win/loss ratio must be positive")

        kelly_fraction = win_probability - (1 - win_probability) / win_loss_ratio

        # Kelly fraction should be between 0 and 1
        return max(0, min(kelly_fraction, 1))

    def calculate_fractional_kelly(
        self,
        win_probability: float,
        win_loss_ratio: float,
        fraction: float = 0.5
    ) -> float:
        """
        Calculate fractional Kelly (more conservative).

        Args:
            win_probability: Probability of winning
            win_loss_ratio: Average win / Average loss
            fraction: Fraction of Kelly to use (0.5 = half Kelly)

        Returns:
            Fractional Kelly position size
        """
        full_kelly = self.calculate_kelly_fraction(win_probability, win_loss_ratio)
        return full_kelly * fraction

    def calculate_from_returns(
        self,
        returns: pd.Series
    ) -> float:
        """
        Calculate Kelly fraction from historical returns.

        Args:
            returns: Series of returns (as decimals, e.g., 0.05 = 5%)

        Returns:
            Kelly fraction
        """
        if len(returns) == 0:
            return 0.0

        # Calculate win probability
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        if len(wins) == 0 or len(losses) == 0:
            return 0.0

        win_probability = len(wins) / len(returns)

        # Calculate average win/loss
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())

        if avg_loss == 0:
            return 0.0

        win_loss_ratio = avg_win / avg_loss

        return self.calculate_kelly_fraction(win_probability, win_loss_ratio)

    def calculate_portfolio_kelly(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate Kelly-optimal portfolio weights.

        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns

        Returns:
            Series of optimal portfolio weights

        Formula:
            f* = Σ^(-1) * μ
            where Σ = covariance matrix, μ = expected returns
        """
        # Kelly weights (can be > 1 due to leverage)
        inv_cov = np.linalg.inv(covariance_matrix.values)
        kelly_weights = np.dot(inv_cov, expected_returns.values)

        return pd.Series(kelly_weights, index=expected_returns.index)

    def calculate_constrained_kelly(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        max_position: float = 0.2,
        allow_short: bool = False
    ) -> pd.Series:
        """
        Calculate Kelly weights with constraints.

        Args:
            expected_returns: Expected returns
            covariance_matrix: Covariance matrix
            max_position: Maximum position size per asset
            allow_short: Allow short positions

        Returns:
            Constrained Kelly weights
        """
        n_assets = len(expected_returns)

        # Objective: minimize -expected log growth
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns.values)
            portfolio_variance = np.dot(
                weights,
                np.dot(covariance_matrix.values, weights)
            )
            # Approximate log utility
            return -(portfolio_return - 0.5 * portfolio_variance)

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Sum to 1
        ]

        # Bounds
        if allow_short:
            bounds = [(-max_position, max_position) for _ in range(n_assets)]
        else:
            bounds = [(0, max_position) for _ in range(n_assets)]

        # Initial guess
        initial_weights = np.ones(n_assets) / n_assets

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            return pd.Series(result.x, index=expected_returns.index)
        else:
            # Return equal weights if optimization fails
            return pd.Series(initial_weights, index=expected_returns.index)

    def calculate_optimal_leverage(
        self,
        expected_return: float,
        volatility: float,
        risk_free_rate: float = 0.0
    ) -> float:
        """
        Calculate optimal leverage using Kelly.

        Args:
            expected_return: Expected return of strategy
            volatility: Standard deviation of returns
            risk_free_rate: Risk-free rate

        Returns:
            Optimal leverage ratio

        Formula:
            L* = (μ - r) / σ²
        """
        if volatility == 0:
            return 0.0

        leverage = (expected_return - risk_free_rate) / (volatility ** 2)

        return max(0, leverage)

    def simulate_growth(
        self,
        returns: pd.Series,
        kelly_fraction: float,
        initial_capital: float = 10000,
        fractional: float = 1.0
    ) -> pd.DataFrame:
        """
        Simulate portfolio growth using Kelly sizing.

        Args:
            returns: Series of returns
            kelly_fraction: Kelly fraction to use
            initial_capital: Starting capital
            fractional: Fraction of Kelly (0.5 = half Kelly)

        Returns:
            DataFrame with portfolio value over time
        """
        position_size = kelly_fraction * fractional
        capital = initial_capital
        portfolio_values = [capital]

        for ret in returns:
            # Kelly position sizing
            capital = capital * (1 + position_size * ret)
            portfolio_values.append(capital)

        return pd.DataFrame({
            'capital': portfolio_values[:-1],
            'returns': returns.values,
            'position_size': position_size,
            'new_capital': portfolio_values[1:]
        }, index=returns.index)

    def backtest_kelly_vs_fixed(
        self,
        returns: pd.Series,
        fixed_fraction: float = 0.1
    ) -> pd.DataFrame:
        """
        Compare Kelly sizing vs fixed fraction.

        Args:
            returns: Series of returns
            fixed_fraction: Fixed position size to compare

        Returns:
            DataFrame comparing both strategies
        """
        kelly_fraction = self.calculate_from_returns(returns)

        # Kelly strategy
        kelly_capital = 10000
        kelly_values = [kelly_capital]

        # Fixed strategy
        fixed_capital = 10000
        fixed_values = [fixed_capital]

        for ret in returns:
            kelly_capital = kelly_capital * (1 + kelly_fraction * ret)
            kelly_values.append(kelly_capital)

            fixed_capital = fixed_capital * (1 + fixed_fraction * ret)
            fixed_values.append(fixed_capital)

        return pd.DataFrame({
            'kelly_portfolio': kelly_values[1:],
            'fixed_portfolio': fixed_values[1:],
            'kelly_fraction': kelly_fraction,
            'fixed_fraction': fixed_fraction
        }, index=returns.index)

    def calculate_drawdown_probability(
        self,
        kelly_fraction: float,
        drawdown_pct: float,
        win_probability: float,
        win_loss_ratio: float
    ) -> float:
        """
        Estimate probability of drawdown.

        Args:
            kelly_fraction: Kelly fraction being used
            drawdown_pct: Drawdown percentage (e.g., 0.2 for 20%)
            win_probability: Win probability
            win_loss_ratio: Win/loss ratio

        Returns:
            Approximate probability of drawdown
        """
        # Simplified approximation
        # More accurate calculation would require simulation

        if kelly_fraction == 0:
            return 0.0

        # Expected number of consecutive losses for drawdown
        avg_loss = kelly_fraction / win_loss_ratio
        num_losses = np.log(1 - drawdown_pct) / np.log(1 - avg_loss)

        # Probability of that many consecutive losses
        prob = (1 - win_probability) ** num_losses

        return prob

    def recommend_position_size(
        self,
        returns: pd.Series,
        risk_tolerance: str = 'moderate'
    ) -> dict:
        """
        Recommend position size based on risk tolerance.

        Args:
            returns: Historical returns
            risk_tolerance: 'conservative', 'moderate', or 'aggressive'

        Returns:
            Dictionary with recommendations
        """
        full_kelly = self.calculate_from_returns(returns)

        # Risk tolerance mapping
        fractions = {
            'conservative': 0.25,   # Quarter Kelly
            'moderate': 0.5,         # Half Kelly
            'aggressive': 0.75       # Three-quarter Kelly
        }

        fraction = fractions.get(risk_tolerance, 0.5)
        recommended_size = full_kelly * fraction

        # Calculate statistics
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        return {
            'full_kelly': full_kelly,
            'recommended_size': recommended_size,
            'risk_tolerance': risk_tolerance,
            'fraction_used': fraction,
            'win_probability': len(wins) / len(returns),
            'avg_win': wins.mean() if len(wins) > 0 else 0,
            'avg_loss': losses.mean() if len(losses) > 0 else 0,
            'win_loss_ratio': abs(wins.mean() / losses.mean()) if len(losses) > 0 and losses.mean() != 0 else 0
        }
