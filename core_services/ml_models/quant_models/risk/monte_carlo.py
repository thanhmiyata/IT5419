"""
Monte Carlo Simulation for Risk Management
===========================================

Monte Carlo methods for portfolio risk assessment.
"""

import numpy as np
import pandas as pd
from scipy import stats


class MonteCarloSimulator:
    """Monte Carlo simulation for risk management."""

    def __init__(
        self,
        n_simulations: int = 10000,
        random_state: int = 42
    ):
        """
        Initialize Monte Carlo simulator.

        Args:
            n_simulations: Number of Monte Carlo simulations
            random_state: Random seed
        """
        self.n_simulations = n_simulations
        self.random_state = random_state
        np.random.seed(random_state)

    def simulate_returns(
        self,
        mean_return: float,
        volatility: float,
        n_periods: int,
        distribution: str = 'normal'
    ) -> np.ndarray:
        """
        Simulate returns using Monte Carlo.

        Args:
            mean_return: Expected return per period
            volatility: Standard deviation of returns
            n_periods: Number of periods to simulate
            distribution: 'normal', 't', or 'empirical'

        Returns:
            Array of simulated returns [n_simulations, n_periods]
        """
        if distribution == 'normal':
            returns = np.random.normal(
                mean_return,
                volatility,
                size=(self.n_simulations, n_periods)
            )
        elif distribution == 't':
            # Student's t-distribution (fatter tails)
            df = 5  # Degrees of freedom
            returns = stats.t.rvs(
                df,
                loc=mean_return,
                scale=volatility,
                size=(self.n_simulations, n_periods)
            )
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        return returns

    def simulate_prices(
        self,
        initial_price: float,
        mean_return: float,
        volatility: float,
        n_periods: int,
        distribution: str = 'normal'
    ) -> np.ndarray:
        """
        Simulate price paths using geometric Brownian motion.

        Args:
            initial_price: Starting price
            mean_return: Expected return (drift)
            volatility: Volatility
            n_periods: Number of periods
            distribution: Distribution type

        Returns:
            Array of simulated prices [n_simulations, n_periods + 1]
        """
        # Simulate returns
        returns = self.simulate_returns(
            mean_return,
            volatility,
            n_periods,
            distribution
        )

        # Convert to prices
        prices = np.zeros((self.n_simulations, n_periods + 1))
        prices[:, 0] = initial_price

        for t in range(1, n_periods + 1):
            prices[:, t] = prices[:, t - 1] * (1 + returns[:, t - 1])

        return prices

    def simulate_portfolio(
        self,
        initial_capital: float,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        n_periods: int
    ) -> np.ndarray:
        """
        Simulate portfolio value paths.

        Args:
            initial_capital: Starting capital
            weights: Portfolio weights
            expected_returns: Expected returns for each asset
            cov_matrix: Covariance matrix
            n_periods: Number of periods

        Returns:
            Array of simulated portfolio values
        """
        # Simulate correlated returns
        portfolio_values = np.zeros((self.n_simulations, n_periods + 1))
        portfolio_values[:, 0] = initial_capital

        for t in range(1, n_periods + 1):
            # Generate correlated returns
            asset_returns = np.random.multivariate_normal(
                expected_returns,
                cov_matrix,
                size=self.n_simulations
            )

            # Portfolio return
            portfolio_return = np.dot(asset_returns, weights)

            # Update portfolio value
            portfolio_values[:, t] = portfolio_values[:, t - 1] * (1 + portfolio_return)

        return portfolio_values

    def calculate_var(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk from simulated returns.

        Args:
            returns: Simulated returns
            confidence_level: Confidence level (e.g., 0.95)

        Returns:
            VaR value
        """
        # Returns can be 1D or 2D
        if returns.ndim == 2:
            # Take final period returns
            final_returns = returns[:, -1]
        else:
            final_returns = returns

        var = np.percentile(final_returns, (1 - confidence_level) * 100)

        return var

    def calculate_cvar(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).

        Args:
            returns: Simulated returns
            confidence_level: Confidence level

        Returns:
            CVaR value
        """
        var = self.calculate_var(returns, confidence_level)

        # Returns below VaR
        if returns.ndim == 2:
            final_returns = returns[:, -1]
        else:
            final_returns = returns

        tail_losses = final_returns[final_returns <= var]

        cvar = tail_losses.mean() if len(tail_losses) > 0 else var

        return cvar

    def calculate_maximum_drawdown(
        self,
        price_paths: np.ndarray
    ) -> np.ndarray:
        """
        Calculate maximum drawdown for each simulation.

        Args:
            price_paths: Simulated price paths

        Returns:
            Array of maximum drawdowns for each simulation
        """
        max_drawdowns = np.zeros(self.n_simulations)

        for i in range(self.n_simulations):
            prices = price_paths[i, :]
            running_max = np.maximum.accumulate(prices)
            drawdown = (prices - running_max) / running_max
            max_drawdowns[i] = drawdown.min()

        return max_drawdowns

    def probability_of_loss(
        self,
        returns: np.ndarray,
        threshold: float = 0.0
    ) -> float:
        """
        Calculate probability of loss exceeding threshold.

        Args:
            returns: Simulated returns
            threshold: Loss threshold (negative value)

        Returns:
            Probability of loss
        """
        if returns.ndim == 2:
            final_returns = returns[:, -1]
        else:
            final_returns = returns

        prob_loss = (final_returns < threshold).sum() / len(final_returns)

        return prob_loss

    def expected_final_value(
        self,
        price_paths: np.ndarray
    ) -> dict:
        """
        Calculate statistics of final values.

        Args:
            price_paths: Simulated price paths

        Returns:
            Dictionary with statistics
        """
        final_values = price_paths[:, -1]

        return {
            'mean': final_values.mean(),
            'median': np.median(final_values),
            'std': final_values.std(),
            'min': final_values.min(),
            'max': final_values.max(),
            'percentile_5': np.percentile(final_values, 5),
            'percentile_25': np.percentile(final_values, 25),
            'percentile_75': np.percentile(final_values, 75),
            'percentile_95': np.percentile(final_values, 95)
        }

    def simulate_with_rebalancing(
        self,
        initial_capital: float,
        target_weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        n_periods: int,
        rebalance_frequency: int = 20
    ) -> np.ndarray:
        """
        Simulate portfolio with periodic rebalancing.

        Args:
            initial_capital: Starting capital
            target_weights: Target portfolio weights
            expected_returns: Expected returns
            cov_matrix: Covariance matrix
            n_periods: Number of periods
            rebalance_frequency: Rebalance every N periods

        Returns:
            Portfolio value paths
        """
        portfolio_values = np.zeros((self.n_simulations, n_periods + 1))
        portfolio_values[:, 0] = initial_capital

        for sim in range(self.n_simulations):
            current_value = initial_capital
            current_weights = target_weights.copy()

            for t in range(1, n_periods + 1):
                # Generate returns
                asset_returns = np.random.multivariate_normal(
                    expected_returns,
                    cov_matrix
                )

                # Calculate new weights after returns
                current_weights = current_weights * (1 + asset_returns)
                current_weights = current_weights / current_weights.sum()

                # Update value
                portfolio_return = np.dot(
                    current_weights / (1 + asset_returns),
                    asset_returns
                )
                current_value = current_value * (1 + portfolio_return)

                # Rebalance
                if t % rebalance_frequency == 0:
                    current_weights = target_weights.copy()

                portfolio_values[sim, t] = current_value

        return portfolio_values

    def stress_test(
        self,
        initial_capital: float,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        stress_scenarios: list
    ) -> pd.DataFrame:
        """
        Perform stress testing on portfolio.

        Args:
            initial_capital: Starting capital
            weights: Portfolio weights
            expected_returns: Expected returns
            cov_matrix: Covariance matrix
            stress_scenarios: List of stress scenarios
                Each scenario: {'name': str, 'return_shock': array, 'vol_multiplier': float}

        Returns:
            DataFrame with stress test results
        """
        results = []

        for scenario in stress_scenarios:
            # Apply shock
            shocked_returns = expected_returns + scenario['return_shock']
            shocked_cov = cov_matrix * scenario['vol_multiplier']

            # Calculate portfolio impact
            portfolio_return = np.dot(weights, shocked_returns)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(shocked_cov, weights)))

            final_value = initial_capital * (1 + portfolio_return)
            loss = initial_capital - final_value

            results.append({
                'scenario': scenario['name'],
                'portfolio_return': portfolio_return,
                'portfolio_volatility': portfolio_vol,
                'final_value': final_value,
                'loss': loss,
                'loss_pct': loss / initial_capital
            })

        return pd.DataFrame(results)

    def calculate_confidence_intervals(
        self,
        price_paths: np.ndarray,
        confidence_levels: list = [0.90, 0.95, 0.99]
    ) -> pd.DataFrame:
        """
        Calculate confidence intervals over time.

        Args:
            price_paths: Simulated price paths
            confidence_levels: List of confidence levels

        Returns:
            DataFrame with confidence intervals
        """
        n_periods = price_paths.shape[1] - 1
        results = []

        for t in range(n_periods + 1):
            period_values = price_paths[:, t]

            ci = {'period': t, 'mean': period_values.mean()}

            for level in confidence_levels:
                alpha = 1 - level
                lower = np.percentile(period_values, alpha / 2 * 100)
                upper = np.percentile(period_values, (1 - alpha / 2) * 100)

                ci[f'lower_{int(level * 100)}'] = lower
                ci[f'upper_{int(level * 100)}'] = upper

            results.append(ci)

        return pd.DataFrame(results)

    def optimal_stopping_time(
        self,
        price_paths: np.ndarray,
        target_return: float
    ) -> dict:
        """
        Calculate optimal stopping time to achieve target return.

        Args:
            price_paths: Simulated price paths
            target_return: Target return threshold

        Returns:
            Dictionary with stopping time statistics
        """
        initial_price = price_paths[:, 0].mean()
        target_price = initial_price * (1 + target_return)

        stopping_times = []

        for sim in range(self.n_simulations):
            # Find first time price exceeds target
            exceeded = np.where(price_paths[sim, :] >= target_price)[0]

            if len(exceeded) > 0:
                stopping_times.append(exceeded[0])
            else:
                # Never reached
                stopping_times.append(price_paths.shape[1])

        stopping_times = np.array(stopping_times)

        return {
            'mean_stopping_time': stopping_times.mean(),
            'median_stopping_time': np.median(stopping_times),
            'probability_reached': (stopping_times < price_paths.shape[1]).sum() / self.n_simulations,
            'min_time': stopping_times.min(),
            'max_time': stopping_times.max()
        }
