"""
Hawkes Process for High-Frequency Trading
==========================================

Self-exciting point process for modeling order flow and price jumps.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize


class HawkesProcess:
    """Hawkes process for HFT modeling."""

    def __init__(self):
        """Initialize Hawkes process."""
        self.params = None

    def simulate(
        self,
        T: float,
        mu: float,
        alpha: float,
        beta: float,
        random_state: int = 42
    ) -> np.ndarray:
        """
        Simulate univariate Hawkes process.

        Args:
            T: Time horizon
            mu: Background intensity
            alpha: Self-excitation parameter
            beta: Decay rate
            random_state: Random seed

        Returns:
            Array of event times
        """
        np.random.seed(random_state)

        events = []
        t = 0
        lambda_star = mu

        while t < T:
            # Generate next event time
            u = np.random.uniform(0, 1)
            t = t - np.log(u) / lambda_star

            if t < T:
                # Accept/reject
                lambda_t = self._intensity(t, events, mu, alpha, beta)
                d = np.random.uniform(0, 1)

                if d * lambda_star <= lambda_t:
                    # Accept event
                    events.append(t)
                    lambda_star = lambda_t + alpha
                else:
                    lambda_star = lambda_t

        return np.array(events)

    def _intensity(
        self,
        t: float,
        events: list,
        mu: float,
        alpha: float,
        beta: float
    ) -> float:
        """
        Calculate intensity at time t.

        Args:
            t: Current time
            events: List of past event times
            mu: Background intensity
            alpha: Self-excitation
            beta: Decay rate

        Returns:
            Intensity value
        """
        if len(events) == 0:
            return mu

        # Sum of exponential kernels
        past_events = np.array([e for e in events if e < t])

        if len(past_events) == 0:
            return mu

        excitation = alpha * np.sum(np.exp(-beta * (t - past_events)))

        return mu + excitation

    def fit(
        self,
        event_times: np.ndarray,
        T: float = None
    ) -> dict:
        """
        Fit Hawkes process to observed events.

        Args:
            event_times: Array of event times
            T: End time (max event time if None)

        Returns:
            Dictionary with fitted parameters
        """
        if T is None:
            T = event_times[-1]

        # Initial parameters
        initial_params = [0.5, 0.5, 1.0]  # [mu, alpha, beta]

        # Negative log-likelihood
        def neg_log_likelihood(params):
            mu, alpha, beta = params

            # Constraints: mu > 0, alpha > 0, beta > 0, alpha < beta (stability)
            if mu <= 0 or alpha <= 0 or beta <= 0 or alpha >= beta:
                return 1e10

            # Log-likelihood
            ll = 0

            # Sum over events
            for i, t_i in enumerate(event_times):
                lambda_i = mu + alpha * np.sum(
                    np.exp(-beta * (t_i - event_times[:i]))
                )
                ll += np.log(lambda_i)

            # Integral term
            integral = mu * T

            for t_i in event_times:
                integral += (alpha / beta) * (1 - np.exp(-beta * (T - t_i)))

            ll -= integral

            return -ll

        # Optimize
        result = minimize(
            neg_log_likelihood,
            initial_params,
            method='L-BFGS-B',
            bounds=[(0.01, None), (0.01, None), (0.01, None)]
        )

        if result.success:
            mu, alpha, beta = result.x

            self.params = {
                'mu': mu,
                'alpha': alpha,
                'beta': beta,
                'branching_ratio': alpha / beta,
                'log_likelihood': -result.fun
            }

            return self.params
        else:
            raise ValueError("Optimization failed")

    def calculate_intensity(
        self,
        t: float,
        event_times: np.ndarray
    ) -> float:
        """
        Calculate intensity at given time.

        Args:
            t: Time point
            event_times: Historical event times

        Returns:
            Intensity value
        """
        if self.params is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return self._intensity(
            t,
            event_times.tolist(),
            self.params['mu'],
            self.params['alpha'],
            self.params['beta']
        )

    def predict_next_event(
        self,
        event_times: np.ndarray,
        n_samples: int = 1000
    ) -> dict:
        """
        Predict distribution of next event time.

        Args:
            event_times: Historical event times
            n_samples: Number of Monte Carlo samples

        Returns:
            Dictionary with prediction statistics
        """
        if self.params is None:
            raise ValueError("Model not fitted. Call fit() first.")

        current_time = event_times[-1]

        # Simulate next event times
        next_times = []

        for _ in range(n_samples):
            t = current_time
            lambda_star = self.calculate_intensity(current_time, event_times)

            while True:
                u = np.random.uniform(0, 1)
                t = t - np.log(u) / lambda_star

                lambda_t = self.calculate_intensity(t, event_times)
                d = np.random.uniform(0, 1)

                if d * lambda_star <= lambda_t:
                    next_times.append(t - current_time)
                    break
                else:
                    lambda_star = lambda_t

        next_times = np.array(next_times)

        return {
            'mean_time': next_times.mean(),
            'median_time': np.median(next_times),
            'std_time': next_times.std(),
            'percentile_25': np.percentile(next_times, 25),
            'percentile_75': np.percentile(next_times, 75)
        }

    def calculate_branching_ratio(self) -> float:
        """
        Calculate branching ratio (stability measure).

        Returns:
            Branching ratio (< 1 for stable process)
        """
        if self.params is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.params['branching_ratio']

    def is_stable(self) -> bool:
        """
        Check if process is stable.

        Returns:
            True if stable (branching ratio < 1)
        """
        return self.calculate_branching_ratio() < 1

    def estimate_clustering(
        self,
        event_times: np.ndarray,
        window: float = 1.0
    ) -> pd.DataFrame:
        """
        Estimate event clustering over time.

        Args:
            event_times: Event times
            window: Time window for clustering

        Returns:
            DataFrame with clustering measures
        """
        clusters = []

        for i, t in enumerate(event_times):
            # Count events in window
            in_window = np.sum(
                (event_times > t) & (event_times <= t + window)
            )

            # Calculate local intensity
            local_intensity = in_window / window

            clusters.append({
                'time': t,
                'events_in_window': in_window,
                'local_intensity': local_intensity
            })

        return pd.DataFrame(clusters)

    def multivariate_simulate(
        self,
        T: float,
        mu: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        n_processes: int = 2
    ) -> list:
        """
        Simulate multivariate Hawkes process.

        Args:
            T: Time horizon
            mu: Background intensities (n_processes,)
            alpha: Cross-excitation matrix (n_processes, n_processes)
            beta: Decay rates (n_processes, n_processes)
            n_processes: Number of processes

        Returns:
            List of event time arrays for each process
        """
        events = [[] for _ in range(n_processes)]
        t = 0
        lambda_star = mu.sum()

        while t < T:
            # Next event time
            u = np.random.uniform(0, 1)
            t = t - np.log(u) / lambda_star

            if t >= T:
                break

            # Which process?
            intensities = np.zeros(n_processes)

            for i in range(n_processes):
                intensities[i] = mu[i]

                for j in range(n_processes):
                    past_j = np.array([e for e in events[j] if e < t])
                    if len(past_j) > 0:
                        intensities[i] += alpha[i, j] * np.sum(
                            np.exp(-beta[i, j] * (t - past_j))
                        )

            # Normalize to get probabilities
            total_intensity = intensities.sum()

            if total_intensity == 0:
                continue

            probs = intensities / total_intensity

            # Select process
            process_idx = np.random.choice(n_processes, p=probs)

            # Accept/reject
            d = np.random.uniform(0, 1)

            if d * lambda_star <= total_intensity:
                events[process_idx].append(t)
                lambda_star = total_intensity + alpha[process_idx, :].sum()
            else:
                lambda_star = total_intensity

        return [np.array(e) for e in events]

    def granger_causality_test(
        self,
        events1: np.ndarray,
        events2: np.ndarray,
        T: float = None
    ) -> dict:
        """
        Test Granger causality between two event processes.

        Args:
            events1: First event series
            events2: Second event series
            T: End time

        Returns:
            Dictionary with test results
        """
        if T is None:
            T = max(events1[-1], events2[-1])

        # Fit univariate model for events2
        self.fit(events2, T)

        # Fit bivariate model (would need multivariate implementation)
        # For simplicity, we'll estimate cross-excitation

        # Check if events1 predicts events2
        # Count events2 shortly after events1

        delta = 0.1  # Time window
        cross_excitation = 0

        for t1 in events1:
            # Count events2 in (t1, t1 + delta]
            count = np.sum((events2 > t1) & (events2 <= t1 + delta))
            cross_excitation += count

        # Expected under independence
        rate2 = len(events2) / T
        expected = len(events1) * rate2 * delta

        # Test statistic (Poisson test)
        if expected > 0:
            test_stat = (cross_excitation - expected) / np.sqrt(expected)
            p_value = 2 * (1 - stats.norm.cdf(abs(test_stat)))
        else:
            test_stat = np.nan
            p_value = np.nan

        return {
            'cross_excitation_count': cross_excitation,
            'expected_count': expected,
            'test_statistic': test_stat,
            'p_value': p_value,
            'reject_independence': p_value < 0.05 if not np.isnan(p_value) else None
        }

    def estimate_market_impact(
        self,
        trade_times: np.ndarray,
        price_changes: np.ndarray
    ) -> dict:
        """
        Estimate market impact using Hawkes framework.

        Args:
            trade_times: Times of trades
            price_changes: Price changes at each trade

        Returns:
            Dictionary with impact estimates
        """
        # Fit Hawkes to trade arrival
        self.fit(trade_times)

        # Estimate impact as function of intensity
        intensities = []

        for i, t in enumerate(trade_times):
            intensity = self.calculate_intensity(t, trade_times[:i])
            intensities.append(intensity)

        intensities = np.array(intensities)

        # Correlation between intensity and price impact
        correlation = np.corrcoef(intensities, np.abs(price_changes))[0, 1]

        # Linear regression
        from scipy.stats import linregress

        slope, intercept, r_value, p_value, std_err = linregress(
            intensities,
            np.abs(price_changes)
        )

        return {
            'correlation': correlation,
            'impact_slope': slope,
            'impact_intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'std_error': std_err
        }
