"""
Extreme Value Theory (EVT)
===========================

Model tail risks and extreme events in financial returns.
"""

import numpy as np
import pandas as pd
from scipy import stats


class ExtremeValueTheory:
    """Extreme Value Theory for tail risk modeling."""

    def __init__(self):
        """Initialize EVT model."""
        self.params = None
        self.threshold = None

    def fit_gev(
        self,
        data: pd.Series,
        method: str = 'block_maxima',
        block_size: int = 20
    ) -> dict:
        """
        Fit Generalized Extreme Value (GEV) distribution.

        Args:
            data: Return series
            method: 'block_maxima' for GEV
            block_size: Size of blocks for maxima

        Returns:
            Dictionary with GEV parameters
        """
        if method == 'block_maxima':
            # Extract block maxima
            n_blocks = len(data) // block_size
            block_maxima = []

            for i in range(n_blocks):
                block = data.iloc[i * block_size:(i + 1) * block_size]
                block_maxima.append(block.max())

            block_maxima = np.array(block_maxima)

            # Fit GEV distribution
            shape, loc, scale = stats.genextreme.fit(block_maxima)

            self.params = {
                'shape': shape,
                'location': loc,
                'scale': scale,
                'method': 'GEV'
            }

            return self.params

    def fit_gpd(
        self,
        data: pd.Series,
        threshold: float = None,
        threshold_quantile: float = 0.95
    ) -> dict:
        """
        Fit Generalized Pareto Distribution (GPD) for peaks over threshold.

        Args:
            data: Return series (negative for losses)
            threshold: Threshold value (None = auto from quantile)
            threshold_quantile: Quantile for auto threshold

        Returns:
            Dictionary with GPD parameters
        """
        # Use losses (negative returns)
        losses = -data

        # Determine threshold
        if threshold is None:
            threshold = losses.quantile(threshold_quantile)

        self.threshold = threshold

        # Extract exceedances
        exceedances = losses[losses > threshold] - threshold

        if len(exceedances) < 10:
            raise ValueError("Too few exceedances. Lower threshold or use more data.")

        # Fit GPD
        shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)

        # Number of exceedances
        n_exceedances = len(exceedances)
        n_total = len(losses)

        self.params = {
            'shape': shape,
            'scale': scale,
            'threshold': threshold,
            'n_exceedances': n_exceedances,
            'exceedance_rate': n_exceedances / n_total,
            'method': 'GPD'
        }

        return self.params

    def calculate_var_evt(
        self,
        confidence_level: float = 0.99
    ) -> float:
        """
        Calculate VaR using EVT.

        Args:
            confidence_level: Confidence level (e.g., 0.99)

        Returns:
            VaR estimate
        """
        if self.params is None:
            raise ValueError("Model not fitted. Call fit_gpd() or fit_gev() first.")

        if self.params['method'] == 'GPD':
            # GPD-based VaR
            shape = self.params['shape']
            scale = self.params['scale']
            threshold = self.params['threshold']
            exceedance_rate = self.params['exceedance_rate']

            # Probability in tail
            p = 1 - confidence_level

            # GPD quantile
            if abs(shape) < 1e-6:
                # Exponential distribution (shape ≈ 0)
                var = threshold + scale * np.log(exceedance_rate / p)
            else:
                var = threshold + (scale / shape) * (
                    ((exceedance_rate / p) ** shape) - 1
                )

            return var

        elif self.params['method'] == 'GEV':
            # GEV-based VaR
            shape = self.params['shape']
            loc = self.params['location']
            scale = self.params['scale']

            # GEV quantile
            var = stats.genextreme.ppf(confidence_level, shape, loc, scale)

            return var

    def calculate_cvar_evt(
        self,
        confidence_level: float = 0.99
    ) -> float:
        """
        Calculate CVaR (Expected Shortfall) using EVT.

        Args:
            confidence_level: Confidence level

        Returns:
            CVaR estimate
        """
        if self.params is None or self.params['method'] != 'GPD':
            raise ValueError("GPD model not fitted. Call fit_gpd() first.")

        shape = self.params['shape']
        scale = self.params['scale']
        threshold = self.params['threshold']

        var = self.calculate_var_evt(confidence_level)

        # Expected shortfall for GPD
        if shape < 1:
            cvar = (var + scale - shape * threshold) / (1 - shape)
        else:
            # Undefined for shape >= 1
            cvar = np.inf

        return cvar

    def estimate_return_level(
        self,
        return_period: int
    ) -> float:
        """
        Estimate return level (e.g., 100-year event).

        Args:
            return_period: Return period in number of observations

        Returns:
            Return level estimate
        """
        if self.params is None:
            raise ValueError("Model not fitted.")

        # Probability
        p = 1 - 1 / return_period

        if self.params['method'] == 'GEV':
            shape = self.params['shape']
            loc = self.params['location']
            scale = self.params['scale']

            if abs(shape) < 1e-6:
                # Gumbel distribution
                return_level = loc - scale * np.log(-np.log(p))
            else:
                return_level = loc + (scale / shape) * (
                    (-np.log(p)) ** (-shape) - 1
                )

        elif self.params['method'] == 'GPD':
            # Convert to exceedance probability
            exceedance_rate = self.params['exceedance_rate']
            p_exceedance = exceedance_rate * (1 - p)

            return_level = self.calculate_var_evt(1 - p_exceedance)

        return return_level

    def mean_excess_plot_data(
        self,
        data: pd.Series,
        thresholds: np.ndarray = None
    ) -> pd.DataFrame:
        """
        Generate data for mean excess plot (to choose threshold).

        Args:
            data: Return series (negative for losses)
            thresholds: Array of threshold values to test

        Returns:
            DataFrame with mean excess values
        """
        losses = -data

        if thresholds is None:
            # Use quantiles
            thresholds = losses.quantile(np.linspace(0.5, 0.99, 50)).values

        mean_excess = []
        n_exceedances = []

        for threshold in thresholds:
            exceedances = losses[losses > threshold] - threshold

            if len(exceedances) > 0:
                mean_excess.append(exceedances.mean())
                n_exceedances.append(len(exceedances))
            else:
                mean_excess.append(np.nan)
                n_exceedances.append(0)

        return pd.DataFrame({
            'threshold': thresholds,
            'mean_excess': mean_excess,
            'n_exceedances': n_exceedances
        })

    def hill_estimator(
        self,
        data: pd.Series,
        k: int = None
    ) -> float:
        """
        Hill estimator for tail index.

        Args:
            data: Return series
            k: Number of order statistics to use

        Returns:
            Hill estimate of tail index
        """
        # Sort data in descending order
        sorted_data = np.sort(-data.values)[::-1]

        if k is None:
            # Use sqrt(n) rule
            k = int(np.sqrt(len(sorted_data)))

        # Hill estimator
        log_ratios = np.log(sorted_data[:k] / sorted_data[k])
        hill_estimate = log_ratios.mean()

        return hill_estimate

    def pickands_estimator(
        self,
        data: pd.Series,
        k: int = None
    ) -> float:
        """
        Pickands estimator for tail index.

        Args:
            data: Return series
            k: Number of order statistics

        Returns:
            Pickands estimate
        """
        sorted_data = np.sort(-data.values)[::-1]

        if k is None:
            k = int(np.sqrt(len(sorted_data)))

        # Pickands estimator
        pickands = (1 / np.log(2)) * np.log(
            (sorted_data[k] - sorted_data[2 * k])
            / (sorted_data[2 * k] - sorted_data[4 * k])
        )

        return pickands

    def tail_index_confidence_interval(
        self,
        data: pd.Series,
        k: int = None,
        confidence_level: float = 0.95
    ) -> tuple:
        """
        Calculate confidence interval for tail index.

        Args:
            data: Return series
            k: Number of order statistics
            confidence_level: Confidence level

        Returns:
            Tuple of (point_estimate, lower_bound, upper_bound)
        """
        if k is None:
            k = int(np.sqrt(len(data)))

        # Hill estimate
        hill = self.hill_estimator(data, k)

        # Asymptotic standard error
        se = hill / np.sqrt(k)

        # Confidence interval
        z = stats.norm.ppf((1 + confidence_level) / 2)
        lower = hill - z * se
        upper = hill + z * se

        return hill, lower, upper

    def test_tail_independence(
        self,
        data1: pd.Series,
        data2: pd.Series,
        threshold_quantile: float = 0.95
    ) -> dict:
        """
        Test for tail independence between two series.

        Args:
            data1: First return series
            data2: Second return series
            threshold_quantile: Quantile for threshold

        Returns:
            Dictionary with test results
        """
        # Get thresholds
        threshold1 = data1.quantile(threshold_quantile)
        threshold2 = data2.quantile(threshold_quantile)

        # Identify joint exceedances
        exceed1 = data1 > threshold1
        exceed2 = data2 > threshold2
        joint_exceed = exceed1 & exceed2

        # Calculate probabilities
        p1 = exceed1.sum() / len(data1)
        p2 = exceed2.sum() / len(data2)
        p_joint = joint_exceed.sum() / len(data1)

        # Under independence: p_joint ≈ p1 * p2
        p_independent = p1 * p2

        # Chi-square test
        chi2_stat = (p_joint - p_independent) ** 2 / p_independent
        p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)

        return {
            'p1': p1,
            'p2': p2,
            'p_joint': p_joint,
            'p_independent': p_independent,
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'reject_independence': p_value < 0.05
        }

    def extreme_value_copula(
        self,
        data1: pd.Series,
        data2: pd.Series,
        threshold_quantile: float = 0.95
    ) -> float:
        """
        Estimate tail dependence coefficient.

        Args:
            data1: First return series
            data2: Second return series
            threshold_quantile: Quantile for threshold

        Returns:
            Tail dependence coefficient
        """
        # Transform to uniform margins
        u1 = data1.rank() / (len(data1) + 1)
        u2 = data2.rank() / (len(data2) + 1)

        # Upper tail dependence
        threshold_u = threshold_quantile

        exceed_u1 = u1 > threshold_u
        exceed_u2 = u2 > threshold_u
        joint_exceed = exceed_u1 & exceed_u2

        # Tail dependence coefficient
        lambda_u = 2 - np.log(joint_exceed.sum() / len(data1)) / np.log(threshold_u)

        return lambda_u

    def backtest_var(
        self,
        returns: pd.Series,
        var_estimates: pd.Series,
        confidence_level: float = 0.99
    ) -> dict:
        """
        Backtest VaR estimates.

        Args:
            returns: Actual returns
            var_estimates: VaR estimates
            confidence_level: Confidence level

        Returns:
            Dictionary with backtest results
        """
        # Violations
        violations = returns < -var_estimates

        n_violations = violations.sum()
        expected_violations = len(returns) * (1 - confidence_level)

        violation_rate = n_violations / len(returns)
        expected_rate = 1 - confidence_level

        # Kupiec test (unconditional coverage)
        p = violation_rate
        p_expected = expected_rate

        if p > 0 and p < 1:
            lr_stat = -2 * (
                expected_violations * np.log(p_expected)
                + (len(returns) - expected_violations) * np.log(1 - p_expected)
                - n_violations * np.log(p)
                - (len(returns) - n_violations) * np.log(1 - p)
            )
            p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
        else:
            lr_stat = np.nan
            p_value = np.nan

        return {
            'n_violations': n_violations,
            'expected_violations': expected_violations,
            'violation_rate': violation_rate,
            'expected_rate': expected_rate,
            'kupiec_statistic': lr_stat,
            'p_value': p_value,
            'reject_model': p_value < 0.05 if not np.isnan(p_value) else None
        }
