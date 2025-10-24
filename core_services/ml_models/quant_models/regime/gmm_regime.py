"""
Gaussian Mixture Models for Regime Detection
=============================================

Detect market regimes using GMM clustering.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class GMMRegimeDetector:
    """Gaussian Mixture Model for market regime detection."""

    def __init__(
        self,
        n_regimes: int = 3,
        covariance_type: str = 'full',
        n_init: int = 10,
        random_state: int = 42
    ):
        """
        Initialize GMM regime detector.

        Args:
            n_regimes: Number of market regimes
            covariance_type: 'full', 'tied', 'diag', 'spherical'
            n_init: Number of initializations
            random_state: Random seed
        """
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_init = n_init
        self.random_state = random_state

        self.model = None
        self.scaler = StandardScaler()
        self.regime_labels = None

    def prepare_features(
        self,
        returns: pd.Series,
        window: int = 20
    ) -> np.ndarray:
        """
        Prepare features for GMM.

        Args:
            returns: Return series
            window: Window for rolling statistics

        Returns:
            Feature matrix
        """
        # Calculate rolling statistics
        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        rolling_skew = returns.rolling(window=window).skew()

        # Fill NaN
        rolling_mean = rolling_mean.fillna(rolling_mean.mean())
        rolling_std = rolling_std.fillna(rolling_std.mean())
        rolling_skew = rolling_skew.fillna(0)

        # Combine features
        features = np.column_stack([
            returns.values,
            rolling_mean.values,
            rolling_std.values,
            rolling_skew.values
        ])

        return features

    def fit(
        self,
        returns: pd.Series,
        feature_window: int = 20
    ):
        """
        Fit GMM to returns data.

        Args:
            returns: Return series
            feature_window: Window for feature calculation
        """
        # Prepare features
        features = self.prepare_features(returns, window=feature_window)

        # Standardize features
        features_scaled = self.scaler.fit_transform(features)

        # Initialize and fit GMM
        self.model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_init=self.n_init,
            random_state=self.random_state
        )

        self.model.fit(features_scaled)

        # Predict regimes
        regimes = self.model.predict(features_scaled)

        # Label regimes based on characteristics
        self._label_regimes(returns, regimes)

        return self

    def _label_regimes(
        self,
        returns: pd.Series,
        regimes: np.ndarray
    ):
        """
        Label regimes based on characteristics.

        Args:
            returns: Return series
            regimes: Predicted regime sequence
        """
        regime_stats = []

        for regime in range(self.n_regimes):
            regime_mask = regimes == regime
            regime_returns = returns[regime_mask]

            mean_return = regime_returns.mean()
            volatility = regime_returns.std()

            regime_stats.append({
                'regime': regime,
                'mean_return': mean_return,
                'volatility': volatility,
                'sharpe': mean_return / volatility if volatility > 0 else 0
            })

        # Sort by mean return
        regime_stats = sorted(regime_stats, key=lambda x: x['mean_return'])

        # Label regimes
        self.regime_labels = {}

        if self.n_regimes == 2:
            self.regime_labels[regime_stats[0]['regime']] = 'Bear'
            self.regime_labels[regime_stats[1]['regime']] = 'Bull'
        elif self.n_regimes == 3:
            self.regime_labels[regime_stats[0]['regime']] = 'Bear'
            self.regime_labels[regime_stats[1]['regime']] = 'Sideways'
            self.regime_labels[regime_stats[2]['regime']] = 'Bull'
        else:
            for i, stat in enumerate(regime_stats):
                self.regime_labels[stat['regime']] = f'Regime_{i}'

    def predict(
        self,
        returns: pd.Series,
        feature_window: int = 20
    ) -> pd.Series:
        """
        Predict regimes for new data.

        Args:
            returns: Return series
            feature_window: Window for feature calculation

        Returns:
            Series of regime predictions
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Prepare features
        features = self.prepare_features(returns, window=feature_window)
        features_scaled = self.scaler.transform(features)

        # Predict
        regimes = self.model.predict(features_scaled)

        # Map to labels
        regime_names = [self.regime_labels[r] for r in regimes]

        return pd.Series(regime_names, index=returns.index)

    def predict_proba(
        self,
        returns: pd.Series,
        feature_window: int = 20
    ) -> pd.DataFrame:
        """
        Predict regime probabilities.

        Args:
            returns: Return series
            feature_window: Window for feature calculation

        Returns:
            DataFrame with probabilities for each regime
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Prepare features
        features = self.prepare_features(returns, window=feature_window)
        features_scaled = self.scaler.transform(features)

        # Predict probabilities
        proba = self.model.predict_proba(features_scaled)

        # Create DataFrame with regime labels
        columns = [self.regime_labels[i] for i in range(self.n_regimes)]

        return pd.DataFrame(proba, index=returns.index, columns=columns)

    def get_regime_statistics(
        self,
        returns: pd.Series,
        regimes: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate statistics for each regime.

        Args:
            returns: Return series
            regimes: Regime predictions

        Returns:
            DataFrame with regime statistics
        """
        stats = []

        for regime in self.regime_labels.values():
            regime_mask = regimes == regime
            regime_returns = returns[regime_mask]

            if len(regime_returns) == 0:
                continue

            stats.append({
                'regime': regime,
                'count': len(regime_returns),
                'frequency': len(regime_returns) / len(returns),
                'mean_return': regime_returns.mean(),
                'volatility': regime_returns.std(),
                'sharpe': regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0,
                'min_return': regime_returns.min(),
                'max_return': regime_returns.max(),
                'skewness': regime_returns.skew(),
                'kurtosis': regime_returns.kurtosis()
            })

        return pd.DataFrame(stats)

    def get_bic_aic(self, returns: pd.Series, max_regimes: int = 10) -> pd.DataFrame:
        """
        Calculate BIC and AIC for different numbers of regimes.

        Args:
            returns: Return series
            max_regimes: Maximum number of regimes to test

        Returns:
            DataFrame with BIC and AIC scores
        """
        features = self.prepare_features(returns)
        features_scaled = self.scaler.fit_transform(features)

        scores = []

        for n in range(1, max_regimes + 1):
            model = GaussianMixture(
                n_components=n,
                covariance_type=self.covariance_type,
                n_init=self.n_init,
                random_state=self.random_state
            )
            model.fit(features_scaled)

            scores.append({
                'n_regimes': n,
                'bic': model.bic(features_scaled),
                'aic': model.aic(features_scaled),
                'log_likelihood': model.score(features_scaled) * len(features_scaled)
            })

        return pd.DataFrame(scores)

    def find_optimal_regimes(
        self,
        returns: pd.Series,
        max_regimes: int = 10,
        criterion: str = 'bic'
    ) -> int:
        """
        Find optimal number of regimes using BIC or AIC.

        Args:
            returns: Return series
            max_regimes: Maximum regimes to test
            criterion: 'bic' or 'aic'

        Returns:
            Optimal number of regimes
        """
        scores_df = self.get_bic_aic(returns, max_regimes)

        if criterion == 'bic':
            optimal_n = scores_df.loc[scores_df['bic'].idxmin(), 'n_regimes']
        else:
            optimal_n = scores_df.loc[scores_df['aic'].idxmin(), 'n_regimes']

        return int(optimal_n)

    def generate_trading_signals(
        self,
        returns: pd.Series,
        long_regimes: list = ['Bull'],
        short_regimes: list = ['Bear']
    ) -> pd.Series:
        """
        Generate trading signals based on regimes.

        Args:
            returns: Return series
            long_regimes: Regimes to go long
            short_regimes: Regimes to go short

        Returns:
            Series of trading signals (1=long, -1=short, 0=neutral)
        """
        regimes = self.predict(returns)

        signals = pd.Series(0, index=returns.index)
        signals[regimes.isin(long_regimes)] = 1
        signals[regimes.isin(short_regimes)] = -1

        return signals

    def backtest_regime_strategy(
        self,
        prices: pd.Series,
        returns: pd.Series,
        long_regimes: list = ['Bull'],
        short_regimes: list = ['Bear']
    ) -> pd.DataFrame:
        """
        Backtest regime-based strategy.

        Args:
            prices: Price series
            returns: Return series
            long_regimes: Regimes to go long
            short_regimes: Regimes to go short

        Returns:
            DataFrame with backtest results
        """
        # Get signals
        signals = self.generate_trading_signals(returns, long_regimes, short_regimes)

        # Calculate strategy returns
        strategy_returns = signals.shift(1) * returns

        # Calculate cumulative returns
        cumulative_returns = (1 + strategy_returns).cumprod()
        buy_hold_returns = (1 + returns).cumprod()

        # Performance metrics
        total_return = cumulative_returns.iloc[-1] - 1
        buy_hold_return = buy_hold_returns.iloc[-1] - 1

        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0

        return pd.DataFrame({
            'regime': self.predict(returns),
            'signal': signals,
            'returns': returns,
            'strategy_returns': strategy_returns,
            'cumulative_strategy': cumulative_returns,
            'cumulative_buy_hold': buy_hold_returns,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'sharpe_ratio': sharpe
        })

    def get_regime_durations(
        self,
        regimes: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate average duration of each regime.

        Args:
            regimes: Regime predictions

        Returns:
            DataFrame with regime durations
        """
        durations = []

        current_regime = None
        current_duration = 0

        for regime in regimes:
            if regime == current_regime:
                current_duration += 1
            else:
                if current_regime is not None:
                    durations.append({
                        'regime': current_regime,
                        'duration': current_duration
                    })
                current_regime = regime
                current_duration = 1

        # Add last regime
        if current_regime is not None:
            durations.append({
                'regime': current_regime,
                'duration': current_duration
            })

        # Calculate statistics
        df = pd.DataFrame(durations)

        stats = df.groupby('regime')['duration'].agg([
            ('avg_duration', 'mean'),
            ('min_duration', 'min'),
            ('max_duration', 'max'),
            ('std_duration', 'std')
        ]).reset_index()

        return stats

    def visualize_regimes(
        self,
        prices: pd.Series,
        regimes: pd.Series
    ) -> pd.DataFrame:
        """
        Prepare data for regime visualization.

        Args:
            prices: Price series
            regimes: Regime predictions

        Returns:
            DataFrame ready for plotting
        """
        return pd.DataFrame({
            'price': prices,
            'regime': regimes
        })
