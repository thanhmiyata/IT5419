"""
Hidden Markov Model for Regime Detection
=========================================

Detect market regimes (bull, bear, sideways) using HMM.
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler


class HMMRegimeDetector:
    """Hidden Markov Model for market regime detection."""

    def __init__(
        self,
        n_regimes: int = 3,
        n_iterations: int = 100,
        random_state: int = 42
    ):
        """
        Initialize HMM regime detector.

        Args:
            n_regimes: Number of market regimes (e.g., 3 for bull/bear/sideways)
            n_iterations: Number of EM iterations
            random_state: Random seed
        """
        self.n_regimes = n_regimes
        self.n_iterations = n_iterations
        self.random_state = random_state

        self.model = None
        self.scaler = StandardScaler()
        self.regime_labels = None

    def prepare_features(
        self,
        returns: pd.Series,
        include_volatility: bool = True,
        volatility_window: int = 20
    ) -> np.ndarray:
        """
        Prepare features for HMM.

        Args:
            returns: Return series
            include_volatility: Include volatility as feature
            volatility_window: Window for volatility calculation

        Returns:
            Feature matrix
        """
        features = returns.values.reshape(-1, 1)

        if include_volatility:
            volatility = returns.rolling(window=volatility_window).std()
            volatility = volatility.fillna(volatility.mean())

            features = np.column_stack([returns.values, volatility.values])

        return features

    def fit(
        self,
        returns: pd.Series,
        include_volatility: bool = True,
        covariance_type: str = 'full'
    ):
        """
        Fit HMM to returns data.

        Args:
            returns: Return series
            include_volatility: Include volatility feature
            covariance_type: 'full', 'tied', 'diag', 'spherical'
        """
        # Prepare features
        features = self.prepare_features(returns, include_volatility)

        # Standardize features
        features_scaled = self.scaler.fit_transform(features)

        # Initialize and fit Gaussian HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=covariance_type,
            n_iter=self.n_iterations,
            random_state=self.random_state
        )

        self.model.fit(features_scaled)

        # Predict regimes
        regimes = self.model.predict(features_scaled)

        # Label regimes based on mean returns
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
                'volatility': volatility
            })

        # Sort by mean return
        regime_stats = sorted(regime_stats, key=lambda x: x['mean_return'])

        # Label: 0 = Bear, 1 = Sideways, 2 = Bull
        self.regime_labels = {
            regime_stats[0]['regime']: 'Bear',
            regime_stats[-1]['regime']: 'Bull'
        }

        if self.n_regimes == 3:
            self.regime_labels[regime_stats[1]['regime']] = 'Sideways'
        elif self.n_regimes > 3:
            for i in range(1, self.n_regimes - 1):
                self.regime_labels[regime_stats[i]['regime']] = f'Regime_{i}'

    def predict(
        self,
        returns: pd.Series,
        include_volatility: bool = True
    ) -> pd.Series:
        """
        Predict regimes for new data.

        Args:
            returns: Return series
            include_volatility: Include volatility feature

        Returns:
            Series of regime predictions
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Prepare features
        features = self.prepare_features(returns, include_volatility)
        features_scaled = self.scaler.transform(features)

        # Predict
        regimes = self.model.predict(features_scaled)

        # Map to labels
        regime_names = [self.regime_labels[r] for r in regimes]

        return pd.Series(regime_names, index=returns.index)

    def predict_proba(
        self,
        returns: pd.Series,
        include_volatility: bool = True
    ) -> pd.DataFrame:
        """
        Predict regime probabilities.

        Args:
            returns: Return series
            include_volatility: Include volatility feature

        Returns:
            DataFrame with probabilities for each regime
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Prepare features
        features = self.prepare_features(returns, include_volatility)
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
                'max_return': regime_returns.max()
            })

        return pd.DataFrame(stats)

    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Get regime transition matrix.

        Returns:
            DataFrame with transition probabilities
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get transition matrix
        trans_mat = self.model.transmat_

        # Create DataFrame with labels
        labels = [self.regime_labels[i] for i in range(self.n_regimes)]

        return pd.DataFrame(
            trans_mat,
            index=labels,
            columns=labels
        )

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

        return pd.DataFrame({
            'regime': self.predict(returns),
            'signal': signals,
            'returns': returns,
            'strategy_returns': strategy_returns,
            'cumulative_strategy': cumulative_returns,
            'cumulative_buy_hold': buy_hold_returns
        })

    def forecast_regime(
        self,
        returns: pd.Series,
        steps: int = 1
    ) -> pd.DataFrame:
        """
        Forecast future regimes.

        Args:
            returns: Return series
            steps: Number of steps to forecast

        Returns:
            DataFrame with regime forecasts and probabilities
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get current state probabilities
        features = self.prepare_features(returns)
        features_scaled = self.scaler.transform(features)

        # Get last state distribution
        last_proba = self.model.predict_proba(features_scaled)[-1]

        # Forecast using transition matrix
        forecasts = []

        current_proba = last_proba
        for step in range(1, steps + 1):
            # Next state probabilities
            next_proba = np.dot(current_proba, self.model.transmat_)

            # Most likely regime
            most_likely_regime = np.argmax(next_proba)

            forecasts.append({
                'step': step,
                'regime': self.regime_labels[most_likely_regime],
                'probability': next_proba[most_likely_regime],
                **{self.regime_labels[i]: next_proba[i] for i in range(self.n_regimes)}
            })

            current_proba = next_proba

        return pd.DataFrame(forecasts)
