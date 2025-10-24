"""
Random Forest Model for Stock Prediction
=========================================

Ensemble tree-based model for price direction and returns prediction.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split


class RandomForestModel:
    """Random Forest for stock market prediction."""

    def __init__(
        self,
        task: str = 'classification',
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42
    ):
        """
        Initialize Random Forest model.

        Args:
            task: 'classification' or 'regression'
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Min samples to split node
            min_samples_leaf: Min samples in leaf node
            random_state: Random seed
        """
        self.task = task
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        if task == 'classification':
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                n_jobs=-1
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                n_jobs=-1
            )

    def create_features(
        self,
        data: pd.DataFrame,
        lags: int = 5
    ) -> pd.DataFrame:
        """
        Create features from price data.

        Args:
            data: DataFrame with OHLCV data
            lags: Number of lagged features

        Returns:
            DataFrame with features
        """
        features = pd.DataFrame(index=data.index)

        # Price features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))

        # Lagged returns
        for i in range(1, lags + 1):
            features[f'return_lag_{i}'] = features['returns'].shift(i)

        # Moving averages
        for window in [5, 10, 20, 50]:
            features[f'sma_{window}'] = data['close'].rolling(window).mean()
            features[f'price_to_sma_{window}'] = data['close'] / features[f'sma_{window}']

        # Volatility
        features['volatility_5'] = features['returns'].rolling(5).std()
        features['volatility_20'] = features['returns'].rolling(20).std()

        # Volume features
        if 'volume' in data.columns:
            features['volume'] = data['volume']
            features['volume_ma_5'] = data['volume'].rolling(5).mean()
            features['volume_ratio'] = data['volume'] / features['volume_ma_5']

        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))

        # Remove NaN
        features = features.dropna()

        return features

    def prepare_data(
        self,
        data: pd.DataFrame,
        target_col: str = 'target',
        test_size: float = 0.2
    ) -> tuple:
        """
        Prepare features and split data.

        Args:
            data: DataFrame with features and target
            target_col: Target column name
            test_size: Test set fraction

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X = data.drop(columns=[target_col])
        y = data[target_col]

        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            shuffle=False  # Preserve time order
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'RandomForestModel':
        """
        Fit Random Forest model.

        Args:
            X_train: Training features
            y_train: Training targets

        Returns:
            Self for method chaining
        """
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predictions array
        """
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities (classification only).

        Args:
            X: Feature matrix

        Returns:
            Probability array
        """
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification")

        return self.model.predict_proba(X)

    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance scores.

        Returns:
            Series of feature importances
        """
        importances = self.model.feature_importances_
        feature_names = self.model.feature_names_in_

        return pd.Series(
            importances,
            index=feature_names
        ).sort_values(ascending=False)

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score,
                                     r2_score, recall_score)

        predictions = self.predict(X_test)

        if self.task == 'classification':
            return {
                'accuracy': accuracy_score(y_test, predictions),
                'precision': precision_score(y_test, predictions, average='weighted'),
                'recall': recall_score(y_test, predictions, average='weighted'),
                'f1_score': f1_score(y_test, predictions, average='weighted')
            }
        else:
            return {
                'mse': mean_squared_error(y_test, predictions),
                'mae': mean_absolute_error(y_test, predictions),
                'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                'r2': r2_score(y_test, predictions)
            }
