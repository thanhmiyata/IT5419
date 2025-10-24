"""
XGBoost Model for Stock Prediction
===================================

Gradient boosting model for price direction and returns prediction.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
import xgboost as xgb


class XGBoostModel:
    """XGBoost for stock market prediction."""

    def __init__(
        self,
        objective: str = 'reg:squarederror',
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42
    ):
        """
        Initialize XGBoost model.

        Args:
            objective: 'reg:squarederror', 'binary:logistic', 'multi:softmax'
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns
            random_state: Random seed
        """
        self.objective = objective
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

        self.model = xgb.XGBRegressor(
            objective=objective,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=-1
        ) if 'reg' in objective else xgb.XGBClassifier(
            objective=objective,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        eval_set: Optional[list] = None,
        early_stopping_rounds: Optional[int] = 10,
        verbose: bool = False
    ) -> 'XGBoostModel':
        """
        Fit XGBoost model.

        Args:
            X_train: Training features
            y_train: Training targets
            eval_set: Evaluation set for early stopping
            early_stopping_rounds: Rounds for early stopping
            verbose: Print training progress

        Returns:
            Self for method chaining
        """
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities (classification only)."""
        if 'reg' in self.objective:
            raise ValueError("predict_proba only for classification")
        return self.model.predict_proba(X)

    def get_feature_importance(
        self,
        importance_type: str = 'gain'
    ) -> pd.Series:
        """
        Get feature importance.

        Args:
            importance_type: 'gain', 'weight', 'cover', 'total_gain', 'total_cover'

        Returns:
            Series of feature importances
        """
        importance = self.model.get_booster().get_score(
            importance_type=importance_type
        )
        return pd.Series(importance).sort_values(ascending=False)

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score,
                                     r2_score, recall_score)

        predictions = self.predict(X_test)

        if 'reg' in self.objective:
            return {
                'mse': mean_squared_error(y_test, predictions),
                'mae': mean_absolute_error(y_test, predictions),
                'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                'r2': r2_score(y_test, predictions)
            }
        else:
            return {
                'accuracy': accuracy_score(y_test, predictions),
                'precision': precision_score(y_test, predictions, average='weighted'),
                'recall': recall_score(y_test, predictions, average='weighted'),
                'f1_score': f1_score(y_test, predictions, average='weighted')
            }
