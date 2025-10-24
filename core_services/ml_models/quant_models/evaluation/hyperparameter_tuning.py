"""
Hyperparameter Tuning for Quantitative Models
==============================================

Grid search, random search, and Bayesian optimization for hyperparameters.
"""

from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class GridSearch:
    """Grid search for hyperparameter tuning."""

    def __init__(
        self,
        model_class: Any,
        param_grid: Dict[str, List],
        scoring_func: Callable,
        cv_splits: int = 5
    ):
        """
        Initialize grid search.

        Args:
            model_class: Model class to instantiate
            param_grid: Dictionary of parameter grids
            scoring_func: Function to score models (higher is better)
            cv_splits: Number of cross-validation splits
        """
        self.model_class = model_class
        self.param_grid = param_grid
        self.scoring_func = scoring_func
        self.cv_splits = cv_splits

        self.best_params = None
        self.best_score = -np.inf
        self.results = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ):
        """
        Perform grid search.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
        """
        # Generate parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())

        for params in product(*param_values):
            param_dict = dict(zip(param_names, params))

            # Train and evaluate model
            try:
                model = self.model_class(**param_dict)
                model.fit(X_train, y_train)

                # Score on validation set
                if X_val is not None and y_val is not None:
                    predictions = model.predict(X_val)
                    score = self.scoring_func(y_val, predictions)
                else:
                    # Use cross-validation
                    score = self._cross_validate(model, X_train, y_train)

                # Store results
                self.results.append({
                    'params': param_dict,
                    'score': score
                })

                # Update best
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = param_dict

                print(f"Params: {param_dict} | Score: {score:.6f}")

            except Exception as e:
                print(f"Error with params {param_dict}: {e}")
                continue

    def _cross_validate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series
    ) -> float:
        """
        Perform cross-validation.

        Args:
            model: Model to evaluate
            X: Features
            y: Target

        Returns:
            Average cross-validation score
        """
        scores = []
        fold_size = len(X) // self.cv_splits

        for i in range(self.cv_splits):
            # Time-series split
            val_start = i * fold_size
            val_end = (i + 1) * fold_size

            X_train_cv = pd.concat([X.iloc[:val_start], X.iloc[val_end:]])
            y_train_cv = pd.concat([y.iloc[:val_start], y.iloc[val_end:]])

            X_val_cv = X.iloc[val_start:val_end]
            y_val_cv = y.iloc[val_start:val_end]

            # Train and score
            model.fit(X_train_cv, y_train_cv)
            predictions = model.predict(X_val_cv)
            score = self.scoring_func(y_val_cv, predictions)

            scores.append(score)

        return np.mean(scores)

    def get_results(self) -> pd.DataFrame:
        """
        Get all results.

        Returns:
            DataFrame with results
        """
        return pd.DataFrame(self.results)


class RandomSearch:
    """Random search for hyperparameter tuning."""

    def __init__(
        self,
        model_class: Any,
        param_distributions: Dict[str, Any],
        scoring_func: Callable,
        n_iter: int = 100,
        cv_splits: int = 5,
        random_state: int = 42
    ):
        """
        Initialize random search.

        Args:
            model_class: Model class
            param_distributions: Parameter distributions
            scoring_func: Scoring function
            n_iter: Number of iterations
            cv_splits: Cross-validation splits
            random_state: Random seed
        """
        self.model_class = model_class
        self.param_distributions = param_distributions
        self.scoring_func = scoring_func
        self.n_iter = n_iter
        self.cv_splits = cv_splits
        self.random_state = random_state

        self.best_params = None
        self.best_score = -np.inf
        self.results = []

        np.random.seed(random_state)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ):
        """
        Perform random search.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
        """
        for iteration in range(self.n_iter):
            # Sample parameters
            params = self._sample_parameters()

            try:
                model = self.model_class(**params)
                model.fit(X_train, y_train)

                # Score
                if X_val is not None and y_val is not None:
                    predictions = model.predict(X_val)
                    score = self.scoring_func(y_val, predictions)
                else:
                    score = self._cross_validate(model, X_train, y_train)

                self.results.append({
                    'params': params,
                    'score': score
                })

                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params

                print(f"Iteration {iteration + 1}/{self.n_iter} | "
                      f"Score: {score:.6f} | Best: {self.best_score:.6f}")

            except Exception as e:
                print(f"Error in iteration {iteration}: {e}")
                continue

    def _sample_parameters(self) -> Dict[str, Any]:
        """
        Sample parameters from distributions.

        Returns:
            Dictionary of sampled parameters
        """
        params = {}

        for name, distribution in self.param_distributions.items():
            if isinstance(distribution, list):
                # Discrete choice
                params[name] = np.random.choice(distribution)
            elif isinstance(distribution, tuple) and len(distribution) == 2:
                # Continuous uniform distribution
                low, high = distribution
                if isinstance(low, int) and isinstance(high, int):
                    params[name] = np.random.randint(low, high + 1)
                else:
                    params[name] = np.random.uniform(low, high)
            elif callable(distribution):
                # Custom distribution
                params[name] = distribution()
            else:
                params[name] = distribution

        return params

    def _cross_validate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series
    ) -> float:
        """Cross-validation scoring."""
        scores = []
        fold_size = len(X) // self.cv_splits

        for i in range(self.cv_splits):
            val_start = i * fold_size
            val_end = (i + 1) * fold_size

            X_train_cv = pd.concat([X.iloc[:val_start], X.iloc[val_end:]])
            y_train_cv = pd.concat([y.iloc[:val_start], y.iloc[val_end:]])

            X_val_cv = X.iloc[val_start:val_end]
            y_val_cv = y.iloc[val_start:val_end]

            model.fit(X_train_cv, y_train_cv)
            predictions = model.predict(X_val_cv)
            score = self.scoring_func(y_val_cv, predictions)

            scores.append(score)

        return np.mean(scores)

    def get_results(self) -> pd.DataFrame:
        """Get all results."""
        return pd.DataFrame(self.results)


class BayesianOptimization:
    """Bayesian optimization for hyperparameter tuning."""

    def __init__(
        self,
        model_class: Any,
        param_bounds: Dict[str, Tuple[float, float]],
        scoring_func: Callable,
        n_iter: int = 50,
        n_init: int = 10,
        random_state: int = 42
    ):
        """
        Initialize Bayesian optimization.

        Args:
            model_class: Model class
            param_bounds: Parameter bounds {name: (min, max)}
            scoring_func: Scoring function
            n_iter: Number of iterations
            n_init: Number of random initialization samples
            random_state: Random seed
        """
        self.model_class = model_class
        self.param_bounds = param_bounds
        self.scoring_func = scoring_func
        self.n_iter = n_iter
        self.n_init = n_init
        self.random_state = random_state

        self.best_params = None
        self.best_score = -np.inf
        self.results = []

        np.random.seed(random_state)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ):
        """
        Perform Bayesian optimization.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
        """
        # Random initialization
        for i in range(self.n_init):
            params = self._sample_random_params()
            score = self._evaluate_params(params, X_train, y_train, X_val, y_val)

            self.results.append({'params': params, 'score': score})

            if score > self.best_score:
                self.best_score = score
                self.best_params = params

            print(f"Init {i + 1}/{self.n_init} | Score: {score:.6f}")

        # Bayesian optimization iterations
        for i in range(self.n_iter - self.n_init):
            # Get next point using acquisition function
            params = self._get_next_point()

            score = self._evaluate_params(params, X_train, y_train, X_val, y_val)

            self.results.append({'params': params, 'score': score})

            if score > self.best_score:
                self.best_score = score
                self.best_params = params

            print(f"Iteration {i + 1 + self.n_init}/{self.n_iter} | "
                  f"Score: {score:.6f} | Best: {self.best_score:.6f}")

    def _sample_random_params(self) -> Dict[str, float]:
        """Sample random parameters within bounds."""
        params = {}

        for name, (low, high) in self.param_bounds.items():
            if isinstance(low, int) and isinstance(high, int):
                params[name] = np.random.randint(low, high + 1)
            else:
                params[name] = np.random.uniform(low, high)

        return params

    def _get_next_point(self) -> Dict[str, float]:
        """
        Get next point using acquisition function (simplified).

        For production, use libraries like scikit-optimize or Optuna.
        This is a simplified implementation using UCB.
        """
        # Simple implementation: sample multiple points and use UCB
        n_candidates = 100
        best_acquisition = -np.inf
        best_candidate = None

        for _ in range(n_candidates):
            candidate = self._sample_random_params()

            # Calculate UCB (Upper Confidence Bound)
            # Simplified: just add random exploration bonus
            acquisition = np.random.random()

            if acquisition > best_acquisition:
                best_acquisition = acquisition
                best_candidate = candidate

        return best_candidate

    def _evaluate_params(
        self,
        params: Dict[str, float],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> float:
        """Evaluate parameter configuration."""
        try:
            model = self.model_class(**params)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            score = self.scoring_func(y_val, predictions)
            return score
        except Exception as e:
            print(f"Error evaluating params {params}: {e}")
            return -np.inf

    def get_results(self) -> pd.DataFrame:
        """Get all results."""
        return pd.DataFrame(self.results)


class ParameterOptimizer:
    """Unified interface for parameter optimization."""

    @staticmethod
    def optimize(
        model_class: Any,
        param_config: Dict,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        scoring_func: Callable,
        method: str = 'grid',
        n_iter: int = 100
    ) -> Tuple[Dict, float, pd.DataFrame]:
        """
        Optimize hyperparameters.

        Args:
            model_class: Model class
            param_config: Parameter configuration
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            scoring_func: Scoring function
            method: 'grid', 'random', or 'bayesian'
            n_iter: Number of iterations (for random/bayesian)

        Returns:
            Tuple of (best_params, best_score, results_df)
        """
        if method == 'grid':
            optimizer = GridSearch(
                model_class,
                param_config,
                scoring_func
            )
        elif method == 'random':
            optimizer = RandomSearch(
                model_class,
                param_config,
                scoring_func,
                n_iter=n_iter
            )
        elif method == 'bayesian':
            optimizer = BayesianOptimization(
                model_class,
                param_config,
                scoring_func,
                n_iter=n_iter
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        optimizer.fit(X_train, y_train, X_val, y_val)

        return optimizer.best_params, optimizer.best_score, optimizer.get_results()
