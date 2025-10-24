"""
Training Utilities for Quantitative Models
===========================================

Utilities for training, validating, and saving models.
"""

import json
import os
import pickle
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


class DataSplitter:
    """Utilities for splitting time-series data."""

    @staticmethod
    def train_test_split_timeseries(
        data: pd.DataFrame,
        test_size: float = 0.2,
        shuffle: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split time-series data into train and test sets.

        Args:
            data: Time-series DataFrame
            test_size: Proportion of data for testing
            shuffle: Whether to shuffle (not recommended for time-series)

        Returns:
            Tuple of (train_data, test_data)
        """
        if shuffle:
            train, test = train_test_split(data, test_size=test_size, shuffle=True)
        else:
            # Time-series split: train on earlier data, test on later
            split_idx = int(len(data) * (1 - test_size))
            train = data.iloc[:split_idx]
            test = data.iloc[split_idx:]

        return train, test

    @staticmethod
    def train_val_test_split_timeseries(
        data: pd.DataFrame,
        val_size: float = 0.15,
        test_size: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split into train, validation, and test sets.

        Args:
            data: Time-series DataFrame
            val_size: Proportion for validation
            test_size: Proportion for testing

        Returns:
            Tuple of (train, val, test)
        """
        # Calculate split indices
        n = len(data)
        test_idx = int(n * (1 - test_size))
        val_idx = int(n * (1 - test_size - val_size))

        train = data.iloc[:val_idx]
        val = data.iloc[val_idx:test_idx]
        test = data.iloc[test_idx:]

        return train, val, test

    @staticmethod
    def rolling_window_split(
        data: pd.DataFrame,
        train_size: int,
        test_size: int,
        step: int = 1
    ):
        """
        Create rolling window splits for walk-forward analysis.

        Args:
            data: Time-series DataFrame
            train_size: Size of training window
            test_size: Size of test window
            step: Step size between windows

        Yields:
            Tuples of (train_data, test_data)
        """
        n = len(data)

        for i in range(0, n - train_size - test_size + 1, step):
            train_end = i + train_size
            test_end = train_end + test_size

            train = data.iloc[i:train_end]
            test = data.iloc[train_end:test_end]

            yield train, test

    @staticmethod
    def expanding_window_split(
        data: pd.DataFrame,
        initial_train_size: int,
        test_size: int,
        step: int = 1
    ):
        """
        Create expanding window splits.

        Args:
            data: Time-series DataFrame
            initial_train_size: Initial training window size
            test_size: Size of test window
            step: Step size

        Yields:
            Tuples of (train_data, test_data)
        """
        n = len(data)

        for i in range(initial_train_size, n - test_size + 1, step):
            train = data.iloc[:i]
            test = data.iloc[i:i + test_size]

            yield train, test


class ModelCheckpoint:
    """Save and load model checkpoints."""

    @staticmethod
    def save_sklearn_model(
        model: Any,
        filepath: str,
        metadata: Optional[Dict] = None
    ):
        """
        Save scikit-learn model.

        Args:
            model: Trained model
            filepath: Path to save file
            metadata: Optional metadata dictionary
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        checkpoint = {
            'model': model,
            'metadata': metadata or {}
        }

        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)

    @staticmethod
    def load_sklearn_model(filepath: str) -> Tuple[Any, Dict]:
        """
        Load scikit-learn model.

        Args:
            filepath: Path to saved file

        Returns:
            Tuple of (model, metadata)
        """
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)

        return checkpoint['model'], checkpoint.get('metadata', {})

    @staticmethod
    def save_pytorch_model(
        model: torch.nn.Module,
        filepath: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Save PyTorch model.

        Args:
            model: PyTorch model
            filepath: Path to save file
            optimizer: Optional optimizer state
            metadata: Optional metadata
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'metadata': metadata or {}
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, filepath)

    @staticmethod
    def load_pytorch_model(
        model: torch.nn.Module,
        filepath: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cpu'
    ) -> Tuple[torch.nn.Module, Dict]:
        """
        Load PyTorch model.

        Args:
            model: Model architecture
            filepath: Path to saved file
            optimizer: Optional optimizer to load state
            device: Device to load model to

        Returns:
            Tuple of (model, metadata)
        """
        checkpoint = torch.load(filepath, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return model, checkpoint.get('metadata', {})


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if should stop.

        Args:
            score: Current validation score

        Returns:
            True if should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class TrainingLogger:
    """Log training progress."""

    def __init__(self, log_file: Optional[str] = None, name: str = 'training'):
        """
        Initialize logger.

        Args:
            log_file: Optional file to save logs
            name: Logger name
        """
        self.log_file = log_file
        self.history = []

        # Initialize Python logger
        import logging
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers
        self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def log(self, epoch: int, metrics: Dict[str, float]):
        """
        Log metrics for an epoch.

        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
        """
        entry = {'epoch': epoch, **metrics}
        self.history.append(entry)

        # Log to Python logger
        metrics_str = ', '.join([f'{k}: {v:.6f}' for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch}: {metrics_str}")

        # Save to file
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def get_history(self) -> pd.DataFrame:
        """
        Get training history as DataFrame.

        Returns:
            DataFrame with training history
        """
        return pd.DataFrame(self.history)

    def plot_history(self, metric: str = 'loss'):
        """
        Plot training history.

        Args:
            metric: Metric to plot
        """
        import matplotlib.pyplot as plt

        df = self.get_history()

        plt.figure(figsize=(10, 6))
        plt.plot(df['epoch'], df[metric], label='Train')

        if f'val_{metric}' in df.columns:
            plt.plot(df['epoch'], df[f'val_{metric}'], label='Validation')

        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()


class CrossValidator:
    """Time-series cross-validation."""

    @staticmethod
    def walk_forward_validation(
        data: pd.DataFrame,
        model_class,
        model_params: Dict,
        train_size: int,
        test_size: int,
        step: int = 1
    ) -> pd.DataFrame:
        """
        Perform walk-forward validation.

        Args:
            data: Time-series data
            model_class: Model class to instantiate
            model_params: Parameters for model
            train_size: Training window size
            test_size: Test window size
            step: Step size

        Returns:
            DataFrame with validation results
        """
        results = []

        splitter = DataSplitter()

        for i, (train, test) in enumerate(
            splitter.rolling_window_split(data, train_size, test_size, step)
        ):
            # Train model
            model = model_class(**model_params)

            # This assumes model has fit/predict interface
            # Adjust based on your model's interface
            try:
                model.fit(train)
                predictions = model.predict(test)

                results.append({
                    'fold': i,
                    'train_start': train.index[0],
                    'train_end': train.index[-1],
                    'test_start': test.index[0],
                    'test_end': test.index[-1],
                    'predictions': predictions
                })
            except Exception as e:
                print(f"Error in fold {i}: {e}")
                continue

        return pd.DataFrame(results)

    @staticmethod
    def expanding_window_validation(
        data: pd.DataFrame,
        model_class,
        model_params: Dict,
        initial_train_size: int,
        test_size: int,
        step: int = 1
    ) -> pd.DataFrame:
        """
        Perform expanding window validation.

        Args:
            data: Time-series data
            model_class: Model class
            model_params: Model parameters
            initial_train_size: Initial training size
            test_size: Test window size
            step: Step size

        Returns:
            DataFrame with validation results
        """
        results = []

        splitter = DataSplitter()

        for i, (train, test) in enumerate(
            splitter.expanding_window_split(data, initial_train_size, test_size, step)
        ):
            model = model_class(**model_params)

            try:
                model.fit(train)
                predictions = model.predict(test)

                results.append({
                    'fold': i,
                    'train_start': train.index[0],
                    'train_end': train.index[-1],
                    'test_start': test.index[0],
                    'test_end': test.index[-1],
                    'predictions': predictions
                })
            except Exception as e:
                print(f"Error in fold {i}: {e}")
                continue

        return pd.DataFrame(results)


class DataPreprocessor:
    """Preprocessing utilities for financial data."""

    @staticmethod
    def normalize(data: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
        """
        Normalize data.

        Args:
            data: DataFrame to normalize
            method: 'minmax' or 'standard'

        Returns:
            Normalized DataFrame
        """
        if method == 'minmax':
            return (data - data.min()) / (data.max() - data.min())
        elif method == 'standard':
            return (data - data.mean()) / data.std()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    @staticmethod
    def handle_missing_values(
        data: pd.DataFrame,
        method: str = 'forward_fill'
    ) -> pd.DataFrame:
        """
        Handle missing values.

        Args:
            data: DataFrame with missing values
            method: 'forward_fill', 'backward_fill', 'interpolate', or 'drop'

        Returns:
            DataFrame with handled missing values
        """
        if method == 'forward_fill':
            return data.fillna(method='ffill')
        elif method == 'backward_fill':
            return data.fillna(method='bfill')
        elif method == 'interpolate':
            return data.interpolate(method='linear')
        elif method == 'drop':
            return data.dropna()
        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def remove_outliers(
        data: pd.Series,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.Series:
        """
        Remove outliers.

        Args:
            data: Series to clean
            method: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection

        Returns:
            Series with outliers removed
        """
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR

            return data[(data >= lower) & (data <= upper)]

        elif method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            return data[z_scores < threshold]

        else:
            raise ValueError(f"Unknown method: {method}")
