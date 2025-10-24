"""
LSTM Model for Stock Price Prediction (PyTorch)
================================================

Long Short-Term Memory neural network for time-series forecasting.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


class LSTMNetwork(nn.Module):
    """PyTorch LSTM network architecture."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 50,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize LSTM network.

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(LSTMNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 25)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(25, 1)

    def forward(self, x):
        """Forward pass."""
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class LSTMPredictor:
    """LSTM model for stock price prediction using PyTorch."""

    def __init__(
        self,
        lookback: int = 60,
        hidden_size: int = 50,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        device: Optional[str] = None
    ):
        """Initialize LSTM predictor."""
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.device = torch.device(device)

        self.model = None
        self.scaler = None
        self.train_losses = []
        self.val_losses = []

    def _create_sequences(
        self,
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences for LSTM."""
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i - self.lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def fit(
        self,
        data: pd.Series,
        validation_split: float = 0.2
    ) -> 'LSTMPredictor':
        """Fit LSTM model to data."""
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))

        X, y = self._create_sequences(scaled_data)

        split_idx = int(len(X) * (1 - validation_split))
        X_train = torch.FloatTensor(X[:split_idx]).to(self.device)
        y_train = torch.FloatTensor(y[:split_idx]).to(self.device)
        X_val = torch.FloatTensor(X[split_idx:]).to(self.device)
        y_val = torch.FloatTensor(y[split_idx:]).to(self.device)

        self.model = LSTMNetwork(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )

        for epoch in range(self.epochs):
            self.model.train()
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.train_losses.append(loss.item())

            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val)
                self.val_losses.append(val_loss.item())

        return self

    def predict(self, data: pd.Series, steps: int = 1) -> np.ndarray:
        """Predict future values."""
        if self.model is None or self.scaler is None:
            raise ValueError("Model must be fitted before prediction")

        self.model.eval()
        last_sequence = data[-self.lookback:].values.reshape(-1, 1)
        scaled_sequence = self.scaler.transform(last_sequence)
        predictions = []

        with torch.no_grad():
            for _ in range(steps):
                X = torch.FloatTensor(
                    scaled_sequence[-self.lookback:].reshape(1, self.lookback, 1)
                ).to(self.device)
                pred_scaled = self.model(X).cpu().numpy()
                pred = self.scaler.inverse_transform(pred_scaled)[0, 0]
                predictions.append(pred)
                scaled_sequence = np.vstack([
                    scaled_sequence,
                    pred_scaled.reshape(-1, 1)
                ])

        return np.array(predictions)
