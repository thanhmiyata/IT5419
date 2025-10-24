"""
Transformer Model for Time-Series Prediction
=============================================

Multi-head attention mechanism for stock price forecasting.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


class TransformerNetwork(nn.Module):
    """Transformer architecture for time-series."""

    def __init__(
        self,
        input_size: int = 1,
        d_model: int = 64,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_length: int = 100
    ):
        """
        Initialize Transformer network.

        Args:
            input_size: Number of input features
            d_model: Dimension of model embeddings
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
        """
        super(TransformerNetwork, self).__init__()

        self.d_model = d_model
        self.input_size = input_size

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(
            max_seq_length, d_model
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # Output projection
        self.fc1 = nn.Linear(d_model, dim_feedforward // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim_feedforward // 2, 1)

        self.relu = nn.ReLU()

    def _create_positional_encoding(
        self,
        max_len: int,
        d_model: int
    ) -> torch.Tensor:
        """Create positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, input_size]

        Returns:
            Output predictions [batch_size, 1]
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_projection(x)

        # Add positional encoding
        pos_encoding = self.positional_encoding[:, :seq_len, :].to(x.device)
        x = x + pos_encoding

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Take last time step
        x = x[:, -1, :]

        # Output projection
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class TransformerForecaster:
    """Transformer model for stock price prediction."""

    def __init__(
        self,
        lookback_period: int = 60,
        d_model: int = 64,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32
    ):
        """
        Initialize Transformer forecaster.

        Args:
            lookback_period: Number of past time steps
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            learning_rate: Learning rate
            epochs: Training epochs
            batch_size: Batch size
        """
        self.lookback_period = lookback_period
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _prepare_data(
        self,
        data: pd.Series
    ) -> tuple:
        """Prepare training data."""
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))

        X, y = [], []
        for i in range(self.lookback_period, len(scaled_data)):
            X.append(scaled_data[i - self.lookback_period:i])
            y.append(scaled_data[i])

        X = np.array(X)
        y = np.array(y)

        return torch.FloatTensor(X), torch.FloatTensor(y)

    def fit(self, data: pd.Series):
        """
        Train the Transformer model.

        Args:
            data: Time series data to train on
        """
        X, y = self._prepare_data(data)
        X, y = X.to(self.device), y.to(self.device)

        # Initialize model
        self.model = TransformerNetwork(
            input_size=1,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            max_seq_length=self.lookback_period
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            num_batches = 0

            for i in range(0, len(X), self.batch_size):
                batch_X = X[i:i + self.batch_size]
                batch_y = y[i:i + self.batch_size]

                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.6f}')

    def predict(
        self,
        data: pd.Series,
        steps: int = 1
    ) -> np.ndarray:
        """
        Make predictions.

        Args:
            data: Recent time series data
            steps: Number of steps to forecast

        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        self.model.eval()

        # Prepare last sequence
        scaled_data = self.scaler.transform(data.values.reshape(-1, 1))
        last_sequence = scaled_data[-self.lookback_period:]

        predictions = []

        with torch.no_grad():
            for _ in range(steps):
                # Prepare input
                X = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)

                # Predict
                pred = self.model(X)
                pred_value = pred.cpu().numpy()[0]

                predictions.append(pred_value)

                # Update sequence
                last_sequence = np.vstack([last_sequence[1:], pred_value])

        # Inverse transform
        predictions = self.scaler.inverse_transform(np.array(predictions))

        return predictions.flatten()

    def forecast_with_confidence(
        self,
        data: pd.Series,
        steps: int = 1,
        num_simulations: int = 100
    ) -> pd.DataFrame:
        """
        Generate forecasts with confidence intervals using Monte Carlo dropout.

        Args:
            data: Recent time series data
            steps: Number of steps to forecast
            num_simulations: Number of Monte Carlo simulations

        Returns:
            DataFrame with mean, lower, and upper bounds
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Enable dropout during inference
        def enable_dropout(m):
            if isinstance(m, nn.Dropout):
                m.train()

        self.model.apply(enable_dropout)

        # Prepare data
        scaled_data = self.scaler.transform(data.values.reshape(-1, 1))
        last_sequence = scaled_data[-self.lookback_period:]

        all_predictions = []

        with torch.no_grad():
            for _ in range(num_simulations):
                predictions = []
                current_sequence = last_sequence.copy()

                for _ in range(steps):
                    X = torch.FloatTensor(current_sequence).unsqueeze(0).to(self.device)
                    pred = self.model(X)
                    pred_value = pred.cpu().numpy()[0]

                    predictions.append(pred_value)
                    current_sequence = np.vstack([current_sequence[1:], pred_value])

                all_predictions.append(predictions)

        # Calculate statistics
        all_predictions = np.array(all_predictions)  # [num_simulations, steps, 1]
        all_predictions = self.scaler.inverse_transform(
            all_predictions.reshape(-1, 1)
        ).reshape(num_simulations, steps)

        mean_forecast = np.mean(all_predictions, axis=0)
        lower_bound = np.percentile(all_predictions, 2.5, axis=0)
        upper_bound = np.percentile(all_predictions, 97.5, axis=0)

        return pd.DataFrame({
            'forecast': mean_forecast,
            'lower_95': lower_bound,
            'upper_95': upper_bound
        })

    def get_attention_weights(
        self,
        data: pd.Series
    ) -> np.ndarray:
        """
        Extract attention weights for interpretability.

        Args:
            data: Time series data

        Returns:
            Attention weights array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        self.model.eval()

        # Prepare data
        scaled_data = self.scaler.transform(data.values.reshape(-1, 1))
        last_sequence = scaled_data[-self.lookback_period:]
        X = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)

        # Hook to capture attention weights
        attention_weights = []

        def hook_fn(module, input_data, output):
            # Capture attention weights from encoder layers
            if hasattr(module, 'self_attn'):
                attention_weights.append(output[1])

        # Register hooks
        for layer in self.model.transformer_encoder.layers:
            layer.register_forward_hook(hook_fn)

        # Forward pass
        with torch.no_grad():
            self.model(X)

        if attention_weights:
            return attention_weights[0].cpu().numpy()
        return None
