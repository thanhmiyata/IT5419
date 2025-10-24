"""
Example: Training LSTM Model for Stock Price Prediction
========================================================

This example demonstrates how to train and evaluate an LSTM model.
"""

import numpy as np
import pandas as pd

from core_services.ml_models.quant_models.deep_learning.lstm_model import LSTMPredictor
from core_services.ml_models.quant_models.evaluation.metrics import ForecastMetrics
from core_services.ml_models.quant_models.evaluation.train_utils import DataSplitter, ModelCheckpoint
from core_services.utils.logger_utils import logger


def generate_sample_data(n_samples=1000):
    """Generate sample stock price data."""
    # Generate synthetic price data with trend and noise
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')

    # Trend + seasonality + noise
    trend = np.linspace(100, 150, n_samples)
    seasonality = 10 * np.sin(np.arange(n_samples) * 2 * np.pi / 365)
    noise = np.random.normal(0, 2, n_samples)

    prices = trend + seasonality + noise

    df = pd.DataFrame({
        'date': dates,
        'close': prices
    })
    df.set_index('date', inplace=True)

    return df


def main():
    """Main training function."""
    logger.info("=" * 60)
    logger.info("LSTM Stock Price Prediction Example")
    logger.info("=" * 60)

    # 1. Load data
    logger.info("\n1. Generating sample data...")
    data = generate_sample_data(n_samples=1000)
    logger.info(f"   Data shape: {data.shape}")
    logger.info(f"   Date range: {data.index[0]} to {data.index[-1]}")

    # 2. Split data
    logger.info("\n2. Splitting data...")
    splitter = DataSplitter()
    train_data, val_data, test_data = splitter.train_val_test_split_timeseries(
        data,
        val_size=0.15,
        test_size=0.15
    )
    logger.info(f"   Train: {len(train_data)} samples")
    logger.info(f"   Val:   {len(val_data)} samples")
    logger.info(f"   Test:  {len(test_data)} samples")

    # 3. Initialize model
    logger.info("\n3. Initializing LSTM model...")
    model = LSTMPredictor(
        lookback_period=60,
        hidden_size=50,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001,
        epochs=50,
        batch_size=32
    )

    # 4. Train model
    logger.info("\n4. Training model...")
    logger.info("-" * 60)

    # Train
    model.fit(train_data['close'])

    logger.info("-" * 60)
    logger.info("   Training completed!")

    # 5. Save model
    logger.info("\n5. Saving model...")
    checkpoint = ModelCheckpoint()
    checkpoint.save_sklearn_model(
        model,
        filepath='models/lstm_model.pkl',
        metadata={
            'lookback_period': 60,
            'train_samples': len(train_data),
            'model_type': 'LSTM'
        }
    )
    logger.info("   Model saved to: models/lstm_model.pkl")

    # 6. Make predictions
    logger.info("\n6. Making predictions...")
    train_pred = model.predict(train_data['close'], steps=len(val_data))
    val_pred = model.predict(
        pd.concat([train_data['close'], val_data['close']]),
        steps=len(test_data)
    )

    # 7. Evaluate
    logger.info("\n7. Evaluating model...")
    logger.info("-" * 60)

    # Validation metrics
    val_metrics = ForecastMetrics.calculate_all_forecast_metrics(
        val_data['close'].values[:len(train_pred)],
        train_pred
    )

    logger.info("   Validation Metrics:")
    for metric, value in val_metrics.items():
        logger.info(f"   {metric:25s}: {value:>12.6f}")

    # Test metrics
    test_metrics = ForecastMetrics.calculate_all_forecast_metrics(
        test_data['close'].values[:len(val_pred)],
        val_pred
    )

    logger.info("\n   Test Metrics:")
    for metric, value in test_metrics.items():
        logger.info(f"   {metric:25s}: {value:>12.6f}")

    logger.info("-" * 60)

    # 8. Summary
    logger.info("\n8. Summary:")
    logger.info(f"   Final RMSE (Validation): {val_metrics['rmse']:.4f}")
    logger.info(f"   Final RMSE (Test):       {test_metrics['rmse']:.4f}")
    logger.info(f"   Directional Accuracy:    {test_metrics['directional_accuracy']:.2%}")

    logger.info("\n" + "=" * 60)
    logger.info("Training completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
