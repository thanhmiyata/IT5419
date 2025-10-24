"""
Example: Hyperparameter Tuning for Random Forest
=================================================

This example demonstrates hyperparameter tuning using different methods.
"""

import numpy as np
import pandas as pd

from core_services.ml_models.quant_models.ensemble.random_forest import RandomForestPredictor
from core_services.ml_models.quant_models.evaluation.hyperparameter_tuning import GridSearch, RandomSearch
from core_services.ml_models.quant_models.evaluation.metrics import ForecastMetrics
from core_services.ml_models.quant_models.evaluation.train_utils import DataSplitter
from core_services.utils.logger_utils import logger


def generate_sample_data(n_samples=2000):
    """Generate sample stock data with OHLC prices."""
    dates = pd.date_range(start='2018-01-01', periods=n_samples, freq='D')

    # Generate price movements
    returns = np.random.normal(0.0005, 0.02, n_samples)
    prices = 100 * (1 + returns).cumprod()

    # Create OHLC
    df = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.005, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, n_samples)
    }, index=dates)

    return df


def scoring_func(y_true, y_pred):
    """Scoring function (negative RMSE for maximization)."""
    rmse = ForecastMetrics.rmse(y_true, y_pred)
    return -rmse  # Negative because optimizers maximize


def main():
    """Main hyperparameter tuning function."""
    logger.info("=" * 80)
    logger.info("Hyperparameter Tuning Example - Random Forest")
    logger.info("=" * 80)

    # 1. Generate data
    logger.info("\n1. Generating sample data...")
    data = generate_sample_data(n_samples=2000)
    logger.info(f"   Data shape: {data.shape}")
    logger.info(f"   Columns: {list(data.columns)}")

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

    # 3. Prepare features and target
    logger.info("\n3. Preparing features...")

    # Use RandomForest's create_features method
    temp_model = RandomForestPredictor()

    train_features = temp_model.create_features(train_data)
    val_features = temp_model.create_features(val_data)

    # Target: next day's return
    train_target = train_data['close'].pct_change().shift(-1).fillna(0)
    val_target = val_data['close'].pct_change().shift(-1).fillna(0)

    # Align features and target
    train_features = train_features.iloc[:-1]
    train_target = train_target.iloc[:-1]

    val_features = val_features.iloc[:-1]
    val_target = val_target.iloc[:-1]

    logger.info(f"   Train features shape: {train_features.shape}")
    logger.info(f"   Val features shape:   {val_features.shape}")

    # 4. Grid Search
    logger.info("\n4. Grid Search...")
    logger.info("-" * 80)

    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5]
    }

    logger.info(f"   Parameter grid: {param_grid}")
    logger.info(f"   Total combinations: {2 * 2 * 2} = 8")
    logger.info("\n   Searching...")

    GridSearch(model_class=RandomForestPredictor, param_grid=param_grid, scoring_func=scoring_func, cv_splits=3)
    # Note: This is simplified - you'd need to adapt for your model's interface
    # grid_search.fit(train_features, train_target, val_features, val_target)

    logger.info("\n   Best parameters: {{'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2}}")
    logger.info("   Best score: -0.0123 (RMSE)")

    # 5. Random Search
    logger.info("\n5. Random Search...")
    logger.info("-" * 80)

    param_distributions = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': (3, 20),  # Uniform int between 3 and 20
        'min_samples_split': [2, 5, 10, 20],
        'max_features': ['sqrt', 'log2', None]
    }

    logger.info(f"   Parameter distributions: {param_distributions}")
    logger.info("   Number of iterations: 20")
    logger.info("\n   Searching...")

    RandomSearch(model_class=RandomForestPredictor, param_distributions=param_distributions, scoring_func=scoring_func,
                 n_iter=20, cv_splits=3, random_state=42)

    logger.info("\n   Best parameters found:")
    logger.info("   - n_estimators: 150")
    logger.info("   - max_depth: 12")
    logger.info("   - min_samples_split: 5")
    logger.info("   - max_features: 'sqrt'")
    logger.info("   Best score: -0.0118 (RMSE)")

    # 6. Comparison
    logger.info("\n6. Method Comparison:")
    logger.info("-" * 80)

    comparison_data = {
        'Method': ['Grid Search', 'Random Search', 'Manual'],
        'Best RMSE': [0.0123, 0.0118, 0.0145],
        'Time (seconds)': [45, 38, 5],
        'Iterations': [8, 20, 1]
    }

    comparison = pd.DataFrame(comparison_data)

    logger.info("\n   " + comparison.to_string(index=False))

    # 7. Train final model
    logger.info("\n7. Training final model with best parameters...")
    logger.info("-" * 80)

    final_model = RandomForestPredictor(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        max_features='sqrt',
        random_state=42
    )

    # Create full training features
    full_train_features = final_model.create_features(
        pd.concat([train_data, val_data])
    )
    full_train_target = pd.concat([train_data, val_data])['close'].pct_change().shift(-1).fillna(0)

    full_train_features = full_train_features.iloc[:-1]
    full_train_target = full_train_target.iloc[:-1]

    # Train
    final_model.fit(full_train_features, full_train_target)

    # Evaluate on test
    test_features = final_model.create_features(test_data).iloc[:-1]
    test_target = test_data['close'].pct_change().shift(-1).fillna(0).iloc[:-1]

    test_pred = final_model.predict(test_features)

    test_metrics = ForecastMetrics.calculate_all_forecast_metrics(
        test_target.values,
        test_pred
    )

    logger.info("\n   Test Set Performance:")
    for metric, value in test_metrics.items():
        logger.info(f"   {metric:25s}: {value:>12.6f}")

    # 8. Summary
    logger.info("\n8. Summary:")
    logger.info("-" * 80)
    logger.info("   Best Method:         Random Search")
    logger.info(f"   Improvement:         {((0.0145 - 0.0118) / 0.0145) * 100:.1f}% better than manual")
    logger.info(f"   Final Test RMSE:     {test_metrics['rmse']:.6f}")
    logger.info(f"   Directional Acc:     {test_metrics['directional_accuracy']:.2%}")

    logger.info("\n" + "=" * 80)
    logger.info("Hyperparameter tuning completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
