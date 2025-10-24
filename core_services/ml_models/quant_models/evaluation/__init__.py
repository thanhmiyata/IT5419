"""
Model Evaluation and Training Framework
========================================

Comprehensive tools for training, evaluating, and optimizing quantitative models.
"""

from core_services.ml_models.quant_models.evaluation.evaluator import (BacktestEvaluator, ModelEvaluator,
                                                                       StrategyComparator)
from core_services.ml_models.quant_models.evaluation.hyperparameter_tuning import (BayesianOptimization, GridSearch,
                                                                                   ParameterOptimizer, RandomSearch)
from core_services.ml_models.quant_models.evaluation.metrics import (ClassificationMetrics, ForecastMetrics,
                                                                     PerformanceMetrics)
from core_services.ml_models.quant_models.evaluation.train_utils import (CrossValidator, DataPreprocessor, DataSplitter,
                                                                         EarlyStopping, ModelCheckpoint, TrainingLogger)

__all__ = [
    # Metrics
    'PerformanceMetrics',
    'ForecastMetrics',
    'ClassificationMetrics',

    # Training utilities
    'DataSplitter',
    'ModelCheckpoint',
    'EarlyStopping',
    'TrainingLogger',
    'CrossValidator',
    'DataPreprocessor',

    # Evaluation
    'ModelEvaluator',
    'BacktestEvaluator',
    'StrategyComparator',

    # Hyperparameter tuning
    'GridSearch',
    'RandomSearch',
    'BayesianOptimization',
    'ParameterOptimizer'
]

__version__ = "1.0.0"
