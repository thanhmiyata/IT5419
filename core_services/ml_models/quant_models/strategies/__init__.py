"""
Trading Strategies
==================

Algorithmic trading strategies:
- Momentum strategies
- Mean reversion strategies
- Pairs trading strategies
"""

from core_services.ml_models.quant_models.strategies.breakout import BreakoutStrategy
from core_services.ml_models.quant_models.strategies.mean_reversion import MeanReversionStrategy
from core_services.ml_models.quant_models.strategies.momentum import MomentumStrategy
from core_services.ml_models.quant_models.strategies.pairs_trading import PairsTradingStrategy

__all__ = [
    "MomentumStrategy",
    "MeanReversionStrategy",
    "PairsTradingStrategy",
    "BreakoutStrategy",
]

__version__ = "1.0.0"
