"""
Portfolio Optimization
======================

Portfolio construction and optimization methods:
- Markowitz Mean-Variance optimization
- Black-Litterman model
- Risk Parity
"""

from core_services.ml_models.quant_models.optimization.black_litterman import BlackLittermanModel
from core_services.ml_models.quant_models.optimization.kelly_criterion import KellyCriterion
from core_services.ml_models.quant_models.optimization.markowitz import MarkowitzOptimizer
from core_services.ml_models.quant_models.optimization.risk_parity import RiskParityOptimizer

__all__ = [
    "MarkowitzOptimizer",
    "BlackLittermanModel",
    "RiskParityOptimizer",
    "KellyCriterion",
]

__version__ = "1.0.0"
