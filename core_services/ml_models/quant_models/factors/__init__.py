"""
Factor Models
=============

Factor-based investment models:
- Fama-French factors
- Momentum factor
- Multi-factor alpha models
"""

from core_services.ml_models.quant_models.factors.fama_french import FamaFrenchModel
from core_services.ml_models.quant_models.factors.momentum_factor import MomentumFactor
from core_services.ml_models.quant_models.factors.multi_factor import MultiFactorModel

__all__ = [
    "FamaFrenchModel",
    "MomentumFactor",
    "MultiFactorModel",
]

__version__ = "1.0.0"
