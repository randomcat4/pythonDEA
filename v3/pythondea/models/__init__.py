"""Built-in v3 model plugins."""

from .sbm import SBMEstimator, SBMMalmquistEstimator, register_sbm_models

__all__ = ["SBMEstimator", "SBMMalmquistEstimator", "register_sbm_models"]
