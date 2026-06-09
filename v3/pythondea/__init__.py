"""Public API for pythonDEA v3."""

from ._version import __version__
from .core.data import DEAData, PanelDEAData
from .core.estimator import Estimator, FitContext
from .core.registry import (
    DEFAULT_REGISTRY,
    ModelSpec,
    get_model,
    list_models,
    register_model,
)
from .core.results import ModelResult, ResultTable
from .core.solver import SciPyHiGHSBackend, SolverBackend, SolverInfo
from .frontier import fit
from .models.sbm import SBMEstimator, SBMMalmquistEstimator, register_sbm_models

register_sbm_models()

__all__ = [
    "DEAData",
    "DEFAULT_REGISTRY",
    "Estimator",
    "FitContext",
    "ModelResult",
    "ModelSpec",
    "PanelDEAData",
    "ResultTable",
    "SBMEstimator",
    "SBMMalmquistEstimator",
    "SciPyHiGHSBackend",
    "SolverBackend",
    "SolverInfo",
    "__version__",
    "fit",
    "get_model",
    "list_models",
    "register_model",
    "register_sbm_models",
]
