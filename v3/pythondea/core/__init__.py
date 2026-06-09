"""Core extension contracts for pythonDEA v3."""

from .data import DEAData, PanelDEAData
from .estimator import Estimator, FitContext
from .registry import DEFAULT_REGISTRY, ModelSpec, ModelRegistry
from .results import ModelResult, ResultTable
from .solver import SciPyHiGHSBackend, SolverBackend, SolverInfo

__all__ = [
    "DEAData",
    "DEFAULT_REGISTRY",
    "Estimator",
    "FitContext",
    "ModelRegistry",
    "ModelResult",
    "ModelSpec",
    "PanelDEAData",
    "ResultTable",
    "SciPyHiGHSBackend",
    "SolverBackend",
    "SolverInfo",
]
