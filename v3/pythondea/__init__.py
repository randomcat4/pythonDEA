"""Public API for pythonDEA v3."""

from ._version import __version__
from .audit import AuditCheck, PublicationAudit, audit_result, model_catalog
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
from .dataframe import dea_from_dataframe, panel_from_dataframe
from .frontier import fit
from .models.sbm import (
    SBMEstimator,
    SBMMalmquistEstimator,
    SBMSuperEfficiencyEstimator,
    register_sbm_models,
)

register_sbm_models()

__all__ = [
    "AuditCheck",
    "DEAData",
    "DEFAULT_REGISTRY",
    "Estimator",
    "FitContext",
    "ModelResult",
    "ModelSpec",
    "PanelDEAData",
    "PublicationAudit",
    "ResultTable",
    "SBMEstimator",
    "SBMMalmquistEstimator",
    "SBMSuperEfficiencyEstimator",
    "SciPyHiGHSBackend",
    "SolverBackend",
    "SolverInfo",
    "__version__",
    "audit_result",
    "dea_from_dataframe",
    "fit",
    "get_model",
    "list_models",
    "model_catalog",
    "panel_from_dataframe",
    "register_model",
    "register_sbm_models",
]
