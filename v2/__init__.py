"""pythonDEA v2 computational core."""

from .data import DEAData
from .panel import PanelDEAData
from .reference import (
    ReferenceSet,
    contemporaneous_reference,
    cross_period_reference,
    global_reference,
    window_reference,
)
from .result import SBMSolution
from .sbm import solve_all_sbm, solve_sbm
from .variables import VariableSpec

__all__ = [
    "DEAData",
    "PanelDEAData",
    "ReferenceSet",
    "SBMSolution",
    "VariableSpec",
    "contemporaneous_reference",
    "cross_period_reference",
    "global_reference",
    "solve_all_sbm",
    "solve_sbm",
    "window_reference",
]
