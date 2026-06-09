"""pythonDEA v2 computational core."""

from .data import DEAData
from .ddf import (
    DDFSolution,
    MalmquistLuenbergerTransition,
    compute_adjacent_malmquist_luenberger,
    compute_malmquist_luenberger_transition,
    solve_all_ddf,
    solve_ddf,
)
from .decomposition import DecompositionResult, TransitionEfficiencies, decompose_transition
from .malmquist import MalmquistTransition, compute_adjacent_malmquist, compute_transition
from .orientation import SBMOrientation
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
    "DDFSolution",
    "DecompositionResult",
    "MalmquistLuenbergerTransition",
    "MalmquistTransition",
    "PanelDEAData",
    "ReferenceSet",
    "SBMSolution",
    "SBMOrientation",
    "TransitionEfficiencies",
    "VariableSpec",
    "contemporaneous_reference",
    "compute_adjacent_malmquist_luenberger",
    "cross_period_reference",
    "compute_malmquist_luenberger_transition",
    "compute_adjacent_malmquist",
    "compute_transition",
    "decompose_transition",
    "global_reference",
    "solve_all_ddf",
    "solve_all_sbm",
    "solve_ddf",
    "solve_sbm",
    "window_reference",
]
