"""SBM-family v3 estimators backed by the v2 computational core."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Literal

from v2 import (
    DEAData,
    PanelDEAData,
    ReferenceSet,
    compute_adjacent_malmquist,
    solve_all_sbm,
)

from .._version import __version__
from ..core.estimator import FitContext
from ..core.registry import DEFAULT_REGISTRY, ModelRegistry, ModelSpec
from ..core.results import ModelResult, ResultTable
from ..core.solver import SciPyHiGHSBackend, SolverBackend

ReturnsToScale = Literal["crs", "vrs"]
SBMOrientation = Literal["non_oriented", "input", "output", "bad_output_adjusted"]


@dataclass(frozen=True)
class SBMEstimator:
    """Slack-based measure estimator with explainable targets and peers."""

    returns_to_scale: ReturnsToScale = "crs"
    orientation: SBMOrientation = "non_oriented"
    reference_set: ReferenceSet | None = None
    exclude_self: bool = False
    peer_tolerance: float = 1e-7
    solver_backend: SolverBackend = SciPyHiGHSBackend()

    model_name: str = "sbm"

    def fit(self, data: DEAData, *, context: FitContext | None = None) -> ModelResult:
        if not isinstance(data, DEAData):
            raise TypeError("SBMEstimator.fit expects a DEAData instance")

        solutions = solve_all_sbm(
            data,
            returns_to_scale=self.returns_to_scale,
            orientation=self.orientation,
            reference_set=self.reference_set,
            exclude_self=self.exclude_self,
            peer_tolerance=self.peer_tolerance,
        )
        score_rows = [
            {
                "dmu": solution.dmu_name,
                "score": solution.score,
                "success": solution.success,
                "status": solution.status,
                "reference": solution.reference_label,
                "peers": ",".join(solution.peers),
            }
            for solution in solutions
        ]
        slack_rows = []
        target_rows = []
        for solution in solutions:
            for name, value in solution.input_slacks.items():
                slack_rows.append({"dmu": solution.dmu_name, "variable": name, "role": "input", "slack": value})
            for name, value in solution.good_output_slacks.items():
                slack_rows.append({"dmu": solution.dmu_name, "variable": name, "role": "good_output", "slack": value})
            for name, value in solution.bad_output_slacks.items():
                slack_rows.append({"dmu": solution.dmu_name, "variable": name, "role": "bad_output", "slack": value})
            for name, value in solution.input_targets.items():
                target_rows.append({"dmu": solution.dmu_name, "variable": name, "role": "input", "target": value})
            for name, value in solution.good_output_targets.items():
                target_rows.append({"dmu": solution.dmu_name, "variable": name, "role": "good_output", "target": value})
            for name, value in solution.bad_output_targets.items():
                target_rows.append({"dmu": solution.dmu_name, "variable": name, "role": "bad_output", "target": value})

        tables = (
            ResultTable.from_rows(
                "efficiency",
                score_rows,
                columns=("dmu", "score", "success", "status", "reference", "peers"),
            ),
            ResultTable.from_rows(
                "slacks",
                slack_rows,
                columns=("dmu", "variable", "role", "slack"),
            ),
            ResultTable.from_rows(
                "targets",
                target_rows,
                columns=("dmu", "variable", "role", "target"),
            ),
        )
        return ModelResult(
            model=self.model_name,
            status="ok" if all(solution.success for solution in solutions) else "partial",
            primary_table="efficiency",
            tables=tables,
            metadata={
                "pythondea_version": __version__,
                "solver_backend": self.solver_backend.info().to_dict(),
                "returns_to_scale": self.returns_to_scale,
                "orientation": self.orientation,
                "exclude_self": self.exclude_self,
                "context": None if context is None else context.metadata,
            },
            artifacts={"solutions": {solution.dmu_name: solution for solution in solutions}},
        )


@dataclass(frozen=True)
class SBMMalmquistEstimator:
    """Adjacent-period SBM-Malmquist estimator with decomposition tables."""

    orientation: SBMOrientation = "non_oriented"
    solver_backend: SolverBackend = SciPyHiGHSBackend()

    model_name: str = "sbm_malmquist"

    def fit(self, data: PanelDEAData, *, context: FitContext | None = None) -> ModelResult:
        if not isinstance(data, PanelDEAData):
            raise TypeError("SBMMalmquistEstimator.fit expects a PanelDEAData instance")

        transitions = compute_adjacent_malmquist(data, orientation=self.orientation)
        rows = []
        for transition in transitions:
            eff = transition.efficiencies
            rows.append(
                {
                    "dmu": eff.dmu,
                    "period_from": eff.period_from,
                    "period_to": eff.period_to,
                    "fglr1992_mlc": transition.decomposition.fglr1992_crs["mlc"],
                    "fglr1992_ecc": transition.decomposition.fglr1992_crs["ecc"],
                    "fglr1992_tcc": transition.decomposition.fglr1992_crs["tcc"],
                    "pl2005_mgc": transition.decomposition.pl2005_crs["mgc"],
                    "pl2005_bpcc": transition.decomposition.pl2005_crs["bpcc"],
                }
            )
        tables = (
            ResultTable.from_rows(
                "malmquist",
                rows,
                columns=(
                    "dmu",
                    "period_from",
                    "period_to",
                    "fglr1992_mlc",
                    "fglr1992_ecc",
                    "fglr1992_tcc",
                    "pl2005_mgc",
                    "pl2005_bpcc",
                ),
            ),
        )
        return ModelResult(
            model=self.model_name,
            status="ok",
            primary_table="malmquist",
            tables=tables,
            metadata={
                "pythondea_version": __version__,
                "solver_backend": self.solver_backend.info().to_dict(),
                "orientation": self.orientation,
                "reference_scheme": "adjacent-period, cross-period, and global frontiers",
                "context": None if context is None else context.metadata,
            },
            artifacts={"transitions": transitions},
        )


@dataclass(frozen=True)
class SBMSuperEfficiencyEstimator:
    """Exclude-self SBM estimator for frontier ranking and sensitivity checks."""

    returns_to_scale: ReturnsToScale = "crs"
    orientation: SBMOrientation = "non_oriented"
    reference_set: ReferenceSet | None = None
    peer_tolerance: float = 1e-7
    solver_backend: SolverBackend = SciPyHiGHSBackend()

    model_name: str = "sbm_super_efficiency"

    def fit(self, data: DEAData, *, context: FitContext | None = None) -> ModelResult:
        base = SBMEstimator(
            returns_to_scale=self.returns_to_scale,
            orientation=self.orientation,
            reference_set=self.reference_set,
            exclude_self=True,
            peer_tolerance=self.peer_tolerance,
            solver_backend=self.solver_backend,
        ).fit(data, context=context)
        metadata = dict(base.metadata)
        metadata["super_efficiency"] = True
        return replace(base, model=self.model_name, metadata=metadata)


def register_sbm_models(registry: ModelRegistry = DEFAULT_REGISTRY) -> None:
    """Register built-in SBM-family models idempotently."""

    for spec in (
        ModelSpec(
            name="sbm",
            estimator=SBMEstimator,
            family="dea",
            summary="Slack-based measure DEA with peers, slacks, targets, and undesirable outputs.",
            citation_hint="Tone (2001) SBM; environmental DEA extensions for bad outputs.",
            keywords=("dea", "sbm", "undesirable-output", "frontier"),
        ),
        ModelSpec(
            name="sbm_malmquist",
            estimator=SBMMalmquistEstimator,
            family="dea-panel",
            summary="Adjacent-period SBM-Malmquist productivity decomposition.",
            citation_hint="FGLR, RD, Zofio, and Pastor-Lovell Malmquist decompositions.",
            keywords=("dea", "sbm", "malmquist", "panel", "productivity"),
        ),
        ModelSpec(
            name="sbm_super_efficiency",
            estimator=SBMSuperEfficiencyEstimator,
            family="dea",
            summary="Exclude-self SBM for ranking efficient DMUs and sensitivity checks.",
            citation_hint="Super-efficiency DEA extensions built on SBM reference exclusion.",
            keywords=("dea", "sbm", "super-efficiency", "ranking"),
            experimental=True,
        ),
    ):
        registry.register(spec, replace=True)
