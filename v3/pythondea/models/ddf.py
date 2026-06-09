"""Directional distance and Malmquist-Luenberger v4 model plugins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from v2 import (
    DEAData,
    PanelDEAData,
    ReferenceSet,
    compute_adjacent_malmquist_luenberger,
    solve_all_ddf,
)

from .._version import __version__
from ..core.estimator import FitContext
from ..core.registry import DEFAULT_REGISTRY, ModelRegistry, ModelSpec
from ..core.results import ModelResult, ResultTable
from ..core.solver import SciPyHiGHSBackend, SolverBackend

ReturnsToScale = Literal["crs", "vrs"]
DDFDirection = Literal["observed", "input", "output", "bad_output", "unit"]


@dataclass(frozen=True)
class DirectionalDistanceEstimator:
    """Directional distance function estimator for environmental DEA."""

    returns_to_scale: ReturnsToScale = "crs"
    direction: DDFDirection = "bad_output"
    reference_set: ReferenceSet | None = None
    exclude_self: bool = False
    peer_tolerance: float = 1e-7
    solver_backend: SolverBackend = SciPyHiGHSBackend()

    model_name: str = "directional_distance"

    def fit(self, data: DEAData, *, context: FitContext | None = None) -> ModelResult:
        if not isinstance(data, DEAData):
            raise TypeError("DirectionalDistanceEstimator.fit expects a DEAData instance")

        solutions = solve_all_ddf(
            data,
            returns_to_scale=self.returns_to_scale,
            direction=self.direction,
            reference_set=self.reference_set,
            exclude_self=self.exclude_self,
            peer_tolerance=self.peer_tolerance,
        )
        distance_rows = [
            {
                "dmu": solution.dmu_name,
                "beta": solution.beta,
                "success": solution.success,
                "status": solution.status,
                "reference": solution.reference_label,
                "peers": ",".join(solution.peers),
            }
            for solution in solutions
        ]
        target_rows = []
        for solution in solutions:
            for name, value in solution.input_targets.items():
                target_rows.append({"dmu": solution.dmu_name, "variable": name, "role": "input", "target": value})
            for name, value in solution.good_output_targets.items():
                target_rows.append({"dmu": solution.dmu_name, "variable": name, "role": "good_output", "target": value})
            for name, value in solution.bad_output_targets.items():
                target_rows.append({"dmu": solution.dmu_name, "variable": name, "role": "bad_output", "target": value})

        return ModelResult(
            model=self.model_name,
            status="ok" if all(solution.success for solution in solutions) else "partial",
            primary_table="directional_distance",
            tables=(
                ResultTable.from_rows(
                    "directional_distance",
                    distance_rows,
                    columns=("dmu", "beta", "success", "status", "reference", "peers"),
                ),
                ResultTable.from_rows(
                    "targets",
                    target_rows,
                    columns=("dmu", "variable", "role", "target"),
                ),
            ),
            metadata={
                "pythondea_version": __version__,
                "solver_backend": self.solver_backend.info().to_dict(),
                "returns_to_scale": self.returns_to_scale,
                "direction": self.direction,
                "exclude_self": self.exclude_self,
                "context": None if context is None else context.metadata,
            },
            artifacts={"solutions": {solution.dmu_name: solution for solution in solutions}},
        )


@dataclass(frozen=True)
class MalmquistLuenbergerEstimator:
    """Adjacent-period Malmquist-Luenberger index based on DDF distances."""

    returns_to_scale: ReturnsToScale = "crs"
    direction: DDFDirection = "bad_output"
    solver_backend: SolverBackend = SciPyHiGHSBackend()

    model_name: str = "malmquist_luenberger"

    def fit(self, data: PanelDEAData, *, context: FitContext | None = None) -> ModelResult:
        if not isinstance(data, PanelDEAData):
            raise TypeError("MalmquistLuenbergerEstimator.fit expects a PanelDEAData instance")

        transitions = compute_adjacent_malmquist_luenberger(
            data,
            returns_to_scale=self.returns_to_scale,
            direction=self.direction,
        )
        rows = [
            {
                "dmu": transition.dmu,
                "period_from": transition.period_from,
                "period_to": transition.period_to,
                "ml_index": transition.ml_index,
                "efficiency_change": transition.efficiency_change,
                "technical_change": transition.technical_change,
                "d_t_t": transition.distances["d_t_t"],
                "d_t1_t1": transition.distances["d_t1_t1"],
                "d_t_t1": transition.distances["d_t_t1"],
                "d_t1_t": transition.distances["d_t1_t"],
            }
            for transition in transitions
        ]
        return ModelResult(
            model=self.model_name,
            status="ok" if all(row["ml_index"] is not None for row in rows) else "partial",
            primary_table="malmquist_luenberger",
            tables=(
                ResultTable.from_rows(
                    "malmquist_luenberger",
                    rows,
                    columns=(
                        "dmu",
                        "period_from",
                        "period_to",
                        "ml_index",
                        "efficiency_change",
                        "technical_change",
                        "d_t_t",
                        "d_t1_t1",
                        "d_t_t1",
                        "d_t1_t",
                    ),
                ),
            ),
            metadata={
                "pythondea_version": __version__,
                "solver_backend": self.solver_backend.info().to_dict(),
                "returns_to_scale": self.returns_to_scale,
                "direction": self.direction,
                "reference_scheme": "adjacent contemporaneous DDF frontiers",
                "context": None if context is None else context.metadata,
            },
            artifacts={"transitions": transitions},
        )


def register_ddf_models(registry: ModelRegistry = DEFAULT_REGISTRY) -> None:
    """Register DDF-family v4 models idempotently."""

    for spec in (
        ModelSpec(
            name="directional_distance",
            estimator=DirectionalDistanceEstimator,
            family="dea-environmental",
            summary="Directional distance function DEA for input contraction, output expansion, and bad-output reduction.",
            citation_hint="Chung, Fare, and Grosskopf style directional distance functions.",
            keywords=("dea", "ddf", "environmental", "bad-output", "directional-distance"),
        ),
        ModelSpec(
            name="malmquist_luenberger",
            estimator=MalmquistLuenbergerEstimator,
            family="dea-panel",
            summary="Adjacent-period Malmquist-Luenberger productivity index based on DDF distances.",
            citation_hint="Malmquist-Luenberger productivity indices for undesirable outputs.",
            keywords=("dea", "ddf", "malmquist-luenberger", "panel", "green-productivity"),
        ),
    ):
        registry.register(spec, replace=True)
