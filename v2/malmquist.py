"""SBM-Malmquist orchestration built on v2 reference sets."""

from __future__ import annotations

from dataclasses import dataclass

from .decomposition import DecompositionResult, TransitionEfficiencies, decompose_transition
from .orientation import SBMOrientation
from .panel import PanelDEAData
from .reference import contemporaneous_reference, global_reference
from .result import SBMSolution
from .sbm import solve_sbm


@dataclass(frozen=True)
class MalmquistTransition:
    """One entity's adjacent-period Malmquist result plus explainable solutions."""

    efficiencies: TransitionEfficiencies
    decomposition: DecompositionResult
    solutions: dict[str, SBMSolution]


def compute_adjacent_malmquist(
    panel: PanelDEAData,
    *,
    orientation: SBMOrientation = "non_oriented",
) -> list[MalmquistTransition]:
    """Compute adjacent-period v2 SBM-Malmquist transitions for all entities."""

    transitions: list[MalmquistTransition] = []
    for period_from, period_to in zip(panel.periods, panel.periods[1:]):
        for entity in panel.entities:
            transitions.append(
                compute_transition(
                    panel,
                    period_from=period_from,
                    period_to=period_to,
                    entity=entity,
                    orientation=orientation,
                )
            )
    return transitions


def compute_transition(
    panel: PanelDEAData,
    *,
    period_from: str,
    period_to: str,
    entity: str,
    orientation: SBMOrientation = "non_oriented",
) -> MalmquistTransition:
    """Compute all efficiency points for one entity and adjacent transition."""

    row_t = panel.row_index(period_from, entity)
    row_t1 = panel.row_index(period_to, entity)
    ref_t = contemporaneous_reference(panel, period_from)
    ref_t1 = contemporaneous_reference(panel, period_to)
    ref_g = global_reference(panel)

    solutions = {
        "crs_tt": solve_sbm(panel.data, row_t, returns_to_scale="crs", orientation=orientation, reference_set=ref_t),
        "vrs_tt": solve_sbm(panel.data, row_t, returns_to_scale="vrs", orientation=orientation, reference_set=ref_t),
        "crs_t1t1": solve_sbm(panel.data, row_t1, returns_to_scale="crs", orientation=orientation, reference_set=ref_t1),
        "vrs_t1t1": solve_sbm(panel.data, row_t1, returns_to_scale="vrs", orientation=orientation, reference_set=ref_t1),
        "crs_tt1": solve_sbm(panel.data, row_t1, returns_to_scale="crs", orientation=orientation, reference_set=ref_t),
        "vrs_tt1": solve_sbm(panel.data, row_t1, returns_to_scale="vrs", orientation=orientation, reference_set=ref_t),
        "crs_t1t": solve_sbm(panel.data, row_t, returns_to_scale="crs", orientation=orientation, reference_set=ref_t1),
        "vrs_t1t": solve_sbm(panel.data, row_t, returns_to_scale="vrs", orientation=orientation, reference_set=ref_t1),
        "crs_gt": solve_sbm(panel.data, row_t, returns_to_scale="crs", orientation=orientation, reference_set=ref_g),
        "vrs_gt": solve_sbm(panel.data, row_t, returns_to_scale="vrs", orientation=orientation, reference_set=ref_g),
        "crs_gt1": solve_sbm(panel.data, row_t1, returns_to_scale="crs", orientation=orientation, reference_set=ref_g),
        "vrs_gt1": solve_sbm(panel.data, row_t1, returns_to_scale="vrs", orientation=orientation, reference_set=ref_g),
    }
    efficiencies = TransitionEfficiencies(
        dmu=str(entity),
        period_from=str(period_from),
        period_to=str(period_to),
        ectt=solutions["crs_tt"].score,
        evtt=solutions["vrs_tt"].score,
        ectt_next=solutions["crs_t1t1"].score,
        evtt_next=solutions["vrs_t1t1"].score,
        ectt1=solutions["crs_tt1"].score,
        evtt1=solutions["vrs_tt1"].score,
        ect1t=solutions["crs_t1t"].score,
        evt1t=solutions["vrs_t1t"].score,
        ecgt=solutions["crs_gt"].score,
        evgt=solutions["vrs_gt"].score,
        ecgt_next=solutions["crs_gt1"].score,
        evgt_next=solutions["vrs_gt1"].score,
    )
    return MalmquistTransition(
        efficiencies=efficiencies,
        decomposition=decompose_transition(efficiencies),
        solutions=solutions,
    )
