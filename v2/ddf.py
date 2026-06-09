"""Directional distance functions and Malmquist-Luenberger indices."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Literal

import numpy as np
from scipy import optimize

from .panel import PanelDEAData
from .problem import DEAData
from .reference import ReferenceSet, contemporaneous_reference

ReturnsToScale = Literal["crs", "vrs"]
DDFDirection = Literal["observed", "input", "output", "bad_output", "unit"]


@dataclass(frozen=True)
class DDFSolution:
    """Result of one directional distance function evaluation."""

    dmu_index: int
    dmu_name: str
    beta: float | None
    returns_to_scale: ReturnsToScale
    direction: DDFDirection
    reference_label: str
    success: bool
    status: int
    message: str
    lambdas: dict[str, float]
    peers: dict[str, float]
    input_targets: dict[str, float]
    good_output_targets: dict[str, float]
    bad_output_targets: dict[str, float]


@dataclass(frozen=True)
class MalmquistLuenbergerTransition:
    """One adjacent-period Malmquist-Luenberger transition."""

    dmu: str
    period_from: str
    period_to: str
    ml_index: float | None
    efficiency_change: float | None
    technical_change: float | None
    distances: dict[str, float | None]
    solutions: dict[str, DDFSolution]


def solve_all_ddf(
    data: DEAData,
    *,
    returns_to_scale: ReturnsToScale = "crs",
    direction: DDFDirection = "bad_output",
    reference_set: ReferenceSet | None = None,
    exclude_self: bool = False,
    peer_tolerance: float = 1e-7,
) -> list[DDFSolution]:
    """Evaluate every DMU with the same directional distance model."""

    return [
        solve_ddf(
            data,
            dmu_index,
            returns_to_scale=returns_to_scale,
            direction=direction,
            reference_set=reference_set,
            exclude_self=exclude_self,
            peer_tolerance=peer_tolerance,
        )
        for dmu_index in range(data.n_dmus)
    ]


def solve_ddf(
    data: DEAData,
    dmu_index: int,
    *,
    returns_to_scale: ReturnsToScale = "crs",
    direction: DDFDirection = "bad_output",
    reference_set: ReferenceSet | None = None,
    exclude_self: bool = False,
    peer_tolerance: float = 1e-7,
) -> DDFSolution:
    """Solve a directional distance function LP for one DMU."""

    if returns_to_scale not in ("crs", "vrs"):
        raise ValueError("returns_to_scale must be 'crs' or 'vrs'")
    if dmu_index < 0 or dmu_index >= data.n_dmus:
        raise ValueError("dmu_index out of bounds")

    base_reference = reference_set or ReferenceSet.all(data.n_dmus)
    effective_reference = base_reference.without(dmu_index) if exclude_self else base_reference
    ref_idx = effective_reference.validate(data.n_dmus)

    x0 = data.inputs[dmu_index]
    y0 = data.good_outputs[dmu_index]
    b0 = None if data.bad_outputs is None else data.bad_outputs[dmu_index]
    gx, gy, gb = _direction_arrays(data, dmu_index, direction)

    lp = _build_ddf_lp(
        data.inputs[ref_idx].T,
        data.good_outputs[ref_idx].T,
        None if data.bad_outputs is None else data.bad_outputs[ref_idx].T,
        x0,
        y0,
        b0,
        gx,
        gy,
        gb,
        returns_to_scale,
    )
    result = optimize.linprog(
        c=lp["c"],
        A_ub=lp["A_ub"],
        b_ub=lp["b_ub"],
        A_eq=lp["A_eq"],
        b_eq=lp["b_eq"],
        bounds=lp["bounds"],
        method="highs",
    )
    if not result.success:
        return _failed_ddf(data, dmu_index, returns_to_scale, direction, effective_reference.label, result)
    return _successful_ddf(
        data=data,
        dmu_index=dmu_index,
        returns_to_scale=returns_to_scale,
        direction=direction,
        reference=effective_reference,
        ref_idx=ref_idx,
        result=result,
        gx=gx,
        gy=gy,
        gb=gb,
        peer_tolerance=peer_tolerance,
    )


def compute_adjacent_malmquist_luenberger(
    panel: PanelDEAData,
    *,
    returns_to_scale: ReturnsToScale = "crs",
    direction: DDFDirection = "bad_output",
) -> list[MalmquistLuenbergerTransition]:
    """Compute adjacent-period Malmquist-Luenberger transitions for all entities."""

    transitions: list[MalmquistLuenbergerTransition] = []
    for period_from, period_to in zip(panel.periods, panel.periods[1:]):
        for entity in panel.entities:
            transitions.append(
                compute_malmquist_luenberger_transition(
                    panel,
                    period_from=period_from,
                    period_to=period_to,
                    entity=entity,
                    returns_to_scale=returns_to_scale,
                    direction=direction,
                )
            )
    return transitions


def compute_malmquist_luenberger_transition(
    panel: PanelDEAData,
    *,
    period_from: str,
    period_to: str,
    entity: str,
    returns_to_scale: ReturnsToScale = "crs",
    direction: DDFDirection = "bad_output",
) -> MalmquistLuenbergerTransition:
    """Compute one adjacent-period Malmquist-Luenberger transition."""

    row_t = panel.row_index(period_from, entity)
    row_t1 = panel.row_index(period_to, entity)
    ref_t = contemporaneous_reference(panel, period_from)
    ref_t1 = contemporaneous_reference(panel, period_to)
    solutions = {
        "d_t_t": solve_ddf(panel.data, row_t, returns_to_scale=returns_to_scale, direction=direction, reference_set=ref_t),
        "d_t1_t1": solve_ddf(panel.data, row_t1, returns_to_scale=returns_to_scale, direction=direction, reference_set=ref_t1),
        "d_t_t1": solve_ddf(panel.data, row_t1, returns_to_scale=returns_to_scale, direction=direction, reference_set=ref_t),
        "d_t1_t": solve_ddf(panel.data, row_t, returns_to_scale=returns_to_scale, direction=direction, reference_set=ref_t1),
    }
    distances = {name: solution.beta for name, solution in solutions.items()}
    ml_index = _sqrt_product(
        _ratio_plus_one(distances["d_t_t"], distances["d_t_t1"]),
        _ratio_plus_one(distances["d_t1_t"], distances["d_t1_t1"]),
    )
    efficiency_change = _ratio_plus_one(distances["d_t_t"], distances["d_t1_t1"])
    technical_change = _div(ml_index, efficiency_change)
    return MalmquistLuenbergerTransition(
        dmu=str(entity),
        period_from=str(period_from),
        period_to=str(period_to),
        ml_index=ml_index,
        efficiency_change=efficiency_change,
        technical_change=technical_change,
        distances=distances,
        solutions=solutions,
    )


def _build_ddf_lp(
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    b_ref: np.ndarray | None,
    x0: np.ndarray,
    y0: np.ndarray,
    b0: np.ndarray | None,
    gx: np.ndarray,
    gy: np.ndarray,
    gb: np.ndarray,
    returns_to_scale: ReturnsToScale,
) -> dict[str, object]:
    n_ref = x_ref.shape[1]
    beta_col_x = gx[:, None]
    x_ub = np.hstack([x_ref, beta_col_x])
    beta_col_y = gy[:, None]
    y_ub = np.hstack([-y_ref, beta_col_y])
    blocks = [x_ub, y_ub]
    rhs = [x0, -y0]
    if b_ref is not None:
        assert b0 is not None
        beta_col_b = gb[:, None]
        blocks.append(np.hstack([b_ref, beta_col_b]))
        rhs.append(b0)

    c = np.zeros(n_ref + 1)
    c[-1] = -1.0
    bounds: list[tuple[float, float | None]] = [(0.0, None)] * n_ref + [(0.0, None)]
    a_eq = None
    b_eq = None
    if returns_to_scale == "vrs":
        a_eq = np.zeros((1, n_ref + 1))
        a_eq[0, :n_ref] = 1.0
        b_eq = np.array([1.0])

    return {
        "c": c,
        "A_ub": np.vstack(blocks),
        "b_ub": np.concatenate(rhs),
        "A_eq": a_eq,
        "b_eq": b_eq,
        "bounds": bounds,
    }


def _direction_arrays(
    data: DEAData,
    dmu_index: int,
    direction: DDFDirection,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0 = data.inputs[dmu_index]
    y0 = data.good_outputs[dmu_index]
    b0 = np.array([]) if data.bad_outputs is None else data.bad_outputs[dmu_index]
    if direction == "observed":
        return x0.copy(), y0.copy(), b0.copy()
    if direction == "input":
        return x0.copy(), np.zeros_like(y0), np.zeros_like(b0)
    if direction == "output":
        return np.zeros_like(x0), y0.copy(), np.zeros_like(b0)
    if direction == "bad_output":
        if data.n_bad_outputs == 0:
            raise ValueError("bad_output direction requires undesirable outputs")
        return np.zeros_like(x0), np.zeros_like(y0), b0.copy()
    if direction == "unit":
        return np.ones_like(x0), np.ones_like(y0), np.ones_like(b0)
    raise ValueError("unexpected DDF direction")


def _successful_ddf(
    *,
    data: DEAData,
    dmu_index: int,
    returns_to_scale: ReturnsToScale,
    direction: DDFDirection,
    reference: ReferenceSet,
    ref_idx: np.ndarray,
    result: optimize.OptimizeResult,
    gx: np.ndarray,
    gy: np.ndarray,
    gb: np.ndarray,
    peer_tolerance: float,
) -> DDFSolution:
    values = np.asarray(result.x, dtype=float)
    beta = float(values[-1])
    lambdas = values[: len(ref_idx)]
    lambda_dict = {
        data.dmu_names[int(original_index)]: float(value)
        for original_index, value in zip(ref_idx, lambdas)
    }
    peers = {name: value for name, value in lambda_dict.items() if abs(value) > peer_tolerance}
    x0 = data.inputs[dmu_index]
    y0 = data.good_outputs[dmu_index]
    b0 = np.array([]) if data.bad_outputs is None else data.bad_outputs[dmu_index]
    return DDFSolution(
        dmu_index=dmu_index,
        dmu_name=data.dmu_names[dmu_index],
        beta=beta,
        returns_to_scale=returns_to_scale,
        direction=direction,
        reference_label=reference.label,
        success=True,
        status=int(result.status),
        message=str(result.message),
        lambdas=lambda_dict,
        peers=peers,
        input_targets=_zip_values(data.input_names, x0 - beta * gx),
        good_output_targets=_zip_values(data.good_output_names, y0 + beta * gy),
        bad_output_targets=_zip_values(data.bad_output_names, b0 - beta * gb),
    )


def _failed_ddf(
    data: DEAData,
    dmu_index: int,
    returns_to_scale: ReturnsToScale,
    direction: DDFDirection,
    reference_label: str,
    result: optimize.OptimizeResult,
) -> DDFSolution:
    return DDFSolution(
        dmu_index=dmu_index,
        dmu_name=data.dmu_names[dmu_index],
        beta=None,
        returns_to_scale=returns_to_scale,
        direction=direction,
        reference_label=reference_label,
        success=False,
        status=int(result.status),
        message=str(result.message),
        lambdas={},
        peers={},
        input_targets={},
        good_output_targets={},
        bad_output_targets={},
    )


def _zip_values(names: list[str], values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(names, values)}


def _ratio_plus_one(numerator_distance: float | None, denominator_distance: float | None) -> float | None:
    if numerator_distance is None or denominator_distance is None:
        return None
    denominator = 1.0 + denominator_distance
    if denominator == 0:
        return None
    return (1.0 + numerator_distance) / denominator


def _div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _sqrt_product(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    product = left * right
    if product < 0:
        return None
    return sqrt(product)
