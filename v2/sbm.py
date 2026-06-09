"""Slack-based measure DEA models for pythonDEA v2."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import optimize

from .orientation import SBMOrientation
from .problem import DEAData
from .reference import ReferenceSet
from .result import SBMSolution

ReturnsToScale = Literal["crs", "vrs"]


def solve_all_sbm(
    data: DEAData,
    *,
    returns_to_scale: ReturnsToScale = "crs",
    orientation: SBMOrientation = "non_oriented",
    reference_set: ReferenceSet | None = None,
    exclude_self: bool = False,
    peer_tolerance: float = 1e-7,
) -> list[SBMSolution]:
    """Evaluate every DMU with the same base reference set."""

    return [
        solve_sbm(
            data,
            dmu_index,
            returns_to_scale=returns_to_scale,
            orientation=orientation,
            reference_set=reference_set,
            exclude_self=exclude_self,
            peer_tolerance=peer_tolerance,
        )
        for dmu_index in range(data.n_dmus)
    ]


def solve_sbm(
    data: DEAData,
    dmu_index: int,
    *,
    returns_to_scale: ReturnsToScale = "crs",
    orientation: SBMOrientation = "non_oriented",
    reference_set: ReferenceSet | None = None,
    exclude_self: bool = False,
    peer_tolerance: float = 1e-7,
) -> SBMSolution:
    """Solve the non-oriented SBM model for one DMU.

    This is the v2 explainable form of the v1 SBM LP: it returns the score plus
    unscaled lambda, slack, peer, and target values.
    """

    if returns_to_scale not in ("crs", "vrs"):
        raise ValueError("returns_to_scale must be 'crs' or 'vrs'")
    if orientation not in ("non_oriented", "input", "output", "bad_output_adjusted"):
        raise ValueError("unexpected SBM orientation")
    if dmu_index < 0 or dmu_index >= data.n_dmus:
        raise ValueError("dmu_index out of bounds")

    base_reference = reference_set or ReferenceSet.all(data.n_dmus)
    effective_reference = base_reference.without(dmu_index) if exclude_self else base_reference
    ref_idx = effective_reference.validate(data.n_dmus)

    x_ref = data.inputs[ref_idx].T
    y_ref = data.good_outputs[ref_idx].T
    b_ref = None if data.bad_outputs is None else data.bad_outputs[ref_idx].T
    x0 = data.inputs[dmu_index]
    y0 = data.good_outputs[dmu_index]
    b0 = None if data.bad_outputs is None else data.bad_outputs[dmu_index]

    input_active, good_active, bad_active = _orientation_slack_masks(data, orientation)
    lp = _build_nonoriented_sbm_lp(
        x_ref,
        y_ref,
        b_ref,
        x0,
        y0,
        b0,
        returns_to_scale,
        input_slack_active=input_active,
        good_slack_active=good_active,
        bad_slack_active=bad_active,
    )
    result = optimize.linprog(
        c=lp["c"],
        A_eq=lp["A_eq"],
        b_eq=lp["b_eq"],
        bounds=lp["bounds"],
        method="highs",
    )

    if not result.success:
        return _failed_solution(data, dmu_index, returns_to_scale, orientation, effective_reference.label, result)

    return _successful_solution(
        data=data,
        dmu_index=dmu_index,
        returns_to_scale=returns_to_scale,
        orientation=orientation,
        reference=effective_reference,
        ref_idx=ref_idx,
        result=result,
        peer_tolerance=peer_tolerance,
    )


def _build_nonoriented_sbm_lp(
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    b_ref: np.ndarray | None,
    x0: np.ndarray,
    y0: np.ndarray,
    b0: np.ndarray | None,
    returns_to_scale: ReturnsToScale,
    input_slack_active: np.ndarray,
    good_slack_active: np.ndarray,
    bad_slack_active: np.ndarray,
) -> dict[str, object]:
    n_ref = x_ref.shape[1]
    n_inputs = x_ref.shape[0]
    n_good = y_ref.shape[0]
    n_bad = 0 if b_ref is None else b_ref.shape[0]
    output_count = n_good + n_bad

    if output_count == 0:
        raise ValueError("at least one desirable or undesirable output is required")

    lambda_block = np.zeros(n_ref)
    input_obj = np.where(input_slack_active, -1.0 / (n_inputs * x0), 0.0)
    good_obj = np.zeros(n_good)
    bad_obj = np.zeros(n_bad)
    c = np.concatenate([lambda_block, input_obj, good_obj, bad_obj, np.array([1.0])])

    total_vars = n_ref + n_inputs + n_good + n_bad + 1
    t_col_x = -x0[:, None]
    x_eq = np.hstack([x_ref, np.eye(n_inputs), np.zeros((n_inputs, n_good + n_bad)), t_col_x])

    t_col_y = -y0[:, None]
    y_eq = np.hstack([
        y_ref,
        np.zeros((n_good, n_inputs)),
        -np.eye(n_good),
        np.zeros((n_good, n_bad)),
        t_col_y,
    ])

    blocks = [x_eq, y_eq]
    rhs = [np.zeros(n_inputs + n_good)]
    if n_bad:
        assert b_ref is not None and b0 is not None
        t_col_b = -b0[:, None]
        bad_eq = np.hstack([
            b_ref,
            np.zeros((n_bad, n_inputs + n_good)),
            np.eye(n_bad),
            t_col_b,
        ])
        blocks.append(bad_eq)
        rhs.append(np.zeros(n_bad))

    norm = np.zeros(total_vars)
    norm[n_ref + n_inputs : n_ref + n_inputs + n_good] = np.where(
        good_slack_active,
        1.0 / (output_count * y0),
        0.0,
    )
    if n_bad:
        assert b0 is not None
        start = n_ref + n_inputs + n_good
        norm[start : start + n_bad] = np.where(
            bad_slack_active,
            1.0 / (output_count * b0),
            0.0,
        )
    norm[-1] = 1.0
    blocks.append(norm.reshape(1, -1))
    rhs.append(np.array([1.0]))

    if returns_to_scale == "vrs":
        vrs = np.zeros(total_vars)
        vrs[:n_ref] = 1.0
        vrs[-1] = -1.0
        blocks.append(vrs.reshape(1, -1))
        rhs.append(np.array([0.0]))

    return {
        "c": c,
        "A_eq": np.vstack(blocks),
        "b_eq": np.concatenate(rhs),
        "bounds": _sbm_bounds(n_ref, input_slack_active, good_slack_active, bad_slack_active),
    }


def _sbm_bounds(
    n_ref: int,
    input_slack_active: np.ndarray,
    good_slack_active: np.ndarray,
    bad_slack_active: np.ndarray,
) -> list[tuple[float, float | None]]:
    bounds: list[tuple[float, float | None]] = [(0.0, None)] * n_ref
    for active in input_slack_active:
        bounds.append((0.0, None if active else 0.0))
    for active in good_slack_active:
        bounds.append((0.0, None if active else 0.0))
    for active in bad_slack_active:
        bounds.append((0.0, None if active else 0.0))
    bounds.append((0.0, None))
    return bounds


def _successful_solution(
    *,
    data: DEAData,
    dmu_index: int,
    returns_to_scale: ReturnsToScale,
    orientation: SBMOrientation,
    reference: ReferenceSet,
    ref_idx: np.ndarray,
    result: optimize.OptimizeResult,
    peer_tolerance: float,
) -> SBMSolution:
    n_ref = len(ref_idx)
    n_inputs = data.n_inputs
    n_good = data.n_good_outputs
    n_bad = data.n_bad_outputs

    values = np.asarray(result.x, dtype=float)
    t_value = float(values[-1])
    if t_value <= 0:
        raise ValueError("SBM transform variable t must be positive")

    lambda_values = values[:n_ref] / t_value
    input_slacks = values[n_ref : n_ref + n_inputs] / t_value
    good_start = n_ref + n_inputs
    good_slacks = values[good_start : good_start + n_good] / t_value
    bad_start = good_start + n_good
    bad_slacks = values[bad_start : bad_start + n_bad] / t_value

    lambda_dict = {
        data.dmu_names[int(original_index)]: float(value)
        for original_index, value in zip(ref_idx, lambda_values)
    }
    peers = {
        name: value
        for name, value in lambda_dict.items()
        if abs(value) > peer_tolerance
    }
    input_slack_dict = _zip_values(data.input_names, input_slacks)
    good_slack_dict = _zip_values(data.good_output_names, good_slacks)
    bad_slack_dict = _zip_values(data.bad_output_names, bad_slacks)

    x0 = data.inputs[dmu_index]
    y0 = data.good_outputs[dmu_index]
    b0 = np.array([]) if data.bad_outputs is None else data.bad_outputs[dmu_index]

    return SBMSolution(
        dmu_index=dmu_index,
        dmu_name=data.dmu_names[dmu_index],
        score=float(result.fun),
        returns_to_scale=returns_to_scale,
        orientation=orientation,
        reference_label=reference.label,
        success=True,
        status=int(result.status),
        message=str(result.message),
        lambdas=lambda_dict,
        peers=peers,
        input_slacks=input_slack_dict,
        good_output_slacks=good_slack_dict,
        bad_output_slacks=bad_slack_dict,
        input_targets=_zip_values(data.input_names, x0 - input_slacks),
        good_output_targets=_zip_values(data.good_output_names, y0 + good_slacks),
        bad_output_targets=_zip_values(data.bad_output_names, b0 - bad_slacks),
        transform_t=t_value,
        variable_attributes=_variable_attributes(data),
    )


def _failed_solution(
    data: DEAData,
    dmu_index: int,
    returns_to_scale: ReturnsToScale,
    orientation: SBMOrientation,
    reference_label: str,
    result: optimize.OptimizeResult,
) -> SBMSolution:
    return SBMSolution(
        dmu_index=dmu_index,
        dmu_name=data.dmu_names[dmu_index],
        score=None,
        returns_to_scale=returns_to_scale,
        orientation=orientation,
        reference_label=reference_label,
        success=False,
        status=int(result.status),
        message=str(result.message),
        lambdas={},
        peers={},
        input_slacks={},
        good_output_slacks={},
        bad_output_slacks={},
        input_targets={},
        good_output_targets={},
        bad_output_targets={},
        transform_t=None,
        variable_attributes=_variable_attributes(data),
    )


def _zip_values(names: list[str], values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(names, values)}


def _orientation_slack_masks(
    data: DEAData,
    orientation: SBMOrientation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_allowed = np.array([spec.allows_slack for spec in data.input_specs], dtype=bool)
    good_allowed = np.array([spec.allows_slack for spec in data.good_output_specs], dtype=bool)
    bad_allowed = np.array([spec.allows_slack for spec in data.bad_output_specs], dtype=bool)

    if orientation == "non_oriented":
        return input_allowed, good_allowed, bad_allowed
    if orientation == "input":
        return input_allowed, np.zeros_like(good_allowed), np.zeros_like(bad_allowed)
    if orientation == "output":
        return np.zeros_like(input_allowed), good_allowed, np.zeros_like(bad_allowed)
    if orientation == "bad_output_adjusted":
        return np.zeros_like(input_allowed), np.zeros_like(good_allowed), bad_allowed
    raise ValueError("unexpected SBM orientation")


def _variable_attributes(data: DEAData) -> dict[str, dict[str, str | bool]]:
    return {
        spec.name: {
            "role": spec.role,
            "controllable": spec.controllable,
            "disposability": spec.disposability,
            "allows_slack": spec.allows_slack,
        }
        for spec in data.variable_specs()
    }
