"""Result objects for pythonDEA v2 solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .orientation import SBMOrientation

ReturnsToScale = Literal["crs", "vrs"]


@dataclass(frozen=True)
class SBMSolution:
    """Result of one SBM evaluation."""

    dmu_index: int
    dmu_name: str
    score: float | None
    returns_to_scale: ReturnsToScale
    orientation: SBMOrientation
    reference_label: str
    success: bool
    status: int
    message: str
    lambdas: dict[str, float]
    peers: dict[str, float]
    input_slacks: dict[str, float]
    good_output_slacks: dict[str, float]
    bad_output_slacks: dict[str, float]
    input_targets: dict[str, float]
    good_output_targets: dict[str, float]
    bad_output_targets: dict[str, float]
    transform_t: float | None
    variable_attributes: dict[str, dict[str, str | bool]]
