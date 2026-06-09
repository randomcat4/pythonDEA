"""Reference-set helpers for DEA v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .panel import PanelDEAData


@dataclass(frozen=True)
class ReferenceSet:
    """A named frontier mask used to evaluate one or more DMUs."""

    indices: tuple[int, ...]
    label: str = "custom"

    @classmethod
    def all(cls, n_dmus: int, label: str = "all") -> "ReferenceSet":
        return cls(tuple(range(n_dmus)), label)

    @classmethod
    def from_indices(cls, indices: Iterable[int], label: str = "custom") -> "ReferenceSet":
        return cls(tuple(int(index) for index in indices), label)

    def without(self, index: int, label: str | None = None) -> "ReferenceSet":
        return ReferenceSet(
            tuple(candidate for candidate in self.indices if candidate != index),
            label or f"{self.label}:exclude-{index}",
        )

    def validate(self, n_dmus: int) -> np.ndarray:
        if not self.indices:
            raise ValueError("reference set must contain at least one DMU")
        array = np.asarray(self.indices, dtype=int)
        if len(set(array.tolist())) != len(array):
            raise ValueError("reference set indices must be unique")
        if np.any(array < 0) or np.any(array >= n_dmus):
            raise ValueError("reference set index out of bounds")
        return array


def contemporaneous_reference(panel: PanelDEAData, period: str | int) -> ReferenceSet:
    """Reference frontier containing all entities in one period."""

    label = str(period)
    return ReferenceSet(panel.period_indices(label), f"period:{label}")


def global_reference(panel: PanelDEAData) -> ReferenceSet:
    """Reference frontier containing all periods and all entities."""

    return ReferenceSet.all(panel.data.n_dmus, "global")


def cross_period_reference(panel: PanelDEAData, reference_period: str | int) -> ReferenceSet:
    """Reference frontier from another period for cross-period efficiency."""

    label = str(reference_period)
    return ReferenceSet(panel.period_indices(label), f"cross-period:{label}")


def window_reference(
    panel: PanelDEAData,
    start_period: str | int,
    end_period: str | int,
) -> ReferenceSet:
    """Reference frontier spanning a closed range of periods."""

    periods = panel.periods_between(start_period, end_period)
    indices: list[int] = []
    for period in periods:
        indices.extend(panel.period_indices(period))
    return ReferenceSet(tuple(indices), f"window:{periods[0]}:{periods[-1]}")
