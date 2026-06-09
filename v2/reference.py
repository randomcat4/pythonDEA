"""Reference-set helpers for DEA v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


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
