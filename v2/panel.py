"""Panel data helpers for v2 Malmquist-style frontiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .problem import DEAData


@dataclass(frozen=True)
class PanelDEAData:
    """Balanced panel wrapper around flattened ``DEAData`` rows."""

    data: DEAData
    periods: tuple[str, ...]
    entities: tuple[str, ...]

    @classmethod
    def from_3d(
        cls,
        *,
        inputs: Sequence[Sequence[Sequence[float]]],
        good_outputs: Sequence[Sequence[Sequence[float]]],
        bad_outputs: Sequence[Sequence[Sequence[float]]] | None = None,
        periods: Sequence[str | int],
        entities: Sequence[str],
        input_names: Sequence[str] | None = None,
        good_output_names: Sequence[str] | None = None,
        bad_output_names: Sequence[str] | None = None,
    ) -> "PanelDEAData":
        """Create a balanced panel from arrays shaped period x entity x variable."""

        period_labels = tuple(str(period) for period in periods)
        entity_labels = tuple(str(entity) for entity in entities)
        if len(set(period_labels)) != len(period_labels):
            raise ValueError("period labels must be unique")
        if len(set(entity_labels)) != len(entity_labels):
            raise ValueError("entity labels must be unique")

        input_array = _as_panel_array("inputs", inputs, len(period_labels), len(entity_labels))
        good_array = _as_panel_array(
            "good_outputs", good_outputs, len(period_labels), len(entity_labels)
        )
        if input_array.shape[:2] != good_array.shape[:2]:
            raise ValueError("inputs and good_outputs must share period/entity dimensions")

        bad_array = None
        if bad_outputs is not None:
            bad_array = _as_panel_array(
                "bad_outputs", bad_outputs, len(period_labels), len(entity_labels)
            )
            if bad_array.shape[:2] != input_array.shape[:2]:
                raise ValueError("bad_outputs must share period/entity dimensions")

        dmu_names = [
            cls.row_name(period, entity)
            for period in period_labels
            for entity in entity_labels
        ]
        data = DEAData(
            inputs=input_array.reshape(len(period_labels) * len(entity_labels), input_array.shape[2]),
            good_outputs=good_array.reshape(len(period_labels) * len(entity_labels), good_array.shape[2]),
            bad_outputs=None
            if bad_array is None
            else bad_array.reshape(len(period_labels) * len(entity_labels), bad_array.shape[2]),
            dmu_names=dmu_names,
            input_names=input_names,
            good_output_names=good_output_names,
            bad_output_names=bad_output_names,
        )
        return cls(data=data, periods=period_labels, entities=entity_labels)

    @staticmethod
    def row_name(period: str, entity: str) -> str:
        return f"{entity}@{period}"

    @property
    def n_periods(self) -> int:
        return len(self.periods)

    @property
    def n_entities(self) -> int:
        return len(self.entities)

    def period_position(self, period: str | int) -> int:
        label = str(period)
        try:
            return self.periods.index(label)
        except ValueError as exc:
            raise ValueError(f"unknown period: {period}") from exc

    def entity_position(self, entity: str) -> int:
        try:
            return self.entities.index(str(entity))
        except ValueError as exc:
            raise ValueError(f"unknown entity: {entity}") from exc

    def row_index(self, period: str | int, entity: str) -> int:
        return self.period_position(period) * self.n_entities + self.entity_position(entity)

    def period_indices(self, period: str | int) -> tuple[int, ...]:
        start = self.period_position(period) * self.n_entities
        return tuple(range(start, start + self.n_entities))

    def periods_between(self, start_period: str | int, end_period: str | int) -> tuple[str, ...]:
        start = self.period_position(start_period)
        end = self.period_position(end_period)
        if start > end:
            raise ValueError("start_period must be at or before end_period")
        return self.periods[start : end + 1]


def _as_panel_array(
    name: str,
    values: Sequence[Sequence[Sequence[float]]],
    expected_periods: int,
    expected_entities: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3D array: period x entity x variable")
    if array.shape[0] != expected_periods or array.shape[1] != expected_entities:
        raise ValueError(
            f"{name} must have shape ({expected_periods}, {expected_entities}, variables)"
        )
    if array.shape[2] == 0:
        raise ValueError(f"{name} must contain at least one variable")
    return array
