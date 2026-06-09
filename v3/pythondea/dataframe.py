"""DataFrame-style adapters for empirical DEA workflows."""

from __future__ import annotations

from typing import Any, Hashable, Sequence

from .core.data import DEAData, PanelDEAData


def dea_from_dataframe(
    frame: Any,
    *,
    inputs: Sequence[str],
    good_outputs: Sequence[str],
    bad_outputs: Sequence[str] | None = None,
    dmu_column: str | None = None,
    non_controllable_inputs: Sequence[str] | None = None,
    non_controllable_good_outputs: Sequence[str] | None = None,
    non_controllable_bad_outputs: Sequence[str] | None = None,
    weakly_disposable_inputs: Sequence[str] | None = None,
    weakly_disposable_good_outputs: Sequence[str] | None = None,
    weakly_disposable_bad_outputs: Sequence[str] | None = None,
) -> DEAData:
    """Build cross-sectional DEA data from a pandas-like table."""

    bad_outputs = tuple(bad_outputs or ())
    _require_columns(frame, [*inputs, *good_outputs, *bad_outputs, *([dmu_column] if dmu_column else [])])
    return DEAData(
        inputs=_matrix(frame, inputs),
        good_outputs=_matrix(frame, good_outputs),
        bad_outputs=None if not bad_outputs else _matrix(frame, bad_outputs),
        dmu_names=None if dmu_column is None else [str(value) for value in _column(frame, dmu_column)],
        input_names=list(inputs),
        good_output_names=list(good_outputs),
        bad_output_names=list(bad_outputs),
        non_controllable_inputs=non_controllable_inputs,
        non_controllable_good_outputs=non_controllable_good_outputs,
        non_controllable_bad_outputs=non_controllable_bad_outputs,
        weakly_disposable_inputs=weakly_disposable_inputs,
        weakly_disposable_good_outputs=weakly_disposable_good_outputs,
        weakly_disposable_bad_outputs=weakly_disposable_bad_outputs,
    )


def panel_from_dataframe(
    frame: Any,
    *,
    period_column: str,
    entity_column: str,
    inputs: Sequence[str],
    good_outputs: Sequence[str],
    bad_outputs: Sequence[str] | None = None,
    periods: Sequence[Hashable] | None = None,
    entities: Sequence[Hashable] | None = None,
    input_names: Sequence[str] | None = None,
    good_output_names: Sequence[str] | None = None,
    bad_output_names: Sequence[str] | None = None,
    non_controllable_inputs: Sequence[str] | None = None,
    non_controllable_good_outputs: Sequence[str] | None = None,
    non_controllable_bad_outputs: Sequence[str] | None = None,
    weakly_disposable_inputs: Sequence[str] | None = None,
    weakly_disposable_good_outputs: Sequence[str] | None = None,
    weakly_disposable_bad_outputs: Sequence[str] | None = None,
) -> PanelDEAData:
    """Build balanced panel data from a pandas-like table."""

    bad_outputs = tuple(bad_outputs or ())
    value_columns = [*inputs, *good_outputs, *bad_outputs]
    _require_columns(frame, [period_column, entity_column, *value_columns])

    period_values = _column(frame, period_column)
    entity_values = _column(frame, entity_column)
    period_labels = tuple(str(value) for value in (periods or _unique_preserve(period_values)))
    entity_labels = tuple(str(value) for value in (entities or _unique_preserve(entity_values)))

    rows: dict[tuple[str, str], int] = {}
    for row_index, key in enumerate(zip(map(str, period_values), map(str, entity_values))):
        if key in rows:
            raise ValueError(f"duplicate panel row for period/entity: {key[0]}, {key[1]}")
        rows[key] = row_index

    input_arrays = []
    good_arrays = []
    bad_arrays = []
    for period in period_labels:
        input_period = []
        good_period = []
        bad_period = []
        for entity in entity_labels:
            row_index = rows.get((period, entity))
            if row_index is None:
                raise ValueError(f"missing balanced panel row for period/entity: {period}, {entity}")
            input_period.append([float(_column(frame, column)[row_index]) for column in inputs])
            good_period.append([float(_column(frame, column)[row_index]) for column in good_outputs])
            if bad_outputs:
                bad_period.append([float(_column(frame, column)[row_index]) for column in bad_outputs])
        input_arrays.append(input_period)
        good_arrays.append(good_period)
        if bad_outputs:
            bad_arrays.append(bad_period)

    return PanelDEAData.from_3d(
        periods=period_labels,
        entities=entity_labels,
        inputs=input_arrays,
        good_outputs=good_arrays,
        bad_outputs=None if not bad_outputs else bad_arrays,
        input_names=input_names or inputs,
        good_output_names=good_output_names or good_outputs,
        bad_output_names=bad_output_names or bad_outputs,
        non_controllable_inputs=non_controllable_inputs,
        non_controllable_good_outputs=non_controllable_good_outputs,
        non_controllable_bad_outputs=non_controllable_bad_outputs,
        weakly_disposable_inputs=weakly_disposable_inputs,
        weakly_disposable_good_outputs=weakly_disposable_good_outputs,
        weakly_disposable_bad_outputs=weakly_disposable_bad_outputs,
    )


def _matrix(frame: Any, columns: Sequence[str]) -> list[list[float]]:
    values = [_column(frame, column) for column in columns]
    return [
        [float(column_values[row_index]) for column_values in values]
        for row_index in range(len(values[0]))
    ]


def _column(frame: Any, name: str) -> list[Any]:
    values = frame[name]
    if hasattr(values, "tolist"):
        return list(values.tolist())
    return list(values)


def _require_columns(frame: Any, columns: Sequence[str | None]) -> None:
    missing = []
    for column in columns:
        if column is None:
            continue
        try:
            frame[column]
        except Exception:
            missing.append(column)
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")


def _unique_preserve(values: Sequence[Any]) -> list[Any]:
    seen: set[str] = set()
    result = []
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
