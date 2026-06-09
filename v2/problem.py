"""Validated data containers for DEA v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .variables import VariableSpec, specs_from_names


def _as_2d_float(name: str, values: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have at least one row and one column")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(array <= 0):
        raise ValueError(f"{name} must contain strictly positive values")
    return array


def _default_names(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{i + 1}" for i in range(count)]


def _validate_names(name: str, values: Sequence[str] | None, count: int) -> list[str]:
    if values is None:
        return _default_names(name, count)
    result = [str(value) for value in values]
    if len(result) != count:
        raise ValueError(f"{name}_names must contain {count} values")
    if len(set(result)) != len(result):
        raise ValueError(f"{name}_names must be unique")
    return result


@dataclass(frozen=True)
class DEAData:
    """DEA data at one frontier grain.

    Rows are DMUs and columns are variables. Bad outputs are optional and are
    treated as undesirable outputs in the SBM model.
    """

    inputs: np.ndarray
    good_outputs: np.ndarray
    bad_outputs: np.ndarray | None = None
    dmu_names: Sequence[str] | None = None
    input_names: Sequence[str] | None = None
    good_output_names: Sequence[str] | None = None
    bad_output_names: Sequence[str] | None = None
    non_controllable_inputs: Sequence[str] | None = None
    non_controllable_good_outputs: Sequence[str] | None = None
    non_controllable_bad_outputs: Sequence[str] | None = None
    weakly_disposable_inputs: Sequence[str] | None = None
    weakly_disposable_good_outputs: Sequence[str] | None = None
    weakly_disposable_bad_outputs: Sequence[str] | None = None

    def __post_init__(self) -> None:
        inputs = _as_2d_float("inputs", self.inputs)
        good_outputs = _as_2d_float("good_outputs", self.good_outputs)
        if inputs.shape[0] != good_outputs.shape[0]:
            raise ValueError("inputs and good_outputs must have the same number of DMUs")

        bad_outputs = None
        if self.bad_outputs is not None:
            bad_outputs = _as_2d_float("bad_outputs", self.bad_outputs)
            if bad_outputs.shape[0] != inputs.shape[0]:
                raise ValueError("bad_outputs must have the same number of DMUs as inputs")

        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "good_outputs", good_outputs)
        object.__setattr__(self, "bad_outputs", bad_outputs)
        object.__setattr__(self, "dmu_names", _validate_names("DMU", self.dmu_names, inputs.shape[0]))
        input_names = _validate_names("input", self.input_names, inputs.shape[1])
        good_output_names = _validate_names("good_output", self.good_output_names, good_outputs.shape[1])
        object.__setattr__(self, "input_names", input_names)
        object.__setattr__(
            self,
            "good_output_names",
            good_output_names,
        )
        bad_count = 0 if bad_outputs is None else bad_outputs.shape[1]
        bad_output_names = _validate_names("bad_output", self.bad_output_names, bad_count)
        object.__setattr__(
            self,
            "bad_output_names",
            bad_output_names,
        )
        object.__setattr__(
            self,
            "input_specs",
            specs_from_names(
                input_names,
                "input",
                non_controllable=self.non_controllable_inputs,
                weakly_disposable=self.weakly_disposable_inputs,
            ),
        )
        object.__setattr__(
            self,
            "good_output_specs",
            specs_from_names(
                good_output_names,
                "good_output",
                non_controllable=self.non_controllable_good_outputs,
                weakly_disposable=self.weakly_disposable_good_outputs,
            ),
        )
        object.__setattr__(
            self,
            "bad_output_specs",
            specs_from_names(
                bad_output_names,
                "bad_output",
                non_controllable=self.non_controllable_bad_outputs,
                weakly_disposable=self.weakly_disposable_bad_outputs,
            ),
        )

    @property
    def n_dmus(self) -> int:
        return self.inputs.shape[0]

    @property
    def n_inputs(self) -> int:
        return self.inputs.shape[1]

    @property
    def n_good_outputs(self) -> int:
        return self.good_outputs.shape[1]

    @property
    def n_bad_outputs(self) -> int:
        return 0 if self.bad_outputs is None else self.bad_outputs.shape[1]

    def variable_specs(self) -> list[VariableSpec]:
        return [*self.input_specs, *self.good_output_specs, *self.bad_output_specs]
