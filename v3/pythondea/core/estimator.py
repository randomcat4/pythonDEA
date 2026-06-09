"""Estimator protocol used by v3 model plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .results import ModelResult


@dataclass(frozen=True)
class FitContext:
    """Execution metadata shared with estimators."""

    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Estimator(Protocol):
    """Small protocol every v3 estimator plugin implements."""

    model_name: str

    def fit(self, data: Any, *, context: FitContext | None = None) -> ModelResult:
        """Fit/evaluate the model on validated data."""
