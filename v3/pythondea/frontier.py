"""Convenience entrypoints for running registered v3 models."""

from __future__ import annotations

from typing import Any

from .core.estimator import FitContext
from .core.registry import DEFAULT_REGISTRY, ModelRegistry
from .core.results import ModelResult


def fit(
    model: str,
    data: Any,
    *,
    registry: ModelRegistry = DEFAULT_REGISTRY,
    context: FitContext | None = None,
    **params: Any,
) -> ModelResult:
    """Create a registered estimator and run it in one call."""

    estimator = registry.create(model, **params)
    return estimator.fit(data, context=context)
