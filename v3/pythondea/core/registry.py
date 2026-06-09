"""Model registry and plugin loader for v3 estimators."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata
from typing import Any, Callable, Iterable


EstimatorFactory = Callable[..., Any]


@dataclass(frozen=True)
class ModelSpec:
    """Metadata for one registered model."""

    name: str
    estimator: EstimatorFactory
    family: str
    summary: str
    citation_hint: str | None = None
    keywords: tuple[str, ...] = field(default_factory=tuple)
    experimental: bool = False


class ModelRegistry:
    """In-memory registry for built-in and third-party model plugins."""

    def __init__(self) -> None:
        self._models: dict[str, ModelSpec] = {}
        self._loaded_entry_points: set[str] = set()

    def register(self, spec: ModelSpec, *, replace: bool = False) -> None:
        name = _normalize_name(spec.name)
        if name in self._models and not replace:
            raise ValueError(f"model already registered: {name}")
        self._models[name] = ModelSpec(
            name=name,
            estimator=spec.estimator,
            family=spec.family,
            summary=spec.summary,
            citation_hint=spec.citation_hint,
            keywords=tuple(spec.keywords),
            experimental=spec.experimental,
        )

    def get(self, name: str) -> ModelSpec:
        key = _normalize_name(name)
        try:
            return self._models[key]
        except KeyError as exc:
            available = ", ".join(sorted(self._models)) or "none"
            raise KeyError(f"unknown model '{name}'. Available models: {available}") from exc

    def create(self, name: str, **params: Any) -> Any:
        spec = self.get(name)
        return spec.estimator(**params)

    def list(self, *, family: str | None = None) -> list[ModelSpec]:
        specs = list(self._models.values())
        if family is not None:
            specs = [spec for spec in specs if spec.family == family]
        return sorted(specs, key=lambda spec: spec.name)

    def load_entry_points(self, group: str = "pythondea.models") -> None:
        """Load third-party plugin registration functions once per entry point."""

        entry_points = metadata.entry_points()
        if hasattr(entry_points, "select"):
            selected: Iterable[metadata.EntryPoint] = entry_points.select(group=group)
        else:
            selected = entry_points.get(group, ())

        for entry_point in selected:
            token = f"{entry_point.module}:{entry_point.attr or ''}"
            if token in self._loaded_entry_points:
                continue
            loaded = entry_point.load()
            if isinstance(loaded, ModelSpec):
                self.register(loaded)
            elif callable(loaded):
                loaded(self)
            else:
                raise TypeError(f"unsupported model entry point: {entry_point.name}")
            self._loaded_entry_points.add(token)


DEFAULT_REGISTRY = ModelRegistry()


def register_model(spec: ModelSpec, *, replace: bool = False) -> None:
    DEFAULT_REGISTRY.register(spec, replace=replace)


def get_model(name: str) -> ModelSpec:
    return DEFAULT_REGISTRY.get(name)


def list_models(*, family: str | None = None) -> list[ModelSpec]:
    return DEFAULT_REGISTRY.list(family=family)


def _normalize_name(name: str) -> str:
    key = str(name).strip().lower()
    if not key:
        raise ValueError("model name must be non-empty")
    return key
