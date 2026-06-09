"""Solver backend contracts for publication-grade estimator metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol


@dataclass(frozen=True)
class SolverInfo:
    """Human-readable solver metadata attached to model results."""

    name: str
    method: str
    version: str | None = None
    options: Mapping[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"name": self.name, "method": self.method}
        if self.version is not None:
            data["version"] = self.version
        if self.options:
            data["options"] = dict(self.options)
        return data


class SolverBackend(Protocol):
    """Protocol for solver backends used by model plugins."""

    def info(self) -> SolverInfo:
        """Return solver metadata suitable for result provenance."""


@dataclass(frozen=True)
class SciPyHiGHSBackend:
    """Current default linear-programming backend used by the v2 SBM core."""

    method: str = "highs"

    def info(self) -> SolverInfo:
        try:
            import scipy

            version = scipy.__version__
        except Exception:
            version = None
        return SolverInfo(name="scipy.optimize.linprog", method=self.method, version=version)
