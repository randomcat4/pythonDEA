"""Publication-readiness checks for v3 model results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .core.registry import DEFAULT_REGISTRY, ModelRegistry
from .core.results import ModelResult


@dataclass(frozen=True)
class AuditCheck:
    """One publication-readiness check."""

    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True)
class PublicationAudit:
    """Auditable checklist for one model result."""

    result_hash: str
    checks: tuple[AuditCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "result_hash": self.result_hash,
            "checks": [check.to_dict() for check in self.checks],
        }


def audit_result(result: ModelResult) -> PublicationAudit:
    """Check whether a result carries the minimum v3 publication metadata."""

    metadata = dict(result.metadata)
    result_hash = result.reproducibility_hash()
    checks = (
        AuditCheck("status", result.status in {"ok", "partial"}, f"status={result.status}"),
        AuditCheck(
            "tables",
            bool(result.tables) and all(table.columns for table in result.tables),
            f"tables={len(result.tables)}",
        ),
        AuditCheck(
            "primary_table",
            any(table.name == result.primary_table for table in result.tables),
            f"primary_table={result.primary_table}",
        ),
        AuditCheck(
            "solver_backend",
            _has_solver_backend(metadata),
            "solver backend metadata is present"
            if _has_solver_backend(metadata)
            else "missing solver backend metadata",
        ),
        AuditCheck(
            "version",
            bool(metadata.get("pythondea_version")),
            f"pythondea_version={metadata.get('pythondea_version')}",
        ),
        AuditCheck("reproducibility_hash", len(result_hash) == 64, result_hash),
    )
    return PublicationAudit(result_hash=result_hash, checks=checks)


def model_catalog(registry: ModelRegistry = DEFAULT_REGISTRY) -> list[Mapping[str, Any]]:
    """Return citation-aware metadata for registered models."""

    return [
        {
            "name": spec.name,
            "family": spec.family,
            "summary": spec.summary,
            "citation_hint": spec.citation_hint,
            "keywords": list(spec.keywords),
            "experimental": spec.experimental,
        }
        for spec in registry.list()
    ]


def _has_solver_backend(metadata: Mapping[str, Any]) -> bool:
    solver = metadata.get("solver_backend")
    return isinstance(solver, Mapping) and bool(solver.get("name")) and bool(solver.get("method"))
