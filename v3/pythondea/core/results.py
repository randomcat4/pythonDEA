"""Standard result containers for v3 estimators."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ResultTable:
    """A named table with rows safe to serialize or convert to pandas."""

    name: str
    rows: tuple[Mapping[str, Any], ...]
    columns: tuple[str, ...]

    @classmethod
    def from_rows(
        cls,
        name: str,
        rows: Sequence[Mapping[str, Any]],
        *,
        columns: Sequence[str] | None = None,
    ) -> "ResultTable":
        inferred = tuple(columns or _infer_columns(rows))
        return cls(name=name, rows=tuple(dict(row) for row in rows), columns=inferred)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "columns": list(self.columns),
            "rows": [dict(row) for row in self.rows],
        }

    def to_pandas(self):
        """Convert table rows to a pandas DataFrame when pandas is installed."""

        import pandas as pd

        return pd.DataFrame([dict(row) for row in self.rows], columns=list(self.columns))


@dataclass(frozen=True)
class ModelResult:
    """Result returned by every public v3 estimator."""

    model: str
    status: str
    primary_table: str
    tables: tuple[ResultTable, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, Any] = field(default_factory=dict)

    def table(self, name: str | None = None) -> ResultTable:
        target = name or self.primary_table
        for table in self.tables:
            if table.name == target:
                return table
        raise KeyError(f"unknown result table: {target}")

    @property
    def rows(self) -> tuple[Mapping[str, Any], ...]:
        return self.table().rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "status": self.status,
            "primary_table": self.primary_table,
            "tables": [table.to_dict() for table in self.tables],
            "metadata": dict(self.metadata),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True, default=str)

    def reproducibility_hash(self) -> str:
        payload = self.to_json(indent=None).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _infer_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    return columns
