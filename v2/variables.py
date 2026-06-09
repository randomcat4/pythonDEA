"""Variable metadata for v2 DEA models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

VariableRole = Literal["input", "good_output", "bad_output"]
Disposability = Literal["strong", "weak"]


@dataclass(frozen=True)
class VariableSpec:
    """Metadata attached to one DEA variable."""

    name: str
    role: VariableRole
    controllable: bool = True
    disposability: Disposability = "strong"

    @property
    def allows_slack(self) -> bool:
        return self.controllable and self.disposability == "strong"


def specs_from_names(
    names: Sequence[str],
    role: VariableRole,
    *,
    non_controllable: Sequence[str] | None = None,
    weakly_disposable: Sequence[str] | None = None,
) -> list[VariableSpec]:
    """Build variable specs from names and attribute name lists."""

    non_controllable_set = set(non_controllable or [])
    weakly_disposable_set = set(weakly_disposable or [])
    unknown = (non_controllable_set | weakly_disposable_set).difference(names)
    if unknown:
        missing = ", ".join(sorted(unknown))
        raise ValueError(f"unknown {role} variable attribute names: {missing}")
    return [
        VariableSpec(
            name=name,
            role=role,
            controllable=name not in non_controllable_set,
            disposability="weak" if name in weakly_disposable_set else "strong",
        )
        for name in names
    ]
