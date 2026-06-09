"""Malmquist decomposition formulas used by v1 and v2."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


Metric = float | None


@dataclass(frozen=True)
class TransitionEfficiencies:
    """Efficiency scores needed for one adjacent-period decomposition."""

    dmu: str
    period_from: str
    period_to: str
    ectt: Metric
    evtt: Metric
    ectt_next: Metric
    evtt_next: Metric
    ectt1: Metric
    evtt1: Metric
    ect1t: Metric
    evt1t: Metric
    ecgt: Metric
    evgt: Metric
    ecgt_next: Metric
    evgt_next: Metric


@dataclass(frozen=True)
class DecompositionResult:
    """All v1-compatible Malmquist decomposition sheets for one transition."""

    fglr1992_crs: dict[str, Metric]
    fglr1992_vrs: dict[str, Metric]
    fglr1994: dict[str, Metric]
    rd1997: dict[str, Metric]
    zofio2007: dict[str, Metric]
    pl2005_crs: dict[str, Metric]
    pl2005_vrs: dict[str, Metric]


def decompose_transition(eff: TransitionEfficiencies) -> DecompositionResult:
    """Compute FGLR/RD/Zofio/PL decompositions from v1 formulas."""

    ecc = _div(eff.ectt_next, eff.ectt)
    tcc = _sqrt_product(
        _div(eff.ectt1, eff.ectt_next),
        _div(eff.ectt, eff.ect1t),
    )
    mlc = _mul(ecc, tcc)

    pec = _div(eff.evtt_next, eff.evtt)
    ptc = _sqrt_product(
        _div(eff.evtt1, eff.evtt_next),
        _div(eff.evtt, eff.evt1t),
    )
    mlv = _mul(pec, ptc)

    sec = _div(_div(eff.ectt_next, eff.evtt_next), _div(eff.ectt, eff.evtt))
    sch = _sqrt_product(
        _div(_div(eff.ectt1, eff.evtt1), _div(eff.ectt, eff.evtt)),
        _div(_div(eff.ectt_next, eff.evtt_next), _div(eff.ect1t, eff.evt1t)),
    )
    stc = _sqrt_product(
        _div(_div(eff.ectt, eff.evtt), _div(eff.ect1t, eff.evt1t)),
        _div(_div(eff.ectt1, eff.evtt1), _div(eff.ectt_next, eff.evtt_next)),
    )

    bpcc = _div(_div(eff.ecgt_next, eff.ectt_next), _div(eff.ecgt, eff.ectt))
    mgc = _mul(ecc, bpcc)
    bpcv = _div(_div(eff.evgt_next, eff.evtt_next), _div(eff.evgt, eff.evtt))
    ecv = _div(eff.evtt_next, eff.evtt)
    mgv = _mul(ecv, bpcv)

    return DecompositionResult(
        fglr1992_crs={"mlc": mlc, "ecc": ecc, "tcc": tcc},
        fglr1992_vrs={"mlv": mlv, "pec": pec, "ptc": ptc},
        fglr1994={"mlc": mlc, "pec": pec, "sec": sec, "tcc": tcc},
        rd1997={"mlc": mlc, "pec": pec, "sch": sch, "ptc": ptc},
        zofio2007={"mlc": mlc, "pec": pec, "sec": sec, "ptc": ptc, "stc": stc},
        pl2005_crs={"mgc": mgc, "ecc": ecc, "bpcc": bpcc},
        pl2005_vrs={"mgv": mgv, "ecv": ecv, "bpcv": bpcv},
    )


def _div(numerator: Metric, denominator: Metric) -> Metric:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _mul(left: Metric, right: Metric) -> Metric:
    if left is None or right is None:
        return None
    return left * right


def _sqrt_product(left: Metric, right: Metric) -> Metric:
    product = _mul(left, right)
    if product is None or product < 0:
        return None
    return sqrt(product)
