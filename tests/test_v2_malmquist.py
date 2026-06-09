import math

from v2 import (
    PanelDEAData,
    TransitionEfficiencies,
    compute_adjacent_malmquist,
    decompose_transition,
)


def test_decompose_transition_reproduces_v1_component_relationships():
    eff = TransitionEfficiencies(
        dmu="A",
        period_from="2020",
        period_to="2021",
        ectt=1.0,
        evtt=1.0,
        ectt_next=0.8,
        evtt_next=0.9,
        ectt1=0.9,
        evtt1=0.95,
        ect1t=1.1,
        evt1t=1.05,
        ecgt=0.95,
        evgt=0.97,
        ecgt_next=0.85,
        evgt_next=0.88,
    )

    result = decompose_transition(eff)

    assert math.isclose(
        result.fglr1992_crs["mlc"],
        result.fglr1992_crs["ecc"] * result.fglr1992_crs["tcc"],
        rel_tol=1e-12,
    )
    assert result.fglr1994["mlc"] == result.fglr1992_crs["mlc"]
    assert result.rd1997["ptc"] == result.fglr1992_vrs["ptc"]
    assert result.zofio2007["sec"] == result.fglr1994["sec"]
    assert math.isclose(
        result.pl2005_crs["mgc"],
        result.pl2005_crs["ecc"] * result.pl2005_crs["bpcc"],
        rel_tol=1e-12,
    )


def test_compute_adjacent_malmquist_keeps_explainable_sbm_solutions():
    panel = PanelDEAData.from_3d(
        periods=["2020", "2021"],
        entities=["A", "B"],
        inputs=[
            [[1.0], [2.0]],
            [[1.0], [2.0]],
        ],
        good_outputs=[
            [[1.0], [1.0]],
            [[1.0], [1.0]],
        ],
    )

    transitions = compute_adjacent_malmquist(panel)

    assert len(transitions) == 2
    first = transitions[0]
    assert first.efficiencies.dmu == "A"
    assert first.efficiencies.period_from == "2020"
    assert first.efficiencies.period_to == "2021"
    assert first.solutions["crs_tt"].success is True
    assert first.solutions["crs_tt"].reference_label == "period:2020"
    assert first.solutions["crs_gt"].reference_label == "global"
    assert "input1" in first.solutions["crs_tt"].input_targets
