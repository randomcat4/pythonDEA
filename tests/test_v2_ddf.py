import math

from v2 import DEAData, PanelDEAData, compute_adjacent_malmquist_luenberger, solve_ddf


def test_input_directional_distance_contracts_inputs():
    data = DEAData(
        inputs=[[1.0], [2.0]],
        good_outputs=[[1.0], [1.0]],
        dmu_names=["A", "B"],
    )

    solution = solve_ddf(data, 1, direction="input")

    assert solution.success is True
    assert math.isclose(solution.beta, 0.5, rel_tol=1e-9)
    assert solution.peers == {"A": 1.0}
    assert math.isclose(solution.input_targets["input1"], 1.0, rel_tol=1e-9)


def test_bad_output_directional_distance_reduces_undesirable_output():
    data = DEAData(
        inputs=[[1.0], [1.0]],
        good_outputs=[[1.0], [1.0]],
        bad_outputs=[[1.0], [2.0]],
        dmu_names=["clean", "dirty"],
        bad_output_names=["co2"],
    )

    solution = solve_ddf(data, 1, direction="bad_output")

    assert solution.success is True
    assert math.isclose(solution.beta, 0.5, rel_tol=1e-9)
    assert solution.peers == {"clean": 1.0}
    assert math.isclose(solution.bad_output_targets["co2"], 1.0, rel_tol=1e-9)


def test_malmquist_luenberger_returns_adjacent_panel_transitions():
    panel = PanelDEAData.from_3d(
        periods=["2020", "2021"],
        entities=["A", "B"],
        inputs=[
            [[1.0], [1.0]],
            [[1.0], [1.0]],
        ],
        good_outputs=[
            [[1.0], [1.0]],
            [[1.0], [1.0]],
        ],
        bad_outputs=[
            [[1.0], [2.0]],
            [[1.0], [1.5]],
        ],
        bad_output_names=["co2"],
    )

    transitions = compute_adjacent_malmquist_luenberger(panel)

    assert len(transitions) == 2
    assert transitions[0].period_from == "2020"
    assert transitions[0].period_to == "2021"
    assert set(transitions[0].distances) == {"d_t_t", "d_t1_t1", "d_t_t1", "d_t1_t"}
    assert transitions[1].ml_index is not None
