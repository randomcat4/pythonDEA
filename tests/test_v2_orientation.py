import math

from v2 import DEAData, solve_sbm


def test_input_orientation_only_moves_inputs():
    data = DEAData(
        inputs=[[1.0], [2.0]],
        good_outputs=[[1.0], [1.0]],
        dmu_names=["A", "B"],
    )

    solution = solve_sbm(data, 1, orientation="input")

    assert solution.success is True
    assert solution.orientation == "input"
    assert math.isclose(solution.score, 0.5, rel_tol=1e-9)
    assert math.isclose(solution.input_targets["input1"], 1.0, rel_tol=1e-9)
    assert math.isclose(solution.good_output_targets["good_output1"], 1.0, rel_tol=1e-9)


def test_output_orientation_only_moves_good_outputs():
    data = DEAData(
        inputs=[[1.0], [1.0]],
        good_outputs=[[2.0], [1.0]],
        dmu_names=["high", "low"],
    )

    solution = solve_sbm(data, 1, orientation="output")

    assert solution.success is True
    assert solution.orientation == "output"
    assert math.isclose(solution.good_output_slacks["good_output1"], 1.0, rel_tol=1e-9)
    assert math.isclose(solution.good_output_targets["good_output1"], 2.0, rel_tol=1e-9)
    assert math.isclose(solution.input_slacks["input1"], 0.0, abs_tol=1e-9)


def test_bad_output_adjusted_orientation_only_moves_bad_outputs():
    data = DEAData(
        inputs=[[1.0], [1.0]],
        good_outputs=[[1.0], [1.0]],
        bad_outputs=[[1.0], [2.0]],
        dmu_names=["clean", "dirty"],
        bad_output_names=["co2"],
    )

    solution = solve_sbm(data, 1, orientation="bad_output_adjusted")

    assert solution.success is True
    assert solution.orientation == "bad_output_adjusted"
    assert math.isclose(solution.bad_output_slacks["co2"], 1.0, rel_tol=1e-9)
    assert math.isclose(solution.bad_output_targets["co2"], 1.0, rel_tol=1e-9)
    assert math.isclose(solution.input_slacks["input1"], 0.0, abs_tol=1e-9)
    assert math.isclose(solution.good_output_slacks["good_output1"], 0.0, abs_tol=1e-9)
