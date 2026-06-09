import math

from v2 import DEAData, ReferenceSet, solve_all_sbm, solve_sbm


def test_nonoriented_sbm_returns_slacks_targets_and_peers():
    data = DEAData(
        inputs=[[1.0], [2.0], [3.0]],
        good_outputs=[[1.0], [1.0], [1.0]],
        dmu_names=["A", "B", "C"],
        input_names=["labor"],
        good_output_names=["service"],
    )

    solution = solve_sbm(data, 1, returns_to_scale="crs")

    assert solution.success is True
    assert math.isclose(solution.score, 0.5, rel_tol=1e-9)
    assert solution.peers == {"A": 1.0}
    assert math.isclose(solution.input_slacks["labor"], 1.0, rel_tol=1e-9)
    assert math.isclose(solution.good_output_slacks["service"], 0.0, abs_tol=1e-9)
    assert math.isclose(solution.input_targets["labor"], 1.0, rel_tol=1e-9)
    assert math.isclose(solution.good_output_targets["service"], 1.0, rel_tol=1e-9)


def test_bad_output_sbm_reduces_undesirable_output_target():
    data = DEAData(
        inputs=[[1.0], [1.0]],
        good_outputs=[[1.0], [1.0]],
        bad_outputs=[[1.0], [2.0]],
        dmu_names=["clean", "dirty"],
        input_names=["capital"],
        good_output_names=["gdp"],
        bad_output_names=["co2"],
    )

    solution = solve_sbm(data, 1, returns_to_scale="crs")

    assert solution.success is True
    assert math.isclose(solution.score, 0.8, rel_tol=1e-9)
    assert solution.peers == {"clean": 1.0}
    assert math.isclose(solution.bad_output_slacks["co2"], 1.0, rel_tol=1e-9)
    assert math.isclose(solution.bad_output_targets["co2"], 1.0, rel_tol=1e-9)


def test_reference_set_can_exclude_self_for_super_style_frontiers():
    data = DEAData(
        inputs=[[1.0], [2.0], [3.0]],
        good_outputs=[[1.0], [1.0], [1.0]],
        dmu_names=["A", "B", "C"],
    )
    reference = ReferenceSet.all(data.n_dmus, label="period")

    solution = solve_sbm(data, 1, reference_set=reference, exclude_self=True)

    assert solution.reference_label == "period:exclude-1"
    assert solution.success is True
    assert solution.peers == {"A": 1.0}


def test_solve_all_sbm_supports_vrs():
    data = DEAData(
        inputs=[[1.0], [2.0], [3.0]],
        good_outputs=[[1.0], [1.0], [1.0]],
        dmu_names=["A", "B", "C"],
    )

    solutions = solve_all_sbm(data, returns_to_scale="vrs")

    assert [solution.dmu_name for solution in solutions] == ["A", "B", "C"]
    assert all(solution.success for solution in solutions)
    assert math.isclose(solutions[0].score, 1.0, rel_tol=1e-9)
    assert math.isclose(solutions[1].score, 0.5, rel_tol=1e-9)
    assert math.isclose(solutions[2].score, 1.0 / 3.0, rel_tol=1e-9)
