import math

import pytest

from v2 import DEAData, solve_sbm


def test_non_controllable_input_target_is_fixed_by_zero_slack():
    data = DEAData(
        inputs=[[1.0, 1.0], [1.0, 2.0]],
        good_outputs=[[1.0], [1.0]],
        dmu_names=["A", "B"],
        input_names=["land", "labor"],
        good_output_names=["service"],
        non_controllable_inputs=["land"],
    )

    solution = solve_sbm(data, 1)

    assert solution.success is True
    assert math.isclose(solution.input_slacks["land"], 0.0, abs_tol=1e-9)
    assert math.isclose(solution.input_targets["land"], 1.0, rel_tol=1e-9)
    assert math.isclose(solution.input_slacks["labor"], 1.0, rel_tol=1e-9)
    assert solution.variable_attributes["land"]["controllable"] is False
    assert solution.variable_attributes["land"]["allows_slack"] is False


def test_weakly_disposable_bad_output_is_traceable_and_not_freely_reduced():
    data = DEAData(
        inputs=[[1.0], [1.0]],
        good_outputs=[[1.0], [1.0]],
        bad_outputs=[[1.0], [2.0]],
        dmu_names=["clean", "dirty"],
        input_names=["capital"],
        good_output_names=["gdp"],
        bad_output_names=["co2"],
        weakly_disposable_bad_outputs=["co2"],
    )

    solution = solve_sbm(data, 1)

    assert solution.success is True
    assert math.isclose(solution.bad_output_slacks["co2"], 0.0, abs_tol=1e-9)
    assert math.isclose(solution.bad_output_targets["co2"], 2.0, rel_tol=1e-9)
    assert solution.variable_attributes["co2"]["disposability"] == "weak"
    assert solution.variable_attributes["co2"]["allows_slack"] is False


def test_unknown_variable_attribute_name_fails_loudly():
    with pytest.raises(ValueError, match="unknown input variable"):
        DEAData(
            inputs=[[1.0]],
            good_outputs=[[1.0]],
            input_names=["labor"],
            non_controllable_inputs=["land"],
        )
