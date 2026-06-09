import math

import pytest

from pythondea import dea_from_dataframe, fit, list_models, panel_from_dataframe


def test_dea_from_dataframe_builds_named_cross_section():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {
            "firm": ["A", "B"],
            "labor": [1.0, 2.0],
            "service": [1.0, 1.0],
            "co2": [1.0, 2.0],
        }
    )

    data = dea_from_dataframe(
        frame,
        dmu_column="firm",
        inputs=["labor"],
        good_outputs=["service"],
        bad_outputs=["co2"],
    )

    assert data.dmu_names == ["A", "B"]
    assert data.input_names == ["labor"]
    assert data.bad_output_names == ["co2"]
    assert fit("sbm", data).rows[1]["dmu"] == "B"


def test_panel_from_dataframe_enforces_balanced_panel():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {
            "year": [2020, 2020, 2021, 2021],
            "firm": ["A", "B", "A", "B"],
            "labor": [1.0, 2.0, 1.1, 2.1],
            "service": [1.0, 1.0, 1.2, 1.1],
        }
    )

    panel = panel_from_dataframe(
        frame,
        period_column="year",
        entity_column="firm",
        inputs=["labor"],
        good_outputs=["service"],
    )

    assert panel.periods == ("2020", "2021")
    assert panel.entities == ("A", "B")
    assert len(fit("sbm_malmquist", panel).rows) == 2


def test_panel_from_dataframe_reports_missing_balanced_rows():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {
            "year": [2020, 2021, 2021],
            "firm": ["A", "A", "B"],
            "labor": [1.0, 1.1, 2.1],
            "service": [1.0, 1.2, 1.1],
        }
    )

    with pytest.raises(ValueError, match="missing balanced panel row"):
        panel_from_dataframe(
            frame,
            period_column="year",
            entity_column="firm",
            inputs=["labor"],
            good_outputs=["service"],
            periods=[2020, 2021],
            entities=["A", "B"],
        )


def test_super_efficiency_model_is_registered_and_excludes_self():
    names = [spec.name for spec in list_models()]
    assert "sbm_super_efficiency" in names

    data = dea_from_dataframe(
        {
            "firm": ["A", "B", "C"],
            "labor": [1.0, 2.0, 3.0],
            "service": [1.0, 1.0, 1.0],
        },
        dmu_column="firm",
        inputs=["labor"],
        good_outputs=["service"],
    )

    result = fit("sbm_super_efficiency", data)
    rows = {row["dmu"]: row for row in result.rows}

    assert result.metadata["super_efficiency"] is True
    assert math.isclose(rows["B"]["score"], 0.5, rel_tol=1e-9)
    assert rows["B"]["reference"] == "all:exclude-1"
