import math

from pythondea import DEAData, PanelDEAData, audit_result, fit, list_models


def test_directional_distance_plugin_reduces_bad_outputs():
    data = DEAData(
        inputs=[[1.0], [1.0]],
        good_outputs=[[1.0], [1.0]],
        bad_outputs=[[1.0], [2.0]],
        dmu_names=["clean", "dirty"],
        bad_output_names=["co2"],
    )

    result = fit("directional_distance", data, direction="bad_output")
    rows = {row["dmu"]: row for row in result.rows}
    dirty_targets = [
        row
        for row in result.table("targets").rows
        if row["dmu"] == "dirty" and row["role"] == "bad_output"
    ]

    assert result.model == "directional_distance"
    assert math.isclose(rows["dirty"]["beta"], 0.5, rel_tol=1e-9)
    assert dirty_targets[0]["target"] == 1.0
    assert audit_result(result).passed is True


def test_malmquist_luenberger_plugin_returns_green_productivity_rows():
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

    result = fit("malmquist_luenberger", panel)

    assert result.model == "malmquist_luenberger"
    assert len(result.rows) == 2
    assert result.table().columns[:6] == (
        "dmu",
        "period_from",
        "period_to",
        "ml_index",
        "efficiency_change",
        "technical_change",
    )
    assert result.rows[1]["ml_index"] is not None
    assert audit_result(result).passed is True


def test_v4_models_are_registered_with_frontier_research_metadata():
    catalog = {spec.name: spec for spec in list_models()}

    assert "directional_distance" in catalog
    assert "malmquist_luenberger" in catalog
    assert catalog["directional_distance"].family == "dea-environmental"
    assert "green-productivity" in catalog["malmquist_luenberger"].keywords
