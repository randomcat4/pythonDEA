import math

from pythondea import DEAData, PanelDEAData, fit
from pythondea.datasets import load_emissions_cross_section, load_productivity_panel


def test_sbm_estimator_returns_standard_v3_tables():
    data = load_emissions_cross_section()

    result = fit("sbm", data, orientation="bad_output_adjusted")

    assert result.model == "sbm"
    assert result.status == "ok"
    assert result.metadata["pythondea_version"] == "3.0.0a2"
    assert result.metadata["solver_backend"]["method"] == "highs"
    assert [row["dmu"] for row in result.rows] == ["clean", "balanced", "dirty"]
    assert result.table("slacks").columns == ("dmu", "variable", "role", "slack")
    assert "efficiency" in result.to_json()
    assert len(result.reproducibility_hash()) == 64
    assert list(result.table("efficiency").to_pandas().columns)[:2] == ["dmu", "score"]
    dirty = [row for row in result.table("targets").rows if row["dmu"] == "dirty" and row["role"] == "bad_output"]
    assert dirty
    assert dirty[0]["target"] < 2.2


def test_sbm_estimator_preserves_v2_score_semantics():
    data = DEAData(
        inputs=[[1.0], [2.0], [3.0]],
        good_outputs=[[1.0], [1.0], [1.0]],
        dmu_names=["A", "B", "C"],
    )

    result = fit("sbm", data, returns_to_scale="crs")
    rows = {row["dmu"]: row for row in result.rows}

    assert math.isclose(rows["A"]["score"], 1.0, rel_tol=1e-9)
    assert math.isclose(rows["B"]["score"], 0.5, rel_tol=1e-9)
    assert math.isclose(rows["C"]["score"], 1.0 / 3.0, rel_tol=1e-9)


def test_malmquist_estimator_returns_panel_decomposition_table():
    panel = load_productivity_panel()

    result = fit("sbm_malmquist", panel)

    assert result.model == "sbm_malmquist"
    assert result.status == "ok"
    assert result.table().columns[:3] == ("dmu", "period_from", "period_to")
    assert len(result.rows) == panel.n_entities
    assert "transitions" in result.artifacts


def test_malmquist_estimator_requires_panel_data():
    data = DEAData(inputs=[[1.0]], good_outputs=[[1.0]])

    try:
        fit("sbm_malmquist", data)
    except TypeError as exc:
        assert "PanelDEAData" in str(exc)
    else:
        raise AssertionError("expected PanelDEAData TypeError")
