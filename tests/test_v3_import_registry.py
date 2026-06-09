import pythondea
from pythondea import (
    DEAData,
    ModelSpec,
    SciPyHiGHSBackend,
    fit,
    get_model,
    list_models,
    register_model,
)


class DummyEstimator:
    model_name = "dummy_frontier"

    def __init__(self, value=1.0):
        self.value = value

    def fit(self, data, *, context=None):
        from pythondea import ModelResult, ResultTable

        return ModelResult(
            model=self.model_name,
            status="ok",
            primary_table="scores",
            tables=(
                ResultTable.from_rows(
                    "scores",
                    [{"dmu": data.dmu_names[0], "score": self.value}],
                    columns=("dmu", "score"),
                ),
            ),
        )


def test_v3_imports_default_models():
    assert pythondea.__version__ == "0.3.0"
    names = [spec.name for spec in list_models()]
    assert "sbm" in names
    assert "sbm_malmquist" in names
    assert get_model("SBM").family == "dea"
    assert SciPyHiGHSBackend().info().method == "highs"


def test_v3_registry_accepts_external_model_specs():
    register_model(
        ModelSpec(
            name="dummy_frontier",
            estimator=DummyEstimator,
            family="test",
            summary="A deterministic test plugin.",
        ),
        replace=True,
    )
    data = DEAData(inputs=[[1.0]], good_outputs=[[1.0]], dmu_names=["A"])

    result = fit("dummy_frontier", data, value=0.75)

    assert result.model == "dummy_frontier"
    assert result.rows[0]["score"] == 0.75
