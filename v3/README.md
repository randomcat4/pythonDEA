# pythonDEA v3

v3 is the importable public package layer for pythonDEA. It wraps the tested v2
computational core with stable model plugins, standard result tables, packaging,
and small deterministic datasets for smoke tests and examples.

## Quickstart

```python
from pythondea import fit
from pythondea.datasets import load_emissions_cross_section

data = load_emissions_cross_section()
result = fit("sbm", data, orientation="bad_output_adjusted")

for row in result.table("efficiency").rows:
    print(row["dmu"], row["score"])
```

## Built-In Models

- `sbm`: CRS/VRS slack-based measure DEA with non-oriented, input-oriented,
  output-oriented, and bad-output-adjusted orientations.
- `sbm_malmquist`: adjacent-period SBM-Malmquist decomposition over balanced
  panel data.

## Plugin Contract

Third-party models register a `ModelSpec` with an estimator class. An estimator
needs a `fit(data, *, context=None)` method returning `ModelResult`. Publication
metadata should include solver information; the built-in SBM plugins report the
current SciPy HiGHS backend.

```python
from pythondea import ModelResult, ModelSpec, ResultTable, register_model


class MyEstimator:
    model_name = "my_model"

    def fit(self, data, *, context=None):
        return ModelResult(
            model=self.model_name,
            status="ok",
            primary_table="scores",
            tables=(ResultTable.from_rows("scores", []),),
        )


register_model(
    ModelSpec(
        name="my_model",
        estimator=MyEstimator,
        family="dea",
        summary="Example third-party model.",
    )
)
```

## Research Scope

The v3 package should prioritize publishable computational depth: undesirable
outputs, panel productivity decompositions, non-radial/directional frontiers,
super-efficiency, and bootstrap or robustness layers. It should not chase GUI
parity or a broad classic DEA toolbox unless a concrete research model requires
that surface.
