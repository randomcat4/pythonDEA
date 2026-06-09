# pythonDEA

`v1/` contains the original SBM-Malmquist implementation exactly as archived.

`v2/` is the new computational core. Its direction is not to clone the full
classic pyDEA model matrix, but to improve the existing SBM-Malmquist line with
research-useful outputs:

- SBM efficiency scores with CRS and VRS frontiers.
- Explainable SBM results: lambdas, peers, slacks, and targets.
- Undesirable-output support for environmental, energy, and carbon studies.
- Reference-set masks for period, global, grouped, and exclude-self frontiers.

`pythondea` is the importable research package layer. It keeps the tested v2
numerical core, then adds stable public APIs, model plugins, standard result
tables, toy datasets, packaging metadata, and publication-oriented smoke tests.

```python
from pythondea import fit
from pythondea.datasets import load_emissions_cross_section

data = load_emissions_cross_section()
result = fit("sbm", data, orientation="bad_output_adjusted")
print(result.table("efficiency").rows)
print(result.reproducibility_hash())
```

Current model plugins:

- `sbm`: slack-based measure DEA with CRS/VRS, orientation modes, undesirable
  outputs, peers, slacks, targets, and variable attributes inherited from v2.
- `sbm_super_efficiency`: exclude-self SBM for ranking and sensitivity checks.
- `sbm_malmquist`: adjacent-period SBM-Malmquist productivity decomposition
  with explainable underlying SBM solutions.
- `directional_distance`: v4 directional distance function for environmental
  frontiers, including bad-output contraction directions.
- `malmquist_luenberger`: v4 adjacent-period green productivity index based on
  DDF distances.

`audit_result(result)` provides a publication-readiness checklist for solver
metadata, result tables, version metadata, and deterministic result hashes.

The roadmap is deliberately research-first. v4 deepens environmental frontier
and panel productivity functionality instead of cloning classic CCR/BCC model
matrices or adding a frontend.

See [TODO.md](TODO.md) for the staged implementation plan.
