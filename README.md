# pythonDEA

`v1/` contains the original SBM-Malmquist implementation exactly as archived.

`v2/` is the new computational core. Its direction is not to clone the full
classic pyDEA model matrix, but to improve the existing SBM-Malmquist line with
research-useful outputs:

- SBM efficiency scores with CRS and VRS frontiers.
- Explainable SBM results: lambdas, peers, slacks, and targets.
- Undesirable-output support for environmental, energy, and carbon studies.
- Reference-set masks for period, global, grouped, and exclude-self frontiers.

`v3/` is the importable research package layer. It keeps the tested v2 numerical
core, then adds stable public APIs, model plugins, standard result tables, toy
datasets, packaging metadata, and publication-oriented smoke tests.

```python
from pythondea import fit
from pythondea.datasets import load_emissions_cross_section

data = load_emissions_cross_section()
result = fit("sbm", data, orientation="bad_output_adjusted")
print(result.table("efficiency").rows)
```

Current v3 model plugins:

- `sbm`: slack-based measure DEA with CRS/VRS, orientation modes, undesirable
  outputs, peers, slacks, targets, and variable attributes inherited from v2.
- `sbm_super_efficiency`: exclude-self SBM for ranking and sensitivity checks.
- `sbm_malmquist`: adjacent-period SBM-Malmquist productivity decomposition
  with explainable underlying SBM solutions.

The v3 roadmap is deliberately research-first. Near-term work should deepen
frontier, environmental, panel, uncertainty, and decomposition functionality.
Classic pyDEA-style GUI/template parity and broad radial DEA model matrices are
not near-term goals.

See [TODO.md](TODO.md) for the staged v2 implementation plan.
