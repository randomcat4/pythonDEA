# pythonDEA

`v1/` contains the original SBM-Malmquist implementation exactly as archived.

`v2/` is the new computational core. Its direction is not to clone the full
classic pyDEA model matrix, but to improve the existing SBM-Malmquist line with
research-useful outputs:

- SBM efficiency scores with CRS and VRS frontiers.
- Explainable SBM results: lambdas, peers, slacks, and targets.
- Undesirable-output support for environmental, energy, and carbon studies.
- Reference-set masks for period, global, grouped, and exclude-self frontiers.

The first v2 milestone is a testable non-oriented SBM solver. Later milestones
will add panel/global orchestration, variable attributes, weak disposability,
and Malmquist integration.

See [TODO.md](TODO.md) for the staged v2 implementation plan.
