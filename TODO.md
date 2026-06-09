# pythonDEA TODO

This roadmap follows the GPT-5.5 Pro review direction: keep v1 frozen and build
around the existing SBM-Malmquist research line instead of cloning the classic
pyDEA model matrix.

## P0: Migration Freeze

- Goal: move the original implementation under `v1/` without changing algorithm
  content or output semantics.
- Files: `v1/**`; v2 importable package skeleton.
- Tests: compile `v1/dea_calculator.py` and `v1/runner.py`; verify git records
  the archive as pure renames.
- Completion: original files are archived under `v1/`; root is ready for v2.
- Push: yes.
- Status: done in commit `6f7b49e`.

## P1: Explainable SBM Results

- Goal: v2 SBM returns score, solver status, lambdas, peers, slacks, and targets.
- Files: `v2/problem.py`, `v2/reference.py`, `v2/sbm.py`, `tests/test_v2_sbm.py`.
- API: `DEAData`, `ReferenceSet`, `solve_sbm`, `solve_all_sbm`, `SBMSolution`.
- Tests: verify `x_target = x - s_minus`, `y_target = y + s_plus`,
  bad-output target reduction, positive lambdas as peers, CRS/VRS solving, and
  exclude-self reference masks.
- Completion: result object fields are stable and covered by tests.
- Push: yes.
- Status: first milestone implemented.

## P2: Reference-Set Abstraction For Panel Frontiers

- Goal: represent contemporaneous, global, cross-period, and window frontiers
  with one reference-set layer for later Malmquist calls.
- Files: extend `v2/reference.py`; add panel data helpers and `v2/malmquist.py`
  skeleton.
- Tests: same-period, all-period, next-period, previous-period, and custom mask
  indices are correct.
- Completion: FGLR, RD, Zofio, and PL decompositions can reuse the same solver
  by swapping reference sets.
- Push: yes.
- Status: first milestone implemented with balanced panel flattening and
  contemporaneous, global, cross-period, and closed-window references.

## P3: Variable Attribute Layer

- Goal: support variable role and modeling attributes: input, good output, bad
  output, controllable, non-controllable, disposable, and weakly disposable.
- Files: add `v2/variables.py`; extend `v2/problem.py` and `v2/sbm.py`.
- Tests: bad-output direction is correct; non-controllable targets are fixed;
  weak-disposability constraints are traceable in the LP.
- Completion: each variable attribute is reflected in LP construction and
  result metadata.
- Push: yes.
- Status: first milestone implemented with variable specs, non-controllable
  slack fixing, weak-disposability slack fixing, and result metadata.

## P4: SBM Orientation Variants

- Goal: implement `non_oriented`, `input`, `output`, and
  `bad_output_adjusted` SBM variants behind one API.
- Files: add `v2/orientation.py`; extend `v2/sbm.py`.
- Tests: small radial-degenerate examples, zero-slack examples, and bad-output
  reduction examples.
- Completion: callers can switch orientation while keeping the same data and
  result shape.
- Push: yes.
- Status: first milestone implemented with stable `orientation` API and tested
  input, output, and bad-output-adjusted slack masks.

## P5: v2 SBM-Malmquist Integration

- Goal: rebuild and enhance SBM-Malmquist decompositions on top of v2:
  FGLR1992, FGLR1994, RD1997, Zofio2007, and PL2005.
- Files: add `v2/malmquist.py` and `v2/decomposition.py`.
- Tests: reproduce v1 index formulas on controlled data; add period-level
  slacks, peers, and targets to outputs.
- Completion: v1-compatible indices plus v2 explanation fields.
- Push: yes, then tag `v2-sbm-malmquist-alpha`.
- Status: first alpha implemented with pure v1-compatible decomposition
  formulas and adjacent-panel orchestration retaining explainable `SBMSolution`
  objects.

## Explicit Non-Goals

- Do not build a pyDEA-style classic DEA toolbox.
- Do not add GUI or frontend work in this v2 computation pass.
- Do not build an Excel template system before the core APIs are stable.
- Do not add network DEA, dynamic DEA, bootstrap DEA, or stochastic frontier
  models in this scope.
- Do not rewrite or mutate v1 algorithms.
- Do not break v1 result compatibility for the sake of a unified interface.

## v3: Importable, Pluggable Research Package

### V3-P0: Package And Plugin Contract

- Goal: make `pythondea` importable after install and usable directly from
  Python.
- Files: `pyproject.toml`, `v3/pythondea/**`, `tests/test_v3_import_registry.py`.
- API: `fit`, `ModelRegistry`, `ModelSpec`, `ModelResult`, `ResultTable`,
  `Estimator`, `SolverBackend`, `DEAData`, `PanelDEAData`.
- Tests: import package, list built-in models, register a third-party style
  model spec, and run it through `fit`.
- Completion: external models can plug into the registry without editing core
  package files.
- Status: implemented in v3 alpha.

### V3-P1: Stable SBM Research Estimators

- Goal: expose v2 SBM and SBM-Malmquist through stable estimator plugins and
  standard result tables.
- Files: `v3/pythondea/models/sbm.py`, `v3/pythondea/datasets.py`,
  `tests/test_v3_sbm_estimators.py`.
- API: `fit("sbm", data, ...)`, `fit("sbm_malmquist", panel, ...)`,
  `SBMEstimator`, `SBMMalmquistEstimator`.
- Tests: preserve v2 score semantics, return efficiency/slack/target tables,
  return Malmquist decomposition rows, and reject wrong data grains.
- Completion: v3 is importable and the current research line is available from
  one stable API.
- Status: implemented in v3 alpha.

### V3-P2: Publishable Data Interfaces

- Goal: accept richer research data without tying the core to Excel templates.
- Worth doing: pandas DataFrame adapters, formula-like column role specs,
  balanced/unbalanced panel validation, grouped frontiers, missing-value policy,
  deterministic provenance/fingerprints.
- Tests: round-trip arrays/DataFrames into identical model results; verify
  error messages identify invalid columns, non-positive values, and panel gaps.
- Completion: empirical scripts can build data objects from common research
  tables with explicit role metadata.
- Status: first milestone implemented with cross-section and balanced-panel
  DataFrame adapters plus missing/duplicate row checks.

### V3-P3: Frontier-Useful Model Extensions

- Worth doing: non-radial directional distance functions, global/window
  Malmquist-Luenberger indices with undesirable outputs, bootstrap confidence
  intervals for DEA scores and productivity changes, and robust/bias-corrected
  reporting.
- Why: these extensions support environmental, energy, productivity, and carbon
  efficiency papers better than a broad catalog of older radial variants.
- Tests: small analytic examples, monotonicity checks, replication fixtures from
  published formulas, and seed-stable bootstrap intervals.
- Status: first milestone includes explicit `sbm_super_efficiency`; DDF,
  Malmquist-Luenberger, and bootstrap remain planned.

### V3-P4: Publication Audit Layer

- Goal: make results defensible in a paper appendix.
- Worth doing: solver status audit, infeasible-DMU report, dual/frontier
  diagnostics where available, peer stability summaries, citation metadata,
  result serialization, exact configuration capture, repository license choice,
  and CI evidence.
- Tests: failed LPs propagate as partial results; serialized results preserve
  all configuration and table data.
- Status: first milestone implemented with `audit_result`, `model_catalog`, JSON
  serialization, reproducibility hashes, and a runnable SBM-Malmquist example.

### V3 Non-Goals

- No frontend or GUI parity.
- No full clone of pyDEA's classic CCR/BCC/additive model matrix unless a
  specific research extension needs it.
- No Excel-template-first workflow before the Python API is stable.
- No stochastic frontier analysis in the DEA package core.
- No broad "all DEA models" promise; v3 should deepen the publishable
  SBM/environmental/panel line first.

## v4: Directional Environmental Frontier Models

### V4-P0: Directional Distance Function

- Goal: add a publishable environmental DEA primitive that can contract inputs,
  expand good outputs, and contract undesirable outputs along explicit
  directions.
- Files: `v2/ddf.py`, `v3/pythondea/models/ddf.py`,
  `tests/test_v2_ddf.py`, `tests/test_v4_ddf_models.py`.
- API: `solve_ddf`, `solve_all_ddf`, `fit("directional_distance", data,
  direction="bad_output")`.
- Result fields: `beta`, peers, reference label, solver status, and projected
  input/good-output/bad-output targets.
- Why v4 does this before CCR/BCC: DDF directly supports carbon, pollution, and
  green productivity papers; CCR/BCC parity would mostly broaden an older model
  matrix without improving the package's research contribution.
- Status: implemented.

### V4-P1: Malmquist-Luenberger Green Productivity

- Goal: compute adjacent-period Malmquist-Luenberger transitions from DDF
  distances for panels with undesirable outputs.
- Files: `v2/ddf.py`, `v3/pythondea/models/ddf.py`,
  `examples/v4_green_productivity_reproduction.py`.
- API: `compute_adjacent_malmquist_luenberger` and
  `fit("malmquist_luenberger", panel, direction="bad_output")`.
- Result fields: `ml_index`, `efficiency_change`, `technical_change`, and the
  four cross-period DDF distances.
- Status: implemented.

### V4 Non-Goals

- No frontend, GUI, or Excel-template work.
- No broad CCR/BCC/additive model catalog.
- No network/dynamic SBM until DDF and Malmquist-Luenberger have stronger
  replication fixtures.
- No bootstrap confidence intervals in v4. They remain valuable, but should be
  implemented as a separate inference layer after model equations are stable.
