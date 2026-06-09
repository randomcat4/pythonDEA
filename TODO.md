# pythonDEA v2 TODO

This roadmap follows the GPT-5.5 Pro review: keep v1 frozen and build v2 around
the existing SBM-Malmquist research line instead of cloning the classic pyDEA
model matrix.

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

## P4: SBM Orientation Variants

- Goal: implement `non_oriented`, `input`, `output`, and
  `bad_output_adjusted` SBM variants behind one API.
- Files: add `v2/orientation.py`; extend `v2/sbm.py`.
- Tests: small radial-degenerate examples, zero-slack examples, and bad-output
  reduction examples.
- Completion: callers can switch orientation while keeping the same data and
  result shape.
- Push: yes.

## P5: v2 SBM-Malmquist Integration

- Goal: rebuild and enhance SBM-Malmquist decompositions on top of v2:
  FGLR1992, FGLR1994, RD1997, Zofio2007, and PL2005.
- Files: add `v2/malmquist.py` and `v2/decomposition.py`.
- Tests: reproduce v1 index formulas on controlled data; add period-level
  slacks, peers, and targets to outputs.
- Completion: v1-compatible indices plus v2 explanation fields.
- Push: yes, then tag `v2-sbm-malmquist-alpha`.

## Explicit Non-Goals

- Do not build a pyDEA-style classic DEA toolbox.
- Do not add GUI or frontend work in this v2 computation pass.
- Do not build an Excel template system before the core APIs are stable.
- Do not add network DEA, dynamic DEA, bootstrap DEA, or stochastic frontier
  models in this scope.
- Do not rewrite or mutate v1 algorithms.
- Do not break v1 result compatibility for the sake of a unified interface.
