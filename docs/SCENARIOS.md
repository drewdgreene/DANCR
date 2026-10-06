# Scenario sets (F2): running a project across many assumptions

A **scenario** is a named set of Input values. A **scenario set** is one of:

- **named** — an explicit list of `{"id", "inputs"}`;
- **sweep** — the cartesian product of input ranges (a grid);
- **monte_carlo** — a seeded random sample within input ranges;
- **sensitivity** — one-at-a-time: the base plus each single input varied.

Every set is deterministic: a sweep is sorted, and Monte Carlo uses an explicit
seed, so the same spec always yields the same scenarios (and the same hashes).

## The runner

`dancr.headless.run_scenarios` runs a target step once per scenario. Each scenario
runs on a **private clone** of the project — the project file is never changed —
with the existing cache, leases and write confinement. It writes one output per
scenario plus a combined table (a `scenario` column) inside the project folder,
never into `.dancr` and never over a source file.

Every scenario records its **plan hash** and **output hash**. Because an Input
value is already part of a step's plan hash (`Executor.inputs_used`), changing one
input recomputes only the steps that use it.

```bash
dancr scenarios shop.json --set cases.json --out-dir results
dancr scenarios shop.json --set '{"sweep": {"rate": [1, 2, 5]}}' --node forecast
dancr --json scenarios shop.json --set cases.json --manifest run.json
```

`cases.json` is a scenario spec:

```json
{"base": {"rate": 1},
 "sweep": {"rate": [1, 2, 5], "limit": 10},
 "monte_carlo": {"demand": {"min": 0, "max": 100}},
 "sensitivity": {"rate": [1, 2, 5]},
 "n": 100, "seed": 0,
 "scenarios": [{"id": "low", "inputs": {"rate": 1}}, {"id": "high", "inputs": {"rate": 5}}]}
```

Only one of `scenarios`/`named`/`sweep`/`monte_carlo`/`sensitivity`/`base` is used;
the first present wins.

## Proof across scenarios

The runner returns an `evidence` block with the per-scenario plan and output
hashes. It is folded into an attestation through the generic extension point
(`docs/adr/0007-hashing-attestation-extension.md`):

```bash
dancr verify shop.json --record att.json --scenarios cases.json
dancr verify shop.json --manifest att.json --scenarios cases.json
```

`--scenarios` runs the set first, folds its hashes into the record, and on verify
recomputes the set and compares it — so a scenario set is reproducible and
provable, and an old attestation without an evidence block still verifies.

## MCP and SDK

- MCP `run_scenarios(path, spec | spec_path, target, out_dir, ext, jobs, force)`.
- SDK `dancr.run_scenarios(pipe, spec, …)` and `dancr.scenario_set(spec)`.

## Honest limits

- Each scenario is a separate run; there is no fan-out inside the LazyFrame
  executor, so a large grid costs proportionally. `--jobs` runs scenarios at once
  (the shared cache is safe for it; progress lines interleave).
- Results are written per scenario, so a very large grid grows the output folder.
- Monte Carlo is uniform (or choice) only; the seed is explicit and stored in the
  spec, never in the project.
