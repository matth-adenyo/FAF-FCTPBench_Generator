# FAF-FCTPBench Generator

FAF-FCTPBench Generator is a reproducible benchmark-generation framework for deterministic and scenario-based fixed-charge transportation problems (FCTPs). It preserves freight-demand structure derived from the U.S. Freight Analysis Framework (FAF5.7.1) while controlling optimization-relevant structural factors through matched comparisons.

The central experimental principle is **matched control**. Geography, commodities, empirical support, and primitive random draws are frozen within an empirical skeleton. Every non-reference treatment changes one declared factor relative to the same-skeleton, same-track reference instance.


## Empirical Scope

The default design uses:

- `FAF5.7.1_HiLoForecasts.zip`;
- pinned SHA-256 `3dcaff4276b330c942e3f9282283963e6476ad84693d4a966b822c9bf2f5467c`;
- forecast year 2030;
- truck mode (`dms_mode = 1`);
- domestic-only trade (`trade_type = 1`);
- SCTG2 commodities;
- FAF low, base, and high forecast quantities;
- distance proxies derived from FAF ton-miles divided by tons.

The benchmark therefore preserves **FAF-derived U.S. domestic truck freight-demand structure**. Capacities, fixed charges, and operational cost perturbations are benchmark calibrations.


## Default Matched Design

The catalogue contains 24 empirical skeletons. Four scale classes, three product-count regimes, and two empirical replicates per scale-product cell. Each skeleton contributes 11 deterministic variants and 13 scenario variants, giving **576 instances** in total (264 deterministic and 312 scenario-based).

The controlled factors are network support, variable-cost variability, capacity tightness, demand concentration, fixed-charge intensity, and scenario uncertainty amplitude. Uncertainty amplitude applies only to the scenario track.


## Scenario-Ordering and Feasibility Safeguards

Demand concentration is applied with an order-preserving construction. The transformed base marginal receives the requested concentration treatment. The low marginal is projected onto a capped simplex below the transformed base, and the high increment is projected onto a nonnegative simplex above it. Scenario totals are retained while enforcing, componentwise,

`low <= base <= high`.

The universal network backbone is constructed from FAF-observed support. A second routing pass on the final backbone builds a simultaneous multi-product lane-load stress envelope covering every configured concentration, uncertainty, and scenario regime. Each instance stores the corresponding `stress_load_ktons` values. Capacity treatments are calibrated above this common constructive envelope.

The independent multicommodity LP is a **secondary cross-check**. Its outcomes are distinguished:

- `independently_certified`;
- `certificate_timeout`;
- `infeasible`;
- `solver_error`.


## Reference Computational Protocol

The standard pipeline uses fresh solves at three MILP horizons:

| Stage | Limit | Purpose |
|---|---:|---|
| LP relaxation | 10 s | initial root/relaxation evidence for standardized screening |
| LP follow-up | 30 s | fresh solve only when the 10 s LP was not certified optimal |
| Screening MILP | 60 s | common computational-hardness observation horizon |
| Reference MILP | 200 s | fresh solve for valid instances unresolved at 60 s |
| Extended MILP | 400 s | fresh solve only for instances unresolved in the 200 s run |

The 200-second and 400-second stages are fresh solves from time zero; they are not continuations of earlier branch-and-bound trees. Difficulty labels remain conditional on the 60-second screening protocol and are not used to select benchmark instances.


## Resumable End-to-End Pipeline

`run-pipeline` is the primary workflow. It performs source verification, catalogue generation, instance materialization, individual validation, matched-family validation, 10-second LP/60-second MILP screening, a targeted fresh 30-second LP follow-up only for LPs not certified at 10 seconds, 200-second reference MILP solves, 400-second extended MILP solves, matched characterization, horizon-persistence reporting, and final compact packaging.

Every expensive unit is checkpointed. Persistent progress is written to:

`results/run_status.json`


## Compact Storage

Canonical numerical data use **Apache Parquet with Zstandard compression level 9 and dictionary encoding**.

Each instance contains:

```text
metadata.json
storage_manifest.json
validation.json
sources.parquet
destinations.parquet
products.parquet
scenarios.parquet
lanes.parquet
supplies.parquet
demands.parquet
variable_costs.parquet
```

## Installation

Use Python 3.11 or newer. From the repository root:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

Verify the CLI:

```powershell
python -m faf_fctp --help
```

## Complete Run

Place the official source archive at:

```text
data/raw/FAF5.7.1_HiLoForecasts.zip
```

Then run:

```powershell
python -m faf_fctp run-pipeline --project-root .
```


The principal final outputs are:

```text
data/catalogue/candidate_catalogue.parquet
data/instances/
results/validation/validation_summary.parquet
results/screening_results.parquet
results/lp_30_results.parquet
results/characterization/lp_relaxation_horizons.parquet
results/characterization/lp_relaxation_summary.json
results/reference_200_results.parquet
results/reference_400_results.parquet
results/characterization/matched_effects.parquet
results/characterization/reference_horizons.parquet
results/characterization/reference_horizon_summary.json
results/pipeline_report.json
results/run_status.json
dist/FAF-FCTPBench.zip
```

Checkpoint files remain under `results/checkpoints/` for local restart capability.


## Documentation

- `docs/methodology.md` — construction, calibration, validation, computational protocol, and restart semantics.
- `docs/data_dictionary.md` — canonical table, metadata, validation, and result fields.


Benchmark instances are archived on Zenodo: https://doi.org/10.5281/zenodo.23425977
