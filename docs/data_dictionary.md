# Data dictionary

## Catalogue

`data/catalogue/candidate_catalogue.parquet` contains one row per benchmark instance. Core identifiers are `instance_id`, `skeleton_id`, `track`, `treatment_factor`, `treatment_level`, `scale_class`, `product_regime`, `replicate`, and `skeleton_seed`. Factor columns are `network_support`, `cost_variability`, `capacity_tightness`, `demand_concentration`, `fixed_charge_intensity`, and `uncertainty_amplitude`.


## Instance Tables

### `sources.parquet`
- `source_id`
- `source_name`

### `destinations.parquet`
- `destination_id`
- `destination_name`

### `products.parquet`
- `product_id`
- `product_name`

### `scenarios.parquet`
- `scenario_id`
- `probability`

Deterministic instances contain only `base` with probability 1. Scenario instances contain `low`, `base`, and `high`.

### `lanes.parquet`
- `lane_id`
- `source_id`
- `destination_id`
- `distance_miles`
- `stress_load_ktons` — universal backbone-routing simultaneous multi-product stress envelope retained as the compact constructive feasibility certificate
- `capacity_ktons`
- `fixed_cost_usd`
- `base_flow_ktons`
- `provenance`

### `supplies.parquet`
- `source_id`
- `product_id`
- `scenario_id`
- `supply_ktons`

### `demands.parquet`
- `destination_id`
- `product_id`
- `scenario_id`
- `demand_ktons`

### `variable_costs.parquet`
- `lane_id`
- `product_id`
- `cost_usd_per_kton`


## Metadata

`metadata.json` contains benchmark and instance identifiers, track and treatment identity, complete catalogue specification, empirical skeleton seed and selected entities, FAF provenance and source SHA-256, operational calibration definitions, realised factor statistics, matched-design declarations, constructive-feasibility metadata, the generation signature used for stale-artifact detection, and storage metadata.


## Validation

`validation.json` records:

- `valid` and `validation_status`;
- `feasibility_status`;
- structural and scenario-ordering checks;
- product-scenario support-feasibility certificate;
- constructive shared-capacity certificate derived from `stress_load_ktons`;
- optional independent multicommodity LP cross-check;
- warnings and errors;
- storage-manifest integrity;
- SHA-256 hashes of instance files.

Independent shared-capacity cross-check statuses are `independently_certified`, `certificate_timeout`, `infeasible`, or `solver_error`. `certificate_timeout` is not an infeasibility certificate and does not invalidate a structurally valid instance with a valid constructive certificate.


## Pipeline and Progress Outputs

### `results/run_status.json`
Persistent status for the current or most recent `run-pipeline` execution. It records stage, completed/total work units, percentage, current instance, elapsed time, estimated remaining time when available, and timestamp.

### `results/pipeline_report.json`
Final machine-readable pipeline summary including the generation signature, solver protocol, output locations, and completion timestamp.

### `results/validation/validation_summary.parquet`
One row per benchmark instance with validity, feasibility status, warning count, and error count.

### `results/validation/family_validation.parquet`
One row per skeleton-track family documenting the realized matched-family isolation check.

### `results/screening_results.parquet`
Standard 10-second LP / 60-second MILP screening results. Principal fields include solution status, 60-second difficulty class, MILP runtime, incumbent objective, best bound, relative gap, node count, LP runtime/objective, provisional screening root gap, factor identifiers, and realized structural metrics.

### `results/lp_30_results.parquet`
Fresh 30-second LP-relaxation results only for valid instances whose original 10-second LP was not certified optimal.

### `results/characterisation/lp_relaxation_horizons.parquet`
Per-instance LP evidence across the 10- and 30-second horizons. `certified_lp_objective_usd`, `certified_lp_horizon_seconds`, and `certified_root_gap` are populated only from an LP solve that reached optimality.

### `results/characterisation/lp_relaxation_summary.json`
Counts of LPs certified at 10 seconds, sent to the 30-second follow-up, newly certified at 30 seconds, and still unresolved after 30 seconds.

### `results/reference_200_results.parquet`
Fresh 200-second MILP results for valid instances unresolved at 60 seconds.

### `results/reference_400_results.parquet`
Fresh 400-second MILP results only for instances unresolved in the fresh 200-second run.

### `results/characterisation/reference_horizons.parquet`
Per-instance joined computational evidence across the 60-, 200-, and 400-second horizons.

### `results/characterisation/reference_horizon_summary.json`
Counts of certified and unresolved instances at each computational horizon.

### `results/checkpoints/`
Per-instance/per-horizon local restart checkpoints. These are operational artifacts for resuming an interrupted run.
