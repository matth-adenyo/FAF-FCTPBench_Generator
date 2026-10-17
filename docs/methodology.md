# Methodology

## 1. Design Principle

FAF-FCTPBench Generator separates two layers:

1. **Empirical skeleton** — geography, commodities, observed OD support, base/low/high FAF demand and distance information.
2. **Optimization treatment** — one controlled change applied to a fixed skeleton.

The empirical skeleton is immutable within a matched family. All primitive pseudo-random quantities used for lane costs, fixed-charge weights, and selection ranking are keyed to the skeleton seed and reused across treatments.

For a skeleton `r` and track `t`, let the reference benchmark be `B(r,t,0)`. A non-reference benchmark `B(r,t,f,l)` changes factor `f` to level `l` and holds every other factor at its reference level.


## 2. FAF Source and Empirical Transformation

The default source is the archive `FAF5.7.1_HiLoForecasts.zip`, verified by SHA-256 before generation.

The default empirical scope is:

- forecast year 2030;
- truck mode (`dms_mode = 1`);
- domestic-only trade (`trade_type = 1`);
- SCTG2 commodities;
- low, base and high tonnage forecasts.

For an observed FAF OD-product record `(i,j,k)`, the normalized quantities are

- `q^L_ijk`: low forecast thousand tons,
- `q^B_ijk`: base forecast thousand tons,
- `q^H_ijk`: high forecast thousand tons.

The lane distance proxy is

`distance_ij = 1000 * ton_miles_ij / tons_ij`,

where ton-miles are reported in millions and tons in thousands.


## 3. Empirical Skeleton Selection

For each scale-product-replicate cell, a fixed skeleton seed is generated from the global design seed. Commodity, origin and destination selection uses reproducible activity-weighted ranking with a common seeded Gumbel perturbation. This creates distinct empirical replicates without changing them across treatment levels.

A skeleton stores:

- selected origins;
- selected destinations;
- selected commodities;
- selected FAF OD-product records;
- observed lane support;
- common lane/product random draws;
- universal feasibility backbone;
- universal shared-capacity stress envelope;
- a common reference variable-cost calibration.

If a selected skeleton cannot support all configured demand/uncertainty treatments on observed FAF lane support, the code advances through deterministic reselection attempts and records the selected attempt.


## 4. Scenario Amplitude

For amplitude `lambda`, row-level low/high quantities are generated around the FAF base value:

`q_low(lambda)  = max(0, q_base + lambda * (min(q_low_FAF,q_base) - q_base))`

`q_high(lambda) = max(q_base, q_base + lambda * (max(q_high_FAF,q_base) - q_base))`.

The configured low/reference/high amplitudes are 0.35, 1.00 and 1.75.

The deterministic track retains only the base scenario. The scenario track retains low/base/high scenarios with configured probabilities.


## 5. Demand Concentration

For a nonnegative marginal vector `d` with total `D`, the unconstrained concentration target is the total-preserving power transform

`d'_j(alpha) = D * d_j^alpha / sum_h d_h^alpha`.

Configured exponents are diffuse 0.75, reference 1.00, and concentrated 1.50.

For scenario instances, independently transforming and renormalising low, base, and high marginals can destroy the required componentwise ordering. The implementation therefore uses an order-preserving construction. The base marginal receives the power transform. The low target is projected onto the capped simplex

`0 <= d_low <= d_base`,  `sum d_low = D_low`,

and the high increment is projected onto the nonnegative simplex

`d_high >= d_base`,  `sum d_high = D_high`.

The projections minimise Euclidean displacement from the corresponding power-transformed targets. The resulting source and destination marginals preserve each product-scenario total and enforce `low <= base <= high` componentwise. The reference exponent 1.00 reproduces the original FAF-derived marginals.


## 6. Universal Observed-Lane Feasibility Backbone

For every configured uncertainty amplitude, demand-concentration level, scenario and product, the code solves a minimum-distance transportation LP on the full observed FAF support. The union of all lanes carrying positive flow in these feasible routings is the **universal feasibility backbone**.

This construction means the restricted network treatment remains feasible for the complete configured demand envelope.

The remaining observed FAF-supported lanes are ranked once using a skeleton-specific common random number. Network treatments are nested:

- restricted: backbone only;
- reference: backbone plus 50% of optional observed lanes;
- expanded: backbone plus all optional observed lanes.

Thus topology treatments never regenerate geography or commodity demand.


## 7. Shared-Capacity Stress Envelope and Constructive Certificate

The first backbone pass identifies the union of lanes used by feasible product-level routings across every configured uncertainty, concentration, and scenario regime. A second routing pass is then performed **on the final backbone itself**. Within each regime, commodity flows are summed lane-wise. If `x^r_a` is the simultaneous multi-product load on lane `a` for routing regime `r`, the universal backbone stress envelope is

`w_a = max_r x^r_a`.

Capacity is calibrated as

`U_a(h) = h * max(w_a, 0.05 * FAF_base_flow_a, epsilon)`.

Configured headroom multipliers are loose 2.00, reference 1.40, and tight 1.05. Because every network treatment contains the universal backbone and every capacity treatment dominates the stored stress load, the generator has a direct constructive all-open feasibility witness. `stress_load_ktons` is retained in the compact lane table so the capacity-to-envelope relationship can be checked without resolving a large multicommodity LP.

An independent all-open multicommodity LP is retained as a secondary cross-check. Its result is recorded as `independently_certified`, `certificate_timeout`, `infeasible`, or `solver_error`. A timeout is not treated as infeasibility. If the constructive certificate and all structural checks pass, an independent-check timeout produces a warning while the instance remains valid. By default the expensive independent LP cross-check is applied to same-skeleton reference instances. All instances receive the constructive certificate and support-feasibility checks.


## 8. Variable Cost Structure

The distance-based baseline variable cost for lane `a` and commodity `k` is

`c^0_ak = beta * distance_a`,

where `beta = 120 USD / (kton-mile)` in the default calibration.

A common lane normal draw `z_a` and commodity normal draw `z_k` are fixed at skeleton construction. For variability level `sigma`,

`c_raw_ak(sigma) = c^0_ak * exp(sigma*z_a + sigma_k*z_k)`.

The entire cost matrix is then normalized against the same reference routing so changing `sigma` changes heterogeneity without changing the aggregate reference calibration.

Configured lane variability levels are 0.00, 0.10 and 0.30.


## 9. Fixed-Charge Intensity

Each observed lane receives a common positive weight

`g_a = distance_a^p * exp(sigma_f * z^f_a)`.

The fixed-charge scale is chosen using the set of lanes active in the fixed reference witness. If `C_ref` is the common baseline reference variable cost, the scale satisfies

`sum_{a in A_ref} f_a / C_ref = rho`,

where `rho` is the requested intensity level.

Configured levels are 0.10, 1.00 and 5.00.

This definition avoids calibrating intensity against every potential lane, many of which may never be opened.


## 10. Mathematical Model

For lane-opening variables `y_a` and commodity flows `x_aks`, the scenario benchmark solves

`min sum_a f_a y_a + sum_s p_s sum_a sum_k c_ak x_aks`

subject to source supply, destination demand, and shared lane-capacity constraints

`sum_k x_aks <= U_a y_a`.

The binary opening decision is shared across scenarios. Flow decisions are scenario-dependent recourse variables.

The deterministic track is the same model with the base scenario only and probability 1.


## 11. Matched Isolation Validation

The family validator compares realized tables, not only catalogue labels. It verifies that:

- empirical skeleton identities are unchanged;
- network treatment changes lane membership only, while common-lane calibration values are retained;
- cost treatment changes variable costs only;
- capacity treatment changes capacities only;
- demand-concentration treatment changes supply/demand marginals only;
- fixed-charge treatment changes fixed charges only;
- uncertainty treatment changes scenario demand marginals only and is absent from the deterministic track.

This prevents implementation drift from silently invalidating the experimental interpretation.


## 12. Computational Difficulty and Multi-Horizon Reference Results

Difficulty is measured after generation. The standardized screening stage solves the LP relaxation for at most 10 seconds and a fresh MILP for at most 60 seconds. Any valid instance whose 10-second LP relaxation is not certified optimal then receives a fresh 30-second LP follow-up. The original 10-second record is preserved. A root gap is treated as certified only when it is computed from an LP relaxation that reached optimality at either the 10-second or 30-second horizon.

Protocol-conditional 60-second labels are descriptive:

- easy: certified optimal within 2 seconds;
- moderate: certified optimal after 2 seconds;
- hard: incumbent available with relative gap at most 0.10;
- extreme: remaining valid open or wider-gap cases.

Every valid instance unresolved at 60 seconds receives a **fresh 200-second MILP solve**. Only instances unresolved in that fresh 200-second run receive a **fresh 400-second MILP solve**. The longer horizons do not replace the 60-second screening evidence. They measure persistence of unresolvedness under materially larger computational budgets.

Solver status, incumbent availability, bound quality, gap, nodes, and persistence across horizons are retained as the computational evidence.


## 13. Effect Estimation

Treatment effects are estimated using within-skeleton, within-track paired comparisons. For response `Y`, the effect observation is

`Delta Y_rfl = Y(r,t,f,l) - Y(r,t,reference)`.

The characterisation output reports paired differences and ratios before aggregation by factor, level and track.


## 14. Resumability and Progress

The complete workflow is executed through `run-pipeline`. Expensive work is checkpointed at the smallest practical unit.