"""FAF-grounded empirical skeletons and independently controlled FCTP variants."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .catalogue import CandidateSpecification, load_design
from .instance import FCTPInstance

TOL = 1e-8


def _stable_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def _rng(seed: int, label: str) -> np.random.Generator:
    return np.random.default_rng(_stable_seed(seed, label))


def _rank_entities(activity: pd.Series, count: int, rng: np.random.Generator) -> list[str]:
    activity = activity.astype(float)
    positive = activity.loc[activity > 0]
    if len(positive) < count:
        raise ValueError(f"only {len(positive)} positive entities available; {count} requested")
    # Gumbel perturbation around log activity gives reproducible empirical sampling
    # without making every replicate a deterministic top-volume subset.
    score = np.log1p(positive.to_numpy()) + rng.gumbel(0.0, 0.35, len(positive))
    order = np.argsort(-score, kind="mergesort")[:count]
    return sorted(positive.index.astype(str).to_numpy()[order].tolist())


def _select_products(flows: pd.DataFrame, count: int, rng: np.random.Generator) -> list[str]:
    activity = flows.groupby("product_id")["tons_base_ktons"].sum()
    return _rank_entities(activity, count, rng)


def _scenario_row_quantities(selected: pd.DataFrame, amplitude: float) -> dict[str, np.ndarray]:
    base = selected["tons_base_ktons"].to_numpy(dtype=float)
    low_observed = np.minimum(selected["tons_low_ktons"].to_numpy(dtype=float), base)
    high_observed = np.maximum(selected["tons_high_ktons"].to_numpy(dtype=float), base)
    low = np.maximum(0.0, base + float(amplitude) * (low_observed - base))
    high = np.maximum(base, base + float(amplitude) * (high_observed - base))
    return {"low": low, "base": base.copy(), "high": high}


def _power_concentrate(values: np.ndarray, exponent: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    total = float(values.sum())
    if total <= TOL or np.isclose(exponent, 1.0):
        return values.copy()
    positive = values > 0
    transformed = np.zeros_like(values)
    transformed[positive] = np.power(values[positive], float(exponent))
    transformed_total = float(transformed.sum())
    if transformed_total <= TOL:
        return values.copy()
    return transformed * (total / transformed_total)


def _project_bounded_simplex(target: np.ndarray, total: float, upper: np.ndarray | None = None) -> np.ndarray:
    """Euclidean projection onto a nonnegative simplex with optional upper bounds."""
    target = np.asarray(target, dtype=float)
    total = float(total)
    if total <= TOL:
        return np.zeros_like(target)
    if upper is None:
        upper_array = np.full_like(target, np.inf, dtype=float)
    else:
        upper_array = np.asarray(upper, dtype=float)
        if float(upper_array.sum()) + 1e-9 < total:
            raise ValueError("bounded-simplex target exceeds available upper-bound mass")
    finite_upper = np.isfinite(upper_array)
    lo = float(np.min(target - np.where(finite_upper, upper_array, total + np.maximum(target, 0.0)))) - abs(total) - 1.0
    hi = float(np.max(target)) + abs(total) + 1.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        candidate = np.minimum(np.maximum(target - mid, 0.0), upper_array)
        if float(candidate.sum()) > total:
            lo = mid
        else:
            hi = mid
    result = np.minimum(np.maximum(target - hi, 0.0), upper_array)
    residual = total - float(result.sum())
    if abs(residual) > 1e-9 * max(1.0, total):
        # Deterministic residual reconciliation on entries with available room.
        if residual > 0:
            room = upper_array - result
            order = np.argsort(-np.where(np.isfinite(room), room, np.inf), kind="mergesort")
            for idx in order:
                add = min(residual, float(room[idx])) if np.isfinite(room[idx]) else residual
                if add > 0:
                    result[idx] += add
                    residual -= add
                if residual <= 1e-10 * max(1.0, total):
                    break
        else:
            order = np.argsort(-result, kind="mergesort")
            for idx in order:
                sub = min(-residual, float(result[idx]))
                if sub > 0:
                    result[idx] -= sub
                    residual += sub
                if residual >= -1e-10 * max(1.0, total):
                    break
    return result


def _ordered_concentrate_triplet(
    low: np.ndarray,
    base: np.ndarray,
    high: np.ndarray,
    exponent: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concentrate three scenario marginals while preserving low <= base <= high.

    The low target is projected onto the capped simplex ``0 <= low <= base`` while
    retaining its original total. The high increment is projected onto a
    nonnegative simplex above the transformed base while retaining its total.
    This preserves the FAF low/base/high ordering componentwise.
    """
    low = np.asarray(low, dtype=float)
    base = np.asarray(base, dtype=float)
    high = np.asarray(high, dtype=float)
    if np.isclose(exponent, 1.0):
        return low.copy(), base.copy(), high.copy()
    target_low = _power_concentrate(low, exponent)
    transformed_base = _power_concentrate(base, exponent)
    target_high = _power_concentrate(high, exponent)
    low_result = _project_bounded_simplex(target_low, float(low.sum()), upper=transformed_base)
    extra_total = max(0.0, float(high.sum()) - float(base.sum()))
    high_increment_target = target_high - transformed_base
    high_increment = _project_bounded_simplex(high_increment_target, extra_total, upper=None)
    high_result = transformed_base + high_increment
    return low_result, transformed_base, high_result


def _raw_margins_from_rows(
    selected: pd.DataFrame,
    row_quantities: np.ndarray,
    source_ids: list[str],
    destination_ids: list[str],
    product_ids: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    work = selected[["source_id", "destination_id", "product_id"]].copy()
    work["quantity"] = row_quantities
    supplies: dict[str, np.ndarray] = {}
    demands: dict[str, np.ndarray] = {}
    for product in product_ids:
        part = work.loc[work["product_id"] == product]
        supplies[product] = (
            part.groupby("source_id")["quantity"].sum().reindex(source_ids, fill_value=0.0).to_numpy(dtype=float)
        )
        demands[product] = (
            part.groupby("destination_id")["quantity"].sum().reindex(destination_ids, fill_value=0.0).to_numpy(dtype=float)
        )
    return supplies, demands


def _ordered_scenario_margins(
    selected: pd.DataFrame,
    row_quantities: dict[str, np.ndarray],
    source_ids: list[str],
    destination_ids: list[str],
    product_ids: list[str],
    concentration_exponent: float,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    """Return ordered low/base/high source and destination marginals by product."""
    raw_supply: dict[str, dict[str, np.ndarray]] = {}
    raw_demand: dict[str, dict[str, np.ndarray]] = {}
    for scenario_id in ("low", "base", "high"):
        supply, demand = _raw_margins_from_rows(
            selected, row_quantities[scenario_id], source_ids, destination_ids, product_ids
        )
        raw_supply[scenario_id] = supply
        raw_demand[scenario_id] = demand
    supplies = {scenario: {} for scenario in ("low", "base", "high")}
    demands = {scenario: {} for scenario in ("low", "base", "high")}
    for product in product_ids:
        sl, sb, sh = _ordered_concentrate_triplet(
            raw_supply["low"][product], raw_supply["base"][product], raw_supply["high"][product], concentration_exponent
        )
        dl, db, dh = _ordered_concentrate_triplet(
            raw_demand["low"][product], raw_demand["base"][product], raw_demand["high"][product], concentration_exponent
        )
        # Both sides preserve the original per-scenario product totals.
        for scenario_id, supply, demand in (("low", sl, dl), ("base", sb, db), ("high", sh, dh)):
            supply_total = float(supply.sum())
            demand_total = float(demand.sum())
            if demand_total > TOL and abs(supply_total - demand_total) > 1e-10 * max(1.0, supply_total):
                demand = demand * (supply_total / demand_total)
            supplies[scenario_id][product] = supply
            demands[scenario_id][product] = demand
    return supplies, demands

def _transportation_route(
    lanes: pd.DataFrame,
    source_ids: list[str],
    destination_ids: list[str],
    supply: np.ndarray,
    demand: np.ndarray,
) -> np.ndarray:
    """Return one deterministic minimum-distance feasible routing."""
    n_lanes = len(lanes)
    if float(demand.sum()) <= TOL:
        return np.zeros(n_lanes, dtype=float)
    source_pos = {s: i for i, s in enumerate(source_ids)}
    destination_pos = {d: i for i, d in enumerate(destination_ids)}
    lane_source = lanes["source_id"].map(source_pos).to_numpy(dtype=int)
    lane_destination = lanes["destination_id"].map(destination_pos).to_numpy(dtype=int)
    active = (supply[lane_source] > TOL) & (demand[lane_destination] > TOL)
    active_idx = np.flatnonzero(active)
    if not len(active_idx):
        raise ValueError("no active FAF-supported lanes for positive margins")
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs: list[float] = []
    row = 0
    # all source equalities
    for i, quantity in enumerate(supply):
        idx = active_idx[lane_source[active_idx] == i]
        for j in idx:
            rows.append(row); cols.append(int(np.searchsorted(active_idx, j))); vals.append(1.0)
        rhs.append(float(quantity)); row += 1
    # all but one destination equality to remove linear dependence
    for i, quantity in enumerate(demand[:-1]):
        idx = active_idx[lane_destination[active_idx] == i]
        for j in idx:
            rows.append(row); cols.append(int(np.searchsorted(active_idx, j))); vals.append(1.0)
        rhs.append(float(quantity)); row += 1
    matrix = coo_matrix((vals, (rows, cols)), shape=(row, len(active_idx))).tocsr()
    result = linprog(
        lanes.loc[active_idx, "distance_miles"].to_numpy(dtype=float),
        A_eq=matrix,
        b_eq=np.asarray(rhs, dtype=float),
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success or result.x is None:
        raise ValueError(f"selected FAF support is infeasible: {result.message}")
    routed = np.zeros(n_lanes, dtype=float)
    routed[active_idx] = result.x
    return routed


@dataclass
class EmpiricalSkeleton:
    skeleton_id: str
    seed: int
    selection_attempt: int
    source_ids: list[str]
    destination_ids: list[str]
    product_ids: list[str]
    selected_flows: pd.DataFrame
    observed_lanes: pd.DataFrame
    backbone_lane_ids: tuple[str, ...]
    optional_lane_ids_ranked: tuple[str, ...]
    stress_load_by_lane: dict[str, float]
    calibration_flow: pd.DataFrame
    lane_cost_z: dict[str, float]
    lane_fixed_z: dict[str, float]
    product_cost_z: dict[str, float]
    baseline_reference_variable_cost: float
    provenance: dict[str, object]


def _lane_table(selected: pd.DataFrame) -> pd.DataFrame:
    work = selected.copy()
    work["weighted_tmiles"] = work["ton_miles_base_millions"]
    grouped = work.groupby(["source_id", "destination_id"], as_index=False).agg(
        base_flow_ktons=("tons_base_ktons", "sum"),
        ton_miles_millions=("weighted_tmiles", "sum"),
    )
    distance = np.divide(
        grouped["ton_miles_millions"].to_numpy(dtype=float) * 1000.0,
        grouped["base_flow_ktons"].to_numpy(dtype=float),
        out=np.full(len(grouped), np.nan),
        where=grouped["base_flow_ktons"].to_numpy(dtype=float) > TOL,
    )
    fallback = float(np.nanmedian(distance[np.isfinite(distance) & (distance > 0)])) if np.any(np.isfinite(distance) & (distance > 0)) else 500.0
    grouped["distance_miles"] = np.where(np.isfinite(distance) & (distance > 0), distance, fallback)
    grouped["lane_id"] = "L:" + grouped["source_id"].astype(str) + ":" + grouped["destination_id"].astype(str)
    return grouped[["lane_id", "source_id", "destination_id", "distance_miles", "base_flow_ktons"]].sort_values("lane_id").reset_index(drop=True)


def build_empirical_skeleton(
    flows: pd.DataFrame,
    specification: CandidateSpecification,
    config: dict[str, Any] | None = None,
    *,
    maximum_selection_attempts: int = 20,
) -> EmpiricalSkeleton:
    """Freeze one FAF-derived skeleton and a universal constructive capacity envelope.

    The backbone is the union of deterministic transportation witnesses covering
    every configured uncertainty/concentration regime. A second routing pass on
    that backbone constructs the lane-wise simultaneous multi-product stress
    envelope used for all capacity treatments.
    """
    if config is None:
        config = load_design()
    for attempt in range(maximum_selection_attempts):
        seed = int(specification.skeleton_seed)
        products = _select_products(flows, int(specification.n_products), _rng(seed, f"products:{attempt}"))
        product_flows = flows.loc[flows["product_id"].isin(products)].copy()
        source_activity = product_flows.groupby("source_id")["tons_base_ktons"].sum()
        destination_activity = product_flows.groupby("destination_id")["tons_base_ktons"].sum()
        try:
            sources = _rank_entities(source_activity, int(specification.n_sources), _rng(seed, f"source:{attempt}"))
            destinations = _rank_entities(destination_activity, int(specification.n_destinations), _rng(seed, f"destination:{attempt}"))
        except ValueError:
            continue
        selected = product_flows.loc[
            product_flows["source_id"].isin(sources) & product_flows["destination_id"].isin(destinations)
        ].copy()
        if selected.empty:
            continue
        lanes = _lane_table(selected)
        if len(lanes) < max(len(sources), len(destinations)):
            continue

        uncertainty_levels = config["factors"]["uncertainty_amplitude"]
        concentration_levels = config["factors"]["demand_concentration"]
        backbone: set[str] = set()
        feasible = True

        # Pass 1: discover one common FAF-supported backbone covering every
        # configured uncertainty/concentration/scenario/product marginal.
        for amplitude in uncertainty_levels.values():
            row_quantities = _scenario_row_quantities(selected, float(amplitude))
            for exponent in concentration_levels.values():
                supplies_by_scenario, demands_by_scenario = _ordered_scenario_margins(
                    selected, row_quantities, sources, destinations, products, float(exponent)
                )
                for scenario_id in ("low", "base", "high"):
                    for product in products:
                        try:
                            route = _transportation_route(
                                lanes, sources, destinations,
                                supplies_by_scenario[scenario_id][product],
                                demands_by_scenario[scenario_id][product],
                            )
                        except ValueError:
                            feasible = False
                            break
                        used = np.flatnonzero(route > 1e-7)
                        backbone.update(lanes.loc[used, "lane_id"].astype(str).tolist())
                    if not feasible:
                        break
                if not feasible:
                    break
            if not feasible:
                break
        if not feasible or not backbone:
            continue

        backbone_lanes = lanes.loc[lanes["lane_id"].isin(backbone)].copy().sort_values("lane_id").reset_index(drop=True)
        stress_backbone = np.zeros(len(backbone_lanes), dtype=float)
        calibration_records: list[dict[str, object]] = []

        # Pass 2: route the complete envelope on the final backbone itself and
        # retain the maximum simultaneous multi-product load on each lane.
        try:
            for uncertainty_level, amplitude in uncertainty_levels.items():
                row_quantities = _scenario_row_quantities(selected, float(amplitude))
                for concentration_level, exponent in concentration_levels.items():
                    supplies_by_scenario, demands_by_scenario = _ordered_scenario_margins(
                        selected, row_quantities, sources, destinations, products, float(exponent)
                    )
                    for scenario_id in ("low", "base", "high"):
                        aggregate = np.zeros(len(backbone_lanes), dtype=float)
                        for product in products:
                            route = _transportation_route(
                                backbone_lanes, sources, destinations,
                                supplies_by_scenario[scenario_id][product],
                                demands_by_scenario[scenario_id][product],
                            )
                            aggregate += route
                            if (
                                uncertainty_level == "reference"
                                and concentration_level == "reference"
                                and scenario_id == "base"
                            ):
                                for pos in np.flatnonzero(route > 1e-7):
                                    calibration_records.append({
                                        "lane_id": str(backbone_lanes.loc[pos, "lane_id"]),
                                        "product_id": product,
                                        "flow_ktons": float(route[pos]),
                                    })
                        stress_backbone = np.maximum(stress_backbone, aggregate)
        except ValueError:
            continue

        optional = [lane for lane in lanes["lane_id"].astype(str) if lane not in backbone]
        optional_rng = _rng(seed, "optional-lane-ranking")
        optional_keys = optional_rng.random(len(optional))
        optional_ranked = [lane for _, lane in sorted(zip(optional_keys, optional), key=lambda pair: (pair[0], pair[1]))]
        lane_cost_z_values = _rng(seed, "lane-cost-noise").normal(size=len(lanes))
        lane_fixed_z_values = _rng(seed, "lane-fixed-noise").normal(size=len(lanes))
        product_z_values = _rng(seed, "product-cost-noise").normal(size=len(products))
        lane_cost_z = dict(zip(lanes["lane_id"].astype(str), lane_cost_z_values))
        lane_fixed_z = dict(zip(lanes["lane_id"].astype(str), lane_fixed_z_values))
        product_cost_z = dict(zip(products, product_z_values))
        stress_by_lane = {lane: 0.0 for lane in lanes["lane_id"].astype(str)}
        stress_by_lane.update(dict(zip(backbone_lanes["lane_id"].astype(str), stress_backbone)))
        calibration_flow = (
            pd.DataFrame(calibration_records)
            .groupby(["lane_id", "product_id"], as_index=False)["flow_ktons"].sum()
        )
        calibration = config["cost_calibration"]
        distance_lookup = lanes.set_index("lane_id")["distance_miles"]
        baseline_variable = 0.0
        for row in calibration_flow.itertuples(index=False):
            baseline_variable += (
                float(row.flow_ktons)
                * float(distance_lookup.loc[row.lane_id])
                * float(calibration["variable_cost_usd_per_kton_mile"])
            )
        provenance = {
            "source_filename": flows.attrs.get("source_filename"),
            "source_sha256": flows.attrs.get("source_sha256"),
            "forecast_year": flows.attrs.get("forecast_year"),
            "mode_code": flows.attrs.get("domestic_mode"),
            "trade_type_code": flows.attrs.get("trade_type"),
        }
        return EmpiricalSkeleton(
            skeleton_id=specification.skeleton_id,
            seed=seed,
            selection_attempt=attempt,
            source_ids=sources,
            destination_ids=destinations,
            product_ids=products,
            selected_flows=selected,
            observed_lanes=lanes,
            backbone_lane_ids=tuple(sorted(backbone)),
            optional_lane_ids_ranked=tuple(optional_ranked),
            stress_load_by_lane=stress_by_lane,
            calibration_flow=calibration_flow,
            lane_cost_z=lane_cost_z,
            lane_fixed_z=lane_fixed_z,
            product_cost_z=product_cost_z,
            baseline_reference_variable_cost=float(baseline_variable),
            provenance=provenance,
        )
    raise ValueError(
        f"could not construct a universally feasible empirical skeleton for {specification.skeleton_id} "
        f"after {maximum_selection_attempts} deterministic attempts"
    )

def _selected_lanes(skeleton: EmpiricalSkeleton, fraction: float) -> pd.DataFrame:
    optional_count = int(round(float(fraction) * len(skeleton.optional_lane_ids_ranked)))
    selected_ids = set(skeleton.backbone_lane_ids) | set(skeleton.optional_lane_ids_ranked[:optional_count])
    return skeleton.observed_lanes.loc[skeleton.observed_lanes["lane_id"].isin(selected_ids)].copy().sort_values("lane_id").reset_index(drop=True)


def _scenario_tables(
    skeleton: EmpiricalSkeleton,
    specification: CandidateSpecification,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if specification.track == "deterministic":
        scenarios = {"base": 1.0}
        amplitude = float(config["factors"]["uncertainty_amplitude"]["reference"])
    else:
        scenarios = {k: float(v) for k, v in config["scenario_probabilities"].items()}
        amplitude = float(config["factors"]["uncertainty_amplitude"][specification.uncertainty_amplitude])
    exponent = float(config["factors"]["demand_concentration"][specification.demand_concentration])
    row_quantities = _scenario_row_quantities(skeleton.selected_flows, amplitude)
    supplies_by_scenario, demands_by_scenario = _ordered_scenario_margins(
        skeleton.selected_flows,
        row_quantities,
        skeleton.source_ids,
        skeleton.destination_ids,
        skeleton.product_ids,
        exponent,
    )
    supply_rows: list[dict[str, object]] = []
    demand_rows: list[dict[str, object]] = []
    for scenario_id in scenarios:
        supplies = supplies_by_scenario[scenario_id]
        demands = demands_by_scenario[scenario_id]
        for product in skeleton.product_ids:
            supply_rows.extend(
                {"source_id": source, "product_id": product, "scenario_id": scenario_id, "supply_ktons": float(value)}
                for source, value in zip(skeleton.source_ids, supplies[product])
            )
            demand_rows.extend(
                {"destination_id": destination, "product_id": product, "scenario_id": scenario_id, "demand_ktons": float(value)}
                for destination, value in zip(skeleton.destination_ids, demands[product])
            )
    scenario_table = pd.DataFrame([{"scenario_id": key, "probability": value} for key, value in scenarios.items()])
    return scenario_table, pd.DataFrame(supply_rows), pd.DataFrame(demand_rows)

def _variable_cost_table(skeleton: EmpiricalSkeleton, lanes: pd.DataFrame, sigma: float, config: dict[str, Any]) -> pd.DataFrame:
    calibration = config["cost_calibration"]
    base_rate = float(calibration["variable_cost_usd_per_kton_mile"])
    product_sigma = float(calibration["commodity_variability_sigma"])
    rows: list[dict[str, object]] = []
    raw_lookup: dict[tuple[str, str], float] = {}
    distance = lanes.set_index("lane_id")["distance_miles"].astype(float)
    for product in skeleton.product_ids:
        for lane in lanes["lane_id"].astype(str):
            multiplier = math.exp(float(sigma) * skeleton.lane_cost_z[lane] + product_sigma * skeleton.product_cost_z[product])
            raw = base_rate * float(distance.loc[lane]) * multiplier
            raw_lookup[(lane, product)] = raw
    # Normalise with the fixed reference witness so cost variability does not change
    # the aggregate reference calibration.
    raw_reference = 0.0
    baseline_reference = 0.0
    full_distance = skeleton.observed_lanes.set_index("lane_id")["distance_miles"].astype(float)
    for row in skeleton.calibration_flow.itertuples(index=False):
        raw_reference += float(row.flow_ktons) * raw_lookup.get((row.lane_id, row.product_id), base_rate * float(full_distance.loc[row.lane_id]))
        baseline_reference += float(row.flow_ktons) * base_rate * float(full_distance.loc[row.lane_id])
    scale = baseline_reference / raw_reference if raw_reference > TOL else 1.0
    for (lane, product), raw in raw_lookup.items():
        rows.append({"lane_id": lane, "product_id": product, "cost_usd_per_kton": float(raw * scale)})
    return pd.DataFrame(rows).sort_values(["product_id", "lane_id"]).reset_index(drop=True)


def _fixed_costs(skeleton: EmpiricalSkeleton, lanes: pd.DataFrame, target: float, config: dict[str, Any]) -> np.ndarray:
    calibration = config["cost_calibration"]
    exponent = float(calibration["fixed_charge_distance_exponent"])
    sigma = float(calibration["fixed_charge_random_sigma"])
    weights = np.array([
        max(float(row.distance_miles), 1.0) ** exponent * math.exp(sigma * skeleton.lane_fixed_z[str(row.lane_id)])
        for row in lanes.itertuples(index=False)
    ])
    calibration_open = set(skeleton.calibration_flow.loc[skeleton.calibration_flow["flow_ktons"] > TOL, "lane_id"].astype(str))
    mask = lanes["lane_id"].astype(str).isin(calibration_open).to_numpy()
    denominator = float(weights[mask].sum())
    if denominator <= TOL:
        denominator = float(weights.sum())
    scale = float(target) * float(skeleton.baseline_reference_variable_cost) / max(denominator, TOL)
    return weights * scale


def _capacities(skeleton: EmpiricalSkeleton, lanes: pd.DataFrame, headroom: float) -> np.ndarray:
    peak_total = 0.0
    for product in skeleton.product_ids:
        peak_total += float(
            skeleton.selected_flows.loc[skeleton.selected_flows["product_id"] == product, "tons_high_ktons"].sum()
        )
    floor = max(peak_total * 1e-6, 1e-6)
    values = []
    for row in lanes.itertuples(index=False):
        stress = float(skeleton.stress_load_by_lane.get(str(row.lane_id), 0.0))
        empirical = float(row.base_flow_ktons)
        proxy = max(stress, empirical * 0.05, floor)
        values.append(float(headroom) * proxy)
    return np.asarray(values, dtype=float)


def _gini(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x) & (x >= 0)]
    if not len(x) or float(x.sum()) <= 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    return float((2 * np.sum((np.arange(1, n + 1)) * x) / (n * x.sum())) - (n + 1) / n)


def build_fctp_instance(
    flows: pd.DataFrame,
    specification: CandidateSpecification,
    config: dict[str, Any] | None = None,
    *,
    skeleton: EmpiricalSkeleton | None = None,
) -> FCTPInstance:
    """Materialise one deterministic or scenario FCTP from a matched skeleton."""
    if config is None:
        config = load_design()
    if skeleton is None:
        skeleton = build_empirical_skeleton(flows, specification, config)
    if skeleton.skeleton_id != specification.skeleton_id:
        raise ValueError("skeleton identifier does not match specification")
    support_fraction = float(config["factors"]["network_support"][specification.network_support])
    lanes = _selected_lanes(skeleton, support_fraction)
    scenario_table, supplies, demands = _scenario_tables(skeleton, specification, config)
    cost_sigma = float(config["factors"]["cost_variability"][specification.cost_variability])
    variable_costs = _variable_cost_table(skeleton, lanes, cost_sigma, config)
    fixed_target = float(config["factors"]["fixed_charge_intensity"][specification.fixed_charge_intensity])
    headroom = float(config["factors"]["capacity_tightness"][specification.capacity_tightness])
    lanes = lanes.copy()
    lanes["stress_load_ktons"] = lanes["lane_id"].astype(str).map(skeleton.stress_load_by_lane).fillna(0.0).astype(float)
    lanes["capacity_ktons"] = _capacities(skeleton, lanes, headroom)
    lanes["fixed_cost_usd"] = _fixed_costs(skeleton, lanes, fixed_target, config)

    zone_names = flows.attrs.get("zone_names", {}) or {}
    product_names = flows.attrs.get("product_names", {}) or {}
    sources = pd.DataFrame({
        "source_id": skeleton.source_ids,
        "source_name": [str(zone_names.get(source, source)) for source in skeleton.source_ids],
    })
    destinations = pd.DataFrame({
        "destination_id": skeleton.destination_ids,
        "destination_name": [str(zone_names.get(destination, destination)) for destination in skeleton.destination_ids],
    })
    products = pd.DataFrame({
        "product_id": skeleton.product_ids,
        "product_name": [str(product_names.get(product, product)) for product in skeleton.product_ids],
    })
    lanes["provenance"] = np.where(
        lanes["lane_id"].isin(skeleton.backbone_lane_ids),
        "FAF observed support; universal feasibility backbone",
        "FAF observed support; optional nested lane",
    )
    lanes = lanes[["lane_id", "source_id", "destination_id", "distance_miles", "stress_load_ktons", "capacity_ktons", "fixed_cost_usd", "base_flow_ktons", "provenance"]]

    base_demand = demands.loc[demands["scenario_id"] == "base"].groupby("destination_id")["demand_ktons"].sum().to_numpy(dtype=float)
    optional_available = len(skeleton.optional_lane_ids_ranked)
    optional_selected = max(0, len(lanes) - len(skeleton.backbone_lane_ids))
    calibration_open = set(skeleton.calibration_flow.loc[skeleton.calibration_flow["flow_ktons"] > TOL, "lane_id"].astype(str))
    realized_fixed = float(lanes.loc[lanes["lane_id"].astype(str).isin(calibration_open), "fixed_cost_usd"].sum()) / max(float(skeleton.baseline_reference_variable_cost), TOL)
    base_rate = float(config["cost_calibration"]["variable_cost_usd_per_kton_mile"])
    lane_distance = lanes.set_index("lane_id")["distance_miles"].astype(float)
    residuals = []
    for row in variable_costs.itertuples(index=False):
        residuals.append(float(row.cost_usd_per_kton) / max(base_rate * float(lane_distance.loc[row.lane_id]), TOL))
    residuals = np.asarray(residuals, dtype=float)
    residual_cv = float(residuals.std(ddof=0) / residuals.mean()) if len(residuals) and residuals.mean() > 0 else 0.0
    stress_headrooms = []
    for row in lanes.itertuples(index=False):
        stress = float(skeleton.stress_load_by_lane.get(str(row.lane_id), 0.0))
        if stress > TOL:
            stress_headrooms.append(float(row.capacity_ktons) / stress)
    minimum_stress_headroom = float(min(stress_headrooms)) if stress_headrooms else float("nan")
    scenario_totals = demands.groupby("scenario_id")["demand_ktons"].sum()
    base_total = float(scenario_totals.get("base", 0.0))
    uncertainty_span_ratio = (float(scenario_totals.get("high", base_total)) - float(scenario_totals.get("low", base_total))) / max(base_total, TOL)
    realized = {
        "lane_count": int(len(lanes)),
        "lane_density": float(len(lanes) / max(len(skeleton.source_ids) * len(skeleton.destination_ids), 1)),
        "backbone_lane_count": int(len(skeleton.backbone_lane_ids)),
        "optional_lanes_available": int(optional_available),
        "optional_lanes_selected": int(optional_selected),
        "optional_support_fraction": float(optional_selected / optional_available) if optional_available else 0.0,
        "capacity_headroom_multiplier": headroom,
        "minimum_stress_headroom": minimum_stress_headroom,
        "maximum_stress_utilization": float(1.0 / minimum_stress_headroom) if np.isfinite(minimum_stress_headroom) and minimum_stress_headroom > 0 else None,
        "demand_gini_base": _gini(base_demand),
        "cost_variability_sigma": cost_sigma,
        "cost_residual_coefficient_of_variation": residual_cv,
        "fixed_charge_intensity_target": fixed_target,
        "fixed_charge_intensity_realized": realized_fixed,
        "uncertainty_amplitude": float(config["factors"]["uncertainty_amplitude"][specification.uncertainty_amplitude]),
        "uncertainty_span_ratio": float(uncertainty_span_ratio),
    }
    metadata = {
        "benchmark_name": config["benchmark_name"],
        "instance_id": specification.instance_id,
        "skeleton_id": specification.skeleton_id,
        "track": specification.track,
        "treatment": {
            "factor": specification.treatment_factor,
            "level": specification.treatment_level,
        },
        "specification": {name: getattr(specification, name) for name in specification.__dataclass_fields__},
        "empirical_skeleton": {
            "selection_attempt": skeleton.selection_attempt,
            "seed": skeleton.seed,
            "products": skeleton.product_ids,
            "sources": skeleton.source_ids,
            "destinations": skeleton.destination_ids,
        },
        "faf_provenance": skeleton.provenance,
        "calibration": {
            "variable_cost_usd_per_kton_mile": float(config["cost_calibration"]["variable_cost_usd_per_kton_mile"]),
            "baseline_reference_variable_cost_usd": float(skeleton.baseline_reference_variable_cost),
            "fixed_charge_intensity_definition": "sum of fixed charges on reference-witness active lanes divided by the common baseline reference variable cost",
            "capacity_tightness_definition": "uniform headroom multiplier applied to a universal backbone-routing simultaneous multi-product stress envelope",
        },
        "constructive_feasibility": {
            "method": "universal_backbone_routing_stress_envelope",
            "certified": bool(np.all(lanes["capacity_ktons"].to_numpy(dtype=float) + TOL >= lanes["stress_load_ktons"].to_numpy(dtype=float))),
            "backbone_lane_count": int(len(skeleton.backbone_lane_ids)),
            "stress_envelope_uses_simultaneous_products": True,
            "covers_all_configured_uncertainty_and_concentration_regimes": True,
            "minimum_capacity_to_stress_ratio": minimum_stress_headroom if np.isfinite(minimum_stress_headroom) else None,
        },
        "realized": realized,
        "matched_design": {
            "common_random_numbers": True,
            "same_skeleton_comparator": True,
            "non_reference_variant_changes_one_declared_factor": specification.treatment_factor != "reference",
        },
    }
    return FCTPInstance(
        sources=sources,
        destinations=destinations,
        products=products,
        scenarios=scenario_table,
        lanes=lanes,
        supplies=supplies,
        demands=demands,
        variable_costs=variable_costs,
        metadata=metadata,
    )


def validate_matched_family(instances: list[FCTPInstance], tolerance: float = 1e-9) -> dict[str, object]:
    """Validate realized one-factor-at-a-time isolation for one skeleton/track family."""
    if not instances:
        raise ValueError("family is empty")
    references = [instance for instance in instances if instance.metadata["treatment"]["factor"] == "reference"]
    if len(references) != 1:
        raise ValueError("family must contain exactly one reference instance")
    reference = references[0]
    errors: list[str] = []

    def table_equal(a: pd.DataFrame, b: pd.DataFrame, keys: list[str]) -> bool:
        aa = a.sort_values(keys).reset_index(drop=True)
        bb = b.sort_values(keys).reset_index(drop=True)
        if list(aa.columns) != list(bb.columns) or aa.shape != bb.shape:
            return False
        for column in aa.columns:
            if pd.api.types.is_numeric_dtype(aa[column]) and pd.api.types.is_numeric_dtype(bb[column]):
                if not np.allclose(aa[column].to_numpy(dtype=float), bb[column].to_numpy(dtype=float), rtol=tolerance, atol=tolerance):
                    return False
            elif not aa[column].astype(str).equals(bb[column].astype(str)):
                return False
        return True

    for instance in instances:
        factor = str(instance.metadata["treatment"]["factor"])
        if factor == "reference":
            continue
        for name, keys in (("sources", ["source_id"]), ("destinations", ["destination_id"]), ("products", ["product_id"])):
            if not table_equal(getattr(reference, name), getattr(instance, name), keys):
                errors.append(f"{instance.metadata['instance_id']}: {name} changed under {factor}")
        if factor not in {"network_support"}:
            left = reference.lanes[["lane_id", "source_id", "destination_id", "distance_miles", "base_flow_ktons"]]
            right = instance.lanes[["lane_id", "source_id", "destination_id", "distance_miles", "base_flow_ktons"]]
            if not table_equal(left, right, ["lane_id"]):
                errors.append(f"{instance.metadata['instance_id']}: topology changed under {factor}")
        if not table_equal(reference.scenarios, instance.scenarios, ["scenario_id"]):
            errors.append(f"{instance.metadata['instance_id']}: scenario probabilities changed under {factor}")
        if factor not in {"demand_concentration", "uncertainty_amplitude"}:
            if not table_equal(reference.supplies, instance.supplies, ["scenario_id", "product_id", "source_id"]):
                errors.append(f"{instance.metadata['instance_id']}: supplies changed under {factor}")
            if not table_equal(reference.demands, instance.demands, ["scenario_id", "product_id", "destination_id"]):
                errors.append(f"{instance.metadata['instance_id']}: demands changed under {factor}")
        common_stress = sorted(set(reference.lanes["lane_id"].astype(str)) & set(instance.lanes["lane_id"].astype(str)))
        stress_a = reference.lanes.loc[reference.lanes["lane_id"].astype(str).isin(common_stress), ["lane_id", "stress_load_ktons"]]
        stress_b = instance.lanes.loc[instance.lanes["lane_id"].astype(str).isin(common_stress), ["lane_id", "stress_load_ktons"]]
        if not table_equal(stress_a, stress_b, ["lane_id"]):
            errors.append(f"{instance.metadata['instance_id']}: constructive stress envelope changed under {factor}")
        if factor != "cost_variability":
            common = sorted(set(reference.variable_costs["lane_id"].astype(str)) & set(instance.variable_costs["lane_id"].astype(str)))
            a = reference.variable_costs.loc[reference.variable_costs["lane_id"].astype(str).isin(common)]
            b = instance.variable_costs.loc[instance.variable_costs["lane_id"].astype(str).isin(common)]
            if not table_equal(a, b, ["product_id", "lane_id"]):
                errors.append(f"{instance.metadata['instance_id']}: variable costs changed under {factor}")
        if factor != "capacity_tightness":
            common = sorted(set(reference.lanes["lane_id"].astype(str)) & set(instance.lanes["lane_id"].astype(str)))
            a = reference.lanes.loc[reference.lanes["lane_id"].astype(str).isin(common), ["lane_id", "capacity_ktons"]]
            b = instance.lanes.loc[instance.lanes["lane_id"].astype(str).isin(common), ["lane_id", "capacity_ktons"]]
            if not table_equal(a, b, ["lane_id"]):
                errors.append(f"{instance.metadata['instance_id']}: capacities changed under {factor}")
        if factor != "fixed_charge_intensity":
            common = sorted(set(reference.lanes["lane_id"].astype(str)) & set(instance.lanes["lane_id"].astype(str)))
            a = reference.lanes.loc[reference.lanes["lane_id"].astype(str).isin(common), ["lane_id", "fixed_cost_usd"]]
            b = instance.lanes.loc[instance.lanes["lane_id"].astype(str).isin(common), ["lane_id", "fixed_cost_usd"]]
            if not table_equal(a, b, ["lane_id"]):
                errors.append(f"{instance.metadata['instance_id']}: fixed costs changed under {factor}")
    if errors:
        raise ValueError("matched-family isolation failed: " + "; ".join(errors[:10]))
    return {"valid": True, "instances": len(instances), "skeleton_id": reference.metadata["skeleton_id"], "track": reference.metadata["track"]}
