"""Reference extensive-form solver for scenario-based multi-product FCTP."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from collections.abc import Collection
import time

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .instance import FCTPInstance


@dataclass
class SolveResult:
    status: str
    success: bool
    objective_usd: float | None
    best_bound_usd: float | None
    relative_gap: float | None
    node_count: int | None
    runtime_seconds: float
    message: str
    relaxation: bool
    opened_lane_ids: list[str] | None = None

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        if self.opened_lane_ids is None:
            result.pop("opened_lane_ids")
        return result


def solve_extensive_form(
    instance: FCTPInstance,
    *,
    time_limit_seconds: float = 200.0,
    mip_relative_gap: float = 1e-4,
    relaxation: bool = False,
    capture_design: bool = False,
    fixed_open_lane_ids: Collection[str] | None = None,
) -> SolveResult:
    """Solve the risk-neutral two-stage extensive form with shared lane openings."""

    lanes = instance.lanes.reset_index(drop=True)
    products = instance.products["product_id"].astype(str).tolist()
    scenarios = instance.scenarios["scenario_id"].astype(str).tolist()
    lane_count, product_count, scenario_count = len(lanes), len(products), len(scenarios)
    flow_count = scenario_count * product_count * lane_count
    variable_count = lane_count + flow_count

    def flow_index(scenario_position: int, product_position: int, lane_position: int) -> int:
        return lane_count + (scenario_position * product_count + product_position) * lane_count + lane_position

    objective = np.zeros(variable_count)
    objective[:lane_count] = lanes["fixed_cost_usd"].to_numpy(dtype=float)
    probabilities = instance.scenarios.set_index("scenario_id")["probability"].astype(float)
    lane_ids = lanes["lane_id"].astype(str).tolist()
    cost_lookup = instance.variable_costs.set_index(
        ["product_id", "lane_id"]
    )["cost_usd_per_kton"]
    expected_cost_index = pd.MultiIndex.from_product(
        [products, lane_ids],
        names=["product_id", "lane_id"],
    )
    ordered_costs = cost_lookup.reindex(expected_cost_index)
    if ordered_costs.isna().any():
        raise ValueError("variable_costs does not cover every product-lane key")
    scenario_probabilities = probabilities.reindex(scenarios).to_numpy(dtype=float)
    objective[lane_count:] = np.tile(
        ordered_costs.to_numpy(dtype=float), scenario_count
    ) * np.repeat(scenario_probabilities, product_count * lane_count)

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    row = 0
    supply_lookup = instance.supplies.set_index(["scenario_id", "product_id", "source_id"])["supply_ktons"]
    demand_lookup = instance.demands.set_index(["scenario_id", "product_id", "destination_id"])["demand_ktons"]
    source_ids = instance.sources["source_id"].astype(str).tolist()
    destination_ids = instance.destinations["destination_id"].astype(str).tolist()
    lane_source = lanes["source_id"].astype(str).to_numpy()
    lane_destination = lanes["destination_id"].astype(str).to_numpy()

    for scenario_position, scenario_id in enumerate(scenarios):
        for product_position, product_id in enumerate(products):
            for source_id in source_ids:
                positions = np.flatnonzero(lane_source == source_id)
                rows.extend([row] * len(positions))
                cols.extend([flow_index(scenario_position, product_position, int(position)) for position in positions])
                values.extend([1.0] * len(positions))
                lower.append(-np.inf)
                upper.append(float(supply_lookup.get((scenario_id, product_id, source_id), 0.0)))
                row += 1
            for destination_id in destination_ids:
                positions = np.flatnonzero(lane_destination == destination_id)
                rows.extend([row] * len(positions))
                cols.extend([flow_index(scenario_position, product_position, int(position)) for position in positions])
                values.extend([1.0] * len(positions))
                demand = float(demand_lookup.get((scenario_id, product_id, destination_id), 0.0))
                lower.append(demand)
                upper.append(demand)
                row += 1
        for lane_position in range(lane_count):
            rows.append(row)
            cols.append(lane_position)
            values.append(-float(lanes.loc[lane_position, "capacity_ktons"]))
            for product_position in range(product_count):
                rows.append(row)
                cols.append(flow_index(scenario_position, product_position, lane_position))
                values.append(1.0)
            lower.append(-np.inf)
            upper.append(0.0)
            row += 1

    matrix = coo_matrix((values, (rows, cols)), shape=(row, variable_count)).tocsr()
    integrality = np.zeros(variable_count, dtype=int)
    if not relaxation and fixed_open_lane_ids is None:
        integrality[:lane_count] = 1
    variable_upper = np.full(variable_count, np.inf); variable_upper[:lane_count] = 1.0
    variable_lower = np.zeros(variable_count)
    if fixed_open_lane_ids is not None:
        opened = set(fixed_open_lane_ids)
        unexpected = opened - set(lane_ids)
        if unexpected:
            raise ValueError(f"fixed design contains {len(unexpected)} unknown lane identifiers")
        fixed_values = np.fromiter((float(lane_id in opened) for lane_id in lane_ids), dtype=float)
        variable_lower[:lane_count] = fixed_values
        variable_upper[:lane_count] = fixed_values
    started = time.perf_counter()
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(variable_lower, variable_upper),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={"time_limit": float(time_limit_seconds), "mip_rel_gap": float(mip_relative_gap)},
    )
    runtime = time.perf_counter() - started
    status_names = {0: "optimal", 1: "limit_reached", 2: "infeasible", 3: "unbounded", 4: "solver_error"}
    objective_value = float(result.fun) if result.fun is not None and np.isfinite(result.fun) else None
    bound = getattr(result, "mip_dual_bound", None)
    gap = getattr(result, "mip_gap", None)
    nodes = getattr(result, "mip_node_count", None)
    if fixed_open_lane_ids is not None and int(result.status) == 0:
        bound = result.fun
        gap = 0.0
    opened_lane_ids = None
    if capture_design and result.x is not None and not relaxation:
        opened_lane_ids = [lane_ids[position] for position in np.flatnonzero(result.x[:lane_count] > 0.5)]
    return SolveResult(
        status=status_names.get(int(result.status), "unknown"),
        success=bool(result.success),
        objective_usd=objective_value,
        best_bound_usd=float(bound) if bound is not None and np.isfinite(bound) else None,
        relative_gap=float(gap) if gap is not None and np.isfinite(gap) else None,
        node_count=int(nodes) if nodes is not None else None,
        runtime_seconds=runtime,
        message=str(result.message),
        relaxation=relaxation,
        opened_lane_ids=opened_lane_ids,
    )
