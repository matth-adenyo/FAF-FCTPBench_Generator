"""Independent transportation-support feasibility certificates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_FEASIBILITY_TOLERANCE_KTONS = 1e-7


@dataclass
class _Edge:
    destination: int
    reverse: int
    residual: float


class _Dinic:
    """Small floating-point Dinic implementation for bipartite support checks."""

    def __init__(self, node_count: int) -> None:
        self.graph: list[list[_Edge]] = [[] for _ in range(node_count)]

    def add_edge(self, source: int, destination: int, capacity: float) -> None:
        forward = _Edge(destination, len(self.graph[destination]), float(capacity))
        reverse = _Edge(source, len(self.graph[source]), 0.0)
        self.graph[source].append(forward)
        self.graph[destination].append(reverse)

    def maximum_flow(self, source: int, sink: int, tolerance: float) -> float:
        total = 0.0
        node_count = len(self.graph)
        while True:
            level = [-1] * node_count
            level[source] = 0
            queue = [source]
            for node in queue:
                for edge in self.graph[node]:
                    if edge.residual > tolerance and level[edge.destination] < 0:
                        level[edge.destination] = level[node] + 1
                        queue.append(edge.destination)
            if level[sink] < 0:
                return total
            cursor = [0] * node_count

            def send(node: int, amount: float) -> float:
                if node == sink:
                    return amount
                while cursor[node] < len(self.graph[node]):
                    edge = self.graph[node][cursor[node]]
                    if edge.residual > tolerance and level[edge.destination] == level[node] + 1:
                        delivered = send(edge.destination, min(amount, edge.residual))
                        if delivered > tolerance:
                            edge.residual -= delivered
                            self.graph[edge.destination][edge.reverse].residual += delivered
                            return delivered
                    cursor[node] += 1
                return 0.0

            while True:
                delivered = send(source, float("inf"))
                if delivered <= tolerance:
                    break
                total += delivered


def transportation_support_deficit(
    lanes: pd.DataFrame,
    supply: pd.Series,
    demand: pd.Series,
    *,
    tolerance_ktons: float = DEFAULT_FEASIBILITY_TOLERANCE_KTONS,
) -> tuple[float, float]:
    """Return ``(routable, deficit)`` for one product-scenario pair."""

    supply = pd.to_numeric(supply, errors="raise").astype(float)
    demand = pd.to_numeric(demand, errors="raise").astype(float)
    supply = supply.loc[supply > tolerance_ktons]
    demand = demand.loc[demand > tolerance_ktons]
    total_supply = float(supply.sum())
    total_demand = float(demand.sum())
    if max(total_supply, total_demand) <= tolerance_ktons:
        return 0.0, 0.0
    marginal_deficit = abs(total_supply - total_demand)
    if marginal_deficit > tolerance_ktons:
        return min(total_supply, total_demand), marginal_deficit

    sources = sorted(supply.index.astype(str))
    destinations = sorted(demand.index.astype(str))
    source_position = {value: position for position, value in enumerate(sources)}
    destination_position = {
        value: position + len(sources) for position, value in enumerate(destinations)
    }
    super_source = len(sources) + len(destinations)
    super_sink = super_source + 1
    network = _Dinic(super_sink + 1)
    for identifier, quantity in supply.items():
        network.add_edge(super_source, source_position[str(identifier)], float(quantity))
    for identifier, quantity in demand.items():
        network.add_edge(destination_position[str(identifier)], super_sink, float(quantity))
    active = lanes.loc[
        lanes["source_id"].astype(str).isin(source_position)
        & lanes["destination_id"].astype(str).isin(destination_position),
        ["source_id", "destination_id"],
    ].drop_duplicates()
    for row in active.itertuples(index=False):
        network.add_edge(
            source_position[str(row.source_id)],
            destination_position[str(row.destination_id)],
            total_demand,
        )
    routable = network.maximum_flow(super_source, super_sink, tolerance_ktons * 0.1)
    deficit = max(0.0, total_demand - routable, marginal_deficit)
    if deficit <= tolerance_ktons:
        deficit = 0.0
    return float(routable), float(deficit)


def support_feasibility_certificate(
    lanes: pd.DataFrame,
    supplies: pd.DataFrame,
    demands: pd.DataFrame,
    *,
    tolerance_ktons: float = DEFAULT_FEASIBILITY_TOLERANCE_KTONS,
) -> dict[str, object]:
    """Certify every product-scenario marginal pair on the available lane support."""

    product_ids = sorted(
        set(supplies["product_id"].astype(str))
        | set(demands["product_id"].astype(str))
    )
    scenario_ids = sorted(
        set(supplies["scenario_id"].astype(str))
        | set(demands["scenario_id"].astype(str))
    )
    failures: list[dict[str, object]] = []
    maximum_deficit = 0.0
    checks = 0
    for scenario_id in scenario_ids:
        for product_id in product_ids:
            supply = (
                supplies.loc[
                    (supplies["scenario_id"].astype(str) == scenario_id)
                    & (supplies["product_id"].astype(str) == product_id)
                ]
                .groupby("source_id")["supply_ktons"]
                .sum()
            )
            demand = (
                demands.loc[
                    (demands["scenario_id"].astype(str) == scenario_id)
                    & (demands["product_id"].astype(str) == product_id)
                ]
                .groupby("destination_id")["demand_ktons"]
                .sum()
            )
            routable, deficit = transportation_support_deficit(
                lanes,
                supply,
                demand,
                tolerance_ktons=tolerance_ktons,
            )
            checks += 1
            maximum_deficit = max(maximum_deficit, deficit)
            if deficit > tolerance_ktons:
                failures.append(
                    {
                        "product_id": product_id,
                        "scenario_id": scenario_id,
                        "routable_ktons": routable,
                        "deficit_ktons": deficit,
                    }
                )
    return {
        "method": "independent_bipartite_maximum_flow",
        "tolerance_ktons": float(tolerance_ktons),
        "checks": checks,
        "failed_checks": len(failures),
        "maximum_deficit_ktons": float(maximum_deficit),
        "all_lanes_open_support_feasible": not failures,
        "failures": failures,
    }
