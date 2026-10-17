"""Validation and independent feasibility checks for FAF-FCTPBench instances."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .feasibility import support_feasibility_certificate
from .instance import FCTPInstance, STORAGE_MANIFEST, TABLE_FILES, table_files_for_directory

REQUIRED_COLUMNS = {
    "sources": {"source_id", "source_name"},
    "destinations": {"destination_id", "destination_name"},
    "products": {"product_id", "product_name"},
    "scenarios": {"scenario_id", "probability"},
    "lanes": {
        "lane_id", "source_id", "destination_id", "distance_miles", "stress_load_ktons",
        "capacity_ktons", "fixed_cost_usd", "base_flow_ktons", "provenance",
    },
    "supplies": {"source_id", "product_id", "scenario_id", "supply_ktons"},
    "demands": {"destination_id", "product_id", "scenario_id", "demand_ktons"},
    "variable_costs": {"lane_id", "product_id", "cost_usd_per_kton"},
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gini(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x) & (x >= 0)]
    if not len(x) or float(x.sum()) <= 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    return float(2.0 * np.dot(np.arange(1, n + 1), x) / (n * x.sum()) - (n + 1) / n)


def structural_fingerprint(instance: FCTPInstance) -> dict[str, float | int]:
    base_demand = (
        instance.demands.loc[instance.demands["scenario_id"].astype(str) == "base"]
        .groupby("destination_id")["demand_ktons"].sum().to_numpy(dtype=float)
    )
    possible = max(len(instance.sources) * len(instance.destinations), 1)
    return {
        "sources": len(instance.sources),
        "destinations": len(instance.destinations),
        "products": len(instance.products),
        "scenarios": len(instance.scenarios),
        "lanes": len(instance.lanes),
        "lane_density": float(len(instance.lanes) / possible),
        "demand_gini_base": _gini(base_demand),
        "total_base_demand_ktons": float(base_demand.sum()),
        "total_lane_capacity_ktons": float(instance.lanes["capacity_ktons"].sum()),
    }


def constructive_capacity_certificate(instance: FCTPInstance, *, tolerance: float = 1e-7) -> dict[str, object]:
    """Verify the compact universal-backbone stress-envelope certificate."""
    if "stress_load_ktons" not in instance.lanes:
        return {
            "method": "universal_backbone_routing_stress_envelope",
            "certified": False,
            "errors": ["stress_load_ktons is missing"],
        }
    stress = pd.to_numeric(instance.lanes["stress_load_ktons"], errors="coerce").to_numpy(dtype=float)
    capacity = pd.to_numeric(instance.lanes["capacity_ktons"], errors="coerce").to_numpy(dtype=float)
    errors: list[str] = []
    if not np.isfinite(stress).all() or np.any(stress < -tolerance):
        errors.append("stress-load certificate must be finite and nonnegative")
    violating = np.flatnonzero(capacity + tolerance < stress)
    if len(violating):
        errors.append(f"{len(violating)} lanes have capacity below the universal stress envelope")
    declaration = instance.metadata.get("constructive_feasibility", {})
    if not isinstance(declaration, dict) or not declaration.get("certified", False):
        errors.append("constructive feasibility declaration is missing or false")
    positive = stress > tolerance
    min_ratio = float(np.min(capacity[positive] / stress[positive])) if np.any(positive) else None
    return {
        "method": "universal_backbone_routing_stress_envelope",
        "certified": not errors,
        "minimum_capacity_to_stress_ratio": min_ratio,
        "positive_stress_lanes": int(np.sum(positive)),
        "errors": errors,
    }


def shared_capacity_feasibility_certificate(
    instance: FCTPInstance,
    *,
    tolerance: float = 1e-7,
    time_limit_seconds: float = 60.0,
) -> dict[str, object]:
    """Cross-check all-open multicommodity feasibility by scenario. """
    lanes = instance.lanes.reset_index(drop=True)
    products = instance.products["product_id"].astype(str).tolist()
    sources = instance.sources["source_id"].astype(str).tolist()
    destinations = instance.destinations["destination_id"].astype(str).tolist()
    lane_source = lanes["source_id"].astype(str).to_numpy()
    lane_destination = lanes["destination_id"].astype(str).to_numpy()
    L, K = len(lanes), len(products)
    capacity = lanes["capacity_ktons"].to_numpy(dtype=float)
    scenario_results: list[dict[str, object]] = []

    for scenario in instance.scenarios["scenario_id"].astype(str):
        nvar = L * K
        eq_rows: list[int] = []
        eq_cols: list[int] = []
        eq_vals: list[float] = []
        rhs: list[float] = []
        row = 0
        supply_lookup = (
            instance.supplies.loc[instance.supplies["scenario_id"].astype(str) == scenario]
            .set_index(["product_id", "source_id"])["supply_ktons"]
        )
        demand_lookup = (
            instance.demands.loc[instance.demands["scenario_id"].astype(str) == scenario]
            .set_index(["product_id", "destination_id"])["demand_ktons"]
        )
        for k, product in enumerate(products):
            for source in sources:
                for lane_pos in np.flatnonzero(lane_source == source):
                    eq_rows.append(row)
                    eq_cols.append(k * L + int(lane_pos))
                    eq_vals.append(1.0)
                rhs.append(float(supply_lookup.get((product, source), 0.0)))
                row += 1
            for destination in destinations[:-1]:
                for lane_pos in np.flatnonzero(lane_destination == destination):
                    eq_rows.append(row)
                    eq_cols.append(k * L + int(lane_pos))
                    eq_vals.append(1.0)
                rhs.append(float(demand_lookup.get((product, destination), 0.0)))
                row += 1
        Aeq = coo_matrix((eq_vals, (eq_rows, eq_cols)), shape=(row, nvar)).tocsr()
        ub_rows: list[int] = []
        ub_cols: list[int] = []
        ub_vals: list[float] = []
        for lane_pos in range(L):
            for k in range(K):
                ub_rows.append(lane_pos)
                ub_cols.append(k * L + lane_pos)
                ub_vals.append(1.0)
        Aub = coo_matrix((ub_vals, (ub_rows, ub_cols)), shape=(L, nvar)).tocsr()
        result = linprog(
            np.zeros(nvar, dtype=float),
            A_ub=Aub,
            b_ub=capacity + tolerance,
            A_eq=Aeq,
            b_eq=np.asarray(rhs, dtype=float),
            bounds=(0.0, None),
            method="highs",
            options={"time_limit": float(time_limit_seconds)},
        )
        message = str(result.message)
        if result.success:
            status = "independently_certified"
        elif int(result.status) == 1 and "time limit" in message.lower():
            status = "certificate_timeout"
        elif int(result.status) == 2:
            status = "infeasible"
        else:
            status = "solver_error"
        scenario_results.append({
            "scenario_id": scenario,
            "status": status,
            "solver_status": int(result.status),
            "message": message,
        })

    statuses = {str(item["status"]) for item in scenario_results}
    if "infeasible" in statuses:
        overall = "infeasible"
    elif "solver_error" in statuses:
        overall = "solver_error"
    elif "certificate_timeout" in statuses:
        overall = "certificate_timeout"
    else:
        overall = "independently_certified"
    return {
        "method": "independent_multicommodity_all_open_linear_feasibility",
        "time_limit_seconds_per_scenario": float(time_limit_seconds),
        "scenarios_checked": int(len(instance.scenarios)),
        "status": overall,
        "independently_certified": overall == "independently_certified",
        "scenario_results": scenario_results,
    }


def validate_instance(
    instance: FCTPInstance,
    *,
    tolerance: float = 1e-7,
    check_shared_capacity: bool = True,
    capacity_check_time_limit_seconds: float = 60.0,
) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    for name, required in REQUIRED_COLUMNS.items():
        missing = required - set(getattr(instance, name).columns)
        if missing:
            errors.append(f"{name} missing columns {sorted(missing)}")
    if errors:
        return {
            "valid": False,
            "validation_status": "invalid_structure",
            "feasibility_status": "invalid_structure",
            "errors": errors,
            "warnings": warnings,
            "dimensions": instance.dimensions,
        }

    ids = {
        "source": set(instance.sources["source_id"].astype(str)),
        "destination": set(instance.destinations["destination_id"].astype(str)),
        "product": set(instance.products["product_id"].astype(str)),
        "scenario": set(instance.scenarios["scenario_id"].astype(str)),
        "lane": set(instance.lanes["lane_id"].astype(str)),
    }
    track = str(instance.metadata.get("track", ""))
    if track == "deterministic":
        if ids["scenario"] != {"base"}:
            errors.append("deterministic instances must contain exactly the base scenario")
        if len(instance.scenarios) == 1 and not np.isclose(float(instance.scenarios.iloc[0]["probability"]), 1.0):
            errors.append("deterministic base scenario probability must equal one")
    elif track == "scenario":
        if ids["scenario"] != {"low", "base", "high"}:
            errors.append("scenario instances must contain exactly low/base/high scenarios")
    else:
        errors.append("metadata track must be deterministic or scenario")

    probabilities = pd.to_numeric(instance.scenarios["probability"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0, atol=tolerance):
        errors.append("scenario probabilities must be finite, nonnegative, and sum to one")

    numeric_checks = [
        (instance.lanes, ["distance_miles", "stress_load_ktons", "capacity_ktons", "fixed_cost_usd", "base_flow_ktons"]),
        (instance.supplies, ["supply_ktons"]),
        (instance.demands, ["demand_ktons"]),
        (instance.variable_costs, ["cost_usd_per_kton"]),
    ]
    for table, columns in numeric_checks:
        for column in columns:
            values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(values).all() or np.any(values < -tolerance):
                errors.append(f"{column} must be finite and nonnegative")
    if np.any(instance.lanes["distance_miles"].to_numpy(dtype=float) <= 0):
        errors.append("distance_miles must be positive")
    if np.any(instance.lanes["capacity_ktons"].to_numpy(dtype=float) <= 0):
        errors.append("capacity_ktons must be positive")

    foreign_checks = [
        (set(instance.lanes["source_id"].astype(str)), ids["source"], "lane source"),
        (set(instance.lanes["destination_id"].astype(str)), ids["destination"], "lane destination"),
        (set(instance.supplies["source_id"].astype(str)), ids["source"], "supply source"),
        (set(instance.demands["destination_id"].astype(str)), ids["destination"], "demand destination"),
        (set(instance.variable_costs["lane_id"].astype(str)), ids["lane"], "variable-cost lane"),
        (set(instance.supplies["product_id"].astype(str)), ids["product"], "supply product"),
        (set(instance.demands["product_id"].astype(str)), ids["product"], "demand product"),
        (set(instance.variable_costs["product_id"].astype(str)), ids["product"], "variable-cost product"),
        (set(instance.supplies["scenario_id"].astype(str)), ids["scenario"], "supply scenario"),
        (set(instance.demands["scenario_id"].astype(str)), ids["scenario"], "demand scenario"),
    ]
    for observed, allowed, label in foreign_checks:
        if not observed <= allowed:
            errors.append(f"{label} contains unknown identifiers")

    expected_cost_rows = len(instance.lanes) * len(instance.products)
    if len(instance.variable_costs) != expected_cost_rows or instance.variable_costs.duplicated(["lane_id", "product_id"]).any():
        errors.append("variable_costs must contain exactly one row per lane-product pair")

    supply_totals = instance.supplies.groupby(["product_id", "scenario_id"])["supply_ktons"].sum()
    demand_totals = instance.demands.groupby(["product_id", "scenario_id"])["demand_ktons"].sum()
    for key in sorted(set(supply_totals.index) | set(demand_totals.index)):
        supply = float(supply_totals.get(key, np.nan))
        demand = float(demand_totals.get(key, np.nan))
        if not np.isfinite(supply) or not np.isfinite(demand) or abs(supply - demand) > tolerance * max(1.0, supply, demand):
            errors.append(f"unbalanced product-scenario {key}: supply={supply}, demand={demand}")

    if track == "scenario":
        for label, table, index in (
            ("supply", instance.supplies, ["source_id", "product_id"]),
            ("demand", instance.demands, ["destination_id", "product_id"]),
        ):
            value_column = "supply_ktons" if label == "supply" else "demand_ktons"
            pivot = table.pivot_table(index=index, columns="scenario_id", values=value_column, aggfunc="sum", fill_value=0.0)
            if {"low", "base", "high"} <= set(pivot.columns):
                if ((pivot["low"] > pivot["base"] + tolerance) | (pivot["base"] > pivot["high"] + tolerance)).any():
                    errors.append(f"{label} scenario ordering must satisfy low <= base <= high")

    support = support_feasibility_certificate(instance.lanes, instance.supplies, instance.demands, tolerance_ktons=tolerance)
    if not support["all_lanes_open_support_feasible"]:
        errors.append("FAF-supported lane topology cannot route every product-scenario marginal")

    constructive = constructive_capacity_certificate(instance, tolerance=tolerance)
    if not constructive["certified"]:
        errors.extend(str(error) for error in constructive.get("errors", []))

    capacity = None
    if check_shared_capacity and not errors:
        capacity = shared_capacity_feasibility_certificate(
            instance,
            tolerance=tolerance,
            time_limit_seconds=capacity_check_time_limit_seconds,
        )
        if capacity["status"] == "infeasible":
            errors.append("independent multicommodity LP certified all-open shared-capacity infeasibility")
        elif capacity["status"] == "certificate_timeout":
            warnings.append("independent shared-capacity LP timed out; constructive certificate remains valid")
        elif capacity["status"] == "solver_error":
            warnings.append("independent shared-capacity LP ended with a solver error; constructive certificate remains valid")

    provenance = instance.metadata.get("faf_provenance", {})
    if not isinstance(provenance, dict) or not provenance.get("source_sha256"):
        warnings.append("FAF source SHA-256 is not recorded in metadata")
    matched = instance.metadata.get("matched_design", {})
    if not isinstance(matched, dict) or not matched.get("common_random_numbers"):
        errors.append("matched-design common-random-number declaration is missing")

    if errors:
        feasibility_status = "infeasible" if any("infeasib" in error.lower() for error in errors) else "invalid_structure"
        validation_status = "invalid"
    elif capacity is None:
        feasibility_status = "constructively_certified"
        validation_status = "valid"
    elif capacity["status"] == "independently_certified":
        feasibility_status = "independently_certified"
        validation_status = "valid"
    elif capacity["status"] == "certificate_timeout":
        feasibility_status = "certificate_timeout"
        validation_status = "valid"
    else:
        feasibility_status = "constructively_certified"
        validation_status = "valid"

    return {
        "valid": not errors,
        "validation_status": validation_status,
        "feasibility_status": feasibility_status,
        "errors": errors,
        "warnings": warnings,
        "dimensions": instance.dimensions,
        "fingerprint": structural_fingerprint(instance),
        "support_feasibility": support,
        "constructive_capacity_feasibility": constructive,
        "shared_capacity_feasibility": capacity,
    }


def directory_checksums(directory: str | Path) -> dict[str, str]:
    root = Path(directory)
    filenames = [*table_files_for_directory(root).values(), "metadata.json"]
    if (root / STORAGE_MANIFEST).is_file():
        filenames.append(STORAGE_MANIFEST)
    return {filename: _sha256_file(root / filename) for filename in filenames}


def validate_storage_manifest(instance: FCTPInstance, directory: str | Path) -> dict[str, object]:
    root = Path(directory)
    errors: list[str] = []
    try:
        table_files_for_directory(root)
    except FileNotFoundError as error:
        return {"valid": False, "errors": [str(error)]}
    path = root / STORAGE_MANIFEST
    if not path.is_file():
        return {"valid": False, "errors": [f"missing {STORAGE_MANIFEST}"]}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != "Apache Parquet" or manifest.get("compression") != "Zstandard":
        errors.append("storage manifest must declare Apache Parquet with Zstandard compression")
    for attribute, filename in TABLE_FILES.items():
        info = manifest.get("tables", {}).get(attribute, {})
        if info.get("file") != filename or info.get("sha256") != _sha256_file(root / filename):
            errors.append(f"storage integrity failure for {filename}")
        if int(info.get("rows", -1)) != len(getattr(instance, attribute)):
            errors.append(f"row count mismatch for {filename}")
    metadata_path = root / "metadata.json"
    if manifest.get("metadata", {}).get("sha256") != _sha256_file(metadata_path):
        errors.append("metadata checksum mismatch")
    return {"valid": not errors, "errors": errors}


def validate_instance_directory(
    directory: str | Path,
    *,
    check_shared_capacity: bool = True,
    capacity_check_time_limit_seconds: float = 60.0,
) -> dict[str, object]:
    instance = FCTPInstance.read(directory)
    report = validate_instance(
        instance,
        check_shared_capacity=check_shared_capacity,
        capacity_check_time_limit_seconds=capacity_check_time_limit_seconds,
    )
    storage = validate_storage_manifest(instance, directory)
    report["storage"] = storage
    report["errors"].extend(storage["errors"])
    report["valid"] = not report["errors"]
    report["validation_status"] = "valid" if report["valid"] else "invalid"
    return report


def write_validation_report(
    instance: FCTPInstance,
    directory: str | Path,
    *,
    check_shared_capacity: bool = True,
    capacity_check_time_limit_seconds: float = 60.0,
) -> Path:
    output = Path(directory) / "validation.json"
    report = validate_instance(
        instance,
        check_shared_capacity=check_shared_capacity,
        capacity_check_time_limit_seconds=capacity_check_time_limit_seconds,
    )
    storage = validate_storage_manifest(instance, directory)
    report["storage"] = storage
    report["errors"].extend(storage["errors"])
    report["valid"] = not report["errors"]
    report["validation_status"] = "valid" if report["valid"] else "invalid"
    report["sha256"] = directory_checksums(directory)
    output.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return output
