"""Matched experimental design for FAF-FCTPBench.

The design is block-matched by empirical skeleton.  A skeleton fixes the FAF
geography, commodities and primitive pseudo-random draws.  Every non-reference
benchmark row changes exactly one declared factor relative to the reference row
for the same skeleton and track.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REFERENCE_LEVEL = "reference"
TRACKS = ("deterministic", "scenario")
FACTOR_ORDER = (
    "network_support",
    "cost_variability",
    "capacity_tightness",
    "demand_concentration",
    "fixed_charge_intensity",
    "uncertainty_amplitude",
)
DETERMINISTIC_FACTORS = FACTOR_ORDER[:-1]
SCENARIO_FACTORS = FACTOR_ORDER


def default_design_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "faf_fctp_design.json"


def load_design(path: str | Path | None = None) -> dict[str, Any]:
    design_path = Path(path) if path is not None else default_design_path()
    payload = json.loads(design_path.read_text(encoding="utf-8"))
    validate_design_config(payload)
    return payload


def validate_design_config(config: dict[str, Any]) -> None:
    required = {"benchmark_name", "faf_source", "design", "factors", "scenario_probabilities", "cost_calibration", "storage", "solver_protocol", "validation"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"design configuration is missing {sorted(missing)}")
    factors = config["factors"]
    if set(factors) != set(FACTOR_ORDER):
        raise ValueError(f"factor set must be exactly {list(FACTOR_ORDER)}")
    for factor, levels in factors.items():
        if REFERENCE_LEVEL not in levels:
            raise ValueError(f"factor {factor} requires a reference level")
        if len(levels) < 3:
            raise ValueError(f"factor {factor} requires at least three levels")
    probabilities = config["scenario_probabilities"]
    if set(probabilities) != {"low", "base", "high"}:
        raise ValueError("scenario probabilities must contain low/base/high")
    if not np.isclose(sum(float(value) for value in probabilities.values()), 1.0):
        raise ValueError("scenario probabilities must sum to one")
    if config["storage"].get("format") != "parquet" or config["storage"].get("compression") != "zstd":
        raise ValueError("canonical storage must be Parquet with Zstandard compression")
    solver = config["solver_protocol"]
    required_solver = {"relaxation_seconds", "screening_milp_seconds", "reference_milp_seconds", "extended_milp_seconds", "mip_relative_gap"}
    if required_solver - set(solver):
        raise ValueError("solver_protocol is incomplete")
    horizons = [float(solver["screening_milp_seconds"]), float(solver["reference_milp_seconds"]), float(solver["extended_milp_seconds"])]
    if not (0 < horizons[0] < horizons[1] < horizons[2]):
        raise ValueError("MILP horizons must satisfy screening < reference < extended")
    validation = config["validation"]
    if validation.get("independent_shared_capacity_check") not in {"none", "reference_only", "all"}:
        raise ValueError("independent_shared_capacity_check must be none, reference_only, or all")


@dataclass(frozen=True)
class CandidateSpecification:
    """One benchmark variant inside a matched empirical skeleton."""

    instance_id: str
    skeleton_id: str
    track: str
    treatment_factor: str
    treatment_level: str
    scale_class: str
    product_regime: str
    replicate: int
    n_sources: int
    n_destinations: int
    n_products: int
    skeleton_seed: int
    network_support: str
    cost_variability: str
    capacity_tightness: str
    demand_concentration: str
    fixed_charge_intensity: str
    uncertainty_amplitude: str

    @classmethod
    def from_row(cls, row: pd.Series | dict[str, object]) -> "CandidateSpecification":
        raw = dict(row)
        integers = {"replicate", "n_sources", "n_destinations", "n_products", "skeleton_seed"}
        values: dict[str, object] = {}
        for name in cls.__dataclass_fields__:
            value = raw[name]
            values[name] = int(value) if name in integers else str(value)
        return cls(**values)


def _factor_variants(config: dict[str, Any], track: str) -> list[tuple[str, str, dict[str, str]]]:
    applicable = DETERMINISTIC_FACTORS if track == "deterministic" else SCENARIO_FACTORS
    reference = {factor: REFERENCE_LEVEL for factor in FACTOR_ORDER}
    rows: list[tuple[str, str, dict[str, str]]] = [("reference", REFERENCE_LEVEL, reference.copy())]
    for factor in applicable:
        for level in config["factors"][factor]:
            if level == REFERENCE_LEVEL:
                continue
            treatment = reference.copy()
            treatment[factor] = str(level)
            rows.append((factor, str(level), treatment))
    return rows


def generate_candidate_catalogue(
    config: dict[str, Any] | None = None,
    *,
    config_path: str | Path | None = None,
) -> pd.DataFrame:
    """Generate the complete matched benchmark catalogue.

    Each skeleton contributes 11 deterministic variants and 13 scenario variants.
    """

    if config is None:
        config = load_design(config_path)
    validate_design_config(config)
    design = config["design"]
    scale_dimensions = design["scale_dimensions"]
    product_counts = design["product_counts"]
    replicates = int(design["replicates_per_scale_product_cell"])
    seed_sequence = np.random.SeedSequence(int(design["base_seed"]))
    skeleton_count = len(scale_dimensions) * len(product_counts) * replicates
    child_seeds = seed_sequence.spawn(skeleton_count)
    rows: list[dict[str, object]] = []
    skeleton_position = 0
    instance_number = 1
    for scale_class, dimensions in scale_dimensions.items():
        n_sources, n_destinations = map(int, dimensions)
        for product_regime, n_products in product_counts.items():
            for replicate in range(1, replicates + 1):
                skeleton_position += 1
                skeleton_id = f"SK-{skeleton_position:03d}"
                skeleton_seed = int(child_seeds[skeleton_position - 1].generate_state(1)[0])
                for track in design.get("tracks", TRACKS):
                    if track not in TRACKS:
                        raise ValueError(f"unsupported track {track}")
                    for treatment_factor, treatment_level, levels in _factor_variants(config, track):
                        rows.append(
                            {
                                "instance_id": f"FAF-FCTP-{instance_number:04d}",
                                "skeleton_id": skeleton_id,
                                "track": track,
                                "treatment_factor": treatment_factor,
                                "treatment_level": treatment_level,
                                "scale_class": scale_class,
                                "product_regime": product_regime,
                                "replicate": replicate,
                                "n_sources": n_sources,
                                "n_destinations": n_destinations,
                                "n_products": int(n_products),
                                "skeleton_seed": skeleton_seed,
                                **levels,
                            }
                        )
                        instance_number += 1
    frame = pd.DataFrame(rows)
    validate_matched_catalogue(frame, config)
    return frame


def validate_matched_catalogue(frame: pd.DataFrame, config: dict[str, Any] | None = None) -> dict[str, object]:
    """Prove the matched-design contract at catalogue level."""

    if config is None:
        config = load_design()
    errors: list[str] = []
    if frame["instance_id"].duplicated().any():
        errors.append("instance identifiers are not unique")
    for (skeleton_id, track), group in frame.groupby(["skeleton_id", "track"], sort=False):
        references = group.loc[group["treatment_factor"] == "reference"]
        if len(references) != 1:
            errors.append(f"{skeleton_id}/{track} must contain exactly one reference row")
            continue
        reference = references.iloc[0]
        applicable = DETERMINISTIC_FACTORS if track == "deterministic" else SCENARIO_FACTORS
        expected = 1 + sum(len(config["factors"][factor]) - 1 for factor in applicable)
        if len(group) != expected:
            errors.append(f"{skeleton_id}/{track} has {len(group)} rows; expected {expected}")
        invariant_columns = [
            "scale_class", "product_regime", "replicate", "n_sources", "n_destinations",
            "n_products", "skeleton_seed",
        ]
        for column in invariant_columns:
            if group[column].nunique(dropna=False) != 1:
                errors.append(f"{skeleton_id}/{track} changes invariant {column}")
        for _, row in group.iterrows():
            if row["treatment_factor"] == "reference":
                if any(str(row[factor]) != REFERENCE_LEVEL for factor in FACTOR_ORDER):
                    errors.append(f"{skeleton_id}/{track} reference row is not all-reference")
                continue
            factor = str(row["treatment_factor"])
            if factor not in applicable:
                errors.append(f"{skeleton_id}/{track} contains inapplicable treatment {factor}")
                continue
            differences = [name for name in FACTOR_ORDER if str(row[name]) != str(reference[name])]
            if differences != [factor]:
                errors.append(
                    f"{row['instance_id']} changes {differences}; expected only {factor}"
                )
            if str(row[factor]) != str(row["treatment_level"]):
                errors.append(f"{row['instance_id']} treatment level is inconsistent")
        if track == "deterministic" and (group["uncertainty_amplitude"] != REFERENCE_LEVEL).any():
            errors.append(f"{skeleton_id}/deterministic varies uncertainty")
    if errors:
        raise ValueError("matched catalogue validation failed: " + "; ".join(errors[:10]))
    return {
        "valid": True,
        "n_instances": int(len(frame)),
        "n_skeletons": int(frame["skeleton_id"].nunique()),
        "deterministic_instances": int((frame["track"] == "deterministic").sum()),
        "scenario_instances": int((frame["track"] == "scenario").sum()),
        "matched_main_effects": True,
    }


def curate_benchmark_core(candidates: pd.DataFrame, skeleton_count: int = 12) -> pd.DataFrame:
    """Return a compact comparison set by retaining complete matched skeletons.
    
    """

    skeletons = sorted(candidates["skeleton_id"].unique())[: int(skeleton_count)]
    core = candidates.loc[candidates["skeleton_id"].isin(skeletons)].copy()
    return core.sort_values("instance_id").reset_index(drop=True)


def validate_design_balance(frame: pd.DataFrame) -> dict[str, object]:
    return validate_matched_catalogue(frame)
