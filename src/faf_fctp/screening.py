"""Checkpointed reference screening and multi-horizon reference solving."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .instance import FCTPInstance
from .optimization import solve_extensive_form
from .runstate import ProgressReporter, atomic_write_json, atomic_write_parquet, utc_now

DIFFICULTY_CLASSES = ("easy", "moderate", "hard", "extreme")


def classify_difficulty(
    *, solution_status: str, runtime_seconds: float | None, relative_gap: float | None,
    objective_usd: float | None, easy_threshold_seconds: float = 2.0,
    hard_gap_threshold: float = 0.10,
) -> str:
    if solution_status == "optimal":
        return "easy" if runtime_seconds is not None and runtime_seconds <= easy_threshold_seconds else "moderate"
    if objective_usd is not None and relative_gap is not None and relative_gap <= hard_gap_threshold:
        return "hard"
    return "extreme"


def _stored_validation(directory: Path) -> dict[str, object]:
    path = directory / "validation.json"
    if not path.is_file():
        return {"valid": False, "errors": ["validation.json is missing"]}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        return {"valid": False, "errors": [f"could not read validation.json: {error}"]}


def screen_instance(
    instance_directory: str | Path,
    *, relaxation_time_limit_seconds: float = 10.0,
    milp_time_limit_seconds: float = 60.0,
    mip_relative_gap: float = 1e-4,
) -> dict[str, object]:
    directory = Path(instance_directory)
    validation = _stored_validation(directory)
    if not bool(validation.get("valid", False)):
        return {
            "instance_id": directory.name,
            "valid": False,
            "difficulty_class": None,
            "validation_status": validation.get("validation_status", "invalid"),
            "feasibility_status": validation.get("feasibility_status"),
            "validation_errors": " | ".join(map(str, validation.get("errors", []))),
        }
    instance = FCTPInstance.read(directory)
    relaxation = solve_extensive_form(
        instance, time_limit_seconds=relaxation_time_limit_seconds,
        mip_relative_gap=0.0, relaxation=True,
    )
    milp = solve_extensive_form(
        instance, time_limit_seconds=milp_time_limit_seconds,
        mip_relative_gap=mip_relative_gap, relaxation=False,
    )
    root_gap = None
    if milp.objective_usd is not None and relaxation.objective_usd is not None and abs(milp.objective_usd) > 1e-12:
        root_gap = max(0.0, (milp.objective_usd - relaxation.objective_usd) / abs(milp.objective_usd))
    difficulty = classify_difficulty(
        solution_status=milp.status, runtime_seconds=milp.runtime_seconds,
        relative_gap=milp.relative_gap, objective_usd=milp.objective_usd,
    )
    realized = instance.metadata.get("realized", {})
    treatment = instance.metadata.get("treatment", {})
    return {
        "instance_id": str(instance.metadata.get("instance_id", directory.name)),
        "skeleton_id": str(instance.metadata.get("skeleton_id", "")),
        "track": str(instance.metadata.get("track", "")),
        "treatment_factor": str(treatment.get("factor", "")),
        "treatment_level": str(treatment.get("level", "")),
        "valid": True,
        "validation_status": validation.get("validation_status", "valid"),
        "feasibility_status": validation.get("feasibility_status"),
        "solution_status": milp.status,
        "difficulty_class": difficulty,
        "milp_runtime_seconds": float(milp.runtime_seconds),
        "objective_usd": milp.objective_usd,
        "best_bound_usd": milp.best_bound_usd,
        "relative_gap": milp.relative_gap,
        "node_count": milp.node_count,
        "relaxation_status": relaxation.status,
        "relaxation_runtime_seconds": float(relaxation.runtime_seconds),
        "relaxation_objective_usd": relaxation.objective_usd,
        "root_gap": root_gap,
        "lane_count": int(len(instance.lanes)),
        "lane_density": float(realized.get("lane_density", np.nan)),
        "demand_gini_base": float(realized.get("demand_gini_base", np.nan)),
        "capacity_headroom_multiplier": float(realized.get("capacity_headroom_multiplier", np.nan)),
        "maximum_stress_utilization": float(realized.get("maximum_stress_utilization", np.nan)) if realized.get("maximum_stress_utilization") is not None else np.nan,
        "cost_variability_sigma": float(realized.get("cost_variability_sigma", np.nan)),
        "fixed_charge_intensity_target": float(realized.get("fixed_charge_intensity_target", np.nan)),
        "uncertainty_amplitude": float(realized.get("uncertainty_amplitude", np.nan)),
        "screened_at_utc": utc_now(),
    }


def _load_checkpoint(path: Path, signature: str) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("protocol_signature") != signature:
        return None
    result = payload.get("result")
    return result if isinstance(result, dict) else None


def screen_library(
    catalogue: pd.DataFrame,
    instance_root: str | Path,
    output_path: str | Path,
    *, relaxation_time_limit_seconds: float = 10.0,
    milp_time_limit_seconds: float = 60.0,
    mip_relative_gap: float = 1e-4,
    checkpoint_directory: str | Path | None = None,
    protocol_signature: str = "",
    progress: ProgressReporter | None = None,
) -> pd.DataFrame:
    root = Path(instance_root)
    output = Path(output_path)
    checkpoint_root = Path(checkpoint_directory) if checkpoint_directory else output.parent / "checkpoints" / "screening_60"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    ids = catalogue["instance_id"].astype(str).tolist()
    if progress:
        progress.begin("screening-60s", len(ids), message="10 s LP + 60 s MILP; checkpointed per instance")
    rows: list[dict[str, object]] = []
    for position, instance_id in enumerate(ids, start=1):
        checkpoint = checkpoint_root / f"{instance_id}.json"
        row = _load_checkpoint(checkpoint, protocol_signature)
        resumed = row is not None
        if row is None:
            row = screen_instance(
                root / instance_id,
                relaxation_time_limit_seconds=relaxation_time_limit_seconds,
                milp_time_limit_seconds=milp_time_limit_seconds,
                mip_relative_gap=mip_relative_gap,
            )
            atomic_write_json(checkpoint, {"protocol_signature": protocol_signature, "result": row})
        rows.append(row)
        atomic_write_parquet(pd.DataFrame(rows), output)
        if progress:
            status = str(row.get("solution_status", row.get("validation_status", "unknown")))
            progress.update(position, current=instance_id, message=("resumed" if resumed else f"status={status}"))
    frame = pd.DataFrame(rows)
    protocol = {
        "purpose": "standardized 60-second screening; difficulty is an observed response",
        "protocol_signature": protocol_signature,
        "relaxation_time_limit_seconds": relaxation_time_limit_seconds,
        "milp_time_limit_seconds": milp_time_limit_seconds,
        "mip_relative_gap": mip_relative_gap,
        "fresh_solve": True,
        "difficulty_policy": {
            "easy": "certified optimal within 2 seconds",
            "moderate": "certified optimal after 2 seconds",
            "hard": "incumbent available with relative gap <= 0.10",
            "extreme": "remaining valid open or wider-gap instances",
        },
    }
    atomic_write_json(output.with_suffix(".protocol.json"), protocol)
    if progress:
        progress.complete(message="60-second screening complete")
    return frame


def _horizon_result(instance: FCTPInstance, time_limit_seconds: float, mip_relative_gap: float) -> dict[str, object]:
    solved = solve_extensive_form(
        instance, time_limit_seconds=time_limit_seconds,
        mip_relative_gap=mip_relative_gap, relaxation=False,
    )
    treatment = instance.metadata.get("treatment", {})
    return {
        "instance_id": str(instance.metadata.get("instance_id", "")),
        "skeleton_id": str(instance.metadata.get("skeleton_id", "")),
        "track": str(instance.metadata.get("track", "")),
        "treatment_factor": str(treatment.get("factor", "")),
        "treatment_level": str(treatment.get("level", "")),
        "time_limit_seconds": float(time_limit_seconds),
        "solution_status": solved.status,
        "runtime_seconds": float(solved.runtime_seconds),
        "objective_usd": solved.objective_usd,
        "best_bound_usd": solved.best_bound_usd,
        "relative_gap": solved.relative_gap,
        "node_count": solved.node_count,
        "solved_at_utc": utc_now(),
    }


def solve_reference_horizon_library(
    catalogue: pd.DataFrame,
    instance_root: str | Path,
    prerequisite: pd.DataFrame,
    output_path: str | Path,
    *, time_limit_seconds: float,
    mip_relative_gap: float = 1e-4,
    checkpoint_directory: str | Path | None = None,
    protocol_signature: str = "",
    stage_name: str,
    progress: ProgressReporter | None = None,
) -> pd.DataFrame:
    """Fresh-solve only the instances unresolved at the preceding horizon."""
    root = Path(instance_root)
    output = Path(output_path)
    checkpoint_root = Path(checkpoint_directory) if checkpoint_directory else output.parent / "checkpoints" / stage_name
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    prereq = prerequisite.copy()
    eligible = prereq.loc[
        prereq.get("valid", pd.Series(True, index=prereq.index)).fillna(False).astype(bool)
        & (prereq["solution_status"].astype(str) != "optimal")
    ]
    eligible_ids = set(eligible["instance_id"].astype(str))
    ordered_ids = [value for value in catalogue["instance_id"].astype(str) if value in eligible_ids]
    if progress:
        progress.begin(stage_name, len(ordered_ids), message=f"fresh {time_limit_seconds:g} s MILP solves for unresolved instances")
    rows: list[dict[str, object]] = []
    for position, instance_id in enumerate(ordered_ids, start=1):
        checkpoint = checkpoint_root / f"{instance_id}.json"
        row = _load_checkpoint(checkpoint, protocol_signature)
        resumed = row is not None
        if row is None:
            instance = FCTPInstance.read(root / instance_id)
            row = _horizon_result(instance, time_limit_seconds, mip_relative_gap)
            atomic_write_json(checkpoint, {"protocol_signature": protocol_signature, "result": row})
        rows.append(row)
        atomic_write_parquet(pd.DataFrame(rows), output)
        if progress:
            progress.update(position, current=instance_id, message=("resumed" if resumed else f"status={row.get('solution_status')}"))
    frame = pd.DataFrame(rows)
    # Keep a valid empty Parquet with stable columns when no instances remain.
    if not len(frame):
        frame = pd.DataFrame(columns=[
            "instance_id", "skeleton_id", "track", "treatment_factor", "treatment_level",
            "time_limit_seconds", "solution_status", "runtime_seconds", "objective_usd",
            "best_bound_usd", "relative_gap", "node_count", "solved_at_utc",
        ])
        atomic_write_parquet(frame, output)
    atomic_write_json(output.with_suffix(".protocol.json"), {
        "purpose": f"fresh reference MILP horizon after unresolved preceding stage",
        "protocol_signature": protocol_signature,
        "time_limit_seconds": float(time_limit_seconds),
        "mip_relative_gap": mip_relative_gap,
        "fresh_solve": True,
        "eligible_instances": int(len(ordered_ids)),
    })
    if progress:
        progress.complete(message=f"{time_limit_seconds:g}-second reference horizon complete")
    return frame


def reclassify_screening_results(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    result = frame.copy()
    result["difficulty_class"] = [
        classify_difficulty(
            solution_status=str(row.solution_status),
            runtime_seconds=float(row.milp_runtime_seconds) if pd.notna(row.milp_runtime_seconds) else None,
            relative_gap=float(row.relative_gap) if pd.notna(row.relative_gap) else None,
            objective_usd=float(row.objective_usd) if pd.notna(row.objective_usd) else None,
        )
        for row in result.itertuples(index=False)
    ]
    counts = result["difficulty_class"].value_counts().reindex(DIFFICULTY_CLASSES, fill_value=0).astype(int).to_dict()
    return result, counts
