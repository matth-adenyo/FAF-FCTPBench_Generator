"""Resumable 30-second LP-relaxation follow-up for censored 10-second screens."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .instance import FCTPInstance
from .optimization import solve_extensive_form
from .runstate import ProgressReporter, atomic_write_json, atomic_write_parquet, utc_now

DEFAULT_FOLLOWUP_SECONDS = 30.0


def followup_signature(generation_signature: str, screening_signature: str, time_limit_seconds: float) -> str:
    """Signature isolated from generation and MILP horizons."""
    payload = {
        "generation_signature": generation_signature,
        "screening_signature": screening_signature,
        "stage": "lp-followup",
        "time_limit_seconds": float(time_limit_seconds),
        "fresh_solve": True,
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\0")
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def eligible_followup_ids(screening: pd.DataFrame) -> list[str]:
    """Return valid instances whose 10-second LP was not certified optimal."""
    valid = screening.get("valid", pd.Series(True, index=screening.index)).fillna(False).astype(bool)
    status = screening.get("relaxation_status", pd.Series("unknown", index=screening.index)).astype(str)
    return screening.loc[valid & (status != "optimal"), "instance_id"].astype(str).tolist()


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


def _solve_followup(instance: FCTPInstance, time_limit_seconds: float) -> dict[str, object]:
    solved = solve_extensive_form(
        instance,
        time_limit_seconds=time_limit_seconds,
        mip_relative_gap=0.0,
        relaxation=True,
    )
    treatment = instance.metadata.get("treatment", {})
    certified = solved.status == "optimal" and solved.objective_usd is not None
    return {
        "instance_id": str(instance.metadata.get("instance_id", "")),
        "skeleton_id": str(instance.metadata.get("skeleton_id", "")),
        "track": str(instance.metadata.get("track", "")),
        "treatment_factor": str(treatment.get("factor", "")),
        "treatment_level": str(treatment.get("level", "")),
        "time_limit_seconds": float(time_limit_seconds),
        "relaxation_status": solved.status,
        "relaxation_runtime_seconds": float(solved.runtime_seconds),
        "relaxation_objective_usd": solved.objective_usd,
        "certified_relaxation_objective_usd": solved.objective_usd if certified else None,
        "solver_message": solved.message,
        "solved_at_utc": utc_now(),
    }


def solve_lp_followup_library(
    catalogue: pd.DataFrame,
    instance_root: str | Path,
    screening: pd.DataFrame,
    output_path: str | Path,
    *,
    time_limit_seconds: float = DEFAULT_FOLLOWUP_SECONDS,
    checkpoint_directory: str | Path | None = None,
    protocol_signature: str = "",
    progress: ProgressReporter | None = None,
) -> pd.DataFrame:
    """Fresh-solve only LP relaxations not certified within the original screen."""
    root = Path(instance_root)
    output = Path(output_path)
    checkpoint_root = Path(checkpoint_directory) if checkpoint_directory else output.parent / "checkpoints" / "lp_30"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    eligible = set(eligible_followup_ids(screening))
    ordered_ids = [value for value in catalogue["instance_id"].astype(str) if value in eligible]
    if progress:
        progress.begin(
            "lp-followup-30s",
            len(ordered_ids),
            message=f"fresh {time_limit_seconds:g} s LP solves only for unresolved 10 s relaxations",
        )

    rows: list[dict[str, object]] = []
    for position, instance_id in enumerate(ordered_ids, start=1):
        checkpoint = checkpoint_root / f"{instance_id}.json"
        row = _load_checkpoint(checkpoint, protocol_signature)
        resumed = row is not None
        if row is None:
            instance = FCTPInstance.read(root / instance_id)
            row = _solve_followup(instance, time_limit_seconds)
            atomic_write_json(checkpoint, {"protocol_signature": protocol_signature, "result": row})
        rows.append(row)
        atomic_write_parquet(pd.DataFrame(rows), output)
        if progress:
            progress.update(
                position,
                current=instance_id,
                message="resumed" if resumed else f"status={row.get('relaxation_status')}",
            )

    frame = pd.DataFrame(rows)
    if not len(frame):
        frame = pd.DataFrame(columns=[
            "instance_id", "skeleton_id", "track", "treatment_factor", "treatment_level",
            "time_limit_seconds", "relaxation_status", "relaxation_runtime_seconds",
            "relaxation_objective_usd", "certified_relaxation_objective_usd",
            "solver_message", "solved_at_utc",
        ])
        atomic_write_parquet(frame, output)

    atomic_write_json(output.with_suffix(".protocol.json"), {
        "purpose": "fresh LP follow-up only for valid instances whose original 10-second relaxation was not certified optimal",
        "protocol_signature": protocol_signature,
        "time_limit_seconds": float(time_limit_seconds),
        "fresh_solve": True,
        "eligible_instances": int(len(ordered_ids)),
        "selection_rule": "valid == true and relaxation_status_10 != optimal",
        "note": "Original 10-second screening records are preserved; this stage does not alter any MILP result.",
    })
    if progress:
        progress.complete(message=f"{time_limit_seconds:g}-second LP follow-up complete")
    return frame


def build_relaxation_horizon_table(screening: pd.DataFrame, followup: pd.DataFrame) -> pd.DataFrame:
    """Combine LP@10 and LP@30 without treating an uncertified LP objective as a root bound."""
    base = screening[[
        "instance_id", "valid", "track", "treatment_factor", "treatment_level",
        "relaxation_status", "relaxation_runtime_seconds", "relaxation_objective_usd",
        "objective_usd",
    ]].copy().rename(columns={
        "relaxation_status": "status_10",
        "relaxation_runtime_seconds": "runtime_10_seconds",
        "relaxation_objective_usd": "objective_10_usd",
        "objective_usd": "milp_incumbent_60_usd",
    })

    if len(followup):
        second = followup[[
            "instance_id", "relaxation_status", "relaxation_runtime_seconds",
            "relaxation_objective_usd", "certified_relaxation_objective_usd",
        ]].rename(columns={
            "relaxation_status": "status_30",
            "relaxation_runtime_seconds": "runtime_30_seconds",
            "relaxation_objective_usd": "objective_30_usd",
            "certified_relaxation_objective_usd": "certified_objective_30_usd",
        })
        base = base.merge(second, on="instance_id", how="left")
    else:
        for column in ("status_30", "runtime_30_seconds", "objective_30_usd", "certified_objective_30_usd"):
            base[column] = np.nan

    certified_objective: list[float | None] = []
    certified_horizon: list[int | None] = []
    certified_root_gap: list[float | None] = []
    for row in base.itertuples(index=False):
        lp_obj = None
        horizon = None
        if str(row.status_10) == "optimal" and pd.notna(row.objective_10_usd) and np.isfinite(float(row.objective_10_usd)):
            lp_obj = float(row.objective_10_usd); horizon = 10
        elif str(row.status_30) == "optimal" and pd.notna(row.certified_objective_30_usd) and np.isfinite(float(row.certified_objective_30_usd)):
            lp_obj = float(row.certified_objective_30_usd); horizon = 30
        certified_objective.append(lp_obj)
        certified_horizon.append(horizon)
        incumbent = row.milp_incumbent_60_usd
        if lp_obj is not None and pd.notna(incumbent) and np.isfinite(float(incumbent)) and abs(float(incumbent)) > 1e-12:
            certified_root_gap.append(max(0.0, (float(incumbent) - lp_obj) / abs(float(incumbent))))
        else:
            certified_root_gap.append(None)

    base["certified_lp_objective_usd"] = certified_objective
    base["certified_lp_horizon_seconds"] = certified_horizon
    base["certified_root_gap"] = certified_root_gap
    base["lp_certified_by_30"] = pd.Series(certified_horizon).notna().to_numpy()
    base["lp_followup_required"] = base["status_10"].astype(str) != "optimal"
    base["lp_unresolved_after_30"] = base["lp_followup_required"] & (base["status_30"].astype(str) != "optimal")
    return base


def write_relaxation_horizon_summary(
    screening_path: str | Path,
    followup_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Path]:
    screening = pd.read_parquet(screening_path, engine="pyarrow")
    followup = pd.read_parquet(followup_path, engine="pyarrow")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    table = build_relaxation_horizon_table(screening, followup)
    table_path = output / "lp_relaxation_horizons.parquet"
    atomic_write_parquet(table, table_path)

    valid = table["valid"].fillna(False).astype(bool)
    valid_table = table.loc[valid]
    status_30_counts = (
        followup["relaxation_status"].astype(str).value_counts(dropna=False).to_dict()
        if len(followup) else {}
    )
    payload = {
        "valid_instances": int(len(valid_table)),
        "certified_optimal_at_10": int((valid_table["status_10"].astype(str) == "optimal").sum()),
        "followup_eligible_after_10": int(valid_table["lp_followup_required"].sum()),
        "fresh_30_second_lp_solves": int(len(followup)),
        "followup_status_counts": {str(k): int(v) for k, v in status_30_counts.items()},
        "newly_certified_at_30": int((followup["relaxation_status"].astype(str) == "optimal").sum()) if len(followup) else 0,
        "certified_by_30": int(valid_table["lp_certified_by_30"].sum()),
        "unresolved_after_30": int(valid_table["lp_unresolved_after_30"].sum()),
        "certified_root_gap_available": int(pd.to_numeric(valid_table["certified_root_gap"], errors="coerce").notna().sum()),
        "interpretation": (
            "The 10-second LP screen is preserved. Only valid instances not certified optimal at 10 seconds receive a fresh 30-second LP solve. "
            "Root-gap analysis uses only a certified optimal LP objective; objectives from time-limited LP solves are not treated as valid lower bounds."
        ),
    }
    summary_path = output / "lp_relaxation_summary.json"
    atomic_write_json(summary_path, payload)
    return {"table": table_path, "summary": summary_path}
