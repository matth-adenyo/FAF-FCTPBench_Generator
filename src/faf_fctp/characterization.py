"""Matched-pair benchmark characterization."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ("milp_runtime_seconds", "relative_gap", "root_gap", "objective_usd")


def matched_effects(catalogue: pd.DataFrame, screening: pd.DataFrame) -> pd.DataFrame:
    merged = catalogue.merge(screening, on=["instance_id", "skeleton_id", "track", "treatment_factor", "treatment_level"], how="inner")
    reference = merged.loc[merged["treatment_factor"] == "reference", ["skeleton_id", "track", *METRICS]].copy()
    reference = reference.rename(columns={metric: f"reference_{metric}" for metric in METRICS})
    treated = merged.loc[merged["treatment_factor"] != "reference"].merge(reference, on=["skeleton_id", "track"], how="left")
    for metric in METRICS:
        treated[f"paired_difference_{metric}"] = pd.to_numeric(treated[metric], errors="coerce") - pd.to_numeric(treated[f"reference_{metric}"], errors="coerce")
        ref = pd.to_numeric(treated[f"reference_{metric}"], errors="coerce")
        treated[f"paired_ratio_{metric}"] = np.divide(
            pd.to_numeric(treated[metric], errors="coerce"),
            ref,
            out=np.full(len(treated), np.nan),
            where=np.isfinite(ref) & (np.abs(ref) > 1e-12),
        )
    return treated


def summarize_matched_effects(effects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (factor, level, track), group in effects.groupby(["treatment_factor", "treatment_level", "track"], dropna=False):
        row: dict[str, object] = {
            "treatment_factor": factor,
            "treatment_level": level,
            "track": track,
            "paired_instances": int(len(group)),
        }
        for metric in METRICS:
            diff = pd.to_numeric(group[f"paired_difference_{metric}"], errors="coerce")
            ratio = pd.to_numeric(group[f"paired_ratio_{metric}"], errors="coerce")
            row[f"mean_paired_difference_{metric}"] = float(diff.mean()) if diff.notna().any() else None
            row[f"median_paired_difference_{metric}"] = float(diff.median()) if diff.notna().any() else None
            row[f"median_paired_ratio_{metric}"] = float(ratio.median()) if ratio.notna().any() else None
        rows.append(row)
    return pd.DataFrame(rows)


def write_characterization(
    catalogue_path: str | Path,
    screening_path: str | Path,
    output_directory: str | Path,
    *,
    relaxation_horizons_path: str | Path | None = None,
) -> dict[str, Path]:
    catalogue = pd.read_parquet(catalogue_path, engine="pyarrow")
    screening = pd.read_parquet(screening_path, engine="pyarrow")
    root_gap_policy = "10-second screening root gap"
    if relaxation_horizons_path is not None and Path(relaxation_horizons_path).is_file():
        lp_horizons = pd.read_parquet(relaxation_horizons_path, engine="pyarrow")
        certified = lp_horizons[["instance_id", "certified_root_gap"]].copy()
        screening = screening.drop(columns=["root_gap"], errors="ignore").merge(certified, on="instance_id", how="left")
        screening = screening.rename(columns={"certified_root_gap": "root_gap"})
        root_gap_policy = "certified LP optimum from 10-second screen or targeted 30-second follow-up"
    else:
        if "root_gap" in screening and "relaxation_status" in screening:
            screening.loc[screening["relaxation_status"].astype(str) != "optimal", "root_gap"] = np.nan
        root_gap_policy = "certified 10-second LP optimum only"
    output = Path(output_directory); output.mkdir(parents=True, exist_ok=True)
    effects = matched_effects(catalogue, screening)
    summary_table = summarize_matched_effects(effects)
    effects_path = output / "matched_effects.parquet"
    summary_table_path = output / "matched_effect_summary.parquet"
    effects.to_parquet(effects_path, index=False, compression="zstd", engine="pyarrow")
    summary_table.to_parquet(summary_table_path, index=False, compression="zstd", engine="pyarrow")
    difficulty_counts = screening["difficulty_class"].value_counts(dropna=False).to_dict() if "difficulty_class" in screening else {}
    payload = {
        "instances": int(len(screening)),
        "matched_effect_rows": int(len(effects)),
        "difficulty_counts": {str(k): int(v) for k, v in difficulty_counts.items()},
        "root_gap_policy": root_gap_policy,
        "certified_root_gap_rows": int(pd.to_numeric(screening.get("root_gap"), errors="coerce").notna().sum()) if "root_gap" in screening else 0,
        "interpretation": (
            "Treatment effects are estimated by within-skeleton, within-track paired comparisons. "
            "Each compared row differs from its reference in one declared factor only. Difficulty "
            "remains solver- and protocol-conditional."
        ),
    }
    json_path = output / "characterization.json"
    json_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return {"matched_effects": effects_path, "matched_effect_summary": summary_table_path, "summary": json_path}


def write_horizon_summary(
    screening_path: str | Path,
    reference_path: str | Path,
    extended_path: str | Path,
    output_directory: str | Path,
) -> Path:
    """Summarize persistence of unresolved instances across 60/200/400 horizons."""
    from .runstate import atomic_write_json, atomic_write_parquet

    screening = pd.read_parquet(screening_path, engine="pyarrow")
    reference = pd.read_parquet(reference_path, engine="pyarrow")
    extended = pd.read_parquet(extended_path, engine="pyarrow")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    base = screening[[
        "instance_id", "valid", "solution_status", "milp_runtime_seconds",
        "objective_usd", "best_bound_usd", "relative_gap", "node_count",
    ]].rename(columns={
        "solution_status": "status_60",
        "milp_runtime_seconds": "runtime_60_seconds",
        "objective_usd": "objective_60_usd",
        "best_bound_usd": "best_bound_60_usd",
        "relative_gap": "relative_gap_60",
        "node_count": "node_count_60",
    })
    if len(reference):
        r = reference[[
            "instance_id", "solution_status", "runtime_seconds", "objective_usd",
            "best_bound_usd", "relative_gap", "node_count",
        ]].rename(columns={
            "solution_status": "status_200", "runtime_seconds": "runtime_200_seconds",
            "objective_usd": "objective_200_usd", "best_bound_usd": "best_bound_200_usd",
            "relative_gap": "relative_gap_200", "node_count": "node_count_200",
        })
        base = base.merge(r, on="instance_id", how="left")
    if len(extended):
        e = extended[[
            "instance_id", "solution_status", "runtime_seconds", "objective_usd",
            "best_bound_usd", "relative_gap", "node_count",
        ]].rename(columns={
            "solution_status": "status_400", "runtime_seconds": "runtime_400_seconds",
            "objective_usd": "objective_400_usd", "best_bound_usd": "best_bound_400_usd",
            "relative_gap": "relative_gap_400", "node_count": "node_count_400",
        })
        base = base.merge(e, on="instance_id", how="left")

    # Consolidate best-known reference evidence across fresh horizons.
    best_objective: list[float | None] = []
    best_bound: list[float | None] = []
    best_gap: list[float | None] = []
    incumbent_horizon: list[int | None] = []
    bound_horizon: list[int | None] = []
    optimal_any: list[bool] = []
    for _, row in base.iterrows():
        objectives: list[tuple[int, float]] = []
        bounds: list[tuple[int, float]] = []
        statuses: list[str] = []
        for horizon in (60, 200, 400):
            status = row.get(f"status_{horizon}")
            if pd.notna(status):
                statuses.append(str(status))
            objective = row.get(f"objective_{horizon}_usd")
            if objective is not None and pd.notna(objective) and np.isfinite(float(objective)):
                objectives.append((horizon, float(objective)))
            bound = row.get(f"best_bound_{horizon}_usd")
            if bound is not None and pd.notna(bound) and np.isfinite(float(bound)):
                bounds.append((horizon, float(bound)))
        if objectives:
            h_obj, obj = min(objectives, key=lambda item: item[1])
            best_objective.append(obj); incumbent_horizon.append(h_obj)
        else:
            obj = None; best_objective.append(None); incumbent_horizon.append(None)
        if bounds:
            h_bnd, bnd = max(bounds, key=lambda item: item[1])
            best_bound.append(bnd); bound_horizon.append(h_bnd)
        else:
            bnd = None; best_bound.append(None); bound_horizon.append(None)
        if obj is not None and bnd is not None and abs(obj) > 1e-12:
            best_gap.append(max(0.0, (obj - bnd) / abs(obj)))
        else:
            best_gap.append(None)
        optimal_any.append("optimal" in statuses)
    base["best_known_objective_usd"] = best_objective
    base["best_known_bound_usd"] = best_bound
    base["best_known_relative_gap"] = best_gap
    base["best_incumbent_horizon_seconds"] = incumbent_horizon
    base["best_bound_horizon_seconds"] = bound_horizon
    base["certified_optimal_by_400"] = optimal_any

    table_path = output / "reference_horizons.parquet"
    atomic_write_parquet(base, table_path)
    valid = screening.loc[screening["valid"].fillna(False).astype(bool)]
    open_60 = valid.loc[valid["solution_status"].astype(str) != "optimal"]
    open_200 = reference.loc[reference["solution_status"].astype(str) != "optimal"] if len(reference) else reference
    open_400 = extended.loc[extended["solution_status"].astype(str) != "optimal"] if len(extended) else extended
    payload = {
        "valid_instances": int(len(valid)),
        "certified_optimal_at_60": int((valid["solution_status"].astype(str) == "optimal").sum()),
        "unresolved_after_60": int(len(open_60)),
        "fresh_200_second_solves": int(len(reference)),
        "certified_optimal_at_200_fresh_run": int((reference["solution_status"].astype(str) == "optimal").sum()) if len(reference) else 0,
        "unresolved_after_200": int(len(open_200)),
        "fresh_400_second_solves": int(len(extended)),
        "certified_optimal_at_400_fresh_run": int((extended["solution_status"].astype(str) == "optimal").sum()) if len(extended) else 0,
        "unresolved_after_400": int(len(open_400)),
        "certified_optimal_by_400_any_fresh_horizon": int(base["certified_optimal_by_400"].sum()),
        "instances_with_finite_best_known_incumbent": int(pd.Series(base["best_known_objective_usd"]).notna().sum()),
        "interpretation": "Each horizon is a fresh solve. The 200-second stage contains only valid instances unresolved at 60 seconds, the 400-second stage contains only instances unresolved in the fresh 200-second run.",
    }
    json_path = output / "reference_horizon_summary.json"
    atomic_write_json(json_path, payload)
    return json_path
