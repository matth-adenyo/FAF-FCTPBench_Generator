"""Single-entry, resumable end-to-end FAF-FCTPBench pipeline."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import pandas as pd

from .catalogue import CandidateSpecification, generate_candidate_catalogue, load_design
from .characterization import write_characterization, write_horizon_summary
from .construction import build_empirical_skeleton, build_fctp_instance, validate_matched_family
from .distribution import package_benchmark
from .empirical import load_faf_forecast_flows, verify_faf_source
from .instance import FCTPInstance
from .runstate import ProgressReporter, atomic_write_json, atomic_write_parquet, utc_now
from .screening import screen_library, solve_reference_horizon_library
from .relaxation_followup import (
    DEFAULT_FOLLOWUP_SECONDS, followup_signature, solve_lp_followup_library,
    write_relaxation_horizon_summary,
)
from .validation import write_validation_report


def _sha256_bytes(parts: list[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
        digest.update(b"\0")
    return digest.hexdigest()


def _implementation_signature(config: dict[str, Any]) -> str:
    module_root = Path(__file__).resolve().parent
    files = ["catalogue.py", "construction.py", "instance.py", "validation.py"]
    parts = [json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")]
    parts.extend((module_root / name).read_bytes() for name in files)
    return _sha256_bytes(parts)


def _solver_signature(generation_signature: str, *, stage: str, seconds: float, lp_seconds: float | None, mip_gap: float) -> str:
    module_root = Path(__file__).resolve().parent
    payload = {
        "generation_signature": generation_signature,
        "stage": stage,
        "seconds": float(seconds),
        "lp_seconds": None if lp_seconds is None else float(lp_seconds),
        "mip_gap": float(mip_gap),
        "fresh_solve": True,
    }
    return _sha256_bytes([
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        (module_root / "optimization.py").read_bytes(),
        (module_root / "screening.py").read_bytes(),
    ])



def _load_completed_solver_table(path: Path, expected_signature: str) -> pd.DataFrame | None:
    """Reuse a completed signed solver table even if per-instance checkpoints were removed."""
    protocol_path = path.with_suffix(".protocol.json")
    if not (path.is_file() and protocol_path.is_file()):
        return None
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol.get("protocol_signature") != expected_signature:
            return None
        return pd.read_parquet(path, engine="pyarrow")
    except Exception:
        return None


def _announce_reused_table(progress: ProgressReporter, stage: str, frame: pd.DataFrame, path: Path) -> None:
    progress.begin(stage, 1, message="reusing completed signed result table")
    progress.update(1, current=path.name, message=f"reused {len(frame)} rows; no solver rerun")


def _instance_reusable(directory: Path, generation_signature: str) -> bool:
    metadata_path = directory / "metadata.json"
    validation_path = directory / "validation.json"
    manifest_path = directory / "storage_manifest.json"
    if not (metadata_path.is_file() and validation_path.is_file() and manifest_path.is_file()):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return metadata.get("generation_signature") == generation_signature and bool(validation.get("valid", False))


def _replace_directory_atomicish(source: Path, destination: Path) -> None:
    # Directory replacement is made restart-safe.
    if destination.exists():
        shutil.rmtree(destination)
    os.replace(source, destination)


def _validation_crosscheck_required(spec: CandidateSpecification, config: dict[str, Any]) -> bool:
    policy = str(config["validation"].get("independent_shared_capacity_check", "reference_only"))
    if policy == "all":
        return True
    if policy == "reference_only":
        return spec.treatment_factor == "reference"
    return False


def _materialize_library(
    flows: pd.DataFrame,
    catalogue: pd.DataFrame,
    instance_root: Path,
    config: dict[str, Any],
    generation_signature: str,
    progress: ProgressReporter,
) -> None:
    instance_root.mkdir(parents=True, exist_ok=True)
    total = len(catalogue)
    progress.begin("materialization", total, message="resumable instance generation and validation")
    completed = 0
    capacity_seconds = float(config["validation"].get("independent_capacity_time_limit_seconds", 60.0))

    for _, skeleton_rows in catalogue.groupby("skeleton_id", sort=False):
        reusable = {
            str(row.instance_id): _instance_reusable(instance_root / str(row.instance_id), generation_signature)
            for row in skeleton_rows.itertuples(index=False)
        }
        skeleton = None
        if not all(reusable.values()):
            representative = CandidateSpecification.from_row(skeleton_rows.iloc[0])
            skeleton = build_empirical_skeleton(flows, representative, config)

        for _, row in skeleton_rows.iterrows():
            spec = CandidateSpecification.from_row(row)
            directory = instance_root / spec.instance_id
            if reusable[spec.instance_id]:
                completed += 1
                progress.update(completed, current=spec.instance_id, message="resumed existing valid instance")
                continue
            assert skeleton is not None
            instance = build_fctp_instance(flows, spec, config, skeleton=skeleton)
            instance.metadata["generation_signature"] = generation_signature
            temporary = instance_root / f".{spec.instance_id}.building"
            if temporary.exists():
                shutil.rmtree(temporary)
            instance.write(
                temporary,
                compression_level=int(config["storage"]["compression_level"]),
                dictionary_encoding=bool(config["storage"]["dictionary_encoding"]),
            )
            write_validation_report(
                instance,
                temporary,
                check_shared_capacity=_validation_crosscheck_required(spec, config),
                capacity_check_time_limit_seconds=capacity_seconds,
            )
            validation = json.loads((temporary / "validation.json").read_text(encoding="utf-8"))
            if not validation.get("valid", False):
                raise ValueError(f"generated instance {spec.instance_id} failed validation: {validation.get('errors')}")
            _replace_directory_atomicish(temporary, directory)
            completed += 1
            progress.update(
                completed,
                current=spec.instance_id,
                message=f"generated; feasibility={validation.get('feasibility_status')}",
            )
    progress.complete(message="all instances materialized and individually valid")


def _validate_families(
    catalogue: pd.DataFrame,
    instance_root: Path,
    results_dir: Path,
    generation_signature: str,
    progress: ProgressReporter,
) -> Path:
    family_root = results_dir / "validation" / "families"
    family_root.mkdir(parents=True, exist_ok=True)
    groups = list(catalogue.groupby(["skeleton_id", "track"], sort=False))
    progress.begin("matched-family-validation", len(groups), message="realized one-factor isolation checks")
    records: list[dict[str, object]] = []
    for position, ((skeleton_id, track), rows) in enumerate(groups, start=1):
        checkpoint = family_root / f"{skeleton_id}-{track}.json"
        record = None
        resumed = False
        if checkpoint.is_file():
            try:
                payload = json.loads(checkpoint.read_text(encoding="utf-8"))
                if payload.get("generation_signature") == generation_signature and payload.get("valid") is True:
                    record = payload
                    resumed = True
            except Exception:
                record = None
        if record is None:
            instances = [FCTPInstance.read(instance_root / str(instance_id)) for instance_id in rows["instance_id"].astype(str)]
            proof = validate_matched_family(instances)
            record = {"generation_signature": generation_signature, **proof, "validated_at_utc": utc_now()}
            atomic_write_json(checkpoint, record)
        records.append(record)
        progress.update(position, current=f"{skeleton_id}/{track}", message="resumed" if resumed else "validated")
    summary = results_dir / "validation" / "family_validation.parquet"
    atomic_write_parquet(pd.DataFrame(records), summary)
    progress.complete(message="all matched families validated")
    return summary


def _write_validation_summary(catalogue: pd.DataFrame, instance_root: Path, results_dir: Path) -> Path:
    rows: list[dict[str, object]] = []
    for instance_id in catalogue["instance_id"].astype(str):
        report = json.loads((instance_root / instance_id / "validation.json").read_text(encoding="utf-8"))
        rows.append({
            "instance_id": instance_id,
            "valid": bool(report.get("valid", False)),
            "validation_status": report.get("validation_status"),
            "feasibility_status": report.get("feasibility_status"),
            "warning_count": len(report.get("warnings", [])),
            "error_count": len(report.get("errors", [])),
        })
    path = results_dir / "validation" / "validation_summary.parquet"
    atomic_write_parquet(pd.DataFrame(rows), path)
    return path


def execute_pipeline(
    project_root: str | Path,
    *,
    design_path: str | Path | None = None,
    source_path: str | Path | None = None,
    relaxation_time_limit_seconds: float | None = None,
    milp_time_limit_seconds: float | None = None,
    relaxation_followup_time_limit_seconds: float | None = None,
    reference_time_limit_seconds: float | None = None,
    extended_time_limit_seconds: float | None = None,
    mip_relative_gap: float | None = None,
    package_output: bool = True,
) -> dict[str, Path]:
    root = Path(project_root).resolve()
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressReporter(results_dir / "run_status.json")
    try:
        config = load_design(design_path)
        solver = config["solver_protocol"]
        lp_seconds = float(relaxation_time_limit_seconds if relaxation_time_limit_seconds is not None else solver["relaxation_seconds"])
        screen_seconds = float(milp_time_limit_seconds if milp_time_limit_seconds is not None else solver["screening_milp_seconds"])
        lp_followup_seconds = float(relaxation_followup_time_limit_seconds if relaxation_followup_time_limit_seconds is not None else DEFAULT_FOLLOWUP_SECONDS)
        if lp_followup_seconds <= lp_seconds:
            raise ValueError("LP follow-up horizon must be greater than the screening LP horizon")
        reference_seconds = float(reference_time_limit_seconds if reference_time_limit_seconds is not None else solver["reference_milp_seconds"])
        extended_seconds = float(extended_time_limit_seconds if extended_time_limit_seconds is not None else solver["extended_milp_seconds"])
        gap = float(mip_relative_gap if mip_relative_gap is not None else solver["mip_relative_gap"])
        if not (0 < screen_seconds < reference_seconds < extended_seconds):
            raise ValueError("MILP horizons must satisfy screening < reference < extended")

        progress.begin("source-verification", 1, message="verifying pinned FAF archive")
        source = Path(source_path) if source_path else root / "data" / "raw" / str(config["faf_source"]["archive_name"])
        verify_faf_source(source, config)
        progress.update(1, current=source.name, message="SHA-256 verified")
        flows = load_faf_forecast_flows(
            source,
            year=int(config["faf_source"]["forecast_year"]),
            domestic_mode=str(config["faf_source"]["mode_code"]),
            trade_type=str(config["faf_source"]["trade_type_code"]),
        )

        generation_signature = _implementation_signature(config)
        progress.begin("catalogue", 1, message="building matched catalogue")
        catalogue = generate_candidate_catalogue(config)
        catalogue_dir = root / "data" / "catalogue"
        catalogue_path = catalogue_dir / "candidate_catalogue.parquet"
        atomic_write_parquet(catalogue, catalogue_path)
        atomic_write_json(catalogue_dir / "catalogue_signature.json", {
            "generation_signature": generation_signature,
            "instances": int(len(catalogue)),
            "created_at_utc": utc_now(),
        })
        progress.update(1, current=catalogue_path.name, message=f"{len(catalogue)} instances")

        instance_root = root / "data" / "instances"
        _materialize_library(flows, catalogue, instance_root, config, generation_signature, progress)
        family_validation = _validate_families(catalogue, instance_root, results_dir, generation_signature, progress)
        validation_summary = _write_validation_summary(catalogue, instance_root, results_dir)

        screening_path = results_dir / "screening_results.parquet"
        screen_signature = _solver_signature(generation_signature, stage="screening-60s", seconds=screen_seconds, lp_seconds=lp_seconds, mip_gap=gap)
        screening = _load_completed_solver_table(screening_path, screen_signature)
        if screening is None:
            screening = screen_library(
                catalogue, instance_root, screening_path,
                relaxation_time_limit_seconds=lp_seconds,
                milp_time_limit_seconds=screen_seconds,
                mip_relative_gap=gap,
                checkpoint_directory=results_dir / "checkpoints" / "screening_60",
                protocol_signature=screen_signature,
                progress=progress,
            )
        else:
            _announce_reused_table(progress, "screening-60s", screening, screening_path)

        # Targeted LP follow-up.
        lp_followup_path = results_dir / "lp_30_results.parquet"
        lp_followup_sig = followup_signature(generation_signature, screen_signature, lp_followup_seconds)
        lp_followup = _load_completed_solver_table(lp_followup_path, lp_followup_sig)
        if lp_followup is None:
            lp_followup = solve_lp_followup_library(
                catalogue, instance_root, screening, lp_followup_path,
                time_limit_seconds=lp_followup_seconds,
                checkpoint_directory=results_dir / "checkpoints" / "lp_30",
                protocol_signature=lp_followup_sig,
                progress=progress,
            )
        else:
            _announce_reused_table(progress, "lp-followup-30s", lp_followup, lp_followup_path)

        reference_path = results_dir / "reference_200_results.parquet"
        reference_signature = _solver_signature(generation_signature, stage="reference-200s", seconds=reference_seconds, lp_seconds=None, mip_gap=gap)
        reference = _load_completed_solver_table(reference_path, reference_signature)
        if reference is None:
            reference = solve_reference_horizon_library(
                catalogue, instance_root, screening, reference_path,
                time_limit_seconds=reference_seconds,
                mip_relative_gap=gap,
                checkpoint_directory=results_dir / "checkpoints" / "reference_200",
                protocol_signature=reference_signature,
                stage_name="reference-200s",
                progress=progress,
            )
        else:
            _announce_reused_table(progress, "reference-200s", reference, reference_path)

        extended_path = results_dir / "reference_400_results.parquet"
        extended_signature = _solver_signature(generation_signature, stage="extended-400s", seconds=extended_seconds, lp_seconds=None, mip_gap=gap)
        extended = _load_completed_solver_table(extended_path, extended_signature)
        if extended is None:
            extended = solve_reference_horizon_library(
                catalogue, instance_root, reference, extended_path,
                time_limit_seconds=extended_seconds,
                mip_relative_gap=gap,
                checkpoint_directory=results_dir / "checkpoints" / "reference_400",
                protocol_signature=extended_signature,
                stage_name="extended-400s",
                progress=progress,
            )
        else:
            _announce_reused_table(progress, "extended-400s", extended, extended_path)

        progress.begin("characterization", 1, message="matched effects and horizon persistence")
        characterization_dir = results_dir / "characterization"
        lp_relaxation = write_relaxation_horizon_summary(screening_path, lp_followup_path, characterization_dir)
        write_characterization(
            catalogue_path, screening_path, characterization_dir,
            relaxation_horizons_path=lp_relaxation["table"],
        )
        horizon_summary = write_horizon_summary(screening_path, reference_path, extended_path, characterization_dir)
        progress.update(1, current="characterization", message="reports written")

        outputs: dict[str, Path] = {
            "catalogue": catalogue_path,
            "instances": instance_root,
            "validation_summary": validation_summary,
            "family_validation": family_validation,
            "screening_60": screening_path,
            "lp_followup_30": lp_followup_path,
            "lp_relaxation_horizons": lp_relaxation["table"],
            "lp_relaxation_summary": lp_relaxation["summary"],
            "reference_200": reference_path,
            "reference_400": extended_path,
            "characterization": characterization_dir,
            "horizon_summary": horizon_summary,
        }
        archive_path = root / "dist" / "FAF-FCTPBench.zip"
        if package_output:
            outputs["archive"] = archive_path
        report = {
            "status": "complete",
            "generation_signature": generation_signature,
            "instances": int(len(catalogue)),
            "solver_protocol": {
                "relaxation_seconds": lp_seconds,
                "relaxation_followup_seconds": lp_followup_seconds,
                "screening_milp_seconds": screen_seconds,
                "reference_milp_seconds": reference_seconds,
                "extended_milp_seconds": extended_seconds,
                "mip_relative_gap": gap,
                "fresh_solve_each_horizon": True,
            },
            "completed_at_utc": utc_now(),
            "outputs": {key: str(value) for key, value in outputs.items()},
        }
        atomic_write_json(results_dir / "pipeline_report.json", report)
        if package_output:
            progress.begin("packaging", 1, message="building compact benchmark archive")
            archive = package_benchmark(
                instance_root, catalogue_path, archive_path, results_directory=results_dir,
            )
            outputs["archive"] = archive
            progress.update(1, current=archive.name, message="Parquet stored without redundant ZIP compression")

        progress.pipeline_complete({key: str(value) for key, value in outputs.items()})
        return outputs
    except BaseException as error:
        progress.fail(error)
        raise
