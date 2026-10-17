"""Command-line interface for FAF-FCTPBench."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .catalogue import CandidateSpecification, generate_candidate_catalogue, load_design, validate_matched_catalogue
from .characterization import write_characterization
from .construction import build_empirical_skeleton, build_fctp_instance, validate_matched_family
from .distribution import package_benchmark
from .empirical import inspect_faf_source, load_faf_forecast_flows, verify_faf_source
from .instance import FCTPInstance
from .optimization import solve_extensive_form
from .pipeline import execute_pipeline
from .screening import screen_library
from .validation import validate_instance_directory, write_validation_report


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd", engine="pyarrow")


def _cmd_catalogue(args: argparse.Namespace) -> None:
    config = load_design(args.design)
    frame = generate_candidate_catalogue(config)
    _write_parquet(frame, args.output)
    print(json.dumps(validate_matched_catalogue(frame, config), indent=2))


def _cmd_inspect_source(args: argparse.Namespace) -> None:
    config = load_design(args.design)
    verified = verify_faf_source(args.source, config)
    report = inspect_faf_source(args.source, year=int(config["faf_source"]["forecast_year"]))
    report["verified_source"] = verified
    print(json.dumps(report, indent=2, sort_keys=True))


def _cmd_materialize(args: argparse.Namespace) -> None:
    config = load_design(args.design)
    verify_faf_source(args.source, config)
    flows = load_faf_forecast_flows(
        args.source,
        year=int(config["faf_source"]["forecast_year"]),
        domestic_mode=str(config["faf_source"]["mode_code"]),
        trade_type=str(config["faf_source"]["trade_type_code"]),
    )
    catalogue = pd.read_parquet(args.catalogue, engine="pyarrow")
    validate_matched_catalogue(catalogue, config)
    args.output.mkdir(parents=True, exist_ok=True)
    for _, rows in catalogue.groupby("skeleton_id", sort=False):
        skeleton = build_empirical_skeleton(flows, CandidateSpecification.from_row(rows.iloc[0]), config)
        families = {"deterministic": [], "scenario": []}
        for _, row in rows.iterrows():
            spec = CandidateSpecification.from_row(row)
            directory = args.output / spec.instance_id
            instance = build_fctp_instance(flows, spec, config, skeleton=skeleton)
            instance.write(directory, compression_level=int(config["storage"]["compression_level"]), dictionary_encoding=bool(config["storage"]["dictionary_encoding"]))
            policy = str(config["validation"].get("independent_shared_capacity_check", "reference_only"))
            check_shared = (not args.skip_shared_capacity_check) and (policy == "all" or (policy == "reference_only" and spec.treatment_factor == "reference"))
            write_validation_report(
                instance, directory, check_shared_capacity=check_shared,
                capacity_check_time_limit_seconds=float(config["validation"].get("independent_capacity_time_limit_seconds", 60.0)),
            )
            families[spec.track].append(instance)
        for family in families.values():
            validate_matched_family(family)


def _cmd_validate(args: argparse.Namespace) -> None:
    single = (args.instance / "metadata.json").is_file()
    directories = [args.instance] if single else sorted(
        p for p in args.instance.iterdir() if p.is_dir() and (p / "metadata.json").is_file()
    )
    failed = []
    family_failed = []
    family_paths: dict[tuple[str, str], list[Path]] = {}
    for directory in directories:
        report = validate_instance_directory(directory, check_shared_capacity=not args.skip_shared_capacity_check)
        if not report["valid"]:
            failed.append({"instance": directory.name, "errors": report["errors"]})
        if not single:
            metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
            key = (str(metadata.get("skeleton_id", "")), str(metadata.get("track", "")))
            family_paths.setdefault(key, []).append(directory)
    if not single:
        for (skeleton_id, track), paths in family_paths.items():
            try:
                validate_matched_family([FCTPInstance.read(path) for path in paths])
            except Exception as error:
                family_failed.append({"skeleton_id": skeleton_id, "track": track, "error": str(error)})
    print(json.dumps({
        "instances_checked": len(directories),
        "instance_failures": failed,
        "matched_family_failures": family_failed,
    }, indent=2))
    if failed or family_failed:
        raise SystemExit(1)


def _cmd_solve(args: argparse.Namespace) -> None:
    instance = FCTPInstance.read(args.instance)
    result = solve_extensive_form(instance, time_limit_seconds=args.time_limit, mip_relative_gap=args.mip_gap, relaxation=args.relaxation, capture_design=args.capture_design)
    print(json.dumps(result.to_dict(), indent=2))


def _cmd_screen(args: argparse.Namespace) -> None:
    catalogue = pd.read_parquet(args.catalogue, engine="pyarrow")
    screen_library(catalogue, args.instances, args.output, relaxation_time_limit_seconds=args.lp_time, milp_time_limit_seconds=args.milp_time, mip_relative_gap=args.mip_gap)


def _cmd_characterize(args: argparse.Namespace) -> None:
    paths = write_characterization(args.catalogue, args.screening, args.output)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


def _cmd_package(args: argparse.Namespace) -> None:
    path = package_benchmark(args.instances, args.catalogue, args.output, screening_path=args.screening, characterization_directory=args.characterization)
    print(path)


def _cmd_pipeline(args: argparse.Namespace) -> None:
    paths = execute_pipeline(
        args.project_root,
        design_path=args.design,
        source_path=args.source,
        relaxation_time_limit_seconds=args.lp_time,
        milp_time_limit_seconds=args.milp_time,
        relaxation_followup_time_limit_seconds=args.lp_followup_time,
        reference_time_limit_seconds=args.reference_time,
        extended_time_limit_seconds=args.extended_time,
        mip_relative_gap=args.mip_gap,
        package_output=not args.no_package,
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


def _cmd_status(args: argparse.Namespace) -> None:
    path = args.project_root / "results" / "run_status.json"
    if not path.is_file():
        print(json.dumps({"status": "not_started", "path": str(path)}, indent=2))
        return
    print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="faf-fctp")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("catalogue", help="generate the matched benchmark catalogue")
    p.add_argument("--design", type=Path, default=None)
    p.add_argument("--output", type=Path, default=Path("data/catalogue/candidate_catalogue.parquet"))
    p.set_defaults(func=_cmd_catalogue)

    p = sub.add_parser("inspect-source", help="verify and inspect the FAF source archive")
    p.add_argument("source", type=Path)
    p.add_argument("--design", type=Path, default=None)
    p.set_defaults(func=_cmd_inspect_source)

    p = sub.add_parser("materialize", help="materialize all catalogue instances")
    p.add_argument("source", type=Path)
    p.add_argument("catalogue", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--design", type=Path, default=None)
    p.add_argument("--skip-shared-capacity-check", action="store_true")
    p.set_defaults(func=_cmd_materialize)

    p = sub.add_parser("validate", help="validate one instance or an instance library")
    p.add_argument("instance", type=Path)
    p.add_argument("--skip-shared-capacity-check", action="store_true")
    p.set_defaults(func=_cmd_validate)

    p = sub.add_parser("solve", help="run the reference extensive-form solver")
    p.add_argument("instance", type=Path)
    p.add_argument("--time-limit", type=float, default=200.0)
    p.add_argument("--mip-gap", type=float, default=1e-4)
    p.add_argument("--relaxation", action="store_true")
    p.add_argument("--capture-design", action="store_true")
    p.set_defaults(func=_cmd_solve)

    p = sub.add_parser("screen", help="run standardized reference screening")
    p.add_argument("catalogue", type=Path); p.add_argument("instances", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--lp-time", type=float, default=10.0); p.add_argument("--milp-time", type=float, default=60.0); p.add_argument("--mip-gap", type=float, default=1e-4)
    p.set_defaults(func=_cmd_screen)

    p = sub.add_parser("characterize", help="compute matched-pair treatment effects")
    p.add_argument("catalogue", type=Path); p.add_argument("screening", type=Path); p.add_argument("output", type=Path)
    p.set_defaults(func=_cmd_characterize)

    p = sub.add_parser("package", help="build a compact benchmark archive")
    p.add_argument("instances", type=Path); p.add_argument("catalogue", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--screening", type=Path); p.add_argument("--characterization", type=Path)
    p.set_defaults(func=_cmd_package)

    p = sub.add_parser("run-pipeline", help="run the complete resumable benchmark workflow from source verification through packaging")
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--design", type=Path, default=None)
    p.add_argument("--source", type=Path, default=None)
    p.add_argument("--lp-time", type=float, default=None, help="override configured LP screening limit")
    p.add_argument("--lp-followup-time", type=float, default=None, help="override targeted LP follow-up horizon (default: 30 seconds)")
    p.add_argument("--milp-time", type=float, default=None, help="override configured 60-second screening horizon")
    p.add_argument("--reference-time", type=float, default=None, help="override configured 200-second reference horizon")
    p.add_argument("--extended-time", type=float, default=None, help="override configured 400-second extended horizon")
    p.add_argument("--mip-gap", type=float, default=None)
    p.add_argument("--no-package", action="store_true", help="skip final compact ZIP packaging")
    p.set_defaults(func=_cmd_pipeline)

    p = sub.add_parser("status", help="show the latest persistent run-pipeline progress status")
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.set_defaults(func=_cmd_status)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)
