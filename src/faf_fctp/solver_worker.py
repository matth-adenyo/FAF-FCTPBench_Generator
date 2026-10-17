"""Isolated reference-solver worker used by the screening protocol."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from .instance import FCTPInstance
from .optimization import solve_extensive_form


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("instance", type=Path)
    parser.add_argument("--time-limit", type=float, required=True)
    parser.add_argument("--mip-gap", type=float, required=True)
    parser.add_argument("--relaxation", action="store_true")
    parser.add_argument("--capture-design", action="store_true")
    parser.add_argument("--scenario", choices=("all", "base", "high"), default="all")
    parser.add_argument("--capacity-multiplier", type=float, default=1.0)
    parser.add_argument("--fixed-open-lanes-path", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.capacity_multiplier < 1.0:
        raise ValueError("capacity multiplier must be at least one")
    instance = FCTPInstance.read(args.instance)
    if args.scenario != "all":
        scenario_ids = instance.scenarios["scenario_id"].astype(str)
        selected = instance.scenarios.loc[scenario_ids == args.scenario].copy()
        if len(selected) != 1:
            raise ValueError(f"missing scenario: {args.scenario}")
        selected["probability"] = 1.0
        instance = replace(
            instance, scenarios=selected,
            supplies=instance.supplies.loc[instance.supplies["scenario_id"].astype(str) == args.scenario].copy(),
            demands=instance.demands.loc[instance.demands["scenario_id"].astype(str) == args.scenario].copy(),
        )
    if args.capacity_multiplier != 1.0:
        lanes = instance.lanes.copy()
        lanes["capacity_ktons"] *= args.capacity_multiplier
        instance = replace(instance, lanes=lanes)
    fixed_design = None
    if args.fixed_open_lanes_path is not None:
        fixed_design = json.loads(args.fixed_open_lanes_path.read_text(encoding="utf-8"))
        if not isinstance(fixed_design, list) or not all(isinstance(lane, str) for lane in fixed_design):
            raise ValueError("fixed-open-lanes-path must contain a JSON list of lane identifiers")
    result = solve_extensive_form(
        instance,
        time_limit_seconds=args.time_limit,
        mip_relative_gap=args.mip_gap,
        relaxation=args.relaxation,
        capture_design=args.capture_design,
        fixed_open_lane_ids=fixed_design,
    )
    print(json.dumps(result.to_dict(), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
