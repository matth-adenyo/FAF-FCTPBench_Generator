#!/usr/bin/env python3
"""Generate figures for Analysis.

Outputs
-------
Six figures are written to ``figures/`` as both PDF and PNG:

1. benchmark_design.pdf/png
2. control_calibration.pdf/png
3. difficulty_distribution.pdf/png
4. regime_heatmap.pdf/png
5. horizon_persistence.pdf/png
6. matched_effects_ci.pdf/png

Derived numerical tables used by the figures are written to ``data/derived/``.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIG = HERE / "figures"
DERIVED = DATA / "derived"

FIG.mkdir(parents=True, exist_ok=True)
DERIVED.mkdir(parents=True, exist_ok=True)

DIFFICULTY_ORDER = ["easy", "moderate", "hard", "extreme"]
DIFFICULTY_SCORE = {name: idx for idx, name in enumerate(DIFFICULTY_ORDER)}
SCALE_ORDER = ["small", "medium", "large", "very_large"]
SCALE_LABEL = {"small": "Small", "medium": "Medium", "large": "Large", "very_large": "Very large"}
PRODUCT_ORDER = [1, 5, 20]


def fail(message: str) -> "None":
    raise SystemExit(f"\nERROR: {message}\n")


def resolve_results_dir() -> Path:
    direct = DATA / "results"
    nested = direct / "results"
    for candidate in (direct, nested):
        if (candidate / "screening_results.parquet").is_file():
            return candidate
    fail(
        "Completed results were not found. Copy the benchmark results folder to "
        f"'{direct}'. The file data/results/screening_results.parquet must exist."
    )


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.is_file():
        fail(f"Required result file is missing: {path.relative_to(HERE)}")
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except ImportError:
        fail(
            "Reading Parquet requires pyarrow. Install plotting dependencies with:\n"
            "    python -m pip install pandas pyarrow numpy matplotlib"
        )
    except Exception as exc:
        fail(f"Could not read {path}: {exc}")


def read_json(path: Path) -> dict:
    if not path.is_file():
        fail(f"Required result file is missing: {path.relative_to(HERE)}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def fmt_level(value: str) -> str:
    return str(value).replace("_", " ").strip().title()


def bootstrap_mean_ci(values: np.ndarray, *, seed: int, n_boot: int = 10000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan, math.nan
    mean = float(values.mean())
    if values.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return mean, float(lo), float(hi)


def load_inputs(results: Path) -> dict[str, object]:
    screening = read_parquet(results / "screening_results.parquet")
    reference_200 = read_parquet(results / "reference_200_results.parquet")
    reference_400 = read_parquet(results / "reference_400_results.parquet")
    lp30 = read_parquet(results / "lp_30_results.parquet")
    matched = read_parquet(results / "characterization" / "matched_effects.parquet")
    characterization = read_json(results / "characterization" / "characterization.json")
    horizon_summary = read_json(results / "characterization" / "reference_horizon_summary.json")
    lp_summary = read_json(results / "characterization" / "lp_relaxation_summary.json")
    pipeline = read_json(results / "pipeline_report.json")

    required_screening = {
        "instance_id", "skeleton_id", "track", "treatment_factor", "treatment_level",
        "difficulty_class", "solution_status", "relaxation_status", "milp_runtime_seconds",
        "relative_gap", "lane_count", "capacity_headroom_multiplier", "cost_variability_sigma",
        "demand_gini_base", "fixed_charge_intensity_target", "uncertainty_amplitude",
    }
    missing = required_screening - set(screening.columns)
    if missing:
        fail(f"screening_results.parquet is missing expected columns: {sorted(missing)}")

    required_matched = {"skeleton_id", "scale_class", "n_products"}
    missing = required_matched - set(matched.columns)
    if missing:
        fail(f"matched_effects.parquet is missing expected columns: {sorted(missing)}")

    if int(characterization.get("instances", -1)) != len(screening):
        fail("characterization.json and screening_results.parquet disagree on instance count")

    # Invariant skeleton metadata is present in every treated row of matched_effects.
    skeleton_meta = (
        matched[["skeleton_id", "scale_class", "n_products"]]
        .drop_duplicates()
        .sort_values("skeleton_id")
    )
    duplicate_meta = skeleton_meta.groupby("skeleton_id").size()
    if (duplicate_meta > 1).any():
        fail("Matched-effect table contains inconsistent scale/product metadata within a skeleton")
    screening = screening.merge(skeleton_meta, on="skeleton_id", how="left", validate="many_to_one")
    if screening[["scale_class", "n_products"]].isna().any().any():
        fail("Could not reconstruct scale/product metadata for every screening instance")

    return {
        "screening": screening,
        "reference_200": reference_200,
        "reference_400": reference_400,
        "lp30": lp30,
        "matched": matched,
        "characterization": characterization,
        "horizon_summary": horizon_summary,
        "lp_summary": lp_summary,
        "pipeline": pipeline,
    }


def make_design_figure(screening: pd.DataFrame, characterization: dict, pipeline: dict) -> None:
    n_instances = len(screening)
    n_skeletons = screening["skeleton_id"].nunique()
    n_det = int((screening["track"] == "deterministic").sum())
    n_scen = int((screening["track"] == "scenario").sum())
    n_pairs = int(characterization.get("matched_effect_rows", 0))

    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.035, 0.58, 0.25, 0.28, "FAF5.7.1", "2030 domestic truck\nOD commodity flows"),
        (0.375, 0.58, 0.25, 0.28, "Empirical skeletons", f"{n_skeletons} matched blocks\ngeography + commodities fixed"),
        (0.715, 0.58, 0.25, 0.28, "Optimization tracks", f"Deterministic: {n_det}\nScenario: {n_scen}"),
        (0.075, 0.12, 0.405, 0.31, "One-factor treatments", "Network support · Cost variability\nCapacity tightness · Demand concentration\nFixed-charge intensity · Uncertainty amplitude"),
        (0.545, 0.12, 0.405, 0.31, "Validated benchmark evidence", f"{n_instances} valid instances · {n_pairs} matched comparisons\nLP horizons: 10/30 s\nMILP horizons: 60/200/400 s"),
    ]

    for x, y, w, h, title, body in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            linewidth=1.0, fill=False,
        )
        ax.add_patch(patch)
        title_size = 10.2 if title == "Validated benchmark evidence" else 11.0
        ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center", fontsize=title_size, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.35, body, ha="center", va="center", fontsize=8.8, linespacing=1.25)

    # Top-row flow.
    ax.add_patch(FancyArrowPatch((0.288, 0.72), (0.368, 0.72), arrowstyle="-|>", mutation_scale=14, linewidth=1.0))
    ax.add_patch(FancyArrowPatch((0.628, 0.72), (0.708, 0.72), arrowstyle="-|>", mutation_scale=14, linewidth=1.0))
    # Tracks feed the controlled treatments; treatments produce benchmark evidence.
    ax.add_patch(FancyArrowPatch((0.79, 0.57), (0.48, 0.39), arrowstyle="-|>", mutation_scale=14, linewidth=1.0, connectionstyle="arc3,rad=0.12"))
    ax.add_patch(FancyArrowPatch((0.482, 0.275), (0.542, 0.275), arrowstyle="-|>", mutation_scale=14, linewidth=1.0))
    save_figure(fig, "benchmark_design")

def factor_rows(screening: pd.DataFrame, factor: str) -> pd.DataFrame:
    d = screening[(screening["treatment_factor"] == "reference") | (screening["treatment_factor"] == factor)].copy()
    d["factor_level"] = np.where(d["treatment_factor"] == "reference", "reference", d["treatment_level"].astype(str))
    return d


def make_calibration_figure(screening: pd.DataFrame) -> pd.DataFrame:
    records: list[pd.DataFrame] = []

    d = factor_rows(screening, "network_support")
    piv = d.pivot_table(index=["skeleton_id", "track"], columns="factor_level", values="lane_count", aggfunc="first")
    if {"restricted", "reference", "expanded"}.issubset(piv.columns):
        denom = (piv["expanded"] - piv["restricted"]).replace(0, np.nan)
        pos = piv.sub(piv["restricted"], axis=0).div(denom, axis=0)
        long = pos[["restricted", "reference", "expanded"]].stack().rename("value").reset_index()
        long["factor"] = "Network support"
        long["metric"] = "Normalized optional-support position"
        records.append(long.rename(columns={"factor_level": "level"}))

    specs = [
        ("cost_variability", "cost_variability_sigma", "Cost variability", "Cost-noise σ"),
        ("capacity_tightness", "capacity_headroom_multiplier", "Capacity tightness", "Headroom multiplier"),
        ("demand_concentration", "demand_gini_base", "Demand concentration", "Base-demand Gini"),
        ("fixed_charge_intensity", "fixed_charge_intensity_target", "Fixed-charge intensity", "Intensity ratio"),
        ("uncertainty_amplitude", "uncertainty_amplitude", "Uncertainty amplitude", "Amplitude multiplier"),
    ]
    for factor, column, label, metric in specs:
        d = factor_rows(screening, factor)
        if factor == "uncertainty_amplitude":
            d = d[d["track"] == "scenario"].copy()
        block = d[["skeleton_id", "track", "factor_level", column]].rename(columns={"factor_level": "level", column: "value"})
        block["factor"] = label
        block["metric"] = metric
        records.append(block)

    cal = pd.concat(records, ignore_index=True)
    cal.to_csv(DERIVED / "control_calibration_points.csv", index=False)

    level_orders = {
        "Network support": ["restricted", "reference", "expanded"],
        "Cost variability": ["low", "reference", "high"],
        "Capacity tightness": ["tight", "reference", "loose"],
        "Demand concentration": ["diffuse", "reference", "concentrated"],
        "Fixed-charge intensity": ["low", "reference", "high"],
        "Uncertainty amplitude": ["low", "reference", "high"],
    }
    factors = list(level_orders)
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.35))
    axes = axes.ravel()
    rng = np.random.default_rng(20250925)

    for ax, factor in zip(axes, factors):
        g = cal[cal["factor"] == factor]
        order = level_orders[factor]
        for x, level in enumerate(order):
            vals = pd.to_numeric(g.loc[g["level"] == level, "value"], errors="coerce").dropna().to_numpy()
            if vals.size == 0:
                continue
            jitter = rng.uniform(-0.08, 0.08, size=vals.size)
            ax.scatter(np.full(vals.size, x) + jitter, vals, s=17, alpha=0.52)
            ax.plot([x - 0.16, x + 0.16], [vals.mean(), vals.mean()], linewidth=2.0)
        ax.set_xticks(range(len(order)), [fmt_level(x) for x in order], rotation=15 if factor in {"Network support", "Demand concentration"} else 0)
        ax.set_title(factor, fontsize=10.5, fontweight="bold")
        metric = g["metric"].iloc[0]
        ax.set_ylabel(metric, fontsize=9)
        ax.grid(axis="y", linewidth=0.4, alpha=0.35)
        ax.tick_params(labelsize=8.5)
    fig.tight_layout(pad=1.1)
    save_figure(fig, "control_calibration")
    return cal


def make_difficulty_figure(screening: pd.DataFrame) -> pd.DataFrame:
    counts = screening["difficulty_class"].value_counts().reindex(DIFFICULTY_ORDER, fill_value=0)
    table = pd.DataFrame({"difficulty": counts.index, "count": counts.values})
    table["percent"] = 100 * table["count"] / len(screening)
    table.to_csv(DERIVED / "difficulty_distribution.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    bars = ax.bar([fmt_level(x) for x in counts.index], counts.values)
    ax.set_ylabel("Instances")
    ax.set_ylim(0, max(counts.values) * 1.18)
    ax.grid(axis="y", linewidth=0.4, alpha=0.3)
    for bar, count, pct in zip(bars, table["count"], table["percent"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts.values)*0.025, f"{count}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    save_figure(fig, "difficulty_distribution")
    return table


def make_regime_heatmap(screening: pd.DataFrame, reference_200: pd.DataFrame, reference_400: pd.DataFrame) -> pd.DataFrame:
    optimal_any = set(screening.loc[screening["solution_status"].astype(str) == "optimal", "instance_id"])
    optimal_any |= set(reference_200.loc[reference_200["solution_status"].astype(str) == "optimal", "instance_id"])
    optimal_any |= set(reference_400.loc[reference_400["solution_status"].astype(str) == "optimal", "instance_id"])
    d = screening[["instance_id", "scale_class", "n_products"]].copy()
    d["optimal_60"] = screening["solution_status"].astype(str).eq("optimal").astype(int).to_numpy()
    d["optimal_400"] = d["instance_id"].isin(optimal_any).astype(int)
    reg = d.groupby(["scale_class", "n_products"], as_index=False).agg(n=("instance_id", "size"), optimal_60=("optimal_60", "mean"), optimal_400=("optimal_400", "mean"))
    reg["optimal_60_pct"] = 100 * reg["optimal_60"]
    reg["optimal_400_pct"] = 100 * reg["optimal_400"]
    reg.to_csv(DERIVED / "regime_optimality.csv", index=False)

    matrix = np.full((len(SCALE_ORDER), len(PRODUCT_ORDER)), np.nan)
    matrix60 = np.full_like(matrix, np.nan)
    for i, scale in enumerate(SCALE_ORDER):
        for j, products in enumerate(PRODUCT_ORDER):
            row = reg[(reg["scale_class"] == scale) & (reg["n_products"] == products)]
            if len(row):
                matrix[i, j] = row.iloc[0]["optimal_400_pct"]
                matrix60[i, j] = row.iloc[0]["optimal_60_pct"]

    fig, ax = plt.subplots(figsize=(6.7, 4.5))
    im = ax.imshow(matrix, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(PRODUCT_ORDER)), [str(x) for x in PRODUCT_ORDER])
    ax.set_yticks(range(len(SCALE_ORDER)), [SCALE_LABEL[x] for x in SCALE_ORDER])
    ax.set_xlabel("Commodity count")
    ax.set_ylabel("Scale class")
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("Certified optimal by 400 s (%)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                text_color = "white" if matrix[i, j] < 45 else "black"
                ax.text(j, i, f"{matrix[i,j]:.0f}%\n({matrix60[i,j]:.0f}% @60 s)", ha="center", va="center", fontsize=8.5, color=text_color)
    fig.tight_layout()
    save_figure(fig, "regime_heatmap")
    return reg


def make_horizon_figure(horizon_summary: dict, lp_summary: dict) -> pd.DataFrame:
    lp_rows = [
        {"problem": "LP", "horizon_seconds": 10, "certified": int(lp_summary["certified_optimal_at_10"]), "unresolved": int(lp_summary["valid_instances"] - lp_summary["certified_optimal_at_10"])},
        {"problem": "LP", "horizon_seconds": 30, "certified": int(lp_summary["certified_by_30"]), "unresolved": int(lp_summary["unresolved_after_30"])},
    ]
    milp_rows = [
        {"problem": "MILP", "horizon_seconds": 60, "certified": int(horizon_summary["certified_optimal_at_60"]), "unresolved": int(horizon_summary["unresolved_after_60"])},
        {"problem": "MILP", "horizon_seconds": 200, "certified": int(horizon_summary["certified_optimal_at_60"] + horizon_summary["certified_optimal_at_200_fresh_run"]), "unresolved": int(horizon_summary["unresolved_after_200"])},
        {"problem": "MILP", "horizon_seconds": 400, "certified": int(horizon_summary["certified_optimal_by_400_any_fresh_horizon"]), "unresolved": int(horizon_summary["unresolved_after_400"])},
    ]
    table = pd.DataFrame(lp_rows + milp_rows)
    table.to_csv(DERIVED / "horizon_persistence.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.15), sharey=True)
    for ax, problem, rows in [(axes[0], "LP relaxation", lp_rows), (axes[1], "MILP", milp_rows)]:
        x = np.arange(len(rows))
        certified = np.array([r["certified"] for r in rows])
        unresolved = np.array([r["unresolved"] for r in rows])
        ax.bar(x, certified, label="Certified optimal")
        ax.bar(x, unresolved, bottom=certified, label="Unresolved")
        ax.set_xticks(x, [f"{r['horizon_seconds']} s" for r in rows])
        ax.set_title(problem, fontsize=10.5, fontweight="bold")
        ax.grid(axis="y", linewidth=0.4, alpha=0.3)
        for xi, c, u in zip(x, certified, unresolved):
            ax.text(xi, c/2, str(c), ha="center", va="center", fontsize=9)
            if u:
                ax.text(xi, c + u/2, str(u), ha="center", va="center", fontsize=9)
    axes[0].set_ylabel("Instances")
    axes[1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save_figure(fig, "horizon_persistence")
    return table


def make_matched_effect_figure(screening: pd.DataFrame) -> pd.DataFrame:
    d = screening.copy()
    d["difficulty_score"] = d["difficulty_class"].map(DIFFICULTY_SCORE)
    d["optimal_60"] = d["solution_status"].astype(str).eq("optimal").astype(int)
    reference = d[d["treatment_factor"] == "reference"][
        ["skeleton_id", "track", "difficulty_score", "optimal_60"]
    ].rename(columns={"difficulty_score": "reference_difficulty", "optimal_60": "reference_optimal"})
    treated = d[d["treatment_factor"] != "reference"].merge(reference, on=["skeleton_id", "track"], how="left", validate="many_to_one")
    treated["delta_difficulty"] = treated["difficulty_score"] - treated["reference_difficulty"]
    treated["delta_optimal"] = treated["optimal_60"] - treated["reference_optimal"]

    rows = []
    groups = list(treated.groupby(["treatment_factor", "treatment_level"], sort=False))
    for seed, ((factor, level), g) in enumerate(groups, start=1):
        vals = pd.to_numeric(g["delta_difficulty"], errors="coerce").dropna().to_numpy()
        mean, lo, hi = bootstrap_mean_ci(vals, seed=20250925 + seed)
        rows.append({
            "treatment_factor": factor,
            "treatment_level": level,
            "n": len(g),
            "mean_delta_difficulty": mean,
            "ci95_low": lo,
            "ci95_high": hi,
            "easier_fraction": float((g["delta_difficulty"] < 0).mean()),
            "unchanged_fraction": float((g["delta_difficulty"] == 0).mean()),
            "harder_fraction": float((g["delta_difficulty"] > 0).mean()),
            "delta_optimal_rate": float(g["delta_optimal"].mean()),
        })
    summary = pd.DataFrame(rows).sort_values("mean_delta_difficulty").reset_index(drop=True)
    summary["label"] = summary["treatment_factor"].str.replace("_", " ").str.title() + " — " + summary["treatment_level"].map(fmt_level)
    summary.to_csv(DERIVED / "matched_effects_bootstrap.csv", index=False)

    y = np.arange(len(summary))
    mean = summary["mean_delta_difficulty"].to_numpy()
    lo = summary["ci95_low"].to_numpy()
    hi = summary["ci95_high"].to_numpy()
    xerr = np.vstack([mean - lo, hi - mean])

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    ax.errorbar(mean, y, xerr=xerr, fmt="o", capsize=3, linewidth=1.1)
    ax.axvline(0, linewidth=0.9)
    ax.set_yticks(y, summary["label"])
    ax.set_xlabel("Mean paired change in 60-s difficulty ordinal \n Points: mean within-skeleton changes \n Whiskers: deterministic 95% bootstrap confidence intervals")
    ax.grid(axis="x", linewidth=0.4, alpha=0.3)
    ax.invert_yaxis()
    # Append sample size to the right of each interval without encoding a second metric.
    right = np.nanmax(hi) if np.isfinite(hi).any() else 1.0
    pad = max(0.04, (np.nanmax(hi) - np.nanmin(lo)) * 0.03)
    for yi, upper, n in zip(y, hi, summary["n"]):
        ax.text(upper + pad, yi, f"n={int(n)}", va="center", fontsize=8)
    fig.tight_layout()
    save_figure(fig, "matched_effects_ci")
    return summary


def main() -> None:
    results = resolve_results_dir()
    print(f"Reading completed benchmark results from: {results}")
    x = load_inputs(results)
    screening = x["screening"]

    make_design_figure(screening, x["characterization"], x["pipeline"])
    calibration = make_calibration_figure(screening)
    difficulty = make_difficulty_figure(screening)
    regimes = make_regime_heatmap(screening, x["reference_200"], x["reference_400"])
    horizons = make_horizon_figure(x["horizon_summary"], x["lp_summary"])
    effects = make_matched_effect_figure(screening)

    summary = {
        "results_directory": str(results),
        "instances": int(len(screening)),
        "skeletons": int(screening["skeleton_id"].nunique()),
        "tracks": screening["track"].value_counts().to_dict(),
        "figures": [
            "benchmark_design", "control_calibration", "difficulty_distribution",
            "regime_heatmap", "horizon_persistence", "matched_effects_ci",
        ],
        "derived_rows": {
            "calibration": int(len(calibration)),
            "difficulty": int(len(difficulty)),
            "regimes": int(len(regimes)),
            "horizons": int(len(horizons)),
            "effects": int(len(effects)),
        },
    }
    (DERIVED / "figure_generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nGenerated figures:")
    for stem in summary["figures"]:
        print(f"  figures/{stem}.pdf")
        print(f"  figures/{stem}.png")
    print("\nDerived numerical tables were written to data/derived/.")


if __name__ == "__main__":
    main()
