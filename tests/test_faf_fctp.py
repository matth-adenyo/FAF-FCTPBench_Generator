from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

from faf_fctp.catalogue import CandidateSpecification, generate_candidate_catalogue, load_design, validate_matched_catalogue
from faf_fctp.construction import build_empirical_skeleton, build_fctp_instance, validate_matched_family
from faf_fctp.distribution import package_benchmark
from faf_fctp.empirical import verify_faf_source
from faf_fctp.instance import FCTPInstance
from faf_fctp.optimization import solve_extensive_form
from faf_fctp.pipeline import _instance_reusable
from faf_fctp.runstate import ProgressReporter
from faf_fctp.relaxation_followup import eligible_followup_ids, build_relaxation_horizon_table
from faf_fctp.validation import shared_capacity_feasibility_certificate, validate_instance, validate_instance_directory

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_design(ROOT / "configs" / "faf_fctp_design.json")


def synthetic_flows(products: int = 3, zones: int = 6) -> pd.DataFrame:
    rows = []
    for p in range(1, products + 1):
        for i in range(1, zones + 1):
            for j in range(1, zones + 1):
                base = 14.0 + (i + 1) * (j + 2) + 2.5 * p
                distance = 90.0 + 11 * i + 7 * j
                rows.append(
                    {
                        "source_id": f"O{i:03d}",
                        "destination_id": f"D{j:03d}",
                        "product_id": f"SCTG{p:02d}",
                        "tons_low_ktons": base * (0.78 + 0.01 * ((i + j) % 4)),
                        "tons_base_ktons": base,
                        "tons_high_ktons": base * (1.18 + 0.01 * ((i + 2*j) % 4)),
                        "ton_miles_base_millions": base * distance / 1000.0,
                    }
                )
    frame = pd.DataFrame(rows)
    frame.attrs.update(
        source_filename="synthetic-test-source",
        source_sha256="test-sha256",
        forecast_year=2030,
        domestic_mode="1",
        trade_type="1",
        zone_names={},
        product_names={},
    )
    return frame


def base_spec(track: str = "scenario", products: int = 2, seed: int = 12345) -> CandidateSpecification:
    return CandidateSpecification(
        instance_id="TEST-REF",
        skeleton_id="SK-TEST",
        track=track,
        treatment_factor="reference",
        treatment_level="reference",
        scale_class="test",
        product_regime="test",
        replicate=1,
        n_sources=5,
        n_destinations=5,
        n_products=products,
        skeleton_seed=seed,
        network_support="reference",
        cost_variability="reference",
        capacity_tightness="reference",
        demand_concentration="reference",
        fixed_charge_intensity="reference",
        uncertainty_amplitude="reference",
    )


def variant(spec: CandidateSpecification, factor: str, level: str, number: int) -> CandidateSpecification:
    values = spec.__dict__.copy()
    values["instance_id"] = f"TEST-{number:02d}"
    values["treatment_factor"] = factor
    values["treatment_level"] = level
    values[factor] = level
    return CandidateSpecification(**values)


class CatalogueTests(unittest.TestCase):
    def test_default_catalogue_is_matched_and_has_both_tracks(self):
        catalogue = generate_candidate_catalogue(CONFIG)
        report = validate_matched_catalogue(catalogue, CONFIG)
        self.assertEqual(report["n_instances"], 576)
        self.assertEqual(report["n_skeletons"], 24)
        self.assertEqual(report["deterministic_instances"], 264)
        self.assertEqual(report["scenario_instances"], 312)
        self.assertTrue(catalogue["instance_id"].is_unique)


    def test_default_solver_horizons_are_ordered_and_locked(self):
        protocol = CONFIG["solver_protocol"]
        self.assertEqual(float(protocol["relaxation_seconds"]), 10.0)
        self.assertEqual(float(protocol["screening_milp_seconds"]), 60.0)
        self.assertEqual(float(protocol["reference_milp_seconds"]), 200.0)
        self.assertEqual(float(protocol["extended_milp_seconds"]), 400.0)

    def test_each_nonreference_row_changes_one_factor(self):
        catalogue = generate_candidate_catalogue(CONFIG)
        factors = ["network_support", "cost_variability", "capacity_tightness", "demand_concentration", "fixed_charge_intensity", "uncertainty_amplitude"]
        for (_, _), group in catalogue.groupby(["skeleton_id", "track"]):
            reference = group.loc[group["treatment_factor"] == "reference"].iloc[0]
            for _, row in group.loc[group["treatment_factor"] != "reference"].iterrows():
                changed = [factor for factor in factors if row[factor] != reference[factor]]
                self.assertEqual(changed, [row["treatment_factor"]])


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flows = synthetic_flows()

    def _family(self, track: str):
        reference_spec = base_spec(track=track)
        skeleton = build_empirical_skeleton(self.flows, reference_spec, CONFIG)
        family = [build_fctp_instance(self.flows, reference_spec, CONFIG, skeleton=skeleton)]
        applicable = ["network_support", "cost_variability", "capacity_tightness", "demand_concentration", "fixed_charge_intensity"]
        if track == "scenario":
            applicable.append("uncertainty_amplitude")
        number = 1
        for factor in applicable:
            for level in CONFIG["factors"][factor]:
                if level == "reference":
                    continue
                family.append(build_fctp_instance(self.flows, variant(reference_spec, factor, level, number), CONFIG, skeleton=skeleton))
                number += 1
        return skeleton, family

    def test_deterministic_family_is_first_class_and_feasible(self):
        _, family = self._family("deterministic")
        self.assertEqual(len(family), 11)
        for instance in family:
            self.assertEqual(instance.scenarios["scenario_id"].tolist(), ["base"])
            report = validate_instance(instance, check_shared_capacity=True, capacity_check_time_limit_seconds=5)
            self.assertTrue(report["valid"], report["errors"])
        self.assertTrue(validate_matched_family(family)["valid"])

    def test_scenario_family_is_first_class_and_feasible(self):
        _, family = self._family("scenario")
        self.assertEqual(len(family), 13)
        for instance in family:
            self.assertEqual(set(instance.scenarios["scenario_id"]), {"low", "base", "high"})
            report = validate_instance(instance, check_shared_capacity=True, capacity_check_time_limit_seconds=5)
            self.assertTrue(report["valid"], report["errors"])
        self.assertTrue(validate_matched_family(family)["valid"])

    def test_control_levels_are_realized_in_expected_direction(self):
        spec = base_spec(track="scenario")
        skeleton = build_empirical_skeleton(self.flows, spec, CONFIG)
        reference = build_fctp_instance(self.flows, spec, CONFIG, skeleton=skeleton)
        restricted = build_fctp_instance(self.flows, variant(spec, "network_support", "restricted", 1), CONFIG, skeleton=skeleton)
        expanded = build_fctp_instance(self.flows, variant(spec, "network_support", "expanded", 2), CONFIG, skeleton=skeleton)
        self.assertLessEqual(len(restricted.lanes), len(reference.lanes))
        self.assertLessEqual(len(reference.lanes), len(expanded.lanes))

        diffuse = build_fctp_instance(self.flows, variant(spec, "demand_concentration", "diffuse", 3), CONFIG, skeleton=skeleton)
        concentrated = build_fctp_instance(self.flows, variant(spec, "demand_concentration", "concentrated", 4), CONFIG, skeleton=skeleton)
        self.assertLess(diffuse.metadata["realized"]["demand_gini_base"], reference.metadata["realized"]["demand_gini_base"])
        self.assertLess(reference.metadata["realized"]["demand_gini_base"], concentrated.metadata["realized"]["demand_gini_base"])

        loose = build_fctp_instance(self.flows, variant(spec, "capacity_tightness", "loose", 5), CONFIG, skeleton=skeleton)
        tight = build_fctp_instance(self.flows, variant(spec, "capacity_tightness", "tight", 6), CONFIG, skeleton=skeleton)
        self.assertGreater(loose.lanes["capacity_ktons"].sum(), reference.lanes["capacity_ktons"].sum())
        self.assertGreater(reference.lanes["capacity_ktons"].sum(), tight.lanes["capacity_ktons"].sum())

        low_fixed = build_fctp_instance(self.flows, variant(spec, "fixed_charge_intensity", "low", 7), CONFIG, skeleton=skeleton)
        high_fixed = build_fctp_instance(self.flows, variant(spec, "fixed_charge_intensity", "high", 8), CONFIG, skeleton=skeleton)
        self.assertLess(low_fixed.lanes["fixed_cost_usd"].sum(), reference.lanes["fixed_cost_usd"].sum())
        self.assertLess(reference.lanes["fixed_cost_usd"].sum(), high_fixed.lanes["fixed_cost_usd"].sum())

        low_unc = build_fctp_instance(self.flows, variant(spec, "uncertainty_amplitude", "low", 9), CONFIG, skeleton=skeleton)
        high_unc = build_fctp_instance(self.flows, variant(spec, "uncertainty_amplitude", "high", 10), CONFIG, skeleton=skeleton)
        def span(instance):
            totals = instance.demands.groupby("scenario_id")["demand_ktons"].sum()
            return float(totals["high"] - totals["low"])
        self.assertLess(span(low_unc), span(reference))
        self.assertLess(span(reference), span(high_unc))


    def test_scenario_concentration_preserves_componentwise_ordering(self):
        spec = base_spec(track="scenario")
        skeleton = build_empirical_skeleton(self.flows, spec, CONFIG)
        concentrated = build_fctp_instance(
            self.flows, variant(spec, "demand_concentration", "concentrated", 99), CONFIG, skeleton=skeleton
        )
        for table, keys, value in (
            (concentrated.supplies, ["source_id", "product_id"], "supply_ktons"),
            (concentrated.demands, ["destination_id", "product_id"], "demand_ktons"),
        ):
            pivot = table.pivot_table(index=keys, columns="scenario_id", values=value, aggfunc="sum", fill_value=0.0)
            self.assertTrue(np.all(pivot["low"].to_numpy() <= pivot["base"].to_numpy() + 1e-8))
            self.assertTrue(np.all(pivot["base"].to_numpy() <= pivot["high"].to_numpy() + 1e-8))

    def test_constructive_capacity_certificate_covers_tight_variant(self):
        spec = base_spec(track="scenario")
        skeleton = build_empirical_skeleton(self.flows, spec, CONFIG)
        tight = build_fctp_instance(self.flows, variant(spec, "capacity_tightness", "tight", 100), CONFIG, skeleton=skeleton)
        self.assertTrue(np.all(tight.lanes["capacity_ktons"].to_numpy() + 1e-8 >= tight.lanes["stress_load_ktons"].to_numpy()))
        report = validate_instance(tight, check_shared_capacity=False)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["feasibility_status"], "constructively_certified")

    def test_independent_capacity_timeout_is_not_infeasibility(self):
        spec = base_spec(track="deterministic", products=1, seed=818)
        skeleton = build_empirical_skeleton(self.flows, spec, CONFIG)
        instance = build_fctp_instance(self.flows, spec, CONFIG, skeleton=skeleton)

        class FakeResult:
            success = False
            status = 1
            message = "Time limit reached."

        with patch("faf_fctp.validation.linprog", return_value=FakeResult()):
            certificate = shared_capacity_feasibility_certificate(instance, time_limit_seconds=0.01)
            self.assertEqual(certificate["status"], "certificate_timeout")
            report = validate_instance(instance, check_shared_capacity=True, capacity_check_time_limit_seconds=0.01)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["feasibility_status"], "certificate_timeout")
        self.assertTrue(report["warnings"])

    def test_reference_solver_handles_both_tracks(self):
        for track in ("deterministic", "scenario"):
            spec = base_spec(track=track, products=1, seed=777)
            skeleton = build_empirical_skeleton(self.flows, spec, CONFIG)
            instance = build_fctp_instance(self.flows, spec, CONFIG, skeleton=skeleton)
            result = solve_extensive_form(instance, time_limit_seconds=5, mip_relative_gap=1e-4)
            self.assertIn(result.status, {"optimal", "limit_reached"})
            self.assertIsNotNone(result.objective_usd)


class ProvenanceAndStorageTests(unittest.TestCase):
    def test_source_sha256_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / CONFIG["faf_source"]["archive_name"]
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("dummy.csv", "x\n1\n")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_faf_source(source, CONFIG)

    @unittest.skipUnless(importlib.util.find_spec("pyarrow") is not None, "pyarrow is not installed")
    def test_compact_parquet_storage_roundtrip(self):
        flows = synthetic_flows(products=1)
        spec = base_spec(track="deterministic", products=1)
        skeleton = build_empirical_skeleton(flows, spec, CONFIG)
        instance = build_fctp_instance(flows, spec, CONFIG, skeleton=skeleton)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "instance"
            instance.write(directory, compression_level=9, dictionary_encoding=True)
            report = validate_instance_directory(directory, check_shared_capacity=True)
            self.assertTrue(report["valid"], report["errors"])
            stored = FCTPInstance.read(directory)
            self.assertEqual(stored.dimensions, instance.dimensions)

    def test_packaging_does_not_recompress_parquet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instances = root / "instances" / "A"; instances.mkdir(parents=True)
            (instances / "lanes.parquet").write_bytes(b"already-compressed-parquet-bytes")
            (instances / "metadata.json").write_text("{}")
            catalogue = root / "catalogue.parquet"; catalogue.write_bytes(b"catalogue")
            archive = package_benchmark(root / "instances", catalogue, root / "FAF-FCTPBench.zip")
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(z.getinfo("instances/A/lanes.parquet").compress_type, zipfile.ZIP_STORED)
                self.assertEqual(z.getinfo("instances/A/metadata.json").compress_type, zipfile.ZIP_DEFLATED)



class RelaxationFollowupTests(unittest.TestCase):
    def test_followup_selects_only_valid_uncertified_lp_rows(self):
        screening = pd.DataFrame([
            {"instance_id": "A", "valid": True, "relaxation_status": "optimal"},
            {"instance_id": "B", "valid": True, "relaxation_status": "limit_reached"},
            {"instance_id": "C", "valid": False, "relaxation_status": "limit_reached"},
            {"instance_id": "D", "valid": True, "relaxation_status": "solver_error"},
        ])
        self.assertEqual(eligible_followup_ids(screening), ["B", "D"])

    def test_combined_root_gap_uses_only_certified_lp_optimum(self):
        screening = pd.DataFrame([
            {
                "instance_id": "A", "valid": True, "track": "deterministic",
                "treatment_factor": "reference", "treatment_level": "reference",
                "relaxation_status": "optimal", "relaxation_runtime_seconds": 1.0,
                "relaxation_objective_usd": 80.0, "objective_usd": 100.0,
            },
            {
                "instance_id": "B", "valid": True, "track": "scenario",
                "treatment_factor": "reference", "treatment_level": "reference",
                "relaxation_status": "limit_reached", "relaxation_runtime_seconds": 10.0,
                "relaxation_objective_usd": 95.0, "objective_usd": 100.0,
            },
            {
                "instance_id": "C", "valid": True, "track": "scenario",
                "treatment_factor": "reference", "treatment_level": "reference",
                "relaxation_status": "limit_reached", "relaxation_runtime_seconds": 10.0,
                "relaxation_objective_usd": 96.0, "objective_usd": 100.0,
            },
        ])
        followup = pd.DataFrame([
            {"instance_id": "B", "relaxation_status": "optimal", "relaxation_runtime_seconds": 18.0,
             "relaxation_objective_usd": 82.0, "certified_relaxation_objective_usd": 82.0},
            {"instance_id": "C", "relaxation_status": "limit_reached", "relaxation_runtime_seconds": 30.0,
             "relaxation_objective_usd": 90.0, "certified_relaxation_objective_usd": None},
        ])
        table = build_relaxation_horizon_table(screening, followup).set_index("instance_id")
        self.assertAlmostEqual(float(table.loc["A", "certified_root_gap"]), 0.20)
        self.assertEqual(int(table.loc["A", "certified_lp_horizon_seconds"]), 10)
        self.assertAlmostEqual(float(table.loc["B", "certified_root_gap"]), 0.18)
        self.assertEqual(int(table.loc["B", "certified_lp_horizon_seconds"]), 30)
        self.assertTrue(pd.isna(table.loc["C", "certified_root_gap"]))
        self.assertTrue(bool(table.loc["C", "lp_unresolved_after_30"]))



class ResumeAndProgressTests(unittest.TestCase):
    def test_instance_resume_requires_matching_generation_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "A"
            directory.mkdir()
            (directory / "metadata.json").write_text('{"generation_signature":"abc"}')
            (directory / "validation.json").write_text('{"valid":true}')
            (directory / "storage_manifest.json").write_text('{}')
            self.assertTrue(_instance_reusable(directory, "abc"))
            self.assertFalse(_instance_reusable(directory, "different"))

    def test_progress_status_is_persistent_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results" / "run_status.json"
            reporter = ProgressReporter(path)
            reporter.begin("screening-60s", 10)
            reporter.update(4, current="FAF-FCTP-0004", message="status=limit_reached")
            payload = __import__("json").loads(path.read_text())
            self.assertEqual(payload["stage"], "screening-60s")
            self.assertEqual(payload["completed"], 4)
            self.assertEqual(payload["total"], 10)
            self.assertEqual(payload["current"], "FAF-FCTP-0004")

if __name__ == "__main__":
    unittest.main()
