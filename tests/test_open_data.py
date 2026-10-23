"""Public-source preservation, unit conversion, and adapter-boundary tests."""
import copy
import csv
from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest

from aidc.data import canonical_hash, validate_data
from aidc.datasets import build_scenario, verify_raw_files
from aidc.datasets.open_data import hourly_records
from aidc.experiments import tiny_data
from aidc.model import solve
from aidc.paths import RAW, ROOT


class OpenDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = build_scenario()
        with (RAW / "helios/saturn_vc_jobs.csv").open(encoding="utf-8", newline="") as handle:
            cls.raw_jobs = {r["job_id"]: r for r in csv.DictReader(handle)}
        with (RAW / "opsd/gb_hourly.csv").open(encoding="utf-8", newline="") as handle:
            cls.energy = list(csv.DictReader(handle))

    def test_all_shipped_source_checksums_match(self):
        manifest = verify_raw_files()
        self.assertEqual({r["id"] for r in manifest["sources"]}, {"helios", "opsd", "neso"})
        self.assertTrue(next(r for r in manifest["sources"] if r["id"] == "opsd")["matches_official_database_checksum"])

    def test_public_adapter_is_deterministic(self):
        self.assertEqual(canonical_hash(self.d), canonical_hash(build_scenario()))
        self.assertEqual(len(self.d["jobs"]), 26)
        self.assertEqual(self.d["gpu_capacity"], [192] * 96)

    def test_gpu_counts_and_execution_seconds_match_source(self):
        for job in self.d["jobs"]:
            raw = self.raw_jobs[job["source_job_id"]]
            self.assertEqual(job["gpus"], int(raw["gpu_num"]))
            self.assertEqual(job["source_duration_seconds"], int(raw["duration"]))
            self.assertEqual(job["source_submit_time"], raw["submit_time"])
            self.assertEqual(job["parents"], [])

    def test_rounding_is_upward_and_bounded(self):
        origin = datetime(2020, 6, 1)
        for job in self.d["jobs"]:
            submitted = datetime.fromisoformat(job["source_submit_time"])
            release_error = job["release"] * 900 - (submitted - origin).total_seconds()
            duration_error = job["duration"] * 900 - job["source_duration_seconds"]
            self.assertGreaterEqual(release_error, 0)
            self.assertLess(release_error, 900)
            self.assertGreaterEqual(duration_error, 0)
            self.assertLess(duration_error, 900)
        self.assertAlmostEqual(self.d["source_metadata"]["source_gpu_hours"], 693.8644444444444)
        self.assertEqual(self.d["source_metadata"]["discretized_gpu_hours"], 762.)

    def test_price_currency_and_mwh_units(self):
        self.assertAlmostEqual(self.d["price_cny_per_kwh"][0], 16.68 * 9 / 1000)
        self.assertEqual(len(set(self.d["price_cny_per_kwh"][:4])), 1)

    def test_carbon_boundary_and_grams_conversion(self):
        records = json.loads((RAW / "neso/intensity.json").read_text())["data"]
        row = next(r for r in records if r["from"] == "2020-06-01T00:00Z")
        self.assertEqual(self.d["carbon_kg_per_kwh"][:2], [row["intensity"]["actual"] / 1000] * 2)
        self.assertEqual(self.d["forecast_carbon_kg_per_kwh"][:2], [row["intensity"]["forecast"] / 1000] * 2)

    def test_solar_forecast_uses_only_the_previous_day(self):
        past = hourly_records(self.energy, "2020-05-31")
        expected = [float(r["GB_GBN_solar_generation_actual"]) * .004 for r in past for _ in range(4)]
        self.assertEqual(self.d["forecast_renewable_kw"], expected)
        self.assertNotEqual(self.d["forecast_renewable_kw"], self.d["renewable_kw"])

    def test_missing_or_duplicate_source_hours_are_rejected(self):
        with self.assertRaises(ValueError):
            hourly_records(self.energy[1:], "2020-05-31")
        with self.assertRaises(ValueError):
            hourly_records([self.energy[0]] + self.energy, "2020-05-31")

    def test_source_corruption_is_not_silently_regenerated(self):
        scratch = (ROOT / "tmp").resolve()
        self.assertTrue(scratch.is_relative_to(ROOT))
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="open-data-", dir=scratch) as folder:
            root = Path(folder).resolve()
            self.assertTrue(root.is_relative_to(scratch))
            (root / "manifest.json").write_text(json.dumps({"files": [{"path": "observation.csv", "sha256": "0" * 64}]}))
            (root / "observation.csv").write_text("corrupted")
            with self.assertRaisesRegex(ValueError, "checksum failure"):
                verify_raw_files(root)

    def test_negative_public_prices_are_preserved_and_bounded(self):
        d = tiny_data()
        d["price_cny_per_kwh"] = [-2., 1., 3., 2.]
        validate_data(d)
        result = solve(d, storage=False)
        self.assertTrue(result["feasible"])
        self.assertAlmostEqual(result["cost_cny"], -1.)


if __name__ == "__main__":
    unittest.main()
