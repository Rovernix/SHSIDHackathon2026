"""Regression cases for asset identity and fail-closed numerical auditing."""
import copy
import json
import unittest

import numpy as np

from aidc.audit import audit
from aidc.data import validate_data
from aidc.experiments import tiny_data


def storage_case():
    """One 1 kW battery shifts 2 kWh; the fixed facility load is 2 kW."""
    data = tiny_data()
    data.update(jobs=[], base_it_kw=2., price_cny_per_kwh=[1.]*4)
    data["batteries"] = [dict(name="B", capacity_kwh=2., min_kwh=0.,
        max_kwh=2., initial_kwh=0., charge_kw=1., discharge_kw=1.,
        eta_charge=1., eta_discharge=1., wear_cny_per_kwh=0.,
        reserve_kw=0., reserve_kwh=0.)]
    result = dict(starts={}, grid_kw=[3.,3.,1.,1.], renewable_used_kw=[0.]*4,
        facility_kw=[2.]*4, training_gpus=[0.]*4, peak_kw=3.,
        batteries=[dict(name="B", charge_kw=[1.,1.,0.,0.],
            discharge_kw=[0.,0.,1.,1.], energy_kwh=[0.,1.,2.,1.,0.])],
        cost_cny=8., carbon_kg=8.)
    return data, result


class AuditValidationTests(unittest.TestCase):
    def assert_rejected(self, report):
        self.assertFalse(report["passed"])
        self.assertTrue(report["errors"])
        # Failure evidence itself must be safe for the strict JSON writer.
        json.dumps(report, allow_nan=False)

    def test_duplicate_dispatch_cannot_double_one_batterys_rating(self):
        data, result = storage_case()
        result["batteries"].append(copy.deepcopy(result["batteries"][0]))
        result.update(grid_kw=[4.,4.,0.,0.], peak_kw=4.)
        # Every duplicated trace is individually valid and the forged AC bus
        # balances; only unique asset identity prevents the extra 1 kW.
        report = audit(data, result)
        self.assert_rejected(report)
        self.assertIn("duplicate_battery:B", report["errors"])

    def test_unknown_battery_is_rejected_without_keyerror(self):
        data, result = storage_case()
        result["batteries"][0]["name"] = "not-configured"
        report = audit(data, result)
        self.assert_rejected(report)
        self.assertIn("unknown_battery:not-configured", report["errors"])

    def test_valid_distinct_batteries_are_not_treated_as_duplicates(self):
        data, result = storage_case()
        other = copy.deepcopy(data["batteries"][0])
        other["name"] = "C"
        data["batteries"].append(other)
        dispatch = copy.deepcopy(result["batteries"][0])
        dispatch["name"] = "C"
        result["batteries"].append(dispatch)
        result.update(grid_kw=[4.,4.,0.,0.], peak_kw=4.)
        self.assertTrue(audit(data, result)["passed"])

    def test_valid_single_and_disabled_storage_remain_supported(self):
        data, result = storage_case()
        self.assertTrue(audit(data, result)["passed"])
        result.update(batteries=[], grid_kw=[2.]*4, peak_kw=2.)
        self.assertTrue(audit(data, result)["passed"])

    def test_invalid_time_steps_are_rejected_by_input_validation(self):
        for value in [float("nan"), float("inf"), -float("inf"), 0., -1.,
                      True, "0.25", [.25], complex(.25, 0.), 10**400]:
            with self.subTest(value=value):
                data, _ = storage_case()
                data["dt_hours"] = value
                with self.assertRaisesRegex(ValueError, "dt_hours"):
                    validate_data(data)

    def test_finite_positive_time_steps_are_accepted(self):
        for value in [.25, 1, np.float64(.5)]:
            with self.subTest(value=value):
                data, _ = storage_case()
                data["dt_hours"] = value
                validate_data(data)

    def test_audit_revalidates_invalid_time_steps(self):
        for value in [float("nan"), float("inf"), -float("inf"), 0., -1.]:
            with self.subTest(value=value):
                data, result = storage_case()
                data["dt_hours"] = value
                report = audit(data, result)
                self.assert_rejected(report)
                self.assertTrue(any(error.startswith("invalid_input:") for error in report["errors"]))

    def test_nested_and_scalar_time_series_fail_cleanly(self):
        for field in ["pue", "forecast_renewable_kw"]:
            for value in [1., [[1.]]*4]:
                with self.subTest(field=field, value=value):
                    data, result = storage_case()
                    data[field] = value
                    with self.assertRaises(ValueError):
                        validate_data(data)
                    self.assert_rejected(audit(data, result))

    def test_invalid_tolerance_cannot_disable_all_comparisons(self):
        data, result = storage_case()
        result["grid_kw"][0] += 10.
        for value in [float("nan"), float("inf"), -float("inf"), -1., "1", [1.], 10**400]:
            with self.subTest(value=value):
                self.assert_rejected(audit(data, result, tolerance=value))

    def test_nonfinite_budgets_are_rejected(self):
        for field in ["carbon_cap_kg", "cost_cap_cny"]:
            for value in [float("nan"), float("inf"), -float("inf")]:
                with self.subTest(field=field, value=value):
                    data, result = storage_case()
                    result[field] = value
                    self.assert_rejected(audit(data, result))

    def test_malformed_battery_records_fail_cleanly(self):
        for batteries in [None, {}, [None], [{}], [dict(name=[])]]:
            with self.subTest(batteries=batteries):
                data, result = storage_case()
                result["batteries"] = batteries
                self.assert_rejected(audit(data, result))
        data, result = storage_case()
        result["batteries"][0]["energy_kwh"] = ["invalid"]*5
        self.assert_rejected(audit(data, result))

    def test_overflowing_residuals_are_rejected_and_json_safe(self):
        data, result = storage_case()
        result.update(batteries=[], grid_kw=[2.]*4, peak_kw=2.)
        data["price_cny_per_kwh"] = [1e308, 1e308, -1e308, -1e308]
        report = audit(data, result)
        self.assert_rejected(report)
        self.assertIn("nonfinite_residual:cost_accounting", report["errors"])
        self.assertIsNone(report["residuals"]["cost_accounting"])

    def test_overflowing_metrics_fail_even_when_cost_and_carbon_are_zero(self):
        data, result = storage_case()
        data.update(dt_hours=1e308, price_cny_per_kwh=[0.]*4,
                    carbon_kg_per_kwh=[0.]*4)
        result.update(batteries=[], grid_kw=[2.]*4, peak_kw=2.,
                      cost_cny=0., carbon_kg=0.)
        # All constraint/accounting residuals are finite zero; total daily kWh
        # nevertheless overflows. A metrics-only failure must also reject.
        report = audit(data, result)
        self.assert_rejected(report)
        self.assertIn("nonfinite_metric:grid_kwh", report["errors"])
        self.assertIsNone(report["metrics"]["grid_kwh"])


if __name__ == "__main__":
    unittest.main()
