"""Build a deterministic trace-driven scenario from shipped public-data excerpts.

No random arrivals, GPU counts, durations, energy prices, carbon intensities,
or solar profiles are generated here. Deadlines, power, asset sizing, a fixed
currency conversion, and the counterfactual relocation remain explicit study
assumptions. See docs/research.md.
"""
from __future__ import annotations

import copy
import csv
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path

from ..data import validate_data
from ..paths import CONFIG, RAW


SLOT_SECONDS = 900
POOL = "Helios GPU-equivalent"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_raw_files(raw_root=None):
    root = Path(raw_root) if raw_root is not None else RAW
    manifest = read_json(root / "manifest.json")
    failures = []
    for row in manifest["files"]:
        path = root / row["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            failures.append(row["path"])
    if failures:
        raise ValueError("Public-data checksum failure: " + ", ".join(failures))
    return manifest


def csv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def trace_time(value):
    # The publisher does not specify a timezone. Retain its clock and rebase
    # within-day offsets onto the counterfactual British UTC scenario.
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def hourly_records(rows, day):
    out = [r for r in rows if r["utc_timestamp"].startswith(day + "T")]
    expected = [f"{day}T{h:02}:00:00Z" for h in range(24)]
    if [r["utc_timestamp"] for r in out] != expected:
        raise ValueError("Missing, duplicated, or unordered OPSD hour: " + day)
    for row in out:
        for key in ["GB_GBN_price_day_ahead", "GB_GBN_solar_generation_actual"]:
            if not row[key] or not math.isfinite(float(row[key])):
                raise ValueError("Missing/nonfinite public observation: " + key)
        if float(row["GB_GBN_solar_generation_actual"]) < 0:
            raise ValueError("Negative solar generation")
    return out


def background_reservation(rows, day):
    """Conservative slot maxima of recorded short-job GPU concurrency."""
    origin = datetime.fromisoformat(day)
    end_of_day = origin + timedelta(days=1)
    jobs = []
    for row in rows:
        duration = int(row["duration"])
        gpus = int(row["gpu_num"])
        if row["state"] != "COMPLETED" or not 0 < duration < 1800 or gpus <= 0:
            continue
        start, end = trace_time(row["start_time"]), trace_time(row["end_time"])
        if start < end_of_day and end > origin:
            jobs.append((start, end, gpus))
    reservations = []
    for slot in range(96):
        left = origin + timedelta(seconds=SLOT_SECONDS * slot)
        right = left + timedelta(seconds=SLOT_SECONDS)
        events = {}
        for start, end, gpus in jobs:
            if start < right and end > left:
                a, b = max(start, left), min(end, right)
                events[a] = events.get(a, 0) + gpus
                events[b] = events.get(b, 0) - gpus
        active = peak = 0
        for moment in sorted(events):
            active += events[moment]
            peak = max(peak, active)
        reservations.append(peak)
    return reservations


def build_scenario(day=None, *, raw_root=None, config=None):
    root = Path(raw_root) if raw_root is not None else RAW
    cfg = copy.deepcopy(config) if config is not None else read_json(CONFIG)
    day = day or cfg["main_date"]
    origin = datetime.fromisoformat(day)
    previous = (origin - timedelta(days=1)).date().isoformat()
    manifest = verify_raw_files(root)
    rows = csv_rows(root / "helios/saturn_vc_jobs.csv")
    if any(r["vc"] != cfg["helios_virtual_cluster"] for r in rows):
        raise ValueError("The raw virtual-cluster excerpt does not match the configuration")
    capacity_rows = csv_rows(root / "helios/saturn_vc_capacity.csv")
    capacity = next((int(r["gpu_count"]) for r in capacity_rows if r["date"] == day), None)
    if capacity is None or capacity <= 0:
        raise ValueError("Missing public VC allocation: " + day)

    cohort, exclusions = [], {}
    for row in rows:
        if not row["submit_time"].startswith(day + " "):
            continue
        reason = None
        seconds, gpus = int(row["duration"]), int(row["gpu_num"])
        if row["state"] != "COMPLETED":
            reason = "not_completed"
        elif gpus <= 0:
            reason = "not_a_gpu_job"
        elif not cfg["minimum_duration_seconds"] <= seconds <= cfg["maximum_duration_seconds"]:
            reason = "duration_outside_cohort"
        elif trace_time(row["submit_time"]).hour >= cfg["latest_submission_hour"]:
            reason = "submission_at_or_after_cutoff"
        elif gpus > capacity:
            reason = "request_exceeds_vc_allocation"
        if reason:
            exclusions[reason] = exclusions.get(reason, 0) + 1
        else:
            cohort.append(row)
    cohort.sort(key=lambda r: (r["submit_time"], int(r["job_id"])))
    selected = cohort[:cfg["max_jobs"]]
    if not selected:
        raise ValueError("The selected public cohort is empty: " + day)
    exclusions["beyond_fixed_sample_limit"] = len(cohort) - len(selected)
    jobs = []
    for row in selected:
        start, end = trace_time(row["start_time"]), trace_time(row["end_time"])
        seconds = int(row["duration"])
        if (end - start).total_seconds() != seconds:
            raise ValueError("Trace execution duration mismatch: " + row["job_id"])
        release = math.ceil((trace_time(row["submit_time"]) - origin).total_seconds() / SLOT_SECONDS)
        duration = math.ceil(seconds / SLOT_SECONDS)
        deadline = min(96, release + duration + int(cfg["assumed_max_wait_hours"] * 4))
        gpus = int(row["gpu_num"])
        jobs.append(dict(id="H" + row["job_id"], kind="trace_gpu_job", release=release,
                         deadline=deadline, duration=duration, gpus=gpus,
                         power_kw=gpus * cfg["assumed_job_incremental_kw_per_gpu"],
                         parents=[], max_wait=deadline-release-duration, hardware_pool=POOL,
                         source_job_id=row["job_id"], source_csv_row=int(row["source_row"]),
                         source_submit_time=row["submit_time"], source_start_time=row["start_time"],
                         source_end_time=row["end_time"], source_duration_seconds=seconds))

    energy = csv_rows(root / "opsd/gb_hourly.csv")
    current, past = hourly_records(energy, day), hourly_records(energy, previous)
    price = [float(r["GB_GBN_price_day_ahead"]) * cfg["assumed_cny_per_gbp"] / 1000 for r in current]
    scale = cfg["assumed_renewable_scale_kw_per_mw"]
    renewable = [float(r["GB_GBN_solar_generation_actual"]) * scale for r in current]
    forecast = [float(r["GB_GBN_solar_generation_actual"]) * scale for r in past]
    intensity_rows = read_json(root / "neso/intensity.json")["data"]
    carbon, carbon_forecast = [], []
    for half_hour in range(48):
        key = (origin + timedelta(minutes=30 * half_hour)).strftime("%Y-%m-%dT%H:%MZ")
        matches = [r for r in intensity_rows if r["from"] == key]
        if len(matches) != 1 or any(matches[0]["intensity"][k] is None for k in ["actual", "forecast"]):
            raise ValueError("Missing or duplicated NESO interval: " + key)
        carbon += [float(matches[0]["intensity"]["actual"]) / 1000] * 2
        carbon_forecast += [float(matches[0]["intensity"]["forecast"]) / 1000] * 2
    repeat = lambda values: [x for x in values for _ in range(4)]
    reservation = background_reservation(rows, day)
    prior_reservation = background_reservation(rows, previous)
    d = dict(schema_version=2, provenance="public-trace-driven composite benchmark; explicit asset and service assumptions",
             scenario_date_utc=day, hardware_pool=POOL, T=96, dt_hours=.25,
             gpu_capacity=[capacity]*96, base_it_kw=cfg["assumed_base_it_kw"],
             inference_kw_per_gpu=cfg["assumed_background_kw_per_gpu"],
             pue=[cfg["assumed_pue"]]*96, facility_cap_kw=cfg["assumed_facility_cap_kw"],
             grid_cap_kw=[cfg["assumed_grid_cap_kw"]]*96,
             grid_ramp_kw_per_step=cfg["assumed_grid_ramp_kw_per_step"],
             demand_rate_cny_per_kw_day=cfg["assumed_demand_cny_per_kw_day"],
             renewable_cost_cny_per_kwh=cfg["assumed_renewable_cny_per_kwh"],
             price_cny_per_kwh=repeat(price), carbon_kg_per_kwh=carbon,
             forecast_carbon_kg_per_kwh=carbon_forecast,
             renewable_kw=repeat(renewable), forecast_renewable_kw=repeat(forecast),
             inference_gpus=reservation, forecast_inference_gpus=prior_reservation,
             reserved_workload_kind="recorded short GPU jobs; not labeled inference by the publisher",
             jobs=jobs, batteries=cfg["batteries"],
             source_metadata=dict(helios_cluster=cfg["helios_cluster"], virtual_cluster=cfg["helios_virtual_cluster"],
                 eligible_jobs=len(cohort), selected_jobs=len(jobs), exclusions=exclusions,
                 source_gpu_hours=sum(int(r["gpu_num"])*int(r["duration"])/3600 for r in selected),
                 discretized_gpu_hours=sum(j["gpus"]*j["duration"]*.25 for j in jobs),
                 selection="First at most 32 eligible jobs by submission time then source ID; fixed before solving",
                 rounding="Release and execution duration rounded upward to 15 minutes; no interpolation of observations",
                 geography="British electricity/solar/carbon; Chinese GPU workload relocated as a counterfactual using source clock offsets, whose timezone is unspecified",
                 forecast="Previous-day persistence for solar and short-job reservations; archived NESO carbon forecast is not a verified day-ahead vintage",
                 manifest_sha256=hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest(),
                 sources=[s["id"] for s in manifest["sources"]]),
             study_assumptions={k: v for k, v in cfg.items() if k.startswith("assumed_")})
    validate_data(d)
    return d
