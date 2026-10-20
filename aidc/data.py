"""Schema validation, accounting utilities, and synthetic unit-test fixtures.

The public benchmark is built by aidc.datasets.open_data. generate() is retained
for artificial correctness tests and deliberately constructed toy examples.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np


def generate(seed: int = 2026, jobs: int = 28) -> dict:
    rng = np.random.default_rng(seed)
    T, dt = 96, 0.25
    h = np.arange(T) * dt
    price = np.select([h < 7, h < 10, h < 16, h < 21], [.32, .78, .68, 1.28], .55)
    carbon = .58 + .10 * np.cos((h - 3) * np.pi / 12) - .16 * np.exp(-((h - 13) / 3.5) ** 2)
    renewable_forecast = 440 * np.maximum(0, np.sin((h - 6) * np.pi / 12))
    inf_forecast = 128 + 48 * ((h >= 8) & (h < 22)) + 48 * ((h >= 17) & (h < 21))
    # Actual traces differ from the published day-ahead forecasts. No controller may
    # read actual future values. These are scenario observations, not measurements.
    clouds = np.clip(.94 + rng.normal(0, .07, T), .65, 1.12)
    renewable = renewable_forecast * clouds
    inf = (np.ceil((inf_forecast + rng.integers(-8, 17, T)) / 8) * 8).astype(int)
    workload = []
    for i in range(jobs):
        r = int(rng.choice([0, 8, 20, 28, 36, 44, 52, 60]) + rng.integers(0, 4))
        L = int(rng.choice([4, 6, 8, 10, 12]))
        g = int(rng.choice([64, 96, 128, 192, 256]))
        slack = int(rng.choice([12, 20, 28, 36]))
        d = min(T, r + L + slack)
        batch = (i % 4 == 0)
        workload.append(dict(id=f"J{i+1:02}", release=r, deadline=d, duration=L,
                             gpus=g, power_kw=round(g * (.57 if batch else .68), 3),
                             kind="batch" if batch else "training", parents=[],
                             max_wait=d-r-L, hardware_pool="H100-like"))
    # Four real precedence edges, selected so a feasible precedence window exists.
    for child in range(4, min(jobs, 20), 5):
        parent = child - 4
        workload[child]["parents"] = [workload[parent]["id"]]
        earliest = workload[parent]["release"] + workload[parent]["duration"]
        workload[child]["release"] = max(workload[child]["release"], earliest)
        workload[child]["deadline"] = min(T, max(workload[child]["deadline"],
            workload[child]["release"] + workload[child]["duration"] + 16))
        workload[child]["max_wait"] = workload[child]["deadline"] - workload[child]["release"] - workload[child]["duration"]
    return dict(schema_version=1, provenance="synthetic; not a real data-center trace",
        seed=seed, T=T, dt_hours=dt, gpu_capacity=[1024]*T, base_it_kw=180.,
        inference_kw_per_gpu=.50, pue=[1.15]*T, facility_cap_kw=1150.,
        grid_cap_kw=[1000.]*T, grid_ramp_kw_per_step=350.,
        demand_rate_cny_per_kw_day=2., renewable_cost_cny_per_kwh=.06,
        # These are scaled study assets, NOT specifications of branded products.
        batteries=[dict(name="LFP", capacity_kwh=800., min_kwh=80., max_kwh=720.,
                        initial_kwh=400., charge_kw=200., discharge_kw=200.,
                        eta_charge=.96, eta_discharge=.96, wear_cny_per_kwh=.035,
                        reserve_kwh=0., reserve_kw=0.),
                   dict(name="Na-ion", capacity_kwh=120., min_kwh=12., max_kwh=108.,
                        initial_kwh=60., charge_kw=120., discharge_kw=120.,
                        eta_charge=.94, eta_discharge=.94, wear_cny_per_kwh=.055,
                        reserve_kwh=24., reserve_kw=40.)],
        price_cny_per_kwh=price.tolist(), carbon_kg_per_kwh=carbon.tolist(),
        renewable_kw=renewable.tolist(), inference_gpus=inf.tolist(),
        forecast_renewable_kw=renewable_forecast.tolist(),
        forecast_inference_gpus=inf_forecast.tolist(), jobs=workload)


def validate_data(d: dict) -> None:
    T = d["T"]
    if not isinstance(T, int) or T <= 0 or d["dt_hours"] <= 0:
        raise ValueError("T and dt_hours must be positive")
    for field in ["gpu_capacity", "pue", "grid_cap_kw", "price_cny_per_kwh",
                  "carbon_kg_per_kwh", "renewable_kw", "inference_gpus"]:
        a = np.asarray(d[field], dtype=float)
        if len(a) != T or not np.all(np.isfinite(a)) or (field != "price_cny_per_kwh" and np.any(a < 0)):
            raise ValueError(f"Invalid time series: {field}")
    for field in ["forecast_renewable_kw","forecast_inference_gpus","forecast_carbon_kg_per_kwh"]:
        if field in d:
            a=np.asarray(d[field],dtype=float)
            if len(a)!=T or not np.all(np.isfinite(a)) or np.any(a<0):
                raise ValueError(f"Invalid forecast: {field}")
    if np.any(np.asarray(d["pue"])<1):
        raise ValueError("PUE cannot be below one")
    for field in ["gpu_capacity","inference_gpus"]:
        a=np.asarray(d[field],dtype=float)
        if np.any(a!=np.rint(a)):
            raise ValueError("GPU counts must be integer")
    for field in ["base_it_kw","inference_kw_per_gpu","facility_cap_kw",
                  "demand_rate_cny_per_kw_day","renewable_cost_cny_per_kwh"]:
        if not np.isfinite(d[field]) or d[field]<0:
            raise ValueError(f"Invalid nonnegative parameter: {field}")
    if d.get("grid_ramp_kw_per_step") is not None and (not np.isfinite(d["grid_ramp_kw_per_step"]) or d["grid_ramp_kw_per_step"]<0):
        raise ValueError("Invalid grid ramp")
    if np.any(np.asarray(d["inference_gpus"]) > np.asarray(d["gpu_capacity"])):
        raise ValueError("Inference reservation exceeds physical GPU capacity")
    ids = [j["id"] for j in d["jobs"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate job ids")
    by_id = {j["id"]: j for j in d["jobs"]}
    visiting, visited = set(), set()
    def visit(jid):
        if jid in visiting:
            raise ValueError("Dependency cycle")
        if jid in visited:
            return
        visiting.add(jid)
        for parent in by_id[jid]["parents"]:
            if parent not in by_id:
                raise ValueError(f"Unknown dependency: {parent}")
            visit(parent)
        visiting.remove(jid)
        visited.add(jid)
    for j in d["jobs"]:
        vals = [j[k] for k in ("release", "deadline", "duration", "gpus", "max_wait")]
        if any(not isinstance(v, int) for v in vals):
            raise ValueError("Job timing and GPU counts must be integer")
        if j["duration"] <= 0 or j["gpus"] <= 0 or not np.isfinite(j["power_kw"]) or j["power_kw"] < 0 or j["max_wait"] < 0:
            raise ValueError("Invalid job demand")
        if not 0 <= j["release"] < j["deadline"] <= T or j["release"] + j["duration"] > j["deadline"]:
            raise ValueError(f"Empty feasible start window: {j['id']}")
        if j["hardware_pool"] != d.get("hardware_pool", "H100-like"):
            raise ValueError("Jobs must match the explicitly homogeneous modeled GPU pool")
        visit(j["id"])
    if len(set(b["name"] for b in d["batteries"]))!=len(d["batteries"]):
        raise ValueError("Duplicate battery names")
    for b in d["batteries"]:
        for field in ["capacity_kwh","min_kwh","max_kwh","initial_kwh","charge_kw","discharge_kw","wear_cny_per_kwh","reserve_kwh","reserve_kw"]:
            if not np.isfinite(b[field]) or b[field]<0:
                raise ValueError(f"Invalid battery parameter: {field}")
        if not (0 < b["eta_charge"] <= 1 and 0 < b["eta_discharge"] <= 1):
            raise ValueError("Battery efficiency outside (0,1]")
        if not (0 <= b["min_kwh"] <= b["min_kwh"] + b["reserve_kwh"] <= b["initial_kwh"] <= b["max_kwh"] <= b["capacity_kwh"]):
            raise ValueError("Invalid battery energy / reserve limits")
        if not 0 <= b["reserve_kw"] <= min(b["charge_kw"], b["discharge_kw"]):
            raise ValueError("Invalid battery power reserve")


def canonical_hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save_json(path: Path | str, d) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def scenario(data: dict, name: str) -> dict:
    d = copy.deepcopy(data)
    d["constructed_stress_scenario"] = name
    if name == "cloudy":
        d["renewable_kw"] = (np.asarray(d["renewable_kw"]) * .35).tolist()
    elif name == "no_renewables":
        d["renewable_kw"] = [0.]*d["T"]
    elif name == "flat_price":
        d["price_cny_per_kwh"] = [float(np.mean(d["price_cny_per_kwh"]))]*d["T"]
    elif name == "price_carbon_conflict":
        prices = np.asarray(d["price_cny_per_kwh"])
        d["carbon_kg_per_kwh"] = (.30 + .45*(prices < np.median(prices))).tolist()
    elif name == "hot_day":
        d["pue"] = [1.40]*d["T"]
    elif name == "inference_surge":
        extra = max(1, int(min(d["gpu_capacity"]) * .25))
        d["inference_gpus"] = (np.asarray(d["inference_gpus"]) + extra).tolist()
    elif name == "grid_derating":
        d["grid_cap_kw"] = (np.asarray(d["grid_cap_kw"]) * .5).tolist()
    elif name == "zero_flexibility":
        for j in d["jobs"]:
            j["deadline"] = j["release"] + j["duration"]
            j["max_wait"] = 0
    elif name == "storage_expensive":
        for b in d["batteries"]:
            b["wear_cny_per_kwh"] = .65
    elif name == "no_demand_charge":
        d["demand_rate_cny_per_kw_day"] = 0.
    elif name == "high_workload":
        for j in d["jobs"]:
            j["gpus"] = int(j["gpus"]*1.5)
            j["power_kw"] *= 1.5
    elif name == "impossible_grid":
        d["grid_cap_kw"] = [0.]*d["T"]
        d["renewable_kw"] = [0.]*d["T"]
    else:
        raise ValueError(name)
    return d


def energy_diagnostic(d: dict) -> dict:
    """A necessary energy condition under cyclic storage; never sufficient."""
    dt=d["dt_hours"]
    fixed=dt*sum(k*(d["base_it_kw"]+d["inference_kw_per_gpu"]*n)
                 for k,n in zip(d["pue"],d["inference_gpus"]))
    jobs=min(d["pue"])*dt*sum(j["power_kw"]*j["duration"] for j in d["jobs"])
    supply=dt*(sum(d["grid_cap_kw"])+sum(d["renewable_kw"]))
    return dict(necessary_facility_energy_lower_bound_kwh=fixed+jobs,
                maximum_grid_plus_renewable_energy_kwh=supply,
                proven_energy_deficit_kwh=max(0.,fixed+jobs-supply),
                note="A positive deficit proves infeasibility; zero does not prove feasibility")
