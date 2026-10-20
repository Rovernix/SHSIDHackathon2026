"""Benchmark, ablation, Pareto, sensitivity, and infeasibility evidence."""
import csv
import platform
from pathlib import Path
import numpy as np
import scipy

from .data import canonical_hash, generate, save_json, scenario
from .model import fifo, solve, unscheduled
from .audit import audit
from .datasets import build_scenario
from .paths import CONFIG
import json


def attach(d,s):
    s["audit"]=audit(d,s)
    s["feasible"]=bool(s["feasible"] and s["audit"]["passed"])
    return s


def summary(s):
    fields=["method","feasible","status","message","cost_cny","carbon_kg","peak_kw",
            "seconds","lower_bound","mip_gap","variables","binary_variables","constraints"]
    return {k:s[k] for k in fields if k in s}


def benchmark(d, directory, time_limit=20., full=False, rolling=False, progress=print):
    out=Path(directory);out.mkdir(parents=True,exist_ok=True)
    save_json(out/"input.json",d)
    meta=dict(input_sha256=canonical_hash(d),python=platform.python_version(),
              numpy=np.__version__,scipy=scipy.__version__,solver="HiGHS bundled with SciPy",
              time_limit_per_solve=time_limit,requested_relative_gap=.002,
              data_provenance=d["provenance"])
    save_json(out/"environment.json",meta)
    results={}
    f=fifo(d);save_json(out/"fifo_dispatch.json",f)
    baseline=attach(d,unscheduled(d,f["starts"]))
    baseline["feasible"]=bool(baseline["feasible"] and f["feasible"])
    results["fifo"]=baseline
    progress(f"FIFO: feasible={baseline['feasible']}, CNY {baseline['cost_cny']:.2f}")
    for name,kwargs in [("fifo_storage",dict(fixed_starts=f["starts"])),
                        ("compute_only",dict(storage=False)),("joint_cost",{})]:
        if name=="fifo_storage" and not f["feasible"]:
            s=dict(feasible=False,message="FIFO schedule violates service constraints")
        else:
            s=attach(d,solve(d,time_limit=time_limit,**kwargs))
        results[name]=s;save_json(out/f"{name}.json",s)
        progress(f"{name}: {summary(s)}")
    # Policy: improve reported operational carbon by at least 3% against the
    # identical admitted workload; service constraints are hard constraints.
    cap=.97*baseline["carbon_kg"]
    eco=attach(d,solve(d,carbon_cap=cap,time_limit=time_limit))
    results["joint_green"]=eco
    progress(f"joint_green: {summary(eco)}")
    lp=solve(d,carbon_cap=cap,relax=True,time_limit=time_limit)
    save_json(out/"lp_relaxation.json",lp)
    tiny=tiny_certificate()
    save_json(out/"enumeration_certificate.json",tiny)
    for name,s in results.items():
        save_json(out/f"{name}.json",s)
        if s.get("feasible"):
            export(d,s,out,name)
    bundle=dict(environment=meta,strategies={k:summary(v) for k,v in results.items()},
                lp_relaxation=summary(lp),enumeration=tiny,
                policy=dict(carbon_cap_kg=cap,rule="minimize cost, carbon <= 97% of FIFO carbon"))
    if full:
        frontier=[]
        green=attach(d,solve(d,objective="carbon",time_limit=time_limit))
        save_json(out/"minimum_carbon.json",green)
        if green["feasible"] and results["joint_cost"]["feasible"]:
            low,high=green["carbon_kg"],results["joint_cost"]["carbon_kg"]
            for a in [0.,.25,.5,.75,1.]:
                budget=low+(high-low)*a+1e-3
                s=attach(d,solve(d,carbon_cap=budget,time_limit=time_limit))
                frontier.append(dict(fraction=a,carbon_cap_kg=budget,**summary(s)))
                save_json(out/f"pareto_{a:.2f}.json",s)
                progress(f"Pareto {a:.2f}: {summary(s)}")
        bundle["pareto"]=frontier
        stress=[]
        for name in ["cloudy","no_renewables","flat_price","price_carbon_conflict",
                     "hot_day","inference_surge","grid_derating","zero_flexibility",
                     "storage_expensive","impossible_grid"]:
            ds=scenario(d,name)
            ff=fifo(ds)
            bb=attach(ds,unscheduled(ds,ff["starts"]))
            bb["feasible"]=bool(bb["feasible"] and ff["feasible"])
            s=attach(ds,solve(ds,time_limit=time_limit,carbon_cap=.97*bb["carbon_kg"]))
            save_json(out/"stress"/f"{name}.json",dict(input=ds,baseline=bb,result=s))
            row=dict(scenario=name,baseline_feasible=bb["feasible"],baseline_cost_cny=bb["cost_cny"],**summary(s))
            if s["feasible"] and bb["feasible"]:
                row["cost_saving_percent"]=100*(1-s["cost_cny"]/bb["cost_cny"])
                row["carbon_saving_percent"]=100*(1-s["carbon_kg"]/bb["carbon_kg"])
            stress.append(row);progress(f"stress {name}: {row}")
        bundle["stress"]=stress
        days=[]
        for day in json.loads(CONFIG.read_text(encoding="utf-8"))["evaluation_dates"]:
            ds=build_scenario(day)
            ff=fifo(ds);bb=attach(ds,unscheduled(ds,ff["starts"]))
            bb["feasible"]=bool(bb["feasible"] and ff["feasible"])
            s=attach(ds,solve(ds,time_limit=time_limit,carbon_cap=.97*bb["carbon_kg"]))
            save_json(out/"days"/f"{day}.json",dict(input=ds,baseline=bb,result=s))
            row=dict(date=day,baseline_feasible=bb["feasible"],**summary(s))
            if s["feasible"] and bb["feasible"]: row["cost_saving_percent"]=100*(1-s["cost_cny"]/bb["cost_cny"])
            days.append(row);progress(f"public day {day}: {row}")
        bundle["days"]=days
    if rolling:
        from .rolling import replay
        margin=json.loads(CONFIG.read_text(encoding="utf-8"))["online_reserve_gpus"] if "source_metadata" in d else 64
        s=replay(d,margin=margin,progress=progress)
        save_json(out/"rolling.json",s)
        bundle["rolling"]=summary(s)
        if s["feasible"]: export(d,s,out,"rolling")
    save_json(out/"summary.json",bundle)
    progress(f"Saved {out.resolve() / 'summary.json'}")
    return bundle


def export(d,s,out,name):
    with (out/f"{name}_jobs.csv").open("w",newline="",encoding="utf-8-sig") as f:
        fields=["id","kind","release","start","finish","deadline","gpus","parents"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for j in d["jobs"]:
            row={k:j[k] for k in ["id","kind","release","deadline","gpus"]}
            row.update(start=s["starts"][j["id"]],finish=s["starts"][j["id"]]+j["duration"],parents=";".join(j["parents"]))
            w.writerow(row)
    with (out/f"{name}_dispatch.csv").open("w",newline="",encoding="utf-8-sig") as f:
        fields=["slot","hour","grid_kw","renewable_used_kw","facility_kw","training_gpus","inference_gpus"]
        for b in s["batteries"]: fields += [b["name"]+"_charge_kw",b["name"]+"_discharge_kw",b["name"]+"_energy_kwh"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for t in range(d["T"]):
            row={k:s[k][t] for k in ["grid_kw","renewable_used_kw","facility_kw","training_gpus"]}
            row.update(slot=t,hour=t*d["dt_hours"],inference_gpus=d["inference_gpus"][t])
            for b in s["batteries"]:
                for k in ["charge_kw","discharge_kw","energy_kwh"]: row[b["name"]+"_"+k]=b[k][t]
            w.writerow(row)


def tiny_data():
    d=generate(jobs=0)
    d.update(T=4,dt_hours=1.,base_it_kw=0.,inference_kw_per_gpu=0.,facility_cap_kw=100.,
             gpu_capacity=[1]*4,inference_gpus=[0]*4,pue=[1.]*4,grid_cap_kw=[100.]*4,
             price_cny_per_kwh=[4.,1.,3.,2.],carbon_kg_per_kwh=[1.]*4,renewable_kw=[0.]*4,
             forecast_renewable_kw=[0.]*4,forecast_inference_gpus=[0]*4,
             demand_rate_cny_per_kw_day=0.,grid_ramp_kw_per_step=None,batteries=[],
             jobs=[dict(id="A",release=0,deadline=4,duration=1,gpus=1,power_kw=1.,
                        kind="training",parents=[],max_wait=3,hardware_pool="H100-like"),
                   dict(id="B",release=0,deadline=4,duration=1,gpus=1,power_kw=1.,
                        kind="batch",parents=["A"],max_wait=3,hardware_pool="H100-like")])
    return d


def tiny_certificate():
    from itertools import product
    d=tiny_data();items=[]
    for a,b in product(range(4),repeat=2):
        if b>=a+1:
            items.append(dict(A=a,B=b,cost=d["price_cny_per_kwh"][a]+d["price_cny_per_kwh"][b]))
    best=min(items,key=lambda v:v["cost"])
    s=attach(d,solve(d,storage=False,gap=0.))
    return dict(enumerated_feasible_schedules=len(items),all_schedules=items,enumeration_best=best,
                milp_best=summary(s),starts=s.get("starts"),
                passed=bool(s["feasible"] and abs(best["cost"]-s["cost_cny"])<1e-7))
