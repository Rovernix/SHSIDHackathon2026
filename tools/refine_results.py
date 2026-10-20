"""Refine headline certificates and complete the Pareto/assumption evidence."""
from pathlib import Path
import argparse
import copy
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aidc.model import solve, fifo, unscheduled
from aidc.data import save_json,scenario
from aidc.experiments import attach,summary,export
from aidc.paths import RESULTS


def main():
    p=argparse.ArgumentParser();p.add_argument("--results",default=str(RESULTS))
    p.add_argument("--time-limit",type=float,default=90.)
    a=p.parse_args();out=Path(a.results)
    d=json.loads((out/"input.json").read_text(encoding="utf-8"))
    bundle=json.loads((out/"summary.json").read_text(encoding="utf-8"))
    for name,kwargs in [("joint_green",dict(carbon_cap=bundle["policy"]["carbon_cap_kg"])),
                        ("joint_cost",{}),("compute_only",dict(storage=False))]:
        s=attach(d,solve(d,time_limit=a.time_limit,**kwargs))
        if not s["feasible"]:raise RuntimeError((name,s))
        save_json(out/f"{name}.json",s);export(d,s,out,name)
        bundle["strategies"][name]=summary(s)
        print(name,summary(s),flush=True)
        save_json(out/"summary.json",bundle)
    green=attach(d,solve(d,objective="carbon",time_limit=30.))
    if not green["feasible"]:raise RuntimeError(green)
    save_json(out/"minimum_carbon.json",green)
    low,high=green["carbon_kg"],bundle["strategies"]["joint_cost"]["carbon_kg"]
    points=[]
    for alpha in [0.,.25,.5,.75,1.]:
        budget=low+(high-low)*alpha+1e-3
        s=attach(d,solve(d,carbon_cap=budget,time_limit=25.))
        points.append(dict(fraction=alpha,carbon_cap_kg=budget,**summary(s)))
        save_json(out/f"pareto_{alpha:.2f}.json",s)
        print("pareto",alpha,summary(s),flush=True)
    bundle["pareto"]=points
    for name in ["no_demand_charge","high_workload","impossible_grid"]:
        ds=scenario(d,name);ff=fifo(ds);bb=attach(ds,unscheduled(ds,ff["starts"]))
        bb["feasible"]=bool(bb["feasible"] and ff["feasible"])
        s=attach(ds,solve(ds,carbon_cap=.97*bb["carbon_kg"],time_limit=20.))
        save_json(out/"stress"/f"{name}.json",dict(input=ds,baseline=bb,result=s))
        row=dict(scenario=name,baseline_feasible=bb["feasible"],baseline_cost_cny=bb["cost_cny"],**summary(s))
        if s["feasible"] and bb["feasible"]:
            row["cost_saving_percent"]=100*(1-s["cost_cny"]/bb["cost_cny"])
            row["carbon_saving_percent"]=100*(1-s["carbon_kg"]/bb["carbon_kg"])
        bundle["stress"]=[r for r in bundle["stress"] if r["scenario"]!=name]+[row]
        print("stress",name,row,flush=True)
    bundle["environment"]["refinement_time_limit_seconds"]=a.time_limit
    bundle["environment"]["pareto_time_limit_seconds"]=25.
    save_json(out/"summary.json",bundle)
    pool_incumbents(out)


def pool_incumbents(out):
    """A solution to a stricter budget is also feasible for a looser budget.
    Pool executable incumbents while preserving the target problem's own bound.
    This avoids overstating a savings against a weaker time-limited baseline.
    Raw runs remain available for reproducibility.
    """
    out=Path(out);bundle=json.loads((out/"summary.json").read_text(encoding="utf-8"))
    d=json.loads((out/"input.json").read_text(encoding="utf-8"))
    names=["joint_cost","joint_green"]+[f"pareto_{x:.2f}" for x in [0.,.25,.5,.75,1.]]
    candidates={name:json.loads((out/f"{name}.json").read_text(encoding="utf-8")) for name in names}
    for name,target in candidates.items():
        if not target["feasible"]:continue
        cap=None if name=="joint_cost" else target.get("carbon_cap_kg")
        eligible=[(k,v) for k,v in candidates.items() if v["feasible"] and (cap is None or v["carbon_kg"]<=cap+1e-6)]
        donor,source=min(eligible,key=lambda item:item[1]["cost_cny"])
        if source["cost_cny"]>=target["cost_cny"]-1e-7:continue
        save_json(out/f"{name}_raw_solver.json",target)
        r=copy.deepcopy(source)
        for field in ["status","message","seconds","variables","binary_variables","constraints","lower_bound","time_limit_seconds","requested_relative_gap"]:
            if field in target:r[field]=target[field]
        r.update(carbon_cap_kg=cap,cost_cap_cny=None,objective=source["cost_cny"],
                 mip_gap=max(0.,(source["cost_cny"]-target["lower_bound"])/abs(source["cost_cny"])),
                 incumbent_source=donor,bound_source=name+"_raw_solver.json",
                 certificate_note="Pooled feasible incumbent; lower bound from original target problem")
        r=attach(d,r);save_json(out/f"{name}.json",r)
        if name in bundle["strategies"]:
            bundle["strategies"][name]=summary(r);export(d,r,out,name)
        for point in bundle.get("pareto",[]):
            if name==f"pareto_{point['fraction']:.2f}":point.update(summary(r))
    save_json(out/"summary.json",bundle)


if __name__=="__main__":main()
