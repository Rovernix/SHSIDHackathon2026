"""Tighten headline bounds and run policy-matched and battery ablations."""
from pathlib import Path
import copy
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aidc.data import save_json
from aidc.model import solve, fifo
from aidc.experiments import attach,summary,export
from tools.refine_results import pool_incumbents
from aidc.paths import RESULTS


def main():
    out=RESULTS
    d=json.loads((out/"input.json").read_text(encoding="utf-8"))
    bundle=json.loads((out/"summary.json").read_text(encoding="utf-8"))
    cap=bundle["policy"]["carbon_cap_kg"]
    for name,kwargs in [("joint_cost",{}),("joint_green",dict(carbon_cap=cap))]:
        old=json.loads((out/f"{name}.json").read_text(encoding="utf-8"))
        new=attach(d,solve(d,time_limit=180.,gap=.0001,**kwargs))
        save_json(out/f"{name}_extended_search.json",new)
        if new["feasible"]:
            best=copy.deepcopy(new if new["cost_cny"]<=old["cost_cny"] else old)
            best["lower_bound"]=max(old["lower_bound"],new["lower_bound"])
            best["mip_gap"]=max(0.,(best["cost_cny"]-best["lower_bound"])/abs(best["cost_cny"]))
            best["certificate_note"]="Best audited incumbent and strongest valid bound across target-problem runs"
            best["extended_search_seconds"]=new["seconds"]
            save_json(out/f"{name}.json",best);export(d,best,out,name)
            bundle["strategies"][name]=summary(best)
        print(name,summary(new),flush=True)
        save_json(out/"summary.json",bundle)
    ablations={}
    configs=[("fifo_storage_green",d,dict(fixed_starts=fifo(d)["starts"])),
             ("compute_only_green",d,dict(storage=False)),
             ("lfp_only_green",dict(d,batteries=copy.deepcopy(d["batteries"][:1])),{})]
    for name,ds,kwargs in configs:
        s=attach(ds,solve(ds,carbon_cap=cap,time_limit=60.,gap=.001,**kwargs))
        save_json(out/f"{name}.json",s)
        if s["feasible"]:export(ds,s,out,name)
        ablations[name]=summary(s)
        print(name,summary(s),flush=True)
    bundle["policy_matched_ablations"]=ablations
    bundle["environment"]["extended_headline_search_seconds"]=180.
    save_json(out/"summary.json",bundle)
    pool_incumbents(out)


if __name__=="__main__":main()
