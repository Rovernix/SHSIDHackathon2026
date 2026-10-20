"""Causal receding-horizon replay, with actuals revealed one slot at a time."""
import copy
from .model import solve
from .audit import audit


def replay(actual, time_limit=2., margin=64, progress=None):
    """At t, only releases <=t and actual traces <=t are revealed.
    Prices and PUE are available in advance. Future solar/background values use
    the supplied forecast arrays (previous-day persistence in the public case).
    Archived carbon forecasts have no verified issuance vintage and do not enter
    this cost-only controller's objective or constraints. Current-slot realized
    aggregates are optimistically treated as known at the boundary. All past
    actions and launched jobs are irrevocable; re-solving cannot repair history.
    """
    T=actual["T"]
    committed={}; logs=[]
    prefix=dict(grid_kw=[],renewable_used_kw=[],batteries=[dict(charge_kw=[],discharge_kw=[],energy_kwh=[b["initial_kwh"]]) for b in actual["batteries"]])
    last=None
    for t in range(T):
        d=copy.deepcopy(actual)
        d["jobs"]=[j for j in d["jobs"] if j["release"]<=t]
        d["renewable_kw"]=actual["renewable_kw"][:t+1]+actual["forecast_renewable_kw"][t+1:]
        d["inference_gpus"]=actual["inference_gpus"][:t+1]+actual["forecast_inference_gpus"][t+1:]
        if "forecast_carbon_kg_per_kwh" in actual:
            d["carbon_kg_per_kwh"]=actual["carbon_kg_per_kwh"][:t+1]+actual["forecast_carbon_kg_per_kwh"][t+1:]
        s=solve(d, fixed_starts=committed, prefix=prefix, now=t,
                future_gpu_margin=margin, time_limit=time_limit, gap=.02)
        fallback=False
        if not s["feasible"] and margin:
            fallback=True
            s=solve(d,fixed_starts=committed,prefix=prefix,now=t,
                    time_limit=max(3.,time_limit),gap=.02)
        logs.append(dict(t=t,observed_until=t,known_jobs=len(d["jobs"]),
                         committed_before=dict(committed),status=s["status"],
                         feasible=s["feasible"],seconds=s["seconds"],margin_released=fallback,
                         forecast_only_after=t))
        if not s["feasible"]:
            return dict(method="causal rolling MILP",feasible=False,failed_at=t,
                        message="No executable incumbent; no service or energy slack hidden",log=logs)
        for jid,start in s["starts"].items():
            if start<=t:
                committed[jid]=start
        prefix["grid_kw"].append(s["grid_kw"][t]); prefix["renewable_used_kw"].append(s["renewable_used_kw"][t])
        for old,b in zip(prefix["batteries"],s["batteries"]):
            old["charge_kw"].append(b["charge_kw"][t]); old["discharge_kw"].append(b["discharge_kw"][t])
            old["energy_kwh"].append(b["energy_kwh"][t+1])
        last=s
        if progress and t%12==0: progress(f"causal replay {t+1}/{T}")
    # The last solve contains a frozen executed prefix and its last action.
    last.update(method="causal rolling MILP",starts=committed,log=logs,
                seconds=sum(v["seconds"] for v in logs))
    last["audit"]=audit(actual,last)
    last["feasible"]=last["audit"]["passed"]
    # The last solve bound belongs to its final fixed-history subproblem only.
    last.pop("lower_bound",None);last.pop("mip_gap",None)
    last["optimality_note"]="No global online optimum certificate; offline bound is a hindsight reference only"
    return last
