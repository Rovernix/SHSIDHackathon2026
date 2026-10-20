"""Start-indexed MILP using SciPy/HiGHS. Units: kW, kWh, hours, CNY, kg CO2."""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .data import validate_data, energy_diagnostic


class Matrix:
    def __init__(self):
        self.lo, self.hi, self.integer = [], [], []
        self.ii, self.jj, self.vv, self.bl, self.bu, self.names = [], [], [], [], [], []

    def var(self, lo=0., hi=np.inf, integer=False):
        k = len(self.lo)
        self.lo.append(float(lo)); self.hi.append(float(hi)); self.integer.append(int(integer))
        return k

    def row(self, terms, lo=-np.inf, hi=np.inf, name=""):
        r = len(self.bl)
        for k, v in terms:
            if v:
                self.ii.append(r); self.jj.append(k); self.vv.append(float(v))
        self.bl.append(lo); self.bu.append(hi); self.names.append(name)


def solve(data: dict, *, storage: bool = True, fixed_starts: dict | None = None,
          objective: str = "cost", carbon_cap: float | None = None,
          cost_cap: float | None = None, relax: bool = False,
          time_limit: float = 30., gap: float = .002, prefix: dict | None = None,
          future_gpu_margin: int = 0, now: int = 0) -> dict:
    """No slack is silently introduced. A time-limited incumbent is labeled as such.

    Prefix freezes already executed AC actions and SoC during causal replay.
    The caller supplies forecast-only future data; this solver has no hidden access
    to a true future trace. fixed_starts may cover just the already launched jobs.
    """
    validate_data(data)
    if objective not in ("cost", "carbon"):
        raise ValueError("Unknown objective")
    T, dt = data["T"], data["dt_hours"]
    m = Matrix()
    active = [[] for _ in range(T)]
    zs, starts = {}, {}
    fixed_starts = fixed_starts or {}
    for j in data["jobs"]:
        jid, L, r, deadline = j["id"], j["duration"], j["release"], j["deadline"]
        latest = min(deadline - L, r + j["max_wait"])
        choices = [fixed_starts[jid]] if jid in fixed_starts else range(max(r,now), latest+1)
        if any(s < r or s > latest for s in choices):
            raise ValueError(f"Fixed start outside admissible window: {jid}")
        zs[jid], starts[jid] = [], []
        for s in choices:
            k = m.var(hi=1., integer=True)
            zs[jid].append(k); starts[jid].append(s)
            for t in range(s, s+L):
                active[t].append((k, j["gpus"], j["power_kw"]))
        m.row([(k, 1.) for k in zs[jid]], lo=1., hi=1., name=f"complete:{jid}")
    by_id = {j["id"]:j for j in data["jobs"]}
    for j in data["jobs"]:
        for p in j["parents"]:
            terms = [(k, s) for k,s in zip(zs[j["id"]], starts[j["id"]])]
            terms += [(k, -(s+by_id[p]["duration"])) for k,s in zip(zs[p], starts[p])]
            m.row(terms, lo=0., name=f"precedence:{p}->{j['id']}")
    G = [m.var(hi=data["grid_cap_kw"][t]) for t in range(T)]
    R = [m.var(hi=data["renewable_kw"][t]) for t in range(T)]
    peak = m.var(hi=max(data["grid_cap_kw"]))
    bats = data["batteries"] if storage else []
    bv = []
    for b in bats:
        cp, dp = b["charge_kw"]-b["reserve_kw"], b["discharge_kw"]-b["reserve_kw"]
        c = [m.var(hi=cp) for _ in range(T)]
        dis = [m.var(hi=dp) for _ in range(T)]
        e = [m.var(lo=b["min_kwh"]+b["reserve_kwh"], hi=b["max_kwh"]) for _ in range(T+1)]
        mode = [m.var(hi=1., integer=True) for _ in range(T)]
        for t in (0, T):
            m.lo[e[t]] = m.hi[e[t]] = b["initial_kwh"]
        for t in range(T):
            m.row([(e[t+1],1.), (e[t],-1.), (c[t],-b["eta_charge"]*dt),
                   (dis[t],dt/b["eta_discharge"])], lo=0., hi=0., name=f"soc:{b['name']}:{t}")
            m.row([(c[t],1.), (mode[t],-cp)], hi=0., name=f"charge_mode:{b['name']}:{t}")
            m.row([(dis[t],1.), (mode[t],dp)], hi=dp, name=f"discharge_mode:{b['name']}:{t}")
        bv.append(dict(charge=c, discharge=dis, energy=e, mode=mode))
    for t in range(T):
        pu = data["pue"][t]
        base = pu * (data["base_it_kw"] + data["inference_kw_per_gpu"]*data["inference_gpus"][t])
        margin = future_gpu_margin if t > now else 0
        m.row([(k,g) for k,g,p in active[t]], hi=data["gpu_capacity"][t]-data["inference_gpus"][t]-margin,
              name=f"gpu:{t}")
        m.row([(k,p*pu) for k,g,p in active[t]], hi=data["facility_cap_kw"]-base, name=f"facility:{t}")
        terms = [(G[t],1.), (R[t],1.)] + [(k,-p*pu) for k,g,p in active[t]]
        for v in bv:
            terms += [(v["discharge"][t],1.), (v["charge"][t],-1.)]
        m.row(terms, lo=base, hi=base, name=f"ac_balance:{t}")
        m.row([(G[t],1.), (peak,-1.)], hi=0., name=f"peak:{t}")
        if t and data.get("grid_ramp_kw_per_step") is not None:
            ramp = data["grid_ramp_kw_per_step"]
            m.row([(G[t],1.), (G[t-1],-1.)], lo=-ramp, hi=ramp, name=f"grid_ramp:{t}")
    if prefix:
        n = len(prefix["grid_kw"])
        for t in range(n):
            for k,v in ((G[t],prefix["grid_kw"][t]), (R[t],prefix["renewable_used_kw"][t])):
                m.lo[k] = m.hi[k] = float(v)
        for i,v in enumerate(bv):
            old = prefix["batteries"][i]
            for field in ("charge", "discharge"):
                for t in range(n):
                    k=v[field][t]; m.lo[k]=m.hi[k]=float(old[field+"_kw"][t])
            for t in range(n+1):
                k=v["energy"][t]; m.lo[k]=m.hi[k]=float(old["energy_kwh"][t])
    N = len(m.lo)
    cost, carbon = np.zeros(N), np.zeros(N)
    for t in range(T):
        cost[G[t]] = data["price_cny_per_kwh"][t] * dt
        cost[R[t]] = data["renewable_cost_cny_per_kwh"] * dt
        carbon[G[t]] = data["carbon_kg_per_kwh"][t] * dt
    cost[peak] = data["demand_rate_cny_per_kw_day"]
    for b,v in zip(bats,bv):
        for t in range(T):
            cost[v["charge"][t]] = cost[v["discharge"][t]] = b["wear_cny_per_kwh"] * dt
    if carbon_cap is not None:
        m.row(list(enumerate(carbon)), hi=carbon_cap, name="carbon_cap")
    if cost_cap is not None:
        m.row(list(enumerate(cost)), hi=cost_cap, name="cost_cap")
    A = coo_matrix((m.vv,(m.ii,m.jj)), shape=(len(m.bl),N)).tocsc()
    start = time.perf_counter()
    opt = milp(cost if objective=="cost" else carbon,
               integrality=np.zeros(N) if relax else np.array(m.integer),
               bounds=Bounds(m.lo,m.hi), constraints=LinearConstraint(A,m.bl,m.bu),
               options=dict(time_limit=time_limit, mip_rel_gap=gap, presolve=True))
    elapsed = time.perf_counter()-start
    result = dict(method="LP relaxation" if relax else "MILP", objective_kind=objective,
                  status=int(opt.status), message=str(opt.message), seconds=elapsed,
                  variables=N, binary_variables=0 if relax else sum(m.integer),
                  constraints=len(m.bl), feasible=False,
                  carbon_cap_kg=carbon_cap, cost_cap_cny=cost_cap,
                  time_limit_seconds=time_limit,requested_relative_gap=gap,
                  energy_diagnostic=energy_diagnostic(data))
    if opt.x is None:
        return result
    x = opt.x
    violation = max(float(np.maximum(np.array(m.bl)-A@x,0).max(initial=0)),
                    float(np.maximum(A@x-np.array(m.bu),0).max(initial=0)),
                    float(np.maximum(np.array(m.lo)-x,0).max(initial=0)),
                    float(np.maximum(x-np.array(m.hi),0).max(initial=0)))
    integer_error = 0. if relax else max((abs(x[i]-round(x[i])) for i,v in enumerate(m.integer) if v), default=0.)
    result.update(feasible=bool(violation<=1e-4 and integer_error<=1e-5),
                  objective=float(opt.fun), cost_cny=float(cost@x), carbon_kg=float(carbon@x),
                  max_matrix_violation=violation, max_integrality_error=float(integer_error),
                  lower_bound=float(getattr(opt,"mip_dual_bound",opt.fun) or 0.),
                  mip_gap=float(getattr(opt,"mip_gap",0.) or 0.),
                  node_count=int(getattr(opt,"mip_node_count",0) or 0))
    if relax:
        result["lower_bound"]=float(opt.fun)
        return result
    selected = {jid:int(ss[int(np.argmax(x[zs[jid]]))]) for jid,ss in starts.items()}
    gpu = np.zeros(T); power = np.zeros(T)
    for j in data["jobs"]:
        s = selected[j["id"]]
        gpu[s:s+j["duration"]] += j["gpus"]
        power[s:s+j["duration"]] += j["power_kw"]
    facility = np.asarray(data["pue"]) * (data["base_it_kw"] + power + data["inference_kw_per_gpu"]*np.asarray(data["inference_gpus"]))
    result.update(starts=selected, training_gpus=gpu.tolist(), facility_kw=facility.tolist(),
                  grid_kw=x[G].tolist(), renewable_used_kw=x[R].tolist(), peak_kw=float(max(x[G])),
                  batteries=[dict(name=b["name"], charge_kw=x[v["charge"]].tolist(),
                                  discharge_kw=x[v["discharge"]].tolist(), energy_kwh=x[v["energy"]].tolist())
                             for b,v in zip(bats,bv)])
    # With a carbon objective, the peak epigraph need not be tight. Account for
    # the physically observed maximum, rather than charging for arbitrary slack.
    result["cost_epigraph_cny"]=result["cost_cny"]
    result["cost_cny"]-=data["demand_rate_cny_per_kw_day"]*(float(x[peak])-result["peak_kw"])
    return result


def fifo(data: dict) -> dict:
    """Non-backfilling FIFO dispatcher using the current background reservation.
    Later bursts are hidden. The public case derives the reservation from short
    GPU jobs, without inference labels; current-slot aggregates are assumed known.
    Failures stay visible.
    This returns a compute schedule, never assumes storage or relaxed deadlines.
    """
    validate_data(data)
    pending, running, done, starts = [], [], set(), {}
    arrivals = sorted(data["jobs"], key=lambda j:(j["release"],j["id"]))
    for t in range(data["T"]):
        for j in list(running):
            if starts[j["id"]]+j["duration"] == t:
                running.remove(j); done.add(j["id"])
        pending += [j for j in arrivals if j["release"]==t]
        capacity = data["gpu_capacity"][t] - data["inference_gpus"][t]
        used = sum(j["gpus"] for j in running)
        while pending:
            j = pending[0]
            if not set(j["parents"]) <= done or used+j["gpus"] > capacity:
                break
            starts[j["id"]] = t; used += j["gpus"]; running.append(j); pending.pop(0)
    violations = []
    for j in arrivals:
        s=starts.get(j["id"])
        if s is None or s+j["duration"]>j["deadline"] or s-j["release"]>j["max_wait"]:
            violations.append(j["id"])
    # Detect burst capacity violations which a causal FIFO cannot foresee.
    for t in range(data["T"]):
        used=sum(j["gpus"] for j in arrivals if j["id"] in starts and starts[j["id"]]<=t<starts[j["id"]]+j["duration"])
        if used+data["inference_gpus"][t]>data["gpu_capacity"][t]:
            violations.append(f"gpu@{t}")
    return dict(starts=starts, feasible=not violations, violations=violations)


def unscheduled(data: dict, starts: dict) -> dict:
    """Direct grid draw and maximum available renewable use, no storage optimization.
    Physical violations are reported by the independent audit, not repaired here.
    """
    T, dt=data["T"],data["dt_hours"]
    gpu, p=np.zeros(T),np.zeros(T)
    for j in data["jobs"]:
        if j["id"] not in starts:
            continue
        s=starts[j["id"]]
        gpu[s:s+j["duration"]]+=j["gpus"]
        p[s:s+j["duration"]]+=j["power_kw"]
    facility=np.asarray(data["pue"])*(data["base_it_kw"]+p+data["inference_kw_per_gpu"]*np.asarray(data["inference_gpus"]))
    renewable=np.minimum(facility,data["renewable_kw"])
    grid=facility-renewable
    cost=float(dt*(np.dot(grid,data["price_cny_per_kwh"])+renewable.sum()*data["renewable_cost_cny_per_kwh"])+max(grid)*data["demand_rate_cny_per_kw_day"])
    return dict(method="causal FIFO, direct energy", feasible=True, starts=starts,
                training_gpus=gpu.tolist(), facility_kw=facility.tolist(), grid_kw=grid.tolist(),
                renewable_used_kw=renewable.tolist(), peak_kw=float(max(grid)), batteries=[],
                cost_cny=cost, carbon_kg=float(dt*np.dot(grid,data["carbon_kg_per_kwh"])))
