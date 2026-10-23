"""Independent reconstruction of results; does not inspect the MILP matrix."""
from numbers import Real
import numpy as np
from .data import validate_data


def _finite_scalar(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return False
    try:
        return bool(np.isfinite(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _failure(errors):
    return dict(passed=False, errors=errors, max_residual=None, residuals={})


def audit(d, result, tolerance=1e-4):
    """Fail closed on invalid inputs, duplicate assets or nonfinite arithmetic."""
    if not _finite_scalar(tolerance) or tolerance < 0:
        return _failure(["invalid_tolerance"])
    try:
        validate_data(d)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        return _failure(["invalid_input: " + str(error)])
    # Finite operands can still overflow during reconstruction. Check residuals
    # and metrics explicitly instead of relying on comparisons with NaN.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return _audit(d, result, tolerance)


def _audit(d, result, tolerance):
    if not isinstance(result, dict) or "starts" not in result:
        return _failure(["No executable schedule"])
    T,dt=d["T"],d["dt_hours"]
    errors=[]; residuals={}
    traces={}
    for field in ["grid_kw","renewable_used_kw","facility_kw","training_gpus"]:
        try:
            arr=np.asarray(result.get(field,[]),dtype=float)
        except (TypeError, ValueError, OverflowError):
            errors.append("invalid_trace:"+field)
            continue
        if arr.shape!=(T,) or not np.all(np.isfinite(arr)):
            errors.append("invalid_trace:"+field)
        traces[field]=arr
    batteries=result.get("batteries")
    if not isinstance(batteries, list):
        return _failure(errors+["invalid_battery_list"])
    configs={b["name"]:b for b in d["batteries"]}
    seen=set(); battery_traces=[]
    for b in batteries:
        if not isinstance(b, dict) or not isinstance(b.get("name"), str):
            errors.append("invalid_battery_name")
            continue
        name=b["name"]
        if name in seen:
            errors.append("duplicate_battery:"+name)
        seen.add(name)
        if name not in configs:
            errors.append("unknown_battery:"+name)
        parsed={}
        for field in ["charge_kw","discharge_kw","energy_kwh"]:
            try:
                arr=np.asarray(b.get(field,[]),dtype=float)
            except (TypeError, ValueError, OverflowError):
                errors.append("invalid_battery_trace:"+field)
                continue
            if arr.shape!=((T+1,) if field=="energy_kwh" else (T,)) or not np.all(np.isfinite(arr)):
                errors.append("invalid_battery_trace:"+field)
            parsed[field]=arr
        battery_traces.append((name,parsed))
    if not all(_finite_scalar(result.get(field)) for field in ["cost_cny","carbon_kg"]):
        errors.append("nonfinite_accounting")
    for field in ["carbon_cap_kg","cost_cap_cny"]:
        if result.get(field) is not None and not _finite_scalar(result[field]):
            errors.append("nonfinite_budget:"+field)
    if not isinstance(result["starts"], dict):
        return _failure(errors+["invalid_starts"])
    for jid,start in result["starts"].items():
        if isinstance(start, (bool,np.bool_)) or not isinstance(start,(int,np.integer)):
            errors.append("noninteger_start:"+str(jid))
    if errors:
        return _failure(errors)
    def check(label, value):
        try:
            value=float(value)
        except (TypeError, ValueError, OverflowError):
            value=np.nan
        if not np.isfinite(value):
            residuals[label]=None
            errors.append("nonfinite_residual:"+label)
            return
        residuals[label]=float(max(0.,value))
        if value>tolerance: errors.append(label)
    starts=result["starts"]
    if set(starts)!=set(j["id"] for j in d["jobs"]): errors.append("job_set")
    gpu,power=np.zeros(T),np.zeros(T)
    waits=[]
    by_id={j["id"]:j for j in d["jobs"]}
    for j in d["jobs"]:
        if j["id"] not in starts: continue
        s=starts[j["id"]]; L=j["duration"]
        check("release:"+j["id"],j["release"]-s)
        check("deadline:"+j["id"],s+L-j["deadline"])
        check("fairness:"+j["id"],s-j["release"]-j["max_wait"])
        check("integer_start:"+j["id"],abs(s-round(s)))
        for parent in j["parents"]:
            if parent in starts: check("dependency:"+j["id"],starts[parent]+by_id[parent]["duration"]-s)
        if s>=0 and s+L<=T:
            gpu[s:s+L]+=j["gpus"]; power[s:s+L]+=j["power_kw"]
        else: errors.append("horizon:"+j["id"])
        waits.append((s-j["release"])*dt)
    facility=np.asarray(d["pue"])*(d["base_it_kw"]+power+d["inference_kw_per_gpu"]*np.asarray(d["inference_gpus"]))
    check("gpu_capacity",np.max(gpu+np.asarray(d["inference_gpus"])-d["gpu_capacity"]))
    check("facility_capacity",np.max(facility-d["facility_cap_kw"]))
    G=traces["grid_kw"]; R=traces["renewable_used_kw"]
    check("grid_nonnegative",-np.min(G)); check("grid_cap",np.max(G-d["grid_cap_kw"]))
    check("renewable_nonnegative",-np.min(R)); check("renewable_cap",np.max(R-d["renewable_kw"]))
    if T>1 and d.get("grid_ramp_kw_per_step") is not None:
        check("grid_ramp",np.max(np.abs(np.diff(G)))-d["grid_ramp_kw_per_step"])
    totalc,totaldis=np.zeros(T),np.zeros(T)
    wear=0.;loss=0.
    # Configured batteries may be disabled/omitted (e.g. storage=False); every
    # exported dispatch must nevertheless identify one unique configured asset.
    for name,b in battery_traces:
        conf=configs[name]
        c,dis,e=(b[k] for k in ("charge_kw","discharge_kw","energy_kwh"))
        check(name+":nonnegative",max(-np.min(c),-np.min(dis)))
        check(name+":charge",np.max(c)-conf["charge_kw"]+conf["reserve_kw"])
        check(name+":discharge",np.max(dis)-conf["discharge_kw"]+conf["reserve_kw"])
        check(name+":exclusivity",np.max(np.minimum(c,dis)))
        check(name+":soc_min",conf["min_kwh"]+conf["reserve_kwh"]-np.min(e))
        check(name+":soc_max",np.max(e)-conf["max_kwh"])
        check(name+":initial",abs(e[0]-conf["initial_kwh"]))
        check(name+":terminal",abs(e[-1]-conf["initial_kwh"]))
        check(name+":transition",np.max(abs(np.diff(e)-dt*(conf["eta_charge"]*c-dis/conf["eta_discharge"]))))
        totalc+=c; totaldis+=dis
        wear+=dt*(c.sum()+dis.sum())*conf["wear_cny_per_kwh"]
        loss+=dt*(c.sum()-dis.sum())
    check("power_balance",np.max(abs(G+R+totaldis-facility-totalc)))
    check("facility_reconstruction",np.max(abs(facility-traces["facility_kw"])))
    check("gpu_reconstruction",np.max(abs(gpu-traces["training_gpus"])))
    energy_cost=dt*np.dot(G,d["price_cny_per_kwh"])
    renewable_cost=dt*R.sum()*d["renewable_cost_cny_per_kwh"]
    demand_cost=max(G)*d["demand_rate_cny_per_kw_day"]
    cost=energy_cost+renewable_cost+demand_cost+wear
    carbon=dt*np.dot(G,d["carbon_kg_per_kwh"])
    check("cost_accounting",abs(cost-result["cost_cny"]))
    check("carbon_accounting",abs(carbon-result["carbon_kg"]))
    if result.get("carbon_cap_kg") is not None: check("carbon_budget",carbon-result["carbon_cap_kg"])
    if result.get("cost_cap_cny") is not None: check("cost_budget",cost-result["cost_cap_cny"])
    metrics=dict(cost_cny=float(cost),grid_energy_cost_cny=float(energy_cost),
                    renewable_cost_cny=float(renewable_cost),demand_cost_cny=float(demand_cost),
                    battery_wear_cny=float(wear),carbon_kg=float(carbon),peak_kw=float(max(G)),
                    facility_kwh=float(dt*facility.sum()),grid_kwh=float(dt*G.sum()),
                    renewable_used_kwh=float(dt*R.sum()),renewable_available_kwh=float(dt*sum(d["renewable_kw"])),
                    battery_loss_kwh=float(loss),gpu_hours=float(dt*gpu.sum()),
                    jobs_completed=len(starts),max_wait_hours=float(max(waits,default=0.)),
                    mean_wait_hours=float(np.mean(waits)) if waits else 0.)
    for field,value in metrics.items():
        if not np.isfinite(value):
            metrics[field]=None
            errors.append("nonfinite_metric:"+field)
    maximum=None if any(value is None for value in residuals.values()) else max(residuals.values(),default=0.)
    return dict(passed=not errors, errors=errors, max_residual=maximum,
                residuals=residuals, metrics=metrics)
