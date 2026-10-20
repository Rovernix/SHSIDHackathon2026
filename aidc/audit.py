"""Independent reconstruction of results; does not inspect the MILP matrix."""
import numpy as np


def audit(d, result, tolerance=1e-4):
    if "starts" not in result:
        return dict(passed=False, errors=["No executable schedule"], residuals={})
    T,dt=d["T"],d["dt_hours"]
    errors=[]; residuals={}
    for field in ["grid_kw","renewable_used_kw","facility_kw","training_gpus"]:
        arr=np.asarray(result.get(field,[]),dtype=float)
        if arr.shape!=(T,) or not np.all(np.isfinite(arr)):
            errors.append("invalid_trace:"+field)
    for b in result.get("batteries",[]):
        for field in ["charge_kw","discharge_kw","energy_kwh"]:
            arr=np.asarray(b.get(field,[]),dtype=float)
            if arr.shape!=((T+1,) if field=="energy_kwh" else (T,)) or not np.all(np.isfinite(arr)):
                errors.append("invalid_battery_trace:"+field)
    if not all(np.isfinite(result.get(field,np.nan)) for field in ["cost_cny","carbon_kg"]):
        errors.append("nonfinite_accounting")
    for jid,start in result["starts"].items():
        if not isinstance(start,(int,np.integer)):
            errors.append("noninteger_start:"+jid)
    if errors:
        return dict(passed=False,errors=errors,residuals={})
    def check(label, value):
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
    G=np.asarray(result["grid_kw"]); R=np.asarray(result["renewable_used_kw"])
    check("grid_nonnegative",-np.min(G)); check("grid_cap",np.max(G-d["grid_cap_kw"]))
    check("renewable_nonnegative",-np.min(R)); check("renewable_cap",np.max(R-d["renewable_kw"]))
    if T>1 and d.get("grid_ramp_kw_per_step") is not None:
        check("grid_ramp",np.max(np.abs(np.diff(G)))-d["grid_ramp_kw_per_step"])
    totalc,totaldis=np.zeros(T),np.zeros(T)
    wear=0.;loss=0.
    configs={b["name"]:b for b in d["batteries"]}
    for b in result["batteries"]:
        conf=configs[b["name"]]
        c,dis,e=(np.asarray(b[k]) for k in ("charge_kw","discharge_kw","energy_kwh"))
        check(b["name"]+":nonnegative",max(-min(c),-min(dis)))
        check(b["name"]+":charge",max(c)-conf["charge_kw"]+conf["reserve_kw"])
        check(b["name"]+":discharge",max(dis)-conf["discharge_kw"]+conf["reserve_kw"])
        check(b["name"]+":exclusivity",max(np.minimum(c,dis)))
        check(b["name"]+":soc_min",conf["min_kwh"]+conf["reserve_kwh"]-min(e))
        check(b["name"]+":soc_max",max(e)-conf["max_kwh"])
        check(b["name"]+":initial",abs(e[0]-conf["initial_kwh"]))
        check(b["name"]+":terminal",abs(e[-1]-conf["initial_kwh"]))
        check(b["name"]+":transition",max(abs(np.diff(e)-dt*(conf["eta_charge"]*c-dis/conf["eta_discharge"]))))
        totalc+=c; totaldis+=dis
        wear+=dt*(c.sum()+dis.sum())*conf["wear_cny_per_kwh"]
        loss+=dt*(c.sum()-dis.sum())
    check("power_balance",max(abs(G+R+totaldis-facility-totalc)))
    check("facility_reconstruction",max(abs(facility-result["facility_kw"])))
    check("gpu_reconstruction",max(abs(gpu-result["training_gpus"])))
    energy_cost=dt*np.dot(G,d["price_cny_per_kwh"])
    renewable_cost=dt*R.sum()*d["renewable_cost_cny_per_kwh"]
    demand_cost=max(G)*d["demand_rate_cny_per_kw_day"]
    cost=energy_cost+renewable_cost+demand_cost+wear
    carbon=dt*np.dot(G,d["carbon_kg_per_kwh"])
    check("cost_accounting",abs(cost-result["cost_cny"]))
    check("carbon_accounting",abs(carbon-result["carbon_kg"]))
    if result.get("carbon_cap_kg") is not None: check("carbon_budget",carbon-result["carbon_cap_kg"])
    if result.get("cost_cap_cny") is not None: check("cost_budget",cost-result["cost_cap_cny"])
    return dict(passed=not errors, errors=errors, max_residual=max(residuals.values(),default=0.),
                residuals=residuals, metrics=dict(cost_cny=float(cost),grid_energy_cost_cny=float(energy_cost),
                    renewable_cost_cny=float(renewable_cost),demand_cost_cny=float(demand_cost),
                    battery_wear_cny=float(wear),carbon_kg=float(carbon),peak_kw=float(max(G)),
                    facility_kwh=float(dt*facility.sum()),grid_kwh=float(dt*G.sum()),
                    renewable_used_kwh=float(dt*R.sum()),renewable_available_kwh=float(dt*sum(d["renewable_kw"])),
                    battery_loss_kwh=float(loss),gpu_hours=float(dt*gpu.sum()),
                    jobs_completed=len(starts),max_wait_hours=float(max(waits,default=0.)),
                    mean_wait_hours=float(np.mean(waits)) if waits else 0.))
