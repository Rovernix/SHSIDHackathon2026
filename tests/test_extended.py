"""Independent enumeration, metamorphic properties and adverse output checks."""
import copy
from itertools import product
import unittest
import numpy as np
from aidc.data import generate,validate_data,energy_diagnostic
from aidc.model import solve
from aidc.audit import audit
from aidc.experiments import tiny_data
from aidc.rolling import replay

RANDOM_EVIDENCE=[]


def record_random(seed,kind,legal,r):
    field="carbon" if kind=="carbon_under_cost_cap" else "cost"
    target=min((p[field] for p in legal),default=None)
    actual=r.get("carbon_kg" if field=="carbon" else "cost_cny")
    RANDOM_EVIDENCE.append(dict(seed=seed,case=kind,enumerated_legal_schedules=len(legal),
                               expected=target,actual=actual,status=r["status"],
                               passed=(r["status"]==2 if target is None else abs(target-actual)<1e-6)))


def random_tiny(seed):
    rng=np.random.default_rng(seed);d=tiny_data();T=5
    d.update(T=T,gpu_capacity=[3]*T,inference_gpus=rng.integers(0,2,T).tolist(),
        pue=[1.1]*T,grid_cap_kw=[4.]*T,price_cny_per_kwh=rng.choice([.2,1.,3.],T).tolist(),
        carbon_kg_per_kwh=rng.choice([.1,.4,.8],T).tolist(),renewable_kw=rng.uniform(0,.8,T).tolist(),
        forecast_inference_gpus=[0]*T,forecast_renewable_kw=[0.]*T,base_it_kw=.2,
        inference_kw_per_gpu=.1,facility_cap_kw=3.,demand_rate_cny_per_kw_day=.3,jobs=[])
    for i in range(3):
        r=int(rng.integers(0,3));L=int(rng.integers(1,3));deadline=int(rng.integers(r+L,T+1))
        d["jobs"].append(dict(id=str(i),release=r,deadline=deadline,duration=L,max_wait=deadline-r-L,
                              gpus=int(rng.integers(1,3)),power_kw=float(rng.choice([.4,.8,1.2])),
                              parents=["0"] if i==2 and seed%2 else [],kind="training",hardware_pool="H100-like"))
    return d


def enumerate_plans(d):
    """No solver or start-indexed matrix is used in this reference evaluator."""
    jobs=d["jobs"];plans=[]
    windows=[range(j["release"],min(j["deadline"]-j["duration"],j["release"]+j["max_wait"])+1) for j in jobs]
    for starts in product(*windows):
        selected=dict(zip((j["id"] for j in jobs),starts));byid={j["id"]:j for j in jobs}
        if any(selected[j["id"]]<selected[p]+byid[p]["duration"] for j in jobs for p in j["parents"]):continue
        grid=[];renew=[];valid=True
        for t in range(d["T"]):
            running=[j for j,st in zip(jobs,starts) if st<=t<st+j["duration"]]
            if sum(j["gpus"] for j in running)+d["inference_gpus"][t]>d["gpu_capacity"][t]:valid=False;break
            facility=d["pue"][t]*(d["base_it_kw"]+d["inference_kw_per_gpu"]*d["inference_gpus"][t]+sum(j["power_kw"] for j in running))
            rr=min(facility,d["renewable_kw"][t]);gg=facility-rr
            if facility>d["facility_cap_kw"]+1e-9 or gg>d["grid_cap_kw"][t]+1e-9:valid=False;break
            grid.append(gg);renew.append(rr)
        if valid:
            cost=d["dt_hours"]*(sum(g*p for g,p in zip(grid,d["price_cny_per_kwh"]))+sum(renew)*d["renewable_cost_cny_per_kwh"])+max(grid)*d["demand_rate_cny_per_kw_day"]
            carbon=d["dt_hours"]*sum(g*x for g,x in zip(grid,d["carbon_kg_per_kwh"]))
            plans.append(dict(cost=cost,carbon=carbon,starts=selected))
    return plans


class EnumerationTests(unittest.TestCase):
    def test_twenty_random_cost_optima(self):
        for seed in range(20):
            with self.subTest(seed=seed):
                d=random_tiny(seed);plans=enumerate_plans(d);r=solve(d,gap=0.)
                if not plans:self.assertEqual(r["status"],2)
                else:
                    self.assertTrue(audit(d,r)["passed"])
                    self.assertAlmostEqual(r["cost_cny"],min(p["cost"] for p in plans),places=6)
                record_random(seed,"cost",plans,r)

    def test_twenty_random_hard_carbon_budget_optima(self):
        for seed in range(20,40):
            with self.subTest(seed=seed):
                d=random_tiny(seed);plans=enumerate_plans(d)
                cap=float(np.median([p["carbon"] for p in plans])) if plans else 0.
                r=solve(d,carbon_cap=cap,gap=0.)
                legal=[p for p in plans if p["carbon"]<=cap+1e-8]
                if not legal:self.assertEqual(r["status"],2)
                else:
                    self.assertTrue(audit(d,r)["passed"])
                    self.assertAlmostEqual(r["cost_cny"],min(p["cost"] for p in legal),places=6)
                record_random(seed,"cost_under_carbon_cap",legal,r)

    def test_twenty_random_carbon_minima_with_cost_ceilings(self):
        for seed in range(40,60):
            with self.subTest(seed=seed):
                d=random_tiny(seed);plans=enumerate_plans(d)
                cap=float(np.median([p["cost"] for p in plans])) if plans else 0.
                r=solve(d,objective="carbon",cost_cap=cap,gap=0.)
                legal=[p for p in plans if p["cost"]<=cap+1e-8]
                if not legal:self.assertEqual(r["status"],2)
                else:
                    self.assertTrue(audit(d,r)["passed"])
                    self.assertAlmostEqual(r["carbon_kg"],min(p["carbon"] for p in legal),places=6)
                record_random(seed,"carbon_under_cost_cap",legal,r)


class ExtendedTests(unittest.TestCase):
    def test_more_gpus_reduces_congestion_cost(self):
        d=tiny_data();d["jobs"][1]["parents"]=[]
        a=solve(d,gap=0.);d["gpu_capacity"]=[2]*4;b=solve(d,gap=0.)
        self.assertAlmostEqual(a["cost_cny"],3.)
        self.assertAlmostEqual(b["cost_cny"],2.)

    def test_uniform_tariff_scaling_preserves_objective_ratio(self):
        d=tiny_data();a=solve(d,gap=0.)
        d["price_cny_per_kwh"]=[3*x for x in d["price_cny_per_kwh"]]
        b=solve(d,gap=0.);self.assertAlmostEqual(b["cost_cny"],3*a["cost_cny"])

    def test_slot_duration_scales_energy_and_carbon(self):
        d=tiny_data();a=solve(d,gap=0.);d["dt_hours"]*=.5;b=solve(d,gap=0.)
        self.assertAlmostEqual(b["cost_cny"],.5*a["cost_cny"])
        self.assertAlmostEqual(b["carbon_kg"],.5*a["carbon_kg"])

    def test_renewable_surplus_is_curtailed_without_export(self):
        d=tiny_data();d["jobs"]=[];d["base_it_kw"]=1.;d["renewable_kw"]=[10.]*4
        r=solve(d,gap=0.);self.assertTrue(audit(d,r)["passed"])
        self.assertAlmostEqual(sum(r["grid_kw"]),0.)
        self.assertAlmostEqual(sum(r["renewable_used_kw"]),4.)

    def test_zero_ramp_can_make_power_sequence_infeasible(self):
        d=tiny_data();d["grid_ramp_kw_per_step"]=0.
        self.assertEqual(solve(d)["status"],2)

    def test_initial_battery_energy_cannot_fund_a_cyclic_deficit(self):
        d=tiny_data();d["jobs"]=[];d["base_it_kw"]=2.;d["grid_cap_kw"]=[1.]*4
        d["batteries"]=copy.deepcopy(generate()["batteries"])
        self.assertEqual(solve(d)["status"],2)
        self.assertGreater(energy_diagnostic(d)["proven_energy_deficit_kwh"],0.)

    def test_lossless_battery_matches_analytic_arbitrage(self):
        d=tiny_data();d["jobs"]=[];d["base_it_kw"]=1.;d["price_cny_per_kwh"]=[.1,.1,3.,3.]
        battery=copy.deepcopy(generate()["batteries"][0]);battery.update(capacity_kwh=2.,min_kwh=0.,max_kwh=2.,initial_kwh=0.,charge_kw=1.,discharge_kw=1.,eta_charge=1.,eta_discharge=1.,wear_cny_per_kwh=0.)
        d["batteries"]=[battery];r=solve(d,gap=0.)
        self.assertAlmostEqual(r["cost_cny"],.4,places=7)
        self.assertTrue(audit(d,r)["passed"])

    def test_prefix_grid_and_started_jobs_are_irrevocable(self):
        d=tiny_data();old=solve(d,gap=0.)
        prefix=dict(grid_kw=old["grid_kw"][:2],renewable_used_kw=old["renewable_used_kw"][:2],batteries=[])
        d["price_cny_per_kwh"]=[4.,1.,.1,2.]
        new=solve(d,now=2,prefix=prefix,fixed_starts={"A":1},gap=0.)
        self.assertEqual(new["starts"],{"A":1,"B":2})
        self.assertEqual(new["grid_kw"][:2],old["grid_kw"][:2])

    def test_future_actual_renewables_cannot_change_first_action(self):
        d=tiny_data();d["jobs"]=[];d["base_it_kw"]=1.
        b=copy.deepcopy(generate()["batteries"][0]);b.update(capacity_kwh=4.,min_kwh=0.,max_kwh=4.,initial_kwh=2.,charge_kw=2.,discharge_kw=2.)
        d["batteries"]=[b];other=copy.deepcopy(d);other["renewable_kw"][2]=10.
        a=replay(d,margin=0);c=replay(other,margin=0)
        self.assertTrue(a["feasible"] and c["feasible"])
        self.assertAlmostEqual(a["grid_kw"][0],c["grid_kw"][0],places=7)

    def test_terminal_energy_tampering_is_detected(self):
        d=tiny_data();d["batteries"]=copy.deepcopy(generate()["batteries"]);r=solve(d)
        r["batteries"][0]["energy_kwh"][-1]+=1.
        self.assertIn("LFP:terminal",audit(d,r)["errors"])

    def test_short_trace_fails_cleanly(self):
        d=tiny_data();r=solve(d);r["grid_kw"].pop()
        self.assertFalse(audit(d,r)["passed"])

    def test_nonfinite_cost_fails_cleanly(self):
        d=tiny_data();r=solve(d);r["cost_cny"]=float("nan")
        self.assertFalse(audit(d,r)["passed"])

    def test_missing_parent_rejected(self):
        d=tiny_data();d["jobs"][0]["parents"]=["missing"]
        with self.assertRaises(ValueError):validate_data(d)

    def test_duplicate_job_ids_rejected(self):
        d=tiny_data();d["jobs"][1]["id"]="A"
        with self.assertRaises(ValueError):validate_data(d)

    def test_fractional_gpu_capacity_rejected(self):
        d=tiny_data();d["gpu_capacity"][0]=1.2
        with self.assertRaises(ValueError):validate_data(d)

    def test_nan_input_rejected(self):
        d=tiny_data();d["price_cny_per_kwh"][0]=float("nan")
        with self.assertRaises(ValueError):validate_data(d)

    def test_efficiency_above_one_rejected(self):
        d=generate();d["batteries"][0]["eta_charge"]=1.1
        with self.assertRaises(ValueError):validate_data(d)

    def test_forecast_length_rejected(self):
        d=tiny_data();d["forecast_renewable_kw"].pop()
        with self.assertRaises(ValueError):validate_data(d)

    def test_duplicate_battery_names_rejected(self):
        d=generate();d["batteries"][1]["name"]="LFP"
        with self.assertRaises(ValueError):validate_data(d)

    def test_negative_wait_rejected(self):
        d=tiny_data();d["jobs"][0]["max_wait"]=-1
        with self.assertRaises(ValueError):validate_data(d)


if __name__=="__main__":unittest.main(verbosity=2)
