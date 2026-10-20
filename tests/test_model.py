import copy
import unittest
from unittest.mock import patch

from aidc.data import generate, validate_data, canonical_hash
from aidc.model import solve, unscheduled, fifo
from aidc.audit import audit
from aidc.experiments import tiny_data, tiny_certificate
from aidc.rolling import replay


class ModelTests(unittest.TestCase):
    def test_exhaustive_optimality_certificate(self):
        r=tiny_certificate()
        self.assertTrue(r["passed"])
        self.assertEqual(r["enumerated_feasible_schedules"],6)
        self.assertEqual(r["enumeration_best"]["cost"],3.)

    def test_reproducible_data(self):
        self.assertEqual(canonical_hash(generate(2026)),canonical_hash(generate(2026)))
        self.assertNotEqual(canonical_hash(generate(2026)),canonical_hash(generate(2027)))

    def test_parent_cannot_overlap_child(self):
        d=tiny_data()
        with self.assertRaises(ValueError):
            solve(d,fixed_starts={"A":3,"B":4})
        r=solve(d,fixed_starts={"A":2,"B":2})
        self.assertEqual(r["status"],2)

    def test_deadline_exclusive(self):
        d=tiny_data();d["jobs"]=d["jobs"][:1];d["jobs"][0]["duration"]=2
        r=solve(d,storage=False)
        self.assertLessEqual(r["starts"]["A"]+2,4)
        with self.assertRaises(ValueError):solve(d,fixed_starts={"A":3})

    def test_now_cannot_schedule_unlaunched_job_in_past(self):
        d=tiny_data();d["jobs"]=d["jobs"][:1]
        r=solve(d,now=2)
        self.assertGreaterEqual(r["starts"]["A"],2)

    def test_wait_cap(self):
        d=tiny_data();d["jobs"]=d["jobs"][:1];d["jobs"][0]["max_wait"]=0
        r=solve(d)
        self.assertEqual(r["starts"]["A"],0)

    def test_gpu_bottleneck_infeasible(self):
        d=tiny_data()
        for j in d["jobs"]:j.update(release=0,deadline=1,max_wait=0,parents=[])
        self.assertEqual(solve(d)["status"],2)

    def test_bad_dependency_cycle(self):
        d=tiny_data();d["jobs"][0]["parents"]=["B"]
        with self.assertRaisesRegex(ValueError,"cycle"):validate_data(d)

    def test_bad_trace_length(self):
        d=tiny_data();d["pue"]=[1.]
        with self.assertRaisesRegex(ValueError,"time series"):validate_data(d)

    def test_inference_capacity_screen(self):
        d=tiny_data();d["inference_gpus"]=[2]*4
        with self.assertRaisesRegex(ValueError,"Inference"):validate_data(d)

    def test_hardware_compatibility_screen(self):
        d=tiny_data();d["jobs"][0]["hardware_pool"]="unmodeled"
        with self.assertRaisesRegex(ValueError,"homogeneous"):validate_data(d)

    def test_lp_lower_bound(self):
        d=tiny_data();r=solve(d);lp=solve(d,relax=True)
        self.assertLessEqual(lp["objective"],r["objective"]+1e-7)

    def test_carbon_cap_is_hard(self):
        d=tiny_data()
        self.assertEqual(solve(d,carbon_cap=1.)["status"],2)

    def test_cost_cap_is_hard(self):
        d=tiny_data()
        self.assertEqual(solve(d,objective="carbon",cost_cap=2.)["status"],2)

    def test_carbon_and_cost_conflict(self):
        d=tiny_data();d["jobs"]=d["jobs"][:1]
        d["carbon_kg_per_kwh"]=[.1,1.,.2,.8]
        cheap=solve(d);green=solve(d,objective="carbon")
        self.assertLess(cheap["cost_cny"],green["cost_cny"])
        self.assertGreater(cheap["carbon_kg"],green["carbon_kg"])

    def test_audit_detects_fake_grid(self):
        d=tiny_data();r=solve(d);r["grid_kw"][0]+=1
        self.assertIn("power_balance",audit(d,r)["errors"])

    def test_audit_detects_dropped_job(self):
        d=tiny_data();r=solve(d);del r["starts"]["B"]
        self.assertIn("job_set",audit(d,r)["errors"])

    def test_audit_rejects_nonfinite_trace(self):
        d=tiny_data();r=solve(d);r["grid_kw"][0]=float("nan")
        self.assertFalse(audit(d,r)["passed"])

    def test_audit_rejects_fractional_start(self):
        d=tiny_data();r=solve(d);r["starts"]["A"]=1.5
        self.assertFalse(audit(d,r)["passed"])

    def test_negative_wear_is_rejected(self):
        d=generate();d["batteries"][0]["wear_cny_per_kwh"]=-1.
        with self.assertRaisesRegex(ValueError,"battery parameter"):validate_data(d)

    def test_subunit_pue_is_rejected(self):
        d=tiny_data();d["pue"][0]=.9
        with self.assertRaisesRegex(ValueError,"PUE"):validate_data(d)

    def test_battery_round_trip_and_cyclic_energy(self):
        d=tiny_data();d["jobs"]=[];d["base_it_kw"]=1.
        d["price_cny_per_kwh"]=[.1,.1,3.,3.]
        b=copy.deepcopy(generate()["batteries"][0])
        b.update(capacity_kwh=4.,min_kwh=0.,max_kwh=4.,initial_kwh=2.,charge_kw=2.,discharge_kw=2.,eta_charge=.8,eta_discharge=.8,wear_cny_per_kwh=0.)
        d["batteries"]=[b]
        r=solve(d,gap=0.)
        self.assertTrue(audit(d,r)["passed"])
        c=sum(r["batteries"][0]["charge_kw"]);dis=sum(r["batteries"][0]["discharge_kw"])
        self.assertGreater(c,0.)
        self.assertAlmostEqual(dis/c,.64,places=6)
        self.assertAlmostEqual(r["batteries"][0]["energy_kwh"][-1],2.,places=6)

    def test_storage_can_be_idle_when_uneconomic(self):
        d=tiny_data();d["jobs"]=[];d["base_it_kw"]=1.;d["price_cny_per_kwh"]=[1.]*4
        d["batteries"]=copy.deepcopy(generate()["batteries"])
        r=solve(d)
        self.assertLess(sum(sum(b["charge_kw"])+sum(b["discharge_kw"]) for b in r["batteries"]),1e-6)

    def test_grid_energy_deficit_is_infeasible(self):
        d=tiny_data();d["jobs"]=[];d["base_it_kw"]=2.;d["grid_cap_kw"]=[1.]*4
        self.assertEqual(solve(d)["status"],2)

    def test_fifo_does_not_silently_drop_impossible_jobs(self):
        d=tiny_data();d["jobs"][0]["gpus"]=2
        r=fifo(d)
        self.assertFalse(r["feasible"])
        self.assertIn("A",r["violations"])

    def test_demand_charge_exact_reconstruction(self):
        d=tiny_data();d["demand_rate_cny_per_kw_day"]=7.
        r=solve(d)
        self.assertAlmostEqual(audit(d,r)["metrics"]["demand_cost_cny"],7*r["peak_kw"])

    def test_carbon_objective_accounts_for_actual_peak(self):
        d=tiny_data();d["demand_rate_cny_per_kw_day"]=7.
        r=solve(d,objective="carbon")
        self.assertTrue(audit(d,r)["passed"])

    def test_rolling_information_barrier_and_locks(self):
        d=tiny_data();d["jobs"][1].update(release=2,deadline=4,max_wait=1)
        d["forecast_inference_gpus"]=[0]*4;d["forecast_renewable_kw"]=[0.]*4
        observed=[]
        real_solve=solve
        def spy(data,**kwargs):
            t=kwargs["now"]
            self.assertTrue(all(j["release"]<=t for j in data["jobs"]))
            self.assertEqual(data["renewable_kw"][t+1:],d["forecast_renewable_kw"][t+1:])
            observed.append((t,copy.deepcopy(kwargs["fixed_starts"])))
            return real_solve(data,**kwargs)
        with patch("aidc.rolling.solve",side_effect=spy):r=replay(d,margin=0)
        self.assertTrue(r["feasible"],r)
        self.assertEqual(len(r["log"]),4)
        self.assertTrue(r["audit"]["passed"])
        self.assertNotIn("lower_bound",r)
        for _,locked in observed:
            for jid,start in locked.items():self.assertEqual(start,r["starts"][jid])


if __name__=="__main__":unittest.main(verbosity=2)
