"""Run tests and independently verify all saved executable experiment outputs."""
from pathlib import Path
import io
import json
import re
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aidc.audit import audit
from aidc.data import canonical_hash,save_json
from aidc.paths import RESULTS, QA
from aidc.datasets import verify_raw_files,build_scenario

ROOT=Path(__file__).resolve().parents[1];RES=RESULTS


def read(p):return json.loads(p.read_text(encoding="utf-8"))


def main():
    buffer=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"))
    test=unittest.TextTestRunner(stream=buffer,verbosity=2).run(suite)
    random_cases=sys.modules["test_extended"].RANDOM_EVIDENCE
    save_json(QA/"random_enumeration_qa.json",dict(passed=all(r["passed"] for r in random_cases),cases=random_cases))
    evidence=dict(unit_tests=dict(tests_run=test.testsRun,failures=len(test.failures),
                                 errors=len(test.errors),passed=test.wasSuccessful(),log=buffer.getvalue()),
                  numerical_audits=[],input_hash=canonical_hash(read(RES/"input.json")))
    evidence["random_enumeration"]=dict(cases=len(random_cases),passed=all(r["passed"] for r in random_cases))
    d=read(RES/"input.json")
    for p in sorted(RES.glob("*.json")):
        r=read(p)
        if isinstance(r,dict) and "starts" in r and "grid_kw" in r and r.get("feasible"):
            ds=dict(d,batteries=d["batteries"][:1]) if p.stem=="lfp_only_green" else d
            a=audit(ds,r)
            row=dict(file=p.name,passed=a["passed"],max_residual=a["max_residual"],errors=a["errors"])
            if "lower_bound" in r:
                obj=r["carbon_kg"] if r.get("objective_kind")=="carbon" else r["cost_cny"]
                row["bound_valid"]=r["lower_bound"]<=obj+1e-5
                row["passed"] &= row["bound_valid"]
            evidence["numerical_audits"].append(row)
    for folder in ["stress","days"]:
        for p in sorted((RES/folder).glob("*.json")):
            case=read(p)
            for role in ["baseline","result"]:
                r=case[role]
                if r.get("feasible"):
                    a=audit(case["input"],r)
                    evidence["numerical_audits"].append(dict(file=f"{folder}/{p.name}:{role}",passed=a["passed"],
                                                            max_residual=a["max_residual"],errors=a["errors"]))
    impossible=read(RES/"stress"/"impossible_grid.json")["result"]
    if (RES/"policy_limits.json").exists():
        for case in read(RES/"policy_limits.json")["cases"]:
            for role in ["cost_only","minimum_carbon"]:
                result=case[role]
                if result["feasible"]:
                    checked=audit(case["input"],result)
                    evidence["numerical_audits"].append(dict(file=f"policy_limits:{case['case']}:{role}",
                        passed=checked["passed"],max_residual=checked["max_residual"],errors=checked["errors"]))
    evidence["expected_infeasibility"]=dict(passed=impossible["status"]==2 and not impossible["feasible"],
        status=impossible["status"],energy_diagnostic=impossible.get("energy_diagnostic"))
    rolling=read(RES/"rolling.json")
    evidence["rolling_evidence"]=dict(passed=rolling["feasible"] and len(rolling["log"])==d["T"] and
        all(x["observed_until"]==x["t"]==x["forecast_only_after"] for x in rolling["log"]) and
        "lower_bound" not in rolling,decisions=len(rolling["log"]))
    if (RES/"online_study.json").exists():
        study=read(RES/"online_study.json")
        rows=[dict(date=r["date"],margin=r["margin"],stage=r["stage"],feasible=r["result"]["feasible"],
                   passed=(audit(r["input"],r["result"])["passed"] if r["result"]["feasible"] else "failed_at" in r["result"])) for r in study["runs"]]
        evidence["online_study_audits"]=dict(passed=all(r["passed"] for r in rows),runs=rows)
    raw_manifest=verify_raw_files()
    rebuilt=build_scenario(d["scenario_date_utc"])
    evidence["public_data"]=dict(passed=canonical_hash(rebuilt)==canonical_hash(d),
        raw_files=len(raw_manifest["files"]),source_ids=[r["id"] for r in raw_manifest["sources"]],
        source_gpu_hours=d["source_metadata"]["source_gpu_hours"],
        modeled_gpu_hours=d["source_metadata"]["discretized_gpu_hours"])
    files=list((ROOT/"docs").rglob("*.md"))+[ROOT/"README.md"]+list((ROOT/"tools").glob("*.py"))
    cjk=[]
    for p in files:
        if re.search(r"[\u4e00-\u9fff]",p.read_text(encoding="utf-8")):cjk.append(str(p.relative_to(ROOT)))
    evidence["english_deliverables"]=dict(passed=not cjk,files_with_cjk_prose=cjk)
    evidence["passed"]=bool(test.wasSuccessful() and all(r["passed"] for r in evidence["numerical_audits"]) and
        evidence["expected_infeasibility"]["passed"] and evidence["rolling_evidence"]["passed"] and
        evidence["random_enumeration"]["passed"] and
        evidence.get("online_study_audits",{"passed":True})["passed"] and evidence["public_data"]["passed"] and not cjk)
    save_json(QA/"verification.json",evidence)
    print(f"Tests: {test.testsRun}; numerical audits: {len(evidence['numerical_audits'])}; overall pass: {evidence['passed']}")
    if not evidence["passed"]:
        print(buffer.getvalue())
        print([r for r in evidence["numerical_audits"] if not r["passed"]])
        raise SystemExit(1)


if __name__=="__main__":main()
