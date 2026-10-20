"""Separate an infeasible carbon target from infeasible workload service."""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aidc.data import save_json
from aidc.experiments import attach
from aidc.model import solve
from aidc.paths import RESULTS


def main():
    rows = []
    for folder in ["days", "stress"]:
        for path in sorted((RESULTS / folder).glob("*.json")):
            case = json.loads(path.read_text(encoding="utf-8"))
            if case["result"]["feasible"] or not case["baseline"]["feasible"]:
                continue
            data = case["input"]
            cheap = attach(data, solve(data, time_limit=15))
            green = attach(data, solve(data, objective="carbon", time_limit=20, gap=.001))
            cap = .97 * case["baseline"]["carbon_kg"]
            row = dict(case=f"{folder}/{path.name}", input=data, baseline=case["baseline"],
                       cost_only=cheap, minimum_carbon=green, requested_cap_kg=cap,
                       target_proven_unattainable=green.get("lower_bound", -1) > cap + 1e-6)
            rows.append(row)
            print(row["case"], "cost-only feasible", cheap["feasible"],
                  "3% target proven unattainable", row["target_proven_unattainable"], flush=True)
    save_json(RESULTS / "policy_limits.json", dict(cases=rows,
        interpretation="All admitted workload can remain feasible even when an additional 3% carbon policy is impossible. Bounds are specific to the declared model."))


if __name__ == "__main__":
    main()
