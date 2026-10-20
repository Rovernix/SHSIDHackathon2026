"""Exercise public input generation and the CLI's successful/failing exit codes."""
from pathlib import Path
import json
import os
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aidc.data import canonical_hash, save_json
from aidc.experiments import tiny_data
from aidc.paths import ROOT, QA


def main():
    folder = ROOT / "tmp/cli"
    folder.mkdir(parents=True, exist_ok=True)
    save_json(folder/"tiny.json", tiny_data())
    cases = []
    commands = [
        (["generate", "--date", "2020-06-01", "--output", "tmp/cli/public.json"], 0),
        (["solve", "tmp/cli/tiny.json", "--output", "tmp/cli/solution.json"], 0),
        (["audit", "tmp/cli/tiny.json", "tmp/cli/solution.json"], 0),
        (["solve", "tmp/cli/tiny.json", "--carbon-cap", "1", "--output", "tmp/cli/infeasible.json"], 2),
    ]
    for arguments, expected in commands:
        run = subprocess.run([sys.executable, "-B", "-m", "aidc", *arguments], cwd=ROOT,
                             capture_output=True, text=True, timeout=60)
        cases.append(dict(command="python -m aidc " + " ".join(arguments), expected_exit_code=expected,
                          actual_exit_code=run.returncode, passed=run.returncode==expected,
                          output=run.stdout, errors=run.stderr))
    actual = json.loads((folder/"public.json").read_text(encoding="utf-8"))
    saved = json.loads((ROOT/"data/processed/main.json").read_text(encoding="utf-8"))
    matches = canonical_hash(actual) == canonical_hash(saved)
    result = dict(passed=all(row["passed"] for row in cases) and matches,
                  cases=cases, public_input_reproducible=matches,
                  note="Public-data generation is tested on the main date; solve/audit/expected infeasibility use an artificial two-job correctness fixture.")
    save_json(QA/"cli_qa.json", result)
    print(f"CLI cases: {len(cases)}; public input reproducible: {matches}; passed: {result['passed']}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
