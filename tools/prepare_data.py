"""Verify shipped raw observations and build all configured public study days."""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aidc.data import save_json
from aidc.datasets import build_scenario
from aidc.paths import CONFIG, PROCESSED


def main():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for day in [config["main_date"]] + config["evaluation_dates"]:
        data = build_scenario(day)
        save_json(PROCESSED / (day + ".json"), data)
        if day == config["main_date"]:
            save_json(PROCESSED / "main.json", data)
        print(day, len(data["jobs"]), "jobs", data["source_metadata"]["source_gpu_hours"], "source GPU-hours")


if __name__ == "__main__":
    main()
