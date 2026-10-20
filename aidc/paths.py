"""Repository paths shared by data, experiment, artifact, and QA tools."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
CONFIG = ROOT / "configs" / "public_study.json"
RESULTS = ROOT / "results" / "main"
QA = ROOT / "results" / "qa"
OUTPUT = ROOT / "output"
FIGURES = OUTPUT / "figures"
REPORTS = OUTPUT / "reports"
DEMO = OUTPUT / "demo"
