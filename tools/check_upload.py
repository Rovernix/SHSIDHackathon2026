"""Validate the standalone upload snapshot and refresh its SHA-256 manifest.

Uses only the standard library. Core tests are run separately with
``python tools/verify_results.py``; saved browser/PDF evidence is checked here,
not regenerated. Caches, local environments, and compiler intermediates are
excluded from the upload manifest.
"""
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
import ast
import tomllib
from build_docs import main as check_documentation, DOCUMENTS


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", ".pytest_cache", "tmp", "archive", "push"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".aux", ".log", ".out", ".toc", ".fls", ".fdb_latexmk"}


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def upload_files():
    files = []
    candidates=[]
    for folder in ["aidc","configs","data","docs","tests","tools","templates","results",
                   "output/reports","output/demo","output/figures"]:
        candidates.extend((ROOT/folder).rglob("*"))
    candidates.extend(ROOT/name for name in ["README.md","requirements.txt","pyproject.toml",".gitignore"])
    for path in candidates:
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        if path.suffix in SKIP_SUFFIXES or path.name.endswith(".synctex.gz"):
            continue
        if relative.parent.as_posix() == "output/reports/latex" and path.name.startswith("compile-pass-"):
            continue
        if path.name.startswith(".env") or path.name in {".DS_Store", "Thumbs.db", "PUSH_MANIFEST.json"}:
            continue
        if relative.as_posix() in {
            "output/reports/latex/AIDC_Modeling_Report.pdf",
            "output/HackSHSID_AIDC_Submission.zip",
            "output/AIDC_LaTeX_Source.zip",
        }:
            continue
        files.append(path)
    return sorted(set(files))


def main():
    checks = []

    def check(name, passed, detail):
        checks.append(dict(name=name, passed=bool(passed), detail=detail))

    required = [*DOCUMENTS, ".gitignore", "requirements.txt",
                "aidc/__main__.py", "data/processed/main.json", "data/raw/manifest.json", "tests/test_open_data.py",
                "tests/test_audit_validation.py",
                "configs/public_study.json", "pyproject.toml", "data/source_documents.json", "templates/results_explorer.html",
                "output/reports/AIDC_Modeling_Report.pdf", "output/demo/Results_Explorer.html",
                "output/reports/latex/AIDC_Modeling_Report.tex"]
    missing = [name for name in required if not (ROOT / name).is_file()]
    check("required_files", not missing, missing)
    metadata=tomllib.loads((ROOT/"pyproject.toml").read_text(encoding="utf-8"))
    module=ast.parse((ROOT/"aidc/__init__.py").read_text(encoding="utf-8"))
    version=next(ast.literal_eval(node.value) for node in module.body if isinstance(node,ast.Assign) and
        any(isinstance(target,ast.Name) and target.id=="__version__" for target in node.targets))
    check("package_version_single_source", metadata["tool"]["setuptools"]["dynamic"]["version"]["attr"]=="aidc.__version__" and
        "version" in metadata["project"]["dynamic"],version)

    files = upload_files()
    json_errors = []
    for path in files:
        if path.suffix == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, UnicodeError) as error:
                json_errors.append(dict(file=path.relative_to(ROOT).as_posix(), error=str(error)))
    check("json_files_parse", not json_errors, json_errors)

    broken_links = []
    local_link_count = 0
    generated_outputs = {ROOT / "results/qa/push_validation.json", ROOT / "PUSH_MANIFEST.json"}
    for path in files:
        if path.suffix != ".md" or path.is_relative_to(ROOT/"data/raw"):
            continue
        for target in re.findall(r"\[[^\]\n]*\]\(([^)\n]+)\)", path.read_text(encoding="utf-8")):
            target = target.strip().strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or not parsed.path:
                continue
            local_link_count += 1
            resolved = (path.parent / unquote(parsed.path)).resolve()
            if not resolved.is_relative_to(ROOT) or (not resolved.exists() and resolved not in generated_outputs):
                broken_links.append(dict(file=path.relative_to(ROOT).as_posix(), target=target))
    check("local_markdown_links", not broken_links, dict(checked=local_link_count, broken=broken_links))

    documentation = check_documentation(check_only=True)
    check("generated_documentation_matches_saved_evidence", documentation["passed"], documentation)
    authored = {p.relative_to(ROOT).as_posix() for p in files if p.suffix == ".md" and not p.is_relative_to(ROOT/"data/raw")}
    check("concise_document_set", authored == set(DOCUMENTS), sorted(authored))
    research = (ROOT/"docs/research.md").read_text(encoding="utf-8")
    prose_without_references = research.split("## References", 1)[0] + "\n".join(
        (ROOT/name).read_text(encoding="utf-8") for name in DOCUMENTS if name != "docs/research.md")
    check("supplemental_material_is_citation_only", "IMMC" not in prose_without_references and
          "IMMC" not in (ROOT/"output/demo/Results_Explorer.html").read_text(encoding="utf-8"), "Citation remains in research references only")

    tex = ROOT / "output/reports/latex/AIDC_Modeling_Report.tex"
    figures = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex.read_text(encoding="utf-8"))
    missing_figures = [name for name in figures if not (tex.parent / name).is_file()]
    check("portable_latex_figures", bool(figures) and not missing_figures,
          dict(checked=len(figures), missing=missing_figures))

    verification = read("results/qa/verification.json")
    audit_rows = verification["numerical_audits"]
    check("recorded_core_verification", verification["passed"] and
          verification["unit_tests"]["tests_run"] >= 74 and verification["public_data"]["passed"] and
          all(row["passed"] for row in audit_rows),
          dict(unit_tests=verification["unit_tests"]["tests_run"], numerical_audits=len(audit_rows)))
    browser = read("results/qa/browser_qa.json")
    pdf = read("results/qa/pdf_structure_qa.json")
    check("recorded_browser_and_pdf_evidence", browser["passed"] and len(browser["cases"]) == 12 and
          pdf["passed"] and pdf["page_count"] == 7,
          dict(browser_cases=len(browser["cases"]), pdf_pages=pdf["page_count"],
               fresh_browser_or_pdf_run=False))
    check("report_matches_pdf_qa_hash", sha256(ROOT / "output/reports/AIDC_Modeling_Report.pdf") == pdf["sha256"],
          pdf["sha256"])
    check("latex_source_matches_recorded_pdf_qa", sha256(tex) == pdf.get("tex_sha256"), pdf.get("tex_sha256"))

    check("explorer_matches_browser_qa_hash", sha256(ROOT/"output/demo/Results_Explorer.html")==browser["explorer_sha256"],browser["explorer_sha256"])
    check("explorer_template_matches_recorded_browser_qa", sha256(ROOT/"templates/results_explorer.html")==browser.get("template_sha256"),browser.get("template_sha256"))
    raw_manifest=read("data/raw/manifest.json")
    mismatches=[row["path"] for row in raw_manifest["files"] if sha256(ROOT/"data/raw"/row["path"])!=row["sha256"]]
    check("public_source_excerpt_hashes",not mismatches,dict(files=len(raw_manifest["files"]),mismatches=mismatches))
    canonical=lambda data:hashlib.sha256(json.dumps(data,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    check("processed_input_matches_experiment",canonical(read("data/processed/main.json"))==canonical(read("results/main/input.json")),"Semantic JSON hashes")

    cjk_files = [path.relative_to(ROOT).as_posix() for path in files
                 if path.suffix in {".md", ".py"} and not path.is_relative_to(ROOT/"data/raw") and
                 re.search(r"[\u4e00-\u9fff]", path.read_text(encoding="utf-8"))]
    check("english_prose_and_source", not cjk_files, cjk_files)
    original_documents = [path.relative_to(ROOT).as_posix() for path in files
                          if path.suffix.lower() == ".docx" or
                          (path.suffix.lower() == ".pdf" and
                           path.relative_to(ROOT).as_posix() != "output/reports/AIDC_Modeling_Report.pdf")]
    check("original_problem_documents_excluded", not original_documents, original_documents)

    green = read("results/main/joint_green.json")
    baseline = read("results/main/fifo.json")
    summary = read("results/main/summary.json")["strategies"]["joint_green"]
    check("headline_summary_matches_schedule", all(abs(summary[key] - green[key]) < 1e-7
          for key in ["cost_cny", "carbon_kg", "peak_kw"]),
          {key: green[key] for key in ["cost_cny", "carbon_kg", "peak_kw"]})
    metrics = {key + "_reduction_percent": 100 * (1 - green[key] / baseline[key])
               for key in ["cost_cny", "carbon_kg", "peak_kw"]}
    report = dict(passed=all(row["passed"] for row in checks),
                  checked_at_utc=datetime.now(UTC).isoformat(), checks=checks,
                  headline_metrics=metrics,
                  note="Core verification is run separately. Browser/PDF checks validate saved evidence and unchanged artifact hashes.")
    (ROOT / "results/qa/push_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    files = upload_files()
    manifest = dict(algorithm="SHA-256", excludes_self=True,
                    file_count=len(files), total_bytes=sum(path.stat().st_size for path in files),
                    files=[dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                                sha256=sha256(path)) for path in files])
    (ROOT / "PUSH_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Upload checks: {len(checks)}; passed: {report['passed']}; manifested files: {len(files)}")
    if not report["passed"]:
        print(json.dumps([row for row in checks if not row["passed"]], indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
