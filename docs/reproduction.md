# Reproduction and GitHub Upload

Run from the root with Python 3.11+. Install `requirements.txt`; included excerpts need no API key/network. Optional PDF/render QA: `python -m pip install -e ".[qa]"` (PyMuPDF/Pillow). Compilation needs TeX Live/MiKTeX. Browser QA needs Node, Playwright and Chrome/Edge.

| Task | Command |
|---|---|
| Rebuild public inputs | `python tools/prepare_data.py` |
| Main experiments | `python -m aidc demo --full --rolling --time-limit 15` |
| Tighten incumbents/bounds | `python tools/refine_results.py --time-limit 30`, then `python tools/optimize_headline.py` |
| Failed policies / replay tuning | `python tools/analyze_policy_limits.py`, then `python tools/optimize_online.py` |
| Tests and independent audits | `python tools/verify_results.py` |
| Charts / offline explorer | `python tools/build_artifacts.py` |
| Native LaTeX and PDF | `python tools/build_report.py` (`--source-only` skips compilation) |
| Artifact QA | `python tools/qa_pdf.py`; `python tools/render_report.py` and inspect images; `node tools/qa_browser.cjs`; `python tools/qa_cli.py` |
| Refresh/check five documents | `python tools/build_docs.py`, then `python tools/build_docs.py --check` |
| Prepare upload | `python tools/prepare_push.py` |

Full searches take several minutes; incumbents/runtimes may differ by machine. Report actual bounds/status and audit results. `fetch_open_data.py` optionally redownloads full archives into ignored `tmp/open_data_cache/`; `--offline` uses the cache. No generated-data fallback is used.

Custom day: `python -m aidc generate --date 2020-06-02 --output data/processed/custom.json`, then `solve` and `audit` with explicit paths. Solve/audit exit with 2 on infeasibility/audit failure.

Report source: `output/reports/latex/AIDC_Modeling_Report.tex`, with local figures. Compile independently with `pdflatex -interaction=nonstopmode -halt-on-error AIDC_Modeling_Report.tex` twice in that directory; the canonical PDF is one directory above.

Edit `tools/build_docs.py`, `tools/build_report.py`/`latex_backend.py`, or `templates/results_explorer.html` before regeneration. Recheck changed artifacts; HTML/template and PDF/TeX hashes bind files to QA. Team fields and repository URL survive regeneration.

Upload the **contents of `push/`** as the GitHub root. Suggested name: `aidc-power-compute-scheduling`. It includes code, public excerpts/terms, configuration, processed inputs, results, five documents, report/source and explorer. Originals, archives, caches, environments and compiler intermediates are excluded. Preparation checks copied hashes, reruns tests/audits/CLI inside the snapshot, and writes `PUSH_MANIFEST.json`; browser/PDF evidence is validated by artifact hashes. Fill real contributions and the URL in [submission notes](submission.md). Preparing files does not publish a repository.
