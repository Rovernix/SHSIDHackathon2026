"""Generate five concise English documents from saved public-data evidence."""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aidc.paths import ROOT, RESULTS, QA, CONFIG

GENERATED = {}
DOCUMENTS = ["README.md", "docs/research.md", "docs/results.md", "docs/reproduction.md", "docs/submission.md"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, content):
    GENERATED[path] = content.strip() + "\n"


def results_markdown(d, solutions, bundle):
    baseline, green = solutions["fifo"], solutions["joint_green"]
    lines = ["# Results and Testing Evidence", "",
        f"Saved energy day: **{d['scenario_date_utc']} UTC**. All policies complete the same admitted jobs. Costs are proxies under the [research assumptions](research.md).", "",
        "| Policy | Cost CNY/day | Carbon kg CO2/day | Grid peak kW | Cost saved | MIP gap |", "|---|---:|---:|---:|---:|---:|"]
    labels = {"fifo": "FIFO", "fifo_storage": "FIFO + storage", "compute_only": "Compute only", "joint_cost": "Joint cost", "joint_green": "Joint green"}
    for key, label in labels.items():
        row = solutions[key]
        gap = f"{100*row['mip_gap']:.6f}%" if "mip_gap" in row else "N/A"
        lines.append(f"| {label} | {row['cost_cny']:.2f} | {row['carbon_kg']:.2f} | {row['peak_kw']:.2f} | {100*(1-row['cost_cny']/baseline['cost_cny']):.2f}% | {gap} |")
    lines += ["", f"Green upper/lower bounds: CNY {green['cost_cny']:.9f}/{green['lower_bound']:.9f}; gap {100*green['mip_gap']:.6f}%. The chosen carbon ceiling is 97% of FIFO. [Schedule](../results/main/joint_green_jobs.csv), [dispatch](../results/main/joint_green_dispatch.csv) and [full results](../results/main/summary.json) retain the evidence.", "",
        f"Under the same ceiling, FIFO + storage and compute-only are infeasible. LFP-only costs CNY {read(RESULTS/'lfp_only_green.json')['cost_cny']:.2f}. The second battery's modeled increment does not establish fast response or investment return.", "",
        "## Other dates and stresses", "", "| Public date | FIFO feasible | Green feasible | Cost saved |", "|---|---|---|---:|"]
    for row in bundle["days"]:
        saving = f"{row['cost_saving_percent']:.2f}%" if "cost_saving_percent" in row else "N/A"
        lines.append(f"| {row['date']} | {row['baseline_feasible']} | {row['feasible']} | {saving} |")
    lines += ["", "June 3 and June 8 have feasible cost-only service, but [minimum-carbon bounds](../results/main/policy_limits.json) exclude the extra 3% target. No failed date is removed.", "",
        "| Constructed stress | FIFO feasible | Green feasible | Cost saved |", "|---|---|---|---:|"]
    for row in bundle["stress"]:
        saving = f"{row['cost_saving_percent']:.2f}%" if "cost_saving_percent" in row else "N/A"
        label = "Background GPU surge" if row["scenario"] == "inference_surge" else row["scenario"].replace("_", " ")
        lines.append(f"| {label} | {row['baseline_feasible']} | {row['feasible']} | {saving} |")
    lines += ["", "Stresses are constructed, not historical incidents. Background surge adds 48 GPUs; impossible grid sets both grid and renewable supply to zero. Negative savings/failures remain visible; N/A means no feasible same-scenario comparison.", "",
        "## Online replay", "", "| Policy | Cost CNY | Carbon kg CO2 | Peak kW |", "|---|---:|---:|---:|"]
    cfg, study = read(CONFIG), read(RESULTS/"online_study.json")
    for label, row in [(f"Default reserve {cfg['online_reserve_gpus']} GPUs", read(RESULTS/"rolling.json")), (f"Selected reserve {study['selected_margin']} GPUs", read(RESULTS/"rolling_optimized.json"))]:
        lines.append(f"| {label} | {row['cost_cny']:.2f} | {row['carbon_kg']:.2f} | {row['peak_kw']:.2f} |")
    lines += ["", "Both main replays complete 96 decisions. Candidates 0/2/4/8 are selected on June 2-3 by failures, then cost; June 4-5 provide separate evaluation. The [online study](../results/main/online_study.json) retains every run and constructed shock. The explorer shows the default replay. Neither online policy enforces the offline carbon ceiling or has a global policy certificate.", "", "## Verification", ""]
    evidence = read(QA/"verification.json")
    lines += [f"The saved run passes **{evidence['unit_tests']['tests_run']} unit tests** and **{len(evidence['numerical_audits'])} independent numerical audits**. Sixty brute-force random cases are subcases of three tests. Checks cover source hashes/fields, conversions, timing, complete jobs, GPU/power limits, unique configured batteries, all energy states, finite arithmetic, accounting, causal locks and tampered outputs.", "",
        "| Evidence | Scope |", "|---|---|",
        "| [Numerical verification](../results/qa/verification.json) | Unit log, audit residuals, source rebuild and 96-step replay |",
        "| [Exact oracle](../results/qa/random_enumeration_qa.json) | 60 independent tiny-instance comparisons |",
        "| [CLI](../results/qa/cli_qa.json) | Generation, solve, audit and infeasible exit status |",
        "| [Browser](../results/qa/browser_qa.json) | Six policies at 390/1280 px; cards, search, last slot and layout |",
        "| [PDF](../results/qa/pdf_structure_qa.json), [visual review](../results/qa/render_qa.json) | Current report, fonts, bounds, key results and rendered pages |",
        "| [Upload](../results/qa/push_validation.json) | Five documents, links, artifact hashes and standalone snapshot |", "",
        "These checks validate the declared model, not monthly billing, request latency, electrical transients, heterogeneous placement or causal grid decarbonization. [Reproduce](reproduction.md)."]
    return "\n".join(lines)


def team_fields():
    team = "| Member | Actual contribution | Personal verification |\n|---|---|---|\n| To be completed by entrant | To be completed | To be completed |"
    url = "Public GitHub repository URL: to be completed after publication by the entrant."
    path = ROOT/"docs/submission.md"
    if path.exists():
        content = path.read_text(encoding="utf-8")
        team = content.split("## Team and actual contributions", 1)[1].split("## AI disclosure", 1)[0].strip()
        url = next((line for line in content.splitlines() if line.startswith("Public GitHub repository URL:")), url)
    return team, url


def main(check_only=False):
    GENERATED.clear()
    d, bundle, cfg = read(RESULTS/"input.json"), read(RESULTS/"summary.json"), read(CONFIG)
    solutions = {key: read(RESULTS/f"{key}.json") for key in ["fifo", "fifo_storage", "compute_only", "joint_cost", "joint_green"]}
    base, green = solutions["fifo"], solutions["joint_green"]
    saving, peak = (100*(1-green[key]/base[key]) for key in ["cost_cny", "peak_kw"])
    original, work = d["source_metadata"]["source_gpu_hours"], green["audit"]["metrics"]["gpu_hours"]
    write("README.md", f'''# AI Data Center Power and Compute Co-Scheduling

A HackSHSID 2026 Track 2 prototype that schedules GPU jobs and batteries together. Flexible computing and storage both shift electricity demand across time; coordinating them exposes the cost/carbon trade-off while preserving admitted work.

The benchmark combines **Helios GPU job records**, **OPSD British electricity/solar observations**, and **UK Carbon Intensity estimates**. It is a trace-driven counterfactual: deadlines, power, asset sizes, conversion, solar scaling and relocation remain assumptions.

On {d['scenario_date_utc']}, joint green completes all {len(d['jobs'])} jobs, reduces modeled daily cost from **CNY {base['cost_cny']:.2f} to {green['cost_cny']:.2f} ({saving:.2f}%)**, meets a chosen **3% carbon-accounting reduction**, and lowers peak grid import **{peak:.2f}%**. A start-indexed MILP solved by SciPy/HiGHS provides a **{100*green['mip_gap']:.6f}%** objective-gap certificate. Other dates include infeasible targets and negative stress savings.

[Research, model and references](docs/research.md) · [Results and testing](docs/results.md) · [Reproduction and upload](docs/reproduction.md) · [Submission notes](docs/submission.md)

[English PDF](output/reports/AIDC_Modeling_Report.pdf) · [Native LaTeX](output/reports/latex/AIDC_Modeling_Report.tex) · [Offline explorer](output/demo/Results_Explorer.html)

```powershell
python -m pip install -r requirements.txt
python tools/prepare_data.py
python tools/verify_results.py
```

Included public excerpts support offline reconstruction; missing/corrupted records raise errors. Artificial tiny/random inputs test correctness, not performance. Every admitted job must complete, and batteries must restore initial energy.

Code: `aidc/`; assumptions: `configs/`; data: `data/`; experiments/QA: `results/`; report, figures and explorer: `output/`. The five Markdown documents are generated by `tools/build_docs.py`; `--check` rejects stale text. `tools/prepare_push.py` independently checks the standalone GitHub folder.

OpenAI Codex assisted with research, modeling, programming, extraction, experiments and writing. Entrants must understand/verify the work and record real contributions in the submission notes. Attribution and source-specific terms remain in `data/raw/`.''')
    write("docs/research.md", rf'''# Research, Data and Model

## Data and assumptions

The main case is a composite study, not a measured British facility or complete historical Helios replay. [Configuration](../configs/public_study.json) separates assumptions from [public records](../data/raw/manifest.json).

| Input | Source and transformation | Boundary |
|---|---|---|
| GPU jobs | Helios Saturn / {cfg['helios_virtual_cluster']}; completed GPU jobs lasting {cfg['minimum_duration_seconds']:,}-{cfg['maximum_duration_seconds']:,} s, submitted before {cfg['latest_submission_hour']}:00; first at most {cfg['max_jobs']} by submission/ID | {len(d['jobs'])} jobs on the main day; retain submit/start/end, GPUs, execution seconds and source row. No source deadlines or invented dependencies. |
| Capacity/background | Published {d['gpu_capacity'][0]}-GPU VC allocation; slot maximum of same-VC short-job concurrency | Homogeneous equivalents; short jobs are not labeled inference. Unselected jobs are outside the workload. |
| Price | OPSD `GB_GBN_price_day_ahead`, GBP/MWh, multiplied by assumed {cfg['assumed_cny_per_gbp']:g}/1,000 | Fixed study conversion, not historical FX or a retail tariff; negative prices retained. |
| Solar | OPSD `GB_GBN_solar_generation_actual`, MW, multiplied by assumed {cfg['assumed_renewable_scale_kw_per_mw']:g} kW/MW | Scaled national generation, not an on-site PV meter; missing capacity factors are not filled. |
| Carbon | UK API national `actual`, g CO2/kWh divided by 1,000 | Estimated average intensity, not marginal or causal emissions. |

Energy records align by UTC interval start. Helios has no documented timezone; its within-day offsets are rebased onto the model day. Release/duration round upward to 15 minutes: **{original:.2f} observed GPU-hours become {work:.2f} modeled GPU-hours ({100*(work/original-1):.2f}% padding)**. Assumed deadline is release + duration + up to four hours waiting, capped at day end. Source completion is not an empirical deadline.

Hourly/half-hourly records are held constant. Future solar/background use previous-day same-clock persistence. Online causality is between slots: current-slot aggregates are optimistically assumed known at its boundary. Archived carbon forecasts lack verified issuance vintage and do not affect the online cost objective/constraints.

Assumed base IT load: {cfg['assumed_base_it_kw']:g} kW; active/background GPU power: {cfg['assumed_job_incremental_kw_per_gpu']:g} kW/GPU; PUE: {cfg['assumed_pue']:g}; facility/grid caps: {cfg['assumed_facility_cap_kw']:g}/{cfg['assumed_grid_cap_kw']:g} kW; successive grid ramp: {cfg['assumed_grid_ramp_kw_per_step']:g} kW; renewable cost: {cfg['assumed_renewable_cny_per_kwh']:g} CNY/kWh; daily peak proxy: {cfg['assumed_demand_cny_per_kw_day']:g} CNY/(kW day). Battery ratings/wear are in the configuration.

## Formulation

There are 96 slots with Δt = 0.25 h. Job j has release r, exclusive deadline d, duration L, maximum wait W, demand g GPUs and incremental power p kW. Binary z selects one start; activity a is substituted into the sparse matrix.

$$F_j=\{{s\in\mathbb Z:r_j\le s\le\min(d_j-L_j,r_j+W_j)\}},\quad \sum_{{s\in F_j}}z_{{js}}=1,\quad a_{{jt}}=\sum_{{s\in F_j:s\le t<s+L_j}}z_{{js}}.$$

$$\sum_jg_ja_{{jt}}+n_t\le N_t,\qquad P_t^{{fac}}=\kappa_t(P_{{base}}+q n_t+\sum_jp_ja_{{jt}}).$$

Here n is background reservation, N GPU capacity, κ PUE, G grid import, R renewable use, and c/u battery charge/discharge (kW).

$$G_t+R_t+\sum_bu_{{bt}}=P_t^{{fac}}+\sum_bc_{{bt}},\quad 0\le G_t\le\bar G_t,\quad0\le R_t\le\bar R_t,\quad P_t^{{fac}}\le\bar P^{{fac}},\quad |G_t-G_{{t-1}}|\le\Gamma\;(t\ge1).$$

Each battery has energy E (kWh), efficiencies η, physical limits and reserved energy/power. Binary v forbids simultaneous charge/discharge within that battery; different batteries may operate in opposing directions.

$$E_{{b,t+1}}=E_{{bt}}+\eta_b^cc_{{bt}}\Delta t-u_{{bt}}\Delta t/\eta_b^d,\quad E_b^{{min}}+E_b^{{res}}\le E_{{bt}}\le E_b^{{max}},\quad E_{{b0}}=E_{{bT}}=E_b^{{init}}.$$

$$0\le c_{{bt}}\le(\bar c_b-P_b^{{res}})v_{{bt}},\qquad0\le u_{{bt}}\le(\bar u_b-P_b^{{res}})(1-v_{{bt}}).$$

With price π, renewable cost π_R, throughput wear k, peak proxy λ, and average grid intensity ξ:

$$C=\Delta t\sum_t(\pi_tG_t+\pi_RR_t)+\lambda\max_tG_t+\Delta t\sum_{{b,t}}k_b(c_{{bt}}+u_{{bt}}),\quad M=\Delta t\sum_t\xi_tG_t,\quad\min C\text{{ subject to }}M\le0.97M_{{FIFO}}.$$

The library supports/tests DAG precedence (parent finishes before child starts), stricter waiting, carbon objectives and cost ceilings. The public cohort has no source DAG. Legacy `inference_gpus`/`training_gpus` fields mean background/scheduled occupancy, not verified workload labels.

## Algorithm and validity

SciPy `milp` calls HiGHS branch-and-bound/cut methods. Feasible incumbent U and target-problem bound L bracket the optimum; report (U-L)/|U| and actual status. Status 0 means tolerance reached, 1 a time/resource limit, 2 infeasibility. LP relaxation is a lower bound, not an executable fractional schedule. Incumbents may transfer to looser carbon budgets; bounds remain target-specific.

The independent audit validates inputs, rejects duplicate/unknown batteries and nonfinite arithmetic, then reconstructs jobs, GPU limits, load, power balance, grid/solar/ramp, energy states and cost/carbon without reading the MILP matrix. Online replay locks history, hides future releases and replans each slot; it has no whole-day green cap or global policy certificate.

Grid charging is counted once; batteries restore initial energy. Average-carbon accounting excludes lifecycle emissions. Power, deadlines, storage, daily peak billing and relocated runtimes remain assumptions. Aggregate GPUs do not certify CPU/RAM/topology or inference latency; a 15-minute model does not validate transients. Capex, TCO and ROI are outside scope.

## References

1. [HeliosData](https://github.com/S-Lab-System-Group/HeliosData), SC 2021, *Characterization and Prediction of Deep Learning Workloads in Large-Scale GPU Datacenters*. CC BY 4.0; commit `159f0caeec16600b9b6017862952a36aae01c43f`.
2. [Open Power System Data, Time series 2020-10-06](https://doi.org/10.25832/time_series/2020-10-06). Primary ENTSO-E records; source-specific terms retained. Archived SQLite SHA-256 matches the [publisher checksum](../data/raw/opsd/checksums.txt); the pinned mirror is a transport route.
3. [UK Carbon Intensity API](https://carbonintensity.org.uk/), [API definitions](https://carbon-intensity.github.io/api-definitions/). National estimates/archived forecasts; CC BY 4.0.
4. [SciPy `milp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html). Solver interface, statuses, gaps and bounds.
5. [Radovanovic et al., 2021, Carbon-Aware Computing for Datacenters](https://arxiv.org/abs/2106.11750). Temporal workload-shifting motivation.
6. [HiTHIUM storage announcement](https://www.hithium.com/newsroom/latest/details/86.html). Storage motivation; modeled assets are assumptions.
7. Supplied HackSHSID Track 2 statement, project brief and technical specification. [Source inventory](../data/source_documents.json); originals excluded from the upload snapshot.
8. IMMC 2025 Problem E, *When Watts Meet Bits*. Supplied reference; [source inventory](../data/source_documents.json).

Full URLs, pinned commits, licenses and SHA-256 values are in the [data manifest](../data/raw/manifest.json).''')
    write("docs/results.md", results_markdown(d, solutions, bundle))
    write("docs/reproduction.md", '''# Reproduction and GitHub Upload

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

Upload the **contents of `push/`** as the GitHub root. Suggested name: `aidc-power-compute-scheduling`. It includes code, public excerpts/terms, configuration, processed inputs, results, five documents, report/source and explorer. Originals, archives, caches, environments and compiler intermediates are excluded. Preparation checks copied hashes, reruns tests/audits/CLI inside the snapshot, and writes `PUSH_MANIFEST.json`; browser/PDF evidence is validated by artifact hashes. Fill real contributions and the URL in [submission notes](submission.md). Preparing files does not publish a repository.''')
    team, url = team_fields()
    write("docs/submission.md", f'''# Submission Notes

**AI Data Center Power and Compute Co-Scheduling** coordinates flexible GPU jobs with LFP/sodium-ion storage. Public workload/electricity records support an auditable MILP with an explicit carbon policy; admitted work and terminal battery energy are preserved.

The [README](../README.md) contains the introduction and inspiration; [research](research.md) covers assumptions and equations; [results](results.md) contains tests/failures. Deliver the [PDF](../output/reports/AIDC_Modeling_Report.pdf), [LaTeX](../output/reports/latex/AIDC_Modeling_Report.tex), [explorer](../output/demo/Results_Explorer.html) and curated repository.

Demonstration: identify observations versus assumptions; compare FIFO and green cost; inspect a job and battery states; show the bound/audit; then show an infeasible carbon-target date and online replay versus hindsight. The 3% reduction is selected policy. Results establish modeled performance, not utility savings or deployment readiness.

## Team and actual contributions

{team}

## AI disclosure

OpenAI Codex assisted with research, modeling, programming, extraction, experiments and writing. Entrants remain responsible for understanding, verification and truthful contributions. Enter real identities, publish through the entrant's account, and submit the description, evidence and URL. The event announcement governs the final deadline.

{url}''')
    assert set(GENERATED) == set(DOCUMENTS)
    stale = [name for name, text in GENERATED.items() if not (ROOT/name).is_file() or (ROOT/name).read_text(encoding="utf-8") != text]
    if not check_only:
        for name, content in GENERATED.items():
            path = ROOT/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    return dict(passed=not stale if check_only else True, generated_documents=len(GENERATED), outdated=stale, check_only=check_only)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compare expected documents without writing")
    result = main(check_only=parser.parse_args().check)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
