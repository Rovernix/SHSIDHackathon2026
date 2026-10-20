"""Build the concise English LaTeX report from audited public-data results."""
from latex_backend import *
from xml.sax.saxutils import escape as markup_escape


def build(source_only=False):
    story.clear()
    d, bundle = read(RES/"input.json"), read(RES/"summary.json")
    cfg, checks = read(ROOT/"configs/public_study.json"), read(QA/"verification.json")
    base, green, cheap = (read(RES/f"{key}.json") for key in ["fifo", "joint_green", "joint_cost"])
    study = read(RES/"online_study.json")
    original, work = d["source_metadata"]["source_gpu_hours"], green["audit"]["metrics"]["gpu_hours"]

    text("AI Data Center<br/>Power and Compute Co-Scheduling", "title")
    text(f"HackSHSID 2026 Track 2 | Public-data study | {d['scenario_date_utc']} energy day", "caption")
    sub("Purpose and result")
    text("Flexible computing and batteries both move electricity demand across time. We coordinate non-preemptive GPU job starts and two storage layers to reduce operating cost while preserving admitted work and an explicit carbon policy.")
    text(f"All {len(d['jobs'])} jobs finish before their assumed deadlines. Joint green reduces modeled cost by {pct(green,base,'cost_cny'):.2f}% and peak import by {pct(green,base,'peak_kw'):.2f}%, while meeting a chosen 3% carbon-accounting reduction. Both batteries restore initial energy.")
    table(["Metric", "FIFO", "Joint green"], [
        ["Cost proxy / CNY per day", fmt(base["cost_cny"]), fmt(green["cost_cny"])],
        ["Carbon accounting / kg CO2", fmt(base["carbon_kg"]), fmt(green["carbon_kg"])],
        ["Peak grid import / kW", fmt(base["peak_kw"]), fmt(green["peak_kw"])],
        ["Modeled GPU-hours", fmt(work), fmt(work)],
    ], [245, 119, 119])
    picture("comparison", maxheight=180, caption="Figure 1. Same admitted workload; all five saved policies independently audited.")
    text("The benchmark combines observed Helios GPU jobs, OPSD British electricity/solar records and UK carbon estimates. It is a counterfactual composite, not a measured British facility: deadlines, power, assets, currency conversion, national-solar scaling and workload relocation remain assumptions.")
    text(f"SciPy/HiGHS solves a start-indexed mixed-integer linear program. The green objective gap is {100*green['mip_gap']:.6f}%. Other public dates include unattainable targets; constructed stresses include negative cost savings. Neither failures nor work are discarded.", "small")

    title(2, "Data and study assumptions")
    table(["Input", "Source and transformation"], [
        ["GPU jobs", f"Helios Saturn / {cfg['helios_virtual_cluster']}: completed GPU jobs, 1,800-10,800 s, submitted before 18:00; first at most 32 by submission then source ID"],
        ["Capacity/background", f"{d['gpu_capacity'][0]} published VC GPUs; fixed background is each slot's maximum concurrency from same-VC completed GPU jobs shorter than 1,800 s"],
        ["Electricity", "OPSD GB_GBN_price_day_ahead, GBP/MWh x assumed 9 CNY/GBP / 1,000; negative prices retained"],
        ["Solar", "OPSD actual national solar MW x assumed 0.004 kW/MW; missing capacity factors are not filled"],
        ["Carbon", "UK national API actual field, estimated average g CO2/kWh / 1,000"],
    ], [100, 383], 8.3)
    text(f"The {len(d['jobs'])} source jobs retain IDs, CSV rows, GPU counts, submit/start/end times and execution seconds. Upward 15-minute rounding gives {fmt(work)} modeled GPU-hours from {fmt(original)} observed GPU-hours ({100*(work/original-1):.2f}% padding). Deadlines allow up to four hours waiting after rounded duration, capped at day end. No dependency graph or inference label is invented.")
    text("Energy records align to UTC interval starts. Helios timestamps have no documented timezone; within-day offsets are rebased onto the modeled British day. Hourly/half-hourly observations are held constant, without interpolated noise. The completed-job subset is not a representative sample of all Helios work.")
    table(["Assumption", "Configured value"], [
        ["Base / active or reserved GPU / PUE", "20 kW / 0.25 kW per GPU / 1.20"],
        ["Facility / grid / successive ramp", "150 kW / 120 kW / 50 kW per step"],
        ["Renewable cost / daily peak proxy", "0.06 CNY/kWh / 2 CNY per kW-day"],
        ["LFP / sodium-ion capacity", "40 kWh / 6 kWh"],
    ], [270, 213], 8.5)
    text("Helios is commit/archive-hash pinned. OPSD's database was recovered through a pinned archive mirror and exactly matches the publisher's SHA-256; the mirror is a transport route. Source excerpts, metadata, terms and hashes ship in data/raw/. Missing/corrupted required observations fail explicitly.", "small")
    picture("inputs", maxheight=260, caption="Figure 2. Public signals and derived inputs; solar/background forecasts use previous-day persistence.")

    title(3, "Job scheduling and power balance")
    table(["Symbol", "Meaning and unit"], [
        ["t, s; delta t", "Slot/start indices; slot duration 0.25 h"],
        ["r, d, L, W; g, p", "Release, exclusive deadline, duration, max wait in slots; GPUs, incremental kW"],
        ["z, a; n, N", "Binary start and derived activity; background GPUs and capacity"],
        ["G, R, c, u; E", "Grid, renewables, battery charge/discharge kW; stored energy kWh"],
        ["kappa, pi, xi", "PUE; converted wholesale CNY/kWh; average kg CO2/kWh"],
    ], [145, 338], 8.5)
    eq("starts")
    eq("activity")
    text("Exactly one start executes each admitted job for L consecutive slots; finishing at d is allowed, executing in slot d is not. The main pool is homogeneous GPU equivalents. Host memory/CPU and fabric placement remain outside this aggregate model.")
    eq("dag")
    text("Parent completion precedes child start when a DAG is supplied. This capability is tested; the public cohort has no supplied edges.")
    eq("facility")
    eq("balance")
    eq("limits")
    text("Base IT power and incremental job/background power are distinct. Renewables may be curtailed; exports are forbidden. Grid ramp applies between successive slot averages, with no previous-day boundary. Legacy inference_gpus/training_gpus fields denote background/scheduled occupancy, not source workload labels.", "small")

    title(4, "Storage, objective and solver")
    eq("soc")
    eq("battery_limits")
    eq("battery_modes")
    table(["Asset", "Initial / limits kWh", "Charge/discharge kW", "Efficiencies"], [
        ["LFP", "20 / 4-36", "10 / 10", "0.96 / 0.96"],
        ["Sodium-ion", "3 / 0.6-5.4", "6 / 6", "0.94 / 0.94"],
    ], [90, 140, 140, 113], 8.3)
    text("Sodium-ion retains an extra 1.2 kWh and 2 kW in each direction. Both batteries restore initial energy. Per-battery exclusive modes permit different layers to operate in opposing directions; throughput wear penalizes cycling but does not forbid it under negative prices.")
    eq("cost")
    eq("carbon")
    text("C includes energy, renewable variable cost, a daily peak proxy and AC-throughput wear (LFP 0.035; sodium-ion 0.055 CNY/kWh). M uses average grid-intensity estimates. Charging is counted at import; discharge gets no extra carbon credit. Modeled renewable operational intensity is zero; lifecycle emissions and battery capex are excluded.")
    text("HiGHS uses mixed-integer branch-and-bound/cut methods through SciPy milp. Status 0 means requested tolerance reached, 1 a time/resource limit, 2 infeasibility. An LP relaxation supplies a lower bound but fractional jobs are not executable. Stricter-budget incumbents may improve looser problems; lower bounds remain target-specific.")
    eq("gap")
    text(f"Green U = CNY {fmt(green['cost_cny'],9)}, L = CNY {fmt(green['lower_bound'],9)}, gap = {100*green['mip_gap']:.6f}%. The model has {green['variables']} variables, {green['binary_variables']} binaries and {green['constraints']} constraints. The independent audit reconstructs physical/accounting outcomes without reading the MILP matrix.", "small")

    title(5, "Dispatch and mechanism comparisons")
    picture("dispatch", maxheight=310, caption="Figure 3. Recommended dispatch, battery net powers and all 97 energy states.")
    metrics = green["audit"]["metrics"]
    text(f"Facility demand is {fmt(metrics['facility_kwh'])} kWh; grid import {fmt(metrics['grid_kwh'])} kWh; renewable use {fmt(metrics['renewable_used_kwh'])} kWh. Conversion losses add {fmt(metrics['battery_loss_kwh'])} kWh. Cyclic terminal energy prevents apparent savings from emptying the initial stock.")
    rows = []
    for name, result in bundle["policy_matched_ablations"].items():
        label = {"fifo_storage_green": "FIFO + storage", "compute_only_green": "Compute only", "lfp_only_green": "Joint, LFP only"}.get(name, name.replace("_", " "))
        rows.append([label, fmt(result["cost_cny"]) if result["feasible"] else "Infeasible", f"{100*result['mip_gap']:.6f}%" if result["feasible"] else "Status 2"])
    rows.append(["Joint, both layers", fmt(green["cost_cny"]), f"{100*green['mip_gap']:.6f}%"])
    table(["Same 3% carbon policy", "Cost CNY", "Gap / status"], rows, [285, 95, 103], 8.4)
    text(f"Cost-only joint scheduling costs CNY {fmt(cheap['cost_cny'])}; the selected green requirement adds CNY {fmt(green['cost_cny']-cheap['cost_cny'])}. Compute/storage benefits cannot be added as independent savings because each changes the other's operating opportunities.")
    text("The LFP-only comparison isolates the second layer's modeled economic increment, not fast-response capability or investment return. Sampled cost/carbon budgets expose trade-offs but do not establish a complete exact frontier. Job CSVs and the offline explorer retain every source-ID schedule.")

    title(6, "Public dates and constructed stresses")
    table(["Public date", "FIFO service", "Green policy", "Cost saved"], [
        [row["date"], "Feasible" if row["baseline_feasible"] else "Infeasible", "Feasible" if row["feasible"] else "Infeasible",
         f"{row['cost_saving_percent']:.2f}%" if "cost_saving_percent" in row else "N/A"] for row in bundle["days"]
    ], [135, 115, 120, 113], 8.4)
    text("June 3 and June 8 have feasible cost-only service, but valid minimum-carbon lower bounds exclude the extra 3% target. The same fixed selection rule applies to all dates. These related observations do not form a facility-wide statistical guarantee.")
    table(["Constructed scenario", "FIFO", "Green", "Cost saved"], [
        ["Background GPU surge" if row["scenario"] == "inference_surge" else row["scenario"].replace("_", " "),
         "Yes" if row["baseline_feasible"] else "No", "Yes" if row["feasible"] else "No",
         f"{row['cost_saving_percent']:.2f}%" if "cost_saving_percent" in row else "N/A"] for row in bundle["stress"]
    ], [225, 70, 75, 113], 8.4)
    text("These are perturbations, not historical incidents: cloudy retains 35% solar; background surge adds 48 GPUs; grid derating halves its cap; hot day sets PUE 1.40. Other cases vary waiting, prices, charges, wear and workload. Negative savings and infeasible targets are retained.")
    text("Impossible grid sets both grid and renewable supply to zero. Cyclic batteries cannot create net daily energy; a positive supply-deficit diagnostic proves failure. Passing this necessary energy condition alone does not prove time/resource feasibility. N/A denotes no feasible same-scenario saving comparison.")
    text("Independent minimum-carbon runs also separate unattainable environmental targets from workload failures in constructed cases. All five public dates and twelve stresses remain in results/main/.", "small")

    title(7, "Online replay, validation and references")
    rolling, chosen = read(RES/"rolling.json"), read(RES/"rolling_optimized.json")
    table(["Online policy", "Cost CNY", "Carbon kg", "Peak kW"], [
        [f"Default reserve {cfg['online_reserve_gpus']}", fmt(rolling["cost_cny"]), fmt(rolling["carbon_kg"]), fmt(rolling["peak_kw"])],
        [f"Selected reserve {study['selected_margin']}", fmt(chosen["cost_cny"]), fmt(chosen["carbon_kg"]), fmt(chosen["peak_kw"])],
    ], [210, 90, 90, 93], 8.3)
    text("Both replays complete 96 decisions. Past actions/launched starts are locked and future releases hidden. Future solar/background use previous-day persistence; current-slot aggregates are optimistically assumed known at the boundary. Archived carbon forecasts have unknown vintage and do not affect this cost-only policy.")
    text("Reserve 0 is selected from 0/2/4/8 on June 2-3 by failures then cost; June 4-5 are separate evaluation. Both default and selected margins survive the declared shock (solar 15% in slots 40-55, +48 reserved GPUs in 44-51). Neither replay guarantees the offline carbon cap or a global online optimum.")
    text(f"Validation passes {checks['unit_tests']['tests_run']} unit tests and {len(checks['numerical_audits'])} numerical audits. Sixty brute-force random instances are subcases of three tests. Public-source checks, tampered-result rejection, CLI exits, six-policy desktop/mobile checks and rendered-PDF QA are recorded separately.")
    text("Reproduce from README.md and docs/reproduction.md. Run tools/verify_results.py, rebuild artifacts/report, run browser/PDF QA, refresh tools/build_docs.py and prepare tools/prepare_push.py. The curated snapshot verifies copies and reruns numerical/CLI checks. AI assistance: OpenAI Codex supported research, code, experiments and writing; entrants retain responsibility for understanding and truthful contributions.", "small")
    text("Limits: assumed deadlines/power/storage, homogeneous GPUs, average carbon and daily peak proxies. No verified request latency, CPU/RAM/topology, transient response, monthly utility bill, TCO/ROI or causal grid-emissions reduction.", "small")
    sub("References")
    references = [
        ("https://github.com/S-Lab-System-Group/HeliosData", "HeliosData; Characterization and Prediction of Deep Learning Workloads in Large-Scale GPU Datacenters, SC 2021. CC BY 4.0."),
        ("https://doi.org/10.25832/time_series/2020-10-06", "Open Power System Data, Time series 2020-10-06. Primary ENTSO-E records; source-specific terms retained."),
        ("https://carbonintensity.org.uk/", "UK Carbon Intensity API. National estimates and archived forecasts. CC BY 4.0."),
        ("https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html", "SciPy milp: HiGHS interface, statuses and certificates."),
        ("https://arxiv.org/abs/2106.11750", "Radovanovic et al. 2021. Carbon-Aware Computing for Datacenters."),
        ("https://www.hithium.com/newsroom/latest/details/86.html", "HiTHIUM storage announcement. Motivation; modeled asset ratings remain assumptions."),
    ]
    for index, (url, description) in enumerate(references, 1):
        text(f'<link href="{url}">[{index}] {markup_escape(description)}</link>', "small")
    text("[7] Supplied HackSHSID Track 2 statement, project brief and technical specification. [8] IMMC 2025 Problem E, When Watts Meet Bits. Source identities/hashes: data/source_documents.json. Public-data URLs/hashes/terms: data/raw/manifest.json.", "small")
    finish_report(source_only)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true")
    build(parser.parse_args().source_only)
