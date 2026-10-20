# Research, Data and Model

## Data and assumptions

The main case is a composite study, not a measured British facility or complete historical Helios replay. [Configuration](../configs/public_study.json) separates assumptions from [public records](../data/raw/manifest.json).

| Input | Source and transformation | Boundary |
|---|---|---|
| GPU jobs | Helios Saturn / vcOIr; completed GPU jobs lasting 1,800-10,800 s, submitted before 18:00; first at most 32 by submission/ID | 26 jobs on the main day; retain submit/start/end, GPUs, execution seconds and source row. No source deadlines or invented dependencies. |
| Capacity/background | Published 192-GPU VC allocation; slot maximum of same-VC short-job concurrency | Homogeneous equivalents; short jobs are not labeled inference. Unselected jobs are outside the workload. |
| Price | OPSD `GB_GBN_price_day_ahead`, GBP/MWh, multiplied by assumed 9/1,000 | Fixed study conversion, not historical FX or a retail tariff; negative prices retained. |
| Solar | OPSD `GB_GBN_solar_generation_actual`, MW, multiplied by assumed 0.004 kW/MW | Scaled national generation, not an on-site PV meter; missing capacity factors are not filled. |
| Carbon | UK API national `actual`, g CO2/kWh divided by 1,000 | Estimated average intensity, not marginal or causal emissions. |

Energy records align by UTC interval start. Helios has no documented timezone; its within-day offsets are rebased onto the model day. Release/duration round upward to 15 minutes: **693.86 observed GPU-hours become 762.00 modeled GPU-hours (9.82% padding)**. Assumed deadline is release + duration + up to four hours waiting, capped at day end. Source completion is not an empirical deadline.

Hourly/half-hourly records are held constant. Future solar/background use previous-day same-clock persistence. Online causality is between slots: current-slot aggregates are optimistically assumed known at its boundary. Archived carbon forecasts lack verified issuance vintage and do not affect the online cost objective/constraints.

Assumed base IT load: 20 kW; active/background GPU power: 0.25 kW/GPU; PUE: 1.2; facility/grid caps: 150/120 kW; successive grid ramp: 50 kW; renewable cost: 0.06 CNY/kWh; daily peak proxy: 2 CNY/(kW day). Battery ratings/wear are in the configuration.

## Formulation

There are 96 slots with Δt = 0.25 h. Job j has release r, exclusive deadline d, duration L, maximum wait W, demand g GPUs and incremental power p kW. Binary z selects one start; activity a is substituted into the sparse matrix.

$$F_j=\{s\in\mathbb Z:r_j\le s\le\min(d_j-L_j,r_j+W_j)\},\quad \sum_{s\in F_j}z_{js}=1,\quad a_{jt}=\sum_{s\in F_j:s\le t<s+L_j}z_{js}.$$

$$\sum_jg_ja_{jt}+n_t\le N_t,\qquad P_t^{fac}=\kappa_t(P_{base}+q n_t+\sum_jp_ja_{jt}).$$

Here n is background reservation, N GPU capacity, κ PUE, G grid import, R renewable use, and c/u battery charge/discharge (kW).

$$G_t+R_t+\sum_bu_{bt}=P_t^{fac}+\sum_bc_{bt},\quad 0\le G_t\le\bar G_t,\quad0\le R_t\le\bar R_t,\quad P_t^{fac}\le\bar P^{fac},\quad |G_t-G_{t-1}|\le\Gamma\;(t\ge1).$$

Each battery has energy E (kWh), efficiencies η, physical limits and reserved energy/power. Binary v forbids simultaneous charge/discharge within that battery; different batteries may operate in opposing directions.

$$E_{b,t+1}=E_{bt}+\eta_b^cc_{bt}\Delta t-u_{bt}\Delta t/\eta_b^d,\quad E_b^{min}+E_b^{res}\le E_{bt}\le E_b^{max},\quad E_{b0}=E_{bT}=E_b^{init}.$$

$$0\le c_{bt}\le(\bar c_b-P_b^{res})v_{bt},\qquad0\le u_{bt}\le(\bar u_b-P_b^{res})(1-v_{bt}).$$

With price π, renewable cost π_R, throughput wear k, peak proxy λ, and average grid intensity ξ:

$$C=\Delta t\sum_t(\pi_tG_t+\pi_RR_t)+\lambda\max_tG_t+\Delta t\sum_{b,t}k_b(c_{bt}+u_{bt}),\quad M=\Delta t\sum_t\xi_tG_t,\quad\min C\text{ subject to }M\le0.97M_{FIFO}.$$

The library supports/tests DAG precedence (parent finishes before child starts), stricter waiting, carbon objectives and cost ceilings. The public cohort has no source DAG. Legacy `inference_gpus`/`training_gpus` fields mean background/scheduled occupancy, not verified workload labels.

## Algorithm and validity

SciPy `milp` calls HiGHS branch-and-bound/cut methods. Feasible incumbent U and target-problem bound L bracket the optimum; report (U-L)/|U| and actual status. Status 0 means tolerance reached, 1 a time/resource limit, 2 infeasibility. LP relaxation is a lower bound, not an executable fractional schedule. Incumbents may transfer to looser carbon budgets; bounds remain target-specific.

The independent audit reconstructs jobs, GPU limits, load, power balance, grid/solar/ramp, all battery states and cost/carbon without reading the MILP matrix. Online replay locks past actions/launched jobs, hides future releases, and replans each slot; it has no whole-day green cap or global policy certificate.

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

Full URLs, pinned commits, licenses and SHA-256 values are in the [data manifest](../data/raw/manifest.json).
