# Results and Testing Evidence

Saved energy day: **2020-06-01 UTC**. All policies complete the same admitted jobs. Costs are proxies under the [research assumptions](research.md).

| Policy | Cost CNY/day | Carbon kg CO2/day | Grid peak kW | Cost saved | MIP gap |
|---|---:|---:|---:|---:|---:|
| FIFO | 230.85 | 97.67 | 42.55 | 0.00% | N/A |
| FIFO + storage | 205.39 | 97.10 | 28.88 | 11.03% | 0.000000% |
| Compute only | 206.39 | 97.28 | 31.20 | 10.60% | 0.049882% |
| Joint cost | 198.08 | 96.32 | 25.93 | 14.20% | 0.009982% |
| Joint green | 205.54 | 94.74 | 29.52 | 10.97% | 0.000055% |

Green upper/lower bounds: CNY 205.535943792/205.535831654; gap 0.000055%. The chosen carbon ceiling is 97% of FIFO. [Schedule](../results/main/joint_green_jobs.csv), [dispatch](../results/main/joint_green_dispatch.csv) and [full results](../results/main/summary.json) retain the evidence.

Under the same ceiling, FIFO + storage and compute-only are infeasible. LFP-only costs CNY 206.47. The second battery's modeled increment does not establish fast response or investment return.

## Other dates and stresses

| Public date | FIFO feasible | Green feasible | Cost saved |
|---|---|---|---:|
| 2020-06-02 | True | True | 17.28% |
| 2020-06-03 | True | False | N/A |
| 2020-06-04 | True | True | 0.52% |
| 2020-06-05 | True | True | 13.02% |
| 2020-06-08 | True | False | N/A |

June 3 and June 8 have feasible cost-only service, but [minimum-carbon bounds](../results/main/policy_limits.json) exclude the extra 3% target. No failed date is removed.

| Constructed stress | FIFO feasible | Green feasible | Cost saved |
|---|---|---|---:|
| cloudy | True | True | -6.75% |
| no renewables | True | False | N/A |
| flat price | True | True | 9.87% |
| price carbon conflict | True | True | 13.47% |
| hot day | True | True | 8.30% |
| Background GPU surge | True | False | N/A |
| grid derating | True | True | 10.94% |
| zero flexibility | True | False | N/A |
| storage expensive | True | True | -2.34% |
| no demand charge | True | True | 5.22% |
| high workload | True | True | 12.63% |
| impossible grid | False | False | N/A |

Stresses are constructed, not historical incidents. Background surge adds 48 GPUs; impossible grid sets both grid and renewable supply to zero. Negative savings/failures remain visible; N/A means no feasible same-scenario comparison.

## Online replay

| Policy | Cost CNY | Carbon kg CO2 | Peak kW |
|---|---:|---:|---:|
| Default reserve 4 GPUs | 228.41 | 98.27 | 41.20 |
| Selected reserve 0 GPUs | 226.94 | 98.11 | 40.64 |

Both main replays complete 96 decisions. Candidates 0/2/4/8 are selected on June 2-3 by failures, then cost; June 4-5 provide separate evaluation. The [online study](../results/main/online_study.json) retains every run and constructed shock. The explorer shows the default replay. Neither online policy enforces the offline carbon ceiling or has a global policy certificate.

## Verification

The saved run passes **61 unit tests** and **55 independent numerical audits**. Sixty brute-force random cases are subcases of three tests. Checks cover source hashes/fields, conversions, timing, complete jobs, GPU/power limits, all battery states, accounting, causal locks and tampered outputs.

| Evidence | Scope |
|---|---|
| [Numerical verification](../results/qa/verification.json) | Unit log, audit residuals, source rebuild and 96-step replay |
| [Exact oracle](../results/qa/random_enumeration_qa.json) | 60 independent tiny-instance comparisons |
| [CLI](../results/qa/cli_qa.json) | Generation, solve, audit and infeasible exit status |
| [Browser](../results/qa/browser_qa.json) | Six policies at 390/1280 px; cards, search, last slot and layout |
| [PDF](../results/qa/pdf_structure_qa.json), [visual review](../results/qa/render_qa.json) | Current report, fonts, bounds, key results and rendered pages |
| [Upload](../results/qa/push_validation.json) | Five documents, links, artifact hashes and standalone snapshot |

These checks validate the declared model, not monthly billing, request latency, electrical transients, heterogeneous placement or causal grid decarbonization. [Reproduce](reproduction.md).
