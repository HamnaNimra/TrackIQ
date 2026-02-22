# TrackIQ Full Feature View

Last updated: 2026-02-22 (local artifact set refreshed; `artifacts/ui_png/tabs` regenerated from live Streamlit pages with consistent `1680x1100` viewport and light-theme styling)

This document embeds the generated UI/report artifacts and summarizes what each result is used for in product and ops workflows.

## Artifact Index

- AutoPerfPy UI report: [`autoperfpy_ui_report.html`](./autoperfpy_ui_report.html)
- AutoPerfPy UI data (JSON): [`autoperfpy_ui_report_data.json`](./autoperfpy_ui_report_data.json)
- AutoPerfPy UI data (CSV): [`autoperfpy_ui_report_data.csv`](./autoperfpy_ui_report_data.csv)
- AutoPerfPy bench inference result: [`autoperfpy_bench_inference.json`](./autoperfpy_bench_inference.json)
- AutoPerfPy bench report: [`autoperfpy_bench_inference_report.html`](./autoperfpy_bench_inference_report.html)
- AutoPerfPy bench report data: [`autoperfpy_bench_inference_report_data.json`](./autoperfpy_bench_inference_report_data.json)
- MiniCluster UI report: [`minicluster_ui_report.html`](./minicluster_ui_report.html)
- MiniCluster heatmap report: [`minicluster_heatmap_report.html`](./minicluster_heatmap_report.html)
- MiniCluster fault timeline report: [`minicluster_fault_timeline_report.html`](./minicluster_fault_timeline_report.html)
- MiniCluster cluster-health demo result: [`minicluster_cluster_health_demo.json`](../tmp/minicluster_cluster_health_demo.json)
- TrackIQ Compare UI report: [`trackiq_compare_ui_report.html`](./trackiq_compare_ui_report.html)
- TrackIQ Compare app screenshot: [`ui_trackiq_compare_app.png`](../ui_png/ui_trackiq_compare_app.png)
- TrackIQ Compare standalone top screenshot: [`compare_standalone_top.png`](../ui_png/tabs/compare_standalone_top.png)

## Result Snapshots (What They Are Used For)

| Tool | Key results snapshot | Why these results exist |
|---|---|---|
| AutoPerfPy (inference run) | throughput `17.08 samples/s`, latency `p50 42.62 ms / p95 54.69 ms / p99 55.76 ms`, power `29.07 W`, performance-per-watt `0.455` | Capacity planning, latency tail-risk checks, and perf-per-watt cost decisions. |
| AutoPerfPy (bench-inference) | backend `mock`, prompts `32`, `mean_ttft 124.77 ms`, `p99_ttft 162.96 ms`, `mean_tpot 45.37 ms`, `p99_tpot 69.65 ms`, throughput `22.04 tokens/s` | LLM serving SLO validation where `TTFT` and `TPOT` are primary production latency/throughput signals. |
| MiniCluster (cluster health) | workers `4`, backend `nccl`, workload `transformer`, throughput `127.8 samples/s`, all-reduce `p50 2.62 ms / p95 3.71 ms / p99 4.24 ms`, stdev `0.39 ms`, max `4.43 ms`, scaling efficiency `92.4%` | Distributed straggler detection, collective stability analysis, and scaling-quality validation. |
| TrackIQ Compare | cross-run metric deltas, winner selection, regression checks, variance consistency analysis | Release gating and regression triage when averages look stable but consistency worsens. |

## End-To-End User Flows

| Flow | Entry point | Core actions | Primary outputs | Success criteria |
|---|---|---|---|---|
| Single-node inference validation | `autoperfpy/ui/streamlit_app.py` or `autoperfpy` CLI | Select device, precision, batch, duration; run benchmark; inspect tabs | Canonical JSON + CSV + HTML report | Stable throughput, acceptable P99, no thermal/power anomalies |
| Multi-worker training/fabric validation | `minicluster/ui/streamlit_app.py` or `minicluster` CLI | Set workers/backend/workload/steps; run; inspect cluster-health and training tabs | MiniCluster JSON + heatmap/fault timeline HTML reports | P99 all-reduce and stdev within expected bounds; scaling efficiency near target |
| Cross-run regression approval | `trackiq_compare/ui/streamlit_app.py` or compare CLI | Load result A/B, set regression and variance thresholds, review deltas and consistency | Compare JSON/HTML report with winner + flagged findings | No blocking regressions; consistency findings explained/accepted |
| Unified review for release | `dashboard.py` (`--tool all`/specific) | Load canonical results from multiple tools, validate summary cards and charts | Unified screenshot/report evidence for sign-off | Data completeness + visual/metric consistency across tools |

## Canonical Contract Mapping (TrackiqResult)

| Contract area | Representative fields | Used by | Why it matters |
|---|---|---|---|
| Run identity | `tool_name`, `tool_version`, `timestamp` | All UIs, compare, exports | Ensures reproducibility and traceability of every result |
| Platform metadata | `platform.hardware_name`, `platform.framework`, `platform.framework_version` | Compare + report headers | Prevents invalid A/B comparisons across mismatched environments |
| Workload metadata | `workload.name`, `workload.workload_type`, `workload.batch_size`, `workload.steps` | UI config tables + compare context | Confirms workloads are comparable and correctly labeled |
| Core metrics | `metrics.*` (latency, throughput, power, memory, communication, scaling) | Cards, charts, gating checks | Encodes go/no-go decision signals |
| Tool payload detail | `tool_payload` (samples/per-step/worker details) | Deep charts and debugging | Preserves raw evidence for root-cause analysis |
| Regression summary | `regression.*` | Compare dashboards and reports | Surfaces pass/fail and delta severity in a canonical way |

## Metric Interpretation Guide

| Metric | Interpretation | Typical risk if degraded | Recommended action |
|---|---|---|---|
| `latency.p99_ms` | Tail latency / straggler indicator | User-visible spikes and SLO misses | Check power, temperature, queueing, and batch shape |
| `throughput.mean_fps` or `throughput_samples_per_sec` | Capacity under current settings | Under-provisioning and cost inefficiency | Tune batch, precision, and runtime backend |
| `power.mean_w` + perf-per-watt | Cost and energy efficiency | Rising run cost with flat performance | Compare precision/device settings, cap thermal hotspots |
| `p99_allreduce_ms` | Distributed synchronization tail | Slowest rank throttles the full cluster | Inspect interconnect, worker skew, and straggler ranks |
| `allreduce_stdev_ms` | Collective consistency | Hidden instability despite good averages | Use compare variance check and worker heatmap |
| `scaling_efficiency_pct` | Multi-worker scaling quality | Poor horizontal scaling economics | Investigate communication overhead and memory pressure |
| `mean_ttft_ms` / `p99_ttft_ms` | LLM prefill latency | Slow first token user experience | Optimize prompt path, KV/cache and scheduling |
| `mean_tpot_ms` / `throughput_tokens_per_sec` | LLM decode efficiency | Lower token throughput and poor UX | Tune decode path, batch scheduler, and backend settings |

## Triage Playbooks

| Symptom | Where to verify | Likely causes | First fixes |
|---|---|---|---|
| Throughput stable but P99 worsens | AutoPerfPy latency tab + compare deltas | Tail contention, thermal throttling, scheduling jitter | Re-run with fixed batch, inspect thermal/power traces |
| Mean all-reduce looks fine but training unstable | MiniCluster stdev/P99 + compare consistency section | One or few straggler workers, noisy fabric path | Use heatmap on `p99_allreduce_ms`; isolate offending rank |
| Compare shows many `Infinity` deltas | Compare config + confidence matrix | Missing metrics on one side or non-overlapping payloads | Validate metric coverage, normalize input generation path |
| Cluster-health charts sparse or empty | Unified dashboard + tool payload checks | Incomplete per-step data in input JSON | Regenerate run with full samples/per-step metrics enabled |
| Good perf, bad perf-per-watt | Power/Thermal + throughput tabs | Over-voltage/thermal state, poor precision choice | Evaluate lower precision and cooling/power envelope |

## Data Readability And Usefulness Checklist

| Artifact | Must include | Consumer action enabled |
|---|---|---|
| JSON | Full canonical contract + tool payload | Automation, gating, reproducible comparisons |
| CSV | Flattened sample-level rows with latency/power/throughput | Spreadsheet/BI ad-hoc slicing |
| HTML report | Summary cards, charts, metadata tables, interpretation notes | Human review and approval workflows |
| PNG gallery | Populated UI tabs/sections with readable scales | QA evidence and docs/release communication |

## Artifact Refresh Procedure

Use this sequence when UI/report visuals need a full refresh:

```powershell
python -m pytest tests/test_e2e_cli_workflows.py -q
python -m streamlit run autoperfpy/ui/streamlit_app.py
python -m streamlit run minicluster/ui/streamlit_app.py
python -m streamlit run trackiq_compare/ui/streamlit_app.py
python -m streamlit run dashboard.py -- --tool all
```

Then regenerate report artifacts and refresh screenshot captures in `artifacts/ui_png/` and `artifacts/ui_png/tabs/`.

## Feature Coverage By Tool

| Tool | Major feature area | What the end user can do | Decision enabled |
|---|---|---|---|
| AutoPerfPy | Benchmark execution | Run inference benchmarks by device/precision/batch profile | Choose deployment profile and capacity target |
| AutoPerfPy | Multi-tab diagnostics | Inspect latency, utilization, power/thermal, memory, throughput | Identify bottleneck family before optimization |
| AutoPerfPy | Report export (`html/csv/json`) | Share readable summary + machine-readable data | Support review meetings and CI automation |
| MiniCluster | Distributed run orchestration | Configure workers/backend/workload and execute training validation | Validate distributed readiness before scale-out |
| MiniCluster | Cluster-health analysis | Review all-reduce percentiles, variability, scaling efficiency | Detect stragglers and interconnect degradation |
| MiniCluster | Fault validation | Visualize injected faults and detection latency | Prove monitoring catches production failure modes |
| TrackIQ Compare | A/B metric comparison | Compare canonical outputs and inspect metric deltas | Release gating and regression approval |
| TrackIQ Compare | Consistency analysis | Flag variance regressions even when means look stable | Catch hidden instability and tail-risk increase |
| TrackIQ Core | Canonical result explorer | Inspect schema completeness and payload quality | Enforce result contract integrity across tools |
| Unified Dashboard | Cross-tool navigation | Switch quickly between app surfaces in one launch point | Reduce operator context-switch time |

## Role-Based Usage Guide

| Role | Primary artifact(s) | Recommended view order | Key question answered |
|---|---|---|---|
| Performance engineer | AutoPerfPy tabs + HTML report | Overview -> Latency -> Throughput -> Power | Is this config fast and stable enough? |
| Cluster/SRE operator | MiniCluster cluster-health + heatmap + fault timeline | Health cards -> all-reduce distribution -> fault timeline | Which rank/node is unstable and why? |
| Release approver | TrackIQ Compare report + consistency section | Winner/regressions -> confidence matrix -> consistency findings | Can this build safely ship? |
| Product manager | Full app feature view + summary HTMLs | Snapshot table -> role flow -> report previews | What value does each result provide to users? |
| QA owner | PNG tab gallery + canonical JSON | Visual checks -> contract mapping -> checklist | Are UI/report outputs complete and non-empty? |

## Decision Thresholds Reference

These are pragmatic defaults for day-to-day triage. Teams can tighten or relax based on workload SLOs and hardware class.

| Signal | Suggested threshold | Severity guidance |
|---|---|---|
| Latency regression (compare) | `> 5%` worse vs baseline | `warn`: 5-10%, `block`: >10% |
| Throughput drop | `> 5%` lower vs baseline | `warn`: 5-10%, `block`: >10% |
| P99 all-reduce | `> 1.3x` P50 or abrupt week-over-week increase | Indicates potential straggler/fabric contention |
| All-reduce stdev | sustained increase `> 25%` vs baseline | Consistency regression risk |
| Scaling efficiency | `< 90%` | Communication or memory bottleneck likely |
| TTFT p99 | exceeds product SLO budget | User-visible first-token degradation |
| TPOT mean | drifts below target token budget | Slower streaming responses |
| Perf-per-watt | downtrend with flat throughput | Cost efficiency regression |

## Output Contract By Format

| Format | Minimum content expected | Validation checks |
|---|---|---|
| JSON | Canonical identity, workload config, metric set, payload details | Required keys present, no invalid numeric types |
| CSV | Row-level sample/per-step metrics with stable headers | Non-empty rows, parseable numeric columns |
| HTML | Executive summary, metric cards, charts, metadata/config tables, interpretation text | All sections render, no missing-chart placeholders |
| PNG | Capture of populated UI/report sections at readable viewport | No runtime errors, text and axes readable |

## Comparison Semantics (How Winners Are Chosen)

| Metric family | Better direction | Notes |
|---|---|---|
| Latency (`p50/p95/p99`) | Lower is better | Tail (`p99`) weighted for user-impact risk |
| Throughput | Higher is better | Evaluate with matching batch/config context |
| Power | Lower is better when throughput is comparable | Pair with perf-per-watt to avoid false wins |
| Perf-per-watt | Higher is better | Primary efficiency signal for cost-aware decisions |
| Communication overhead | Lower is better | Key for distributed scalability |
| Consistency / variance | Lower variance is better | Guards against hidden instability |

## Common Anti-Patterns And Fixes

| Anti-pattern | Why it is risky | Fix |
|---|---|---|
| Comparing runs with different workload or batch but treating as apples-to-apples | Produces misleading deltas and winner results | Enforce workload/batch match before gating |
| Using only mean latency for approval | Misses tail regressions and straggler effects | Gate on `p95/p99` and variance |
| Treating missing metrics as pass | Masks instrumentation regressions | Use confidence matrix and fail on critical gaps |
| Running screenshots at mixed viewport/theme settings | Inconsistent docs and unreadable comparisons | Standardize on one viewport + theme per capture set |
| Exporting HTML without linked JSON/CSV artifacts | Breaks reproducibility in postmortems | Always publish artifact bundle together |

## Documentation QA Checklist

- Every linked artifact in the index resolves and opens.
- Every major chart section is populated (no empty/skeleton states in final captures).
- Comparison pages include both regression findings and consistency findings.
- Cluster-health pages include both performance and fault-signal context.
- JSON/CSV/HTML for a run share the same timestamp family or run id.
- Screenshot names follow a stable naming convention per tab/section.

## Standalone UI Gallery

### AutoPerfPy Streamlit

![AutoPerfPy app](../ui_png/ui_autoperfpy_app.png)

Purpose: benchmark execution, upload/demo data exploration, tabbed latency/utilization/power/memory/throughput analysis, and HTML report generation.

### MiniCluster Streamlit

![MiniCluster app](../ui_png/ui_minicluster_app.png)

Purpose: distributed run configuration, cluster-health diagnostics, training/fault analysis views, and export paths.

### TrackIQ Compare Streamlit

![TrackIQ Compare app](../ui_png/ui_trackiq_compare_app.png)
![TrackIQ Compare standalone top](../ui_png/tabs/compare_standalone_top.png)

Purpose: compare two canonical results, show metric deltas, and highlight consistency regressions.

### TrackIQ Core Streamlit Explorer

![TrackIQ Core app](../ui_png/ui_trackiq_core_app.png)

Purpose: generic schema-level exploration of `TrackiqResult` payloads across tools.

## Unified Dashboard Gallery (`dashboard.py`)

### All tools launcher

![Unified dashboard all](../ui_png/dashboard_all.png)

### AutoPerfPy mode

![Unified dashboard autoperfpy](../ui_png/dashboard_autoperfpy.png)

### MiniCluster mode

![Unified dashboard minicluster](../ui_png/dashboard_minicluster.png)

### Compare mode

![Unified dashboard compare](../ui_png/dashboard_compare.png)

### Cluster-health mode

![Unified dashboard cluster health](../ui_png/dashboard_cluster_health.png)

## HTML Report Preview Gallery

### AutoPerfPy UI report

![AutoPerfPy report preview](../ui_png/report_autoperfpy_ui_report.png)

Open report: [`autoperfpy_ui_report.html`](./autoperfpy_ui_report.html)

### AutoPerfPy bench-inference report

![AutoPerfPy bench report preview](../ui_png/report_autoperfpy_bench_inference_report.png)

Open report: [`autoperfpy_bench_inference_report.html`](./autoperfpy_bench_inference_report.html)

### MiniCluster run report

![MiniCluster report preview](../ui_png/report_minicluster_ui_report.png)

Open report: [`minicluster_ui_report.html`](./minicluster_ui_report.html)

### MiniCluster heatmap report

![MiniCluster heatmap preview](../ui_png/report_minicluster_heatmap_report.png)

Open report: [`minicluster_heatmap_report.html`](./minicluster_heatmap_report.html)

### MiniCluster fault timeline report

![MiniCluster fault timeline preview](../ui_png/report_minicluster_fault_timeline_report.png)

Open report: [`minicluster_fault_timeline_report.html`](./minicluster_fault_timeline_report.html)

### TrackIQ Compare report

![TrackIQ Compare report preview](../ui_png/report_trackiq_compare_ui_report.png)

Open report: [`trackiq_compare_ui_report.html`](./trackiq_compare_ui_report.html)

## Tab And Section Captures

Note: the tab/section screenshots below are curated viewport captures for readability in docs. They prioritize key decision surfaces (summary cards, configuration, and metric tables) over full-page scroll length.

### AutoPerfPy tabs

![AutoPerfPy overview tab](../ui_png/tabs/autoperfpy_standalone_tab_overview.png)
![AutoPerfPy latency tab](../ui_png/tabs/autoperfpy_standalone_tab_latency.png)
![AutoPerfPy utilization tab](../ui_png/tabs/autoperfpy_standalone_tab_utilization.png)
![AutoPerfPy power thermal tab](../ui_png/tabs/autoperfpy_standalone_tab_power_thermal.png)
![AutoPerfPy memory tab](../ui_png/tabs/autoperfpy_standalone_tab_memory.png)
![AutoPerfPy throughput tab](../ui_png/tabs/autoperfpy_standalone_tab_throughput.png)

### MiniCluster sections

![MiniCluster top section](../ui_png/tabs/minicluster_standalone_top_viewport.png)
![MiniCluster cluster health section](../ui_png/tabs/minicluster_standalone_cluster_health_viewport.png)
![MiniCluster training graphs section](../ui_png/tabs/minicluster_standalone_training_graphs_viewport.png)
![MiniCluster run config section](../ui_png/tabs/minicluster_standalone_run_config_viewport.png)

### Compare sections

![Compare standalone top section](../ui_png/tabs/compare_standalone_top.png)
![Compare top section](../ui_png/tabs/dashboard_compare_top_viewport.png)
![Compare configuration section](../ui_png/tabs/dashboard_compare_configuration_viewport.png)
![Compare graphs section](../ui_png/tabs/dashboard_compare_graphs_viewport.png)
![Compare consistency section](../ui_png/tabs/dashboard_compare_consistency_viewport.png)

### Cluster-health sections

![Cluster health top section](../ui_png/tabs/dashboard_cluster_health_top_viewport.png)
![Cluster health loss curve section](../ui_png/tabs/dashboard_cluster_health_loss_curve_viewport.png)
![Cluster health histogram section](../ui_png/tabs/dashboard_cluster_health_histogram_viewport.png)
![Cluster health fault timeline section](../ui_png/tabs/dashboard_cluster_health_fault_timeline_viewport.png)

## Result Type Purpose Matrix

| Result type | Purpose | Typical consumer |
|---|---|---|
| Canonical JSON (`TrackiqResult`) | Contract for automation, compare logic, and reproducible analysis | CI/CD gates, data pipelines, backend services |
| HTML report | Human-friendly run narrative with charts/tables | Engineering managers, infra/perf reviewers |
| CSV export | Flat data for ad-hoc filtering/plotting | Analysts, spreadsheet users |
| UI PNG captures | Visual proof that dashboards/charts render and are populated | QA, release notes, demos |
| Heatmap report | Worker-level outlier map to find stragglers quickly | Cluster/SRE operators |
| Fault timeline report | Shows injection vs detection latency for failure modes | Reliability and monitoring owners |
| Compare report | Quantifies regressions and consistency drift between runs | Release approvers, performance owners |

