# TrackIQ Full Feature View

Last updated: 2026-02-22 (local artifact set refreshed; `artifacts/ui_png/tabs` now uses light-theme, readable viewport captures aligned with current Streamlit styling)

This document embeds the generated UI/report artifacts and summarizes what each result is used for in product and ops workflows.

## Artifact Index

- AutoPerfPy UI report: [`autoperfpy_ui_report.html`](./artifacts/reports/autoperfpy_ui_report.html)
- AutoPerfPy UI data (JSON): [`autoperfpy_ui_report_data.json`](./artifacts/reports/autoperfpy_ui_report_data.json)
- AutoPerfPy UI data (CSV): [`autoperfpy_ui_report_data.csv`](./artifacts/reports/autoperfpy_ui_report_data.csv)
- AutoPerfPy bench inference result: [`autoperfpy_bench_inference.json`](./artifacts/reports/autoperfpy_bench_inference.json)
- AutoPerfPy bench report: [`autoperfpy_bench_inference_report.html`](./artifacts/reports/autoperfpy_bench_inference_report.html)
- AutoPerfPy bench report data: [`autoperfpy_bench_inference_report_data.json`](./artifacts/reports/autoperfpy_bench_inference_report_data.json)
- MiniCluster UI report: [`minicluster_ui_report.html`](./artifacts/reports/minicluster_ui_report.html)
- MiniCluster heatmap report: [`minicluster_heatmap_report.html`](./artifacts/reports/minicluster_heatmap_report.html)
- MiniCluster fault timeline report: [`minicluster_fault_timeline_report.html`](./artifacts/reports/minicluster_fault_timeline_report.html)
- MiniCluster cluster-health demo result: [`minicluster_cluster_health_demo.json`](./artifacts/tmp/minicluster_cluster_health_demo.json)
- TrackIQ Compare UI report: [`trackiq_compare_ui_report.html`](./artifacts/reports/trackiq_compare_ui_report.html)
- TrackIQ Compare app screenshot: [`ui_trackiq_compare_app.png`](./artifacts/ui_png/ui_trackiq_compare_app.png)
- TrackIQ Compare standalone top screenshot: [`compare_standalone_top.png`](./artifacts/ui_png/tabs/compare_standalone_top.png)

## Result Snapshots (What They Are Used For)

| Tool | Key results snapshot | Why these results exist |
|---|---|---|
| AutoPerfPy (inference run) | throughput `17.08 samples/s`, latency `p50 42.62 ms / p95 54.69 ms / p99 55.76 ms`, power `29.07 W`, performance-per-watt `0.455` | Capacity planning, latency tail-risk checks, and perf-per-watt cost decisions. |
| AutoPerfPy (bench-inference) | backend `mock`, prompts `32`, `mean_ttft 124.77 ms`, `p99_ttft 162.96 ms`, `mean_tpot 45.37 ms`, `p99_tpot 69.65 ms`, throughput `22.04 tokens/s` | LLM serving SLO validation where `TTFT` and `TPOT` are primary production latency/throughput signals. |
| MiniCluster (cluster health) | workers `4`, backend `nccl`, workload `transformer`, throughput `127.8 samples/s`, all-reduce `p50 2.62 ms / p95 3.71 ms / p99 4.24 ms`, stdev `0.39 ms`, max `4.43 ms`, scaling efficiency `92.4%` | Distributed straggler detection, collective stability analysis, and scaling-quality validation. |
| TrackIQ Compare | cross-run metric deltas, winner selection, regression checks, variance consistency analysis | Release gating and regression triage when averages look stable but consistency worsens. |

## Standalone UI Gallery

### AutoPerfPy Streamlit

![AutoPerfPy app](./artifacts/ui_png/ui_autoperfpy_app.png)

Purpose: benchmark execution, upload/demo data exploration, tabbed latency/utilization/power/memory/throughput analysis, and HTML report generation.

### MiniCluster Streamlit

![MiniCluster app](./artifacts/ui_png/ui_minicluster_app.png)

Purpose: distributed run configuration, cluster-health diagnostics, training/fault analysis views, and export paths.

### TrackIQ Compare Streamlit

![TrackIQ Compare app](./artifacts/ui_png/ui_trackiq_compare_app.png)
![TrackIQ Compare standalone top](./artifacts/ui_png/tabs/compare_standalone_top.png)

Purpose: compare two canonical results, show metric deltas, and highlight consistency regressions.

### TrackIQ Core Streamlit Explorer

![TrackIQ Core app](./artifacts/ui_png/ui_trackiq_core_app.png)

Purpose: generic schema-level exploration of `TrackiqResult` payloads across tools.

## Unified Dashboard Gallery (`dashboard.py`)

### All tools launcher

![Unified dashboard all](./artifacts/ui_png/dashboard_all.png)

### AutoPerfPy mode

![Unified dashboard autoperfpy](./artifacts/ui_png/dashboard_autoperfpy.png)

### MiniCluster mode

![Unified dashboard minicluster](./artifacts/ui_png/dashboard_minicluster.png)

### Compare mode

![Unified dashboard compare](./artifacts/ui_png/dashboard_compare.png)

### Cluster-health mode

![Unified dashboard cluster health](./artifacts/ui_png/dashboard_cluster_health.png)

## HTML Report Preview Gallery

### AutoPerfPy UI report

![AutoPerfPy report preview](./artifacts/ui_png/report_autoperfpy_ui_report.png)

Open report: [`autoperfpy_ui_report.html`](./artifacts/reports/autoperfpy_ui_report.html)

### AutoPerfPy bench-inference report

![AutoPerfPy bench report preview](./artifacts/ui_png/report_autoperfpy_bench_inference_report.png)

Open report: [`autoperfpy_bench_inference_report.html`](./artifacts/reports/autoperfpy_bench_inference_report.html)

### MiniCluster run report

![MiniCluster report preview](./artifacts/ui_png/report_minicluster_ui_report.png)

Open report: [`minicluster_ui_report.html`](./artifacts/reports/minicluster_ui_report.html)

### MiniCluster heatmap report

![MiniCluster heatmap preview](./artifacts/ui_png/report_minicluster_heatmap_report.png)

Open report: [`minicluster_heatmap_report.html`](./artifacts/reports/minicluster_heatmap_report.html)

### MiniCluster fault timeline report

![MiniCluster fault timeline preview](./artifacts/ui_png/report_minicluster_fault_timeline_report.png)

Open report: [`minicluster_fault_timeline_report.html`](./artifacts/reports/minicluster_fault_timeline_report.html)

### TrackIQ Compare report

![TrackIQ Compare report preview](./artifacts/ui_png/report_trackiq_compare_ui_report.png)

Open report: [`trackiq_compare_ui_report.html`](./artifacts/reports/trackiq_compare_ui_report.html)

## Tab And Section Captures

Note: the tab/section screenshots below are curated viewport captures for readability in docs. They prioritize key decision surfaces (summary cards, configuration, and metric tables) over full-page scroll length.

### AutoPerfPy tabs

![AutoPerfPy overview tab](./artifacts/ui_png/tabs/autoperfpy_standalone_tab_overview.png)
![AutoPerfPy latency tab](./artifacts/ui_png/tabs/autoperfpy_standalone_tab_latency.png)
![AutoPerfPy utilization tab](./artifacts/ui_png/tabs/autoperfpy_standalone_tab_utilization.png)
![AutoPerfPy power thermal tab](./artifacts/ui_png/tabs/autoperfpy_standalone_tab_power_thermal.png)
![AutoPerfPy memory tab](./artifacts/ui_png/tabs/autoperfpy_standalone_tab_memory.png)
![AutoPerfPy throughput tab](./artifacts/ui_png/tabs/autoperfpy_standalone_tab_throughput.png)

### MiniCluster sections

![MiniCluster top section](./artifacts/ui_png/tabs/minicluster_standalone_top_viewport.png)
![MiniCluster cluster health section](./artifacts/ui_png/tabs/minicluster_standalone_cluster_health_viewport.png)
![MiniCluster training graphs section](./artifacts/ui_png/tabs/minicluster_standalone_training_graphs_viewport.png)
![MiniCluster run config section](./artifacts/ui_png/tabs/minicluster_standalone_run_config_viewport.png)

### Compare sections

![Compare standalone top section](./artifacts/ui_png/tabs/compare_standalone_top.png)
![Compare top section](./artifacts/ui_png/tabs/dashboard_compare_top_viewport.png)
![Compare configuration section](./artifacts/ui_png/tabs/dashboard_compare_configuration_viewport.png)
![Compare graphs section](./artifacts/ui_png/tabs/dashboard_compare_graphs_viewport.png)
![Compare consistency section](./artifacts/ui_png/tabs/dashboard_compare_consistency_viewport.png)

### Cluster-health sections

![Cluster health top section](./artifacts/ui_png/tabs/dashboard_cluster_health_top_viewport.png)
![Cluster health loss curve section](./artifacts/ui_png/tabs/dashboard_cluster_health_loss_curve_viewport.png)
![Cluster health histogram section](./artifacts/ui_png/tabs/dashboard_cluster_health_histogram_viewport.png)
![Cluster health fault timeline section](./artifacts/ui_png/tabs/dashboard_cluster_health_fault_timeline_viewport.png)

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
