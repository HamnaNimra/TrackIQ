# UI Screenshot Artifacts

This folder contains generated PNG captures for all tools and dashboards.

## Preview Gallery

![AutoPerfPy app](./ui_autoperfpy_app.png)
![MiniCluster app](./ui_minicluster_app.png)
![TrackIQ Compare app](./ui_trackiq_compare_app.png)
![TrackIQ Core app](./ui_trackiq_core_app.png)
![TrackIQ Compare standalone top](./tabs/compare_standalone_top.png)

## Primary Set

Use `artifacts/ui_png/tabs/` as the curated set:

- `autoperfpy_standalone_tab_overview.png`
- `autoperfpy_standalone_tab_latency.png`
- `autoperfpy_standalone_tab_utilization.png`
- `autoperfpy_standalone_tab_power_thermal.png`
- `autoperfpy_standalone_tab_memory.png`
- `autoperfpy_standalone_tab_throughput.png`
- `dashboard_autoperfpy_top_viewport.png`
- `minicluster_standalone_top_viewport.png`
- `minicluster_standalone_cluster_health_viewport.png`
- `minicluster_standalone_run_config_viewport.png`
- `minicluster_standalone_training_graphs_viewport.png`
- `dashboard_compare_top_viewport.png`
- `dashboard_compare_configuration_viewport.png`
- `dashboard_compare_graphs_viewport.png`
- `dashboard_compare_consistency_viewport.png`
- `compare_standalone_top.png`
- `dashboard_cluster_health_top_viewport.png`
- `dashboard_cluster_health_loss_curve_viewport.png`
- `dashboard_cluster_health_histogram_viewport.png`
- `dashboard_cluster_health_fault_timeline_viewport.png`
- `ui_trackiq_core_app.png`

## Notes

- AutoPerfPy standalone app has real Streamlit tabs and was captured tab-by-tab.
- MiniCluster / Compare / Cluster Health dashboard views are section-based (no tab widgets), so they are captured by section/viewport.
- `ui_trackiq_core_app.png` captures the canonical TrackIQ Core explorer for schema-level inspection flows.
- HTML report preview captures are refreshed at 1600px width for improved text/chart readability in docs.
