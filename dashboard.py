"""Unified dashboard launcher for AutoPerfPy, MiniCluster, and trackiq-compare."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from autoperfpy.ui.dashboard import AutoPerfDashboard
from minicluster.runner import RunConfig, run_distributed, save_metrics
from minicluster.ui.dashboard import MiniClusterDashboard
from trackiq_compare.ui.dashboard import CompareDashboard
from trackiq_core.schema import Metrics, PlatformInfo, RegressionInfo, TrackiqResult, WorkloadInfo
from trackiq_core.serializer import load_trackiq_result
from trackiq_core.ui import DARK_THEME, ResultBrowser, run_dashboard


def _validate_path(path: Optional[str], label: str) -> str:
    if not path:
        raise SystemExit(f"{label} is required.")
    if not Path(path).exists():
        raise SystemExit(f"{label} does not exist: {path}")
    return path


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified TrackIQ dashboard launcher")
    parser.add_argument(
        "--tool",
        required=False,
        default="all",
        choices=["all", "autoperfpy", "minicluster", "compare"],
        help="Tool dashboard to launch (default: all)",
    )
    parser.add_argument("--result", help="Single TrackiqResult JSON path")
    parser.add_argument("--result-a", help="Compare mode: result A path")
    parser.add_argument("--result-b", help="Compare mode: result B path")
    parser.add_argument("--label-a", help="Compare mode: display label A")
    parser.add_argument("--label-b", help="Compare mode: display label B")
    return parser.parse_args(argv)


def _placeholder_result(tool_name: str, workload_type: str = "inference") -> TrackiqResult:
    """Create placeholder result for browser-mode dashboards."""
    from datetime import datetime, timezone

    return TrackiqResult(
        tool_name=tool_name,
        tool_version="browser",
        timestamp=datetime.now(timezone.utc),
        platform=PlatformInfo(
            hardware_name="Unknown",
            os="Unknown",
            framework="unknown",
            framework_version="unknown",
        ),
        workload=WorkloadInfo(
            name="browser_mode",
            workload_type=workload_type,  # type: ignore[arg-type]
            batch_size=0,
            steps=0,
        ),
        metrics=Metrics(
            throughput_samples_per_sec=0.0,
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            latency_p99_ms=0.0,
            memory_utilization_percent=0.0,
            communication_overhead_percent=None,
            power_consumption_watts=None,
        ),
        regression=RegressionInfo(
            baseline_id=None,
            delta_percent=0.0,
            status="pass",
            failed_metrics=[],
        ),
    )


def _run_minicluster_once(config: RunConfig) -> TrackiqResult:
    """Run MiniCluster once and return canonical TrackiqResult."""
    metrics = run_distributed(config)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        out_path = handle.name
    try:
        save_metrics(metrics, out_path)
        return load_trackiq_result(out_path)
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


def _run_autoperf_single(
    *,
    device_id: str,
    precision: str,
    batch_size: int,
    duration_seconds: int,
    warmup_runs: int,
    iterations: int,
) -> TrackiqResult:
    """Run one AutoPerfPy benchmark and return canonical TrackiqResult."""
    from autoperfpy.auto_runner import run_single_benchmark
    from autoperfpy.cli import _infer_trackiq_result
    from autoperfpy.device_config import (
        InferenceConfig,
        resolve_device,
        resolve_precision_for_device,
    )

    device = resolve_device(device_id or "cpu_0")
    if device is None:
        raise RuntimeError(f"Device not found: {device_id}")
    effective_precision = resolve_precision_for_device(device, precision)
    config = InferenceConfig(
        precision=effective_precision,
        batch_size=int(batch_size),
        accelerator=device.device_id,
        streams=1,
        warmup_runs=int(warmup_runs),
        iterations=int(iterations),
    )
    payload = run_single_benchmark(
        device,
        config,
        duration_seconds=float(duration_seconds),
        sample_interval_seconds=0.2,
        quiet=True,
    )
    return _infer_trackiq_result(
        payload,
        workload_name=str(payload.get("run_label", "autoperfpy_manual_ui")),
        workload_type="inference",
    )


def _run_autoperf_auto(
    *,
    device_ids: List[str],
    precisions: List[str],
    batch_sizes: List[int],
    duration_seconds: int,
    max_configs_per_device: int,
) -> List[TrackiqResult]:
    """Run AutoPerfPy auto mode and return canonical TrackiqResult list."""
    from autoperfpy.auto_runner import run_auto_benchmarks
    from autoperfpy.cli import _infer_trackiq_result
    from autoperfpy.device_config import get_devices_and_configs_auto

    pairs = get_devices_and_configs_auto(
        precisions=list(precisions),
        batch_sizes=list(batch_sizes),
        max_configs_per_device=int(max_configs_per_device),
        device_ids_filter=list(device_ids) if device_ids else None,
    )
    if not pairs:
        return []
    payloads = run_auto_benchmarks(
        pairs,
        duration_seconds=float(duration_seconds),
        sample_interval_seconds=0.2,
        quiet=True,
    )
    results: List[TrackiqResult] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        results.append(
            _infer_trackiq_result(
                payload,
                workload_name=str(payload.get("run_label", "autoperfpy_auto_ui")),
                workload_type="inference",
            )
        )
    return results


def _render_autoperf_interactive(
    *,
    args: argparse.Namespace,
    key_prefix: str,
    title: Optional[str] = None,
    caption: Optional[str] = None,
) -> int:
    """Render AutoPerfPy with rich run configuration and graph output."""
    import streamlit as st

    from autoperfpy.device_config import (
        DEFAULT_BATCH_SIZES,
        DEFAULT_ITERATIONS,
        DEFAULT_WARMUP_RUNS,
        PRECISION_FP32,
        PRECISIONS,
    )
    from trackiq_core.hardware import get_all_devices

    if title:
        st.title(title)
    if caption:
        st.caption(caption)

    devices = get_all_devices()
    device_ids = [d.device_id for d in devices]
    if not device_ids:
        device_ids = ["cpu_0"]

    with st.sidebar.expander("AutoPerfPy Run Configuration", expanded=True):
        run_mode = st.radio(
            "Run Mode",
            options=["Manual", "Auto"],
            index=0,
            key=f"{key_prefix}_mode",
        )
        duration_seconds = st.number_input(
            "Duration (seconds)",
            min_value=1,
            max_value=300,
            value=10,
            step=1,
            key=f"{key_prefix}_duration",
        )
        warmup_runs = st.number_input(
            "Warmup Runs",
            min_value=0,
            max_value=200,
            value=int(DEFAULT_WARMUP_RUNS),
            step=1,
            key=f"{key_prefix}_warmup",
        )
        iterations = st.number_input(
            "Iterations",
            min_value=1,
            max_value=5000,
            value=int(DEFAULT_ITERATIONS),
            step=1,
            key=f"{key_prefix}_iterations",
        )

        if run_mode == "Manual":
            manual_device = st.selectbox(
                "Device",
                options=device_ids,
                index=0,
                key=f"{key_prefix}_manual_device",
            )
            manual_precision = st.selectbox(
                "Precision",
                options=list(PRECISIONS),
                index=list(PRECISIONS).index(PRECISION_FP32) if PRECISION_FP32 in PRECISIONS else 0,
                key=f"{key_prefix}_manual_precision",
            )
            manual_batch = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=4096,
                value=1,
                step=1,
                key=f"{key_prefix}_manual_batch",
            )
            run_clicked = st.button(
                "Run AutoPerfPy Benchmark",
                use_container_width=True,
                key=f"{key_prefix}_run_manual",
            )
            if run_clicked:
                with st.spinner("Running AutoPerfPy benchmark..."):
                    result = _run_autoperf_single(
                        device_id=str(manual_device),
                        precision=str(manual_precision),
                        batch_size=int(manual_batch),
                        duration_seconds=int(duration_seconds),
                        warmup_runs=int(warmup_runs),
                        iterations=int(iterations),
                    )
                    st.session_state[f"{key_prefix}_results"] = [result]
                    st.session_state[f"{key_prefix}_selected_idx"] = 0
        else:
            selected_devices = st.multiselect(
                "Devices",
                options=device_ids,
                default=device_ids[: min(2, len(device_ids))] or device_ids,
                key=f"{key_prefix}_auto_devices",
            )
            selected_precisions = st.multiselect(
                "Precisions",
                options=list(PRECISIONS),
                default=list(PRECISIONS[:2]) if len(PRECISIONS) >= 2 else list(PRECISIONS),
                key=f"{key_prefix}_auto_precisions",
            )
            auto_batch_sizes = st.multiselect(
                "Batch Sizes",
                options=list(dict.fromkeys(list(DEFAULT_BATCH_SIZES) + [2, 16, 32])),
                default=list(DEFAULT_BATCH_SIZES),
                key=f"{key_prefix}_auto_batches",
            )
            max_configs = st.number_input(
                "Max Configs Per Device",
                min_value=1,
                max_value=20,
                value=6,
                step=1,
                key=f"{key_prefix}_auto_max_cfg",
            )
            run_auto_clicked = st.button(
                "Run AutoPerfPy Auto Benchmarks",
                use_container_width=True,
                key=f"{key_prefix}_run_auto",
            )
            if run_auto_clicked:
                if not selected_devices:
                    st.sidebar.warning("Select at least one device.")
                elif not selected_precisions:
                    st.sidebar.warning("Select at least one precision.")
                elif not auto_batch_sizes:
                    st.sidebar.warning("Select at least one batch size.")
                else:
                    with st.spinner("Running AutoPerfPy auto benchmarks..."):
                        results = _run_autoperf_auto(
                            device_ids=[str(x) for x in selected_devices],
                            precisions=[str(x) for x in selected_precisions],
                            batch_sizes=[int(x) for x in auto_batch_sizes],
                            duration_seconds=int(duration_seconds),
                            max_configs_per_device=int(max_configs),
                        )
                        st.session_state[f"{key_prefix}_results"] = results
                        st.session_state[f"{key_prefix}_selected_idx"] = 0

    with st.sidebar.expander("Load AutoPerfPy Result", expanded=False):
        ResultBrowser(theme=DARK_THEME, allowed_tools=["autoperfpy"]).render()

    results = st.session_state.get(f"{key_prefix}_results")
    selected: Optional[TrackiqResult] = None
    if isinstance(results, list) and results:
        if len(results) > 1:
            labels = [
                f"{idx + 1}. {r.workload.name} | bs{r.workload.batch_size} | {r.platform.hardware_name}"
                for idx, r in enumerate(results)
                if isinstance(r, TrackiqResult)
            ]
            if labels:
                idx = st.sidebar.selectbox(
                    "Displayed Auto Run",
                    options=list(range(len(labels))),
                    index=min(
                        int(st.session_state.get(f"{key_prefix}_selected_idx", 0)),
                        len(labels) - 1,
                    ),
                    format_func=lambda i: labels[i],
                    key=f"{key_prefix}_selected_idx",
                )
                if 0 <= int(idx) < len(results) and isinstance(results[int(idx)], TrackiqResult):
                    selected = results[int(idx)]
        else:
            only = results[0]
            if isinstance(only, TrackiqResult):
                selected = only

    loaded = st.session_state.get("loaded_result")
    if selected is None and isinstance(loaded, TrackiqResult) and str(loaded.tool_name).lower() == "autoperfpy":
        selected = loaded
    if selected is None and args.result and Path(args.result).exists():
        loaded_arg = load_trackiq_result(args.result)
        if str(loaded_arg.tool_name).lower() == "autoperfpy":
            selected = loaded_arg

    if selected is None:
        st.info("Configure and run AutoPerfPy from the sidebar, or load an existing result.")
        return 0

    dashboard = AutoPerfDashboard(result=selected)
    dashboard.apply_theme(dashboard.theme)
    dashboard.render_header()
    dashboard.render_sidebar()
    dashboard.render_body()
    dashboard.render_footer()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for the unified dashboard launcher."""
    args = _parse_args(argv)

    try:
        if args.tool == "all":
            import streamlit as st

            st.set_page_config(
                page_title="TrackIQ Unified Dashboard",
                layout="wide",
                initial_sidebar_state="expanded",
            )
            st.title("TrackIQ Unified Dashboard")
            st.caption("Switch between AutoPerfPy, MiniCluster, and Compare in one app.")

            tool_choice = st.sidebar.selectbox(
                "Tool",
                options=["autoperfpy", "minicluster", "compare"],
                index=0,
                key="trackiq_unified_tool_choice",
            )

            if tool_choice == "autoperfpy":
                return _render_autoperf_interactive(
                    args=args,
                    key_prefix="trackiq_unified_autoperf",
                )

            if tool_choice == "minicluster":
                with st.sidebar.expander("MiniCluster Run Configuration", expanded=True):
                    workers = st.number_input(
                        "Workers",
                        min_value=1,
                        max_value=16,
                        value=1,
                        step=1,
                        key="trackiq_unified_mini_workers",
                    )
                    backend = st.selectbox(
                        "Collective Backend",
                        options=["gloo", "nccl"],
                        index=0,
                        key="trackiq_unified_mini_backend",
                    )
                    workload = st.selectbox(
                        "Workload",
                        options=["mlp", "transformer", "embedding"],
                        index=0,
                        key="trackiq_unified_mini_workload",
                    )
                    steps = st.number_input(
                        "Steps",
                        min_value=1,
                        max_value=5000,
                        value=100,
                        step=1,
                        key="trackiq_unified_mini_steps",
                    )
                    batch_size = st.number_input(
                        "Batch Size",
                        min_value=1,
                        max_value=4096,
                        value=32,
                        step=1,
                        key="trackiq_unified_mini_batch",
                    )
                    learning_rate = st.number_input(
                        "Learning Rate",
                        min_value=0.0001,
                        max_value=1.0,
                        value=0.01,
                        step=0.0001,
                        format="%.4f",
                        key="trackiq_unified_mini_lr",
                    )
                    seed = st.number_input(
                        "Seed",
                        min_value=0,
                        max_value=2_147_483_647,
                        value=42,
                        step=1,
                        key="trackiq_unified_mini_seed",
                    )
                    tdp_watts = st.number_input(
                        "TDP Watts",
                        min_value=10.0,
                        max_value=1000.0,
                        value=150.0,
                        step=1.0,
                        key="trackiq_unified_mini_tdp",
                    )
                    use_baseline = st.checkbox(
                        "Use Baseline Throughput",
                        value=False,
                        key="trackiq_unified_mini_use_baseline",
                    )
                    baseline_throughput = st.number_input(
                        "Baseline Throughput (samples/s)",
                        min_value=0.0,
                        max_value=1_000_000.0,
                        value=100.0,
                        step=1.0,
                        disabled=not bool(use_baseline),
                        key="trackiq_unified_mini_baseline",
                    )
                    run_clicked = st.button(
                        "Run MiniCluster",
                        use_container_width=True,
                        key="trackiq_unified_mini_run",
                    )
                    quick_clicked = st.button(
                        "Quick Smoke Run (20 steps)",
                        use_container_width=True,
                        key="trackiq_unified_mini_quick_run",
                    )

                with st.sidebar.expander("Load MiniCluster Result", expanded=False):
                    ResultBrowser(theme=DARK_THEME, allowed_tools=["minicluster"]).render()

                if run_clicked or quick_clicked:
                    cfg = RunConfig(
                        num_steps=int(20 if quick_clicked else steps),
                        num_processes=int(1 if quick_clicked else workers),
                        batch_size=int(16 if quick_clicked else batch_size),
                        learning_rate=float(0.01 if quick_clicked else learning_rate),
                        seed=int(42 if quick_clicked else seed),
                        tdp_watts=float(tdp_watts),
                        collective_backend=str(backend),
                        workload=str(workload),
                        baseline_throughput=(float(baseline_throughput) if bool(use_baseline) else None),
                    )
                    with st.spinner("Running MiniCluster..."):
                        st.session_state["trackiq_unified_minicluster_result"] = _run_minicluster_once(cfg)

                selected = st.session_state.get("trackiq_unified_minicluster_result")
                loaded = st.session_state.get("loaded_result")
                if (
                    not isinstance(selected, TrackiqResult)
                    and isinstance(loaded, TrackiqResult)
                    and str(loaded.tool_name).lower() == "minicluster"
                ):
                    selected = loaded
                if not isinstance(selected, TrackiqResult) and args.result and Path(args.result).exists():
                    loaded_arg = load_trackiq_result(args.result)
                    if str(loaded_arg.tool_name).lower() == "minicluster":
                        selected = loaded_arg

                if not isinstance(selected, TrackiqResult):
                    st.info("Use sidebar run configuration to generate a MiniCluster run, or load an existing result.")
                    return 0

                dashboard = MiniClusterDashboard(result=selected)
                dashboard.apply_theme(dashboard.theme)
                dashboard.render_header()
                dashboard.render_sidebar()
                dashboard.render_body()
                dashboard.render_footer()
                return 0

            st.sidebar.markdown("---")
            st.sidebar.subheader("Compare Configuration")
            input_mode = st.sidebar.radio(
                "Input Mode",
                options=["Browse discovered results", "Manual paths"],
                index=0,
                key="trackiq_unified_compare_input_mode",
            )
            result_a_path = ""
            result_b_path = ""
            rows = ResultBrowser(theme=DARK_THEME).to_dict()
            if input_mode == "Browse discovered results":
                if not rows:
                    st.sidebar.info("No result files discovered. Switch to manual paths.")
                else:
                    labels = [
                        f"{i + 1}. {row.get('tool_name', '?')} | {row.get('workload_name', '?')} | {row.get('timestamp', '?')}"
                        for i, row in enumerate(rows)
                    ]
                    idx_a = st.sidebar.selectbox(
                        "Result A",
                        options=list(range(len(labels))),
                        format_func=lambda i: labels[i],
                        key="trackiq_unified_compare_a_idx",
                    )
                    remaining = [i for i in range(len(labels)) if i != idx_a] or [idx_a]
                    idx_b = st.sidebar.selectbox(
                        "Result B",
                        options=remaining,
                        format_func=lambda i: labels[i],
                        key="trackiq_unified_compare_b_idx",
                    )
                    result_a_path = str(rows[idx_a]["path"])
                    result_b_path = str(rows[idx_b]["path"])
            else:
                result_a_path = st.sidebar.text_input(
                    "Result A Path",
                    value=args.result_a or "",
                    key="trackiq_unified_result_a",
                )
                result_b_path = st.sidebar.text_input(
                    "Result B Path",
                    value=args.result_b or "",
                    key="trackiq_unified_result_b",
                )

            label_a = st.sidebar.text_input(
                "Label A",
                value=args.label_a or "Result A",
                key="trackiq_unified_label_a",
            )
            label_b = st.sidebar.text_input(
                "Label B",
                value=args.label_b or "Result B",
                key="trackiq_unified_label_b",
            )
            regression_threshold = st.sidebar.slider(
                "Regression Threshold (%)",
                min_value=0.5,
                max_value=50.0,
                value=5.0,
                step=0.5,
                key="trackiq_unified_compare_regression_threshold",
            )

            if st.sidebar.button("Load Comparison", use_container_width=True):
                if not result_a_path or not Path(result_a_path).exists():
                    st.sidebar.error(f"Result A file not found: {result_a_path}")
                elif not result_b_path or not Path(result_b_path).exists():
                    st.sidebar.error(f"Result B file not found: {result_b_path}")
                else:
                    st.session_state["trackiq_unified_compare_a"] = load_trackiq_result(result_a_path)
                    st.session_state["trackiq_unified_compare_b"] = load_trackiq_result(result_b_path)
                    st.session_state["trackiq_unified_compare_label_a"] = label_a
                    st.session_state["trackiq_unified_compare_label_b"] = label_b
                    st.session_state["trackiq_unified_compare_regression_threshold"] = float(regression_threshold)

            result_a = st.session_state.get("trackiq_unified_compare_a")
            result_b = st.session_state.get("trackiq_unified_compare_b")
            if not isinstance(result_a, TrackiqResult) or not isinstance(result_b, TrackiqResult):
                st.info("Configure comparison in the sidebar and click 'Load Comparison'.")
                return 0

            dashboard = CompareDashboard(
                result_a=result_a,
                result_b=result_b,
                label_a=str(st.session_state.get("trackiq_unified_compare_label_a", label_a)),
                label_b=str(st.session_state.get("trackiq_unified_compare_label_b", label_b)),
                regression_threshold_percent=float(
                    st.session_state.get(
                        "trackiq_unified_compare_regression_threshold",
                        regression_threshold,
                    )
                ),
            )
            dashboard.apply_theme(dashboard.theme)
            dashboard.render_header()
            dashboard.render_body()
            dashboard.render_footer()
            return 0

        if args.tool == "autoperfpy":
            import streamlit as st

            if args.result:
                _validate_path(args.result, "--result")

            st.set_page_config(
                page_title="AutoPerfPy Dashboard",
                layout="wide",
                initial_sidebar_state="expanded",
            )
            return _render_autoperf_interactive(
                args=args,
                key_prefix="trackiq_autoperf_only",
                title="AutoPerfPy Interactive Dashboard",
                caption="Configure benchmark runs and visualize graph-rich results.",
            )

        if args.tool == "minicluster":
            if args.result:
                result_path = _validate_path(args.result, "--result")
                run_dashboard(MiniClusterDashboard, result_path=result_path)
            else:

                class _MiniClusterBrowserDashboard(MiniClusterDashboard):
                    def render_body(self) -> None:
                        import streamlit as st

                        loaded = st.session_state.get("loaded_result")
                        if loaded is None:
                            st.info("Select a result file to begin.")
                            ResultBrowser(theme=self.theme).render()
                            return
                        self.result = loaded
                        super().render_body()

                run_dashboard(
                    _MiniClusterBrowserDashboard,
                    result=_placeholder_result("minicluster", workload_type="training"),
                )
            return 0

        if not args.result_a or not args.result_b:
            from trackiq_compare.ui import streamlit_app

            streamlit_app.main()
            return 0

        result_a_path = _validate_path(args.result_a, "--result-a")
        result_b_path = _validate_path(args.result_b, "--result-b")
        result_a = load_trackiq_result(result_a_path)
        result_b = load_trackiq_result(result_b_path)

        class _CompareDashboardAdapter(CompareDashboard):
            def __init__(self, result, theme, title="TrackIQ Compare Dashboard"):
                if not isinstance(result, list) or len(result) != 2:
                    raise ValueError("Compare dashboard requires exactly two loaded results.")
                super().__init__(
                    result_a=result[0],
                    result_b=result[1],
                    label_a=args.label_a,
                    label_b=args.label_b,
                    theme=theme,
                    title=title,
                )

        run_dashboard(_CompareDashboardAdapter, result=[result_a, result_b])  # type: ignore[arg-type]
        return 0
    except Exception as exc:
        raise SystemExit(f"Failed to launch dashboard: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
