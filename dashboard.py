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
    from datetime import UTC, datetime

    return TrackiqResult(
        tool_name=tool_name,
        tool_version="browser",
        timestamp=datetime.now(UTC),
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
                with st.sidebar.expander("Load AutoPerfPy Result", expanded=False):
                    ResultBrowser(theme=DARK_THEME, allowed_tools=["autoperfpy"]).render()
                loaded = st.session_state.get("loaded_result")
                if (
                    loaded is None
                    and args.result
                    and Path(args.result).exists()
                ):
                    loaded = load_trackiq_result(args.result)
                dashboard = AutoPerfDashboard(
                    result=loaded
                    if isinstance(loaded, TrackiqResult)
                    and str(loaded.tool_name).lower() == "autoperfpy"
                    else _placeholder_result("autoperfpy", workload_type="inference")
                )
                dashboard.apply_theme(dashboard.theme)
                dashboard.render_header()
                dashboard.render_sidebar()
                dashboard.render_body()
                dashboard.render_footer()
                return 0

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
                if (
                    not isinstance(selected, TrackiqResult)
                    and args.result
                    and Path(args.result).exists()
                ):
                    loaded_arg = load_trackiq_result(args.result)
                    if str(loaded_arg.tool_name).lower() == "minicluster":
                        selected = loaded_arg

                if not isinstance(selected, TrackiqResult):
                    st.info(
                        "Use sidebar run configuration to generate a MiniCluster run, or load an existing result."
                    )
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
                    st.session_state["trackiq_unified_compare_regression_threshold"] = float(
                        regression_threshold
                    )

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
            if args.result:
                result_path = _validate_path(args.result, "--result")
                run_dashboard(AutoPerfDashboard, result_path=result_path)
            else:
                class _AutoPerfBrowserDashboard(AutoPerfDashboard):
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
                    _AutoPerfBrowserDashboard,
                    result=_placeholder_result("autoperfpy", workload_type="inference"),
                )
            return 0

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
