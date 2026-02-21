"""Unified dashboard launcher for AutoPerfPy, MiniCluster, and trackiq-compare."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from autoperfpy.ui.dashboard import AutoPerfDashboard
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

            if tool_choice in {"autoperfpy", "minicluster"}:
                with st.sidebar.expander("Load Result", expanded=False):
                    ResultBrowser(
                        theme=DARK_THEME,
                        allowed_tools=[tool_choice],
                    ).render()

                loaded = st.session_state.get("loaded_result")
                if tool_choice == "autoperfpy":
                    if loaded is None and args.result and Path(args.result).exists():
                        loaded = load_trackiq_result(args.result)
                    dashboard = AutoPerfDashboard(
                        result=loaded
                        if isinstance(loaded, TrackiqResult)
                        else _placeholder_result("autoperfpy", workload_type="inference")
                    )
                else:
                    if loaded is None and args.result and Path(args.result).exists():
                        loaded = load_trackiq_result(args.result)
                    dashboard = MiniClusterDashboard(
                        result=loaded
                        if isinstance(loaded, TrackiqResult)
                        else _placeholder_result("minicluster", workload_type="training")
                    )
                dashboard.apply_theme(dashboard.theme)
                dashboard.render_header()
                dashboard.render_sidebar()
                dashboard.render_body()
                dashboard.render_footer()
                return 0

            st.sidebar.markdown("---")
            st.sidebar.subheader("Compare Inputs")
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

            result_a = st.session_state.get("trackiq_unified_compare_a")
            result_b = st.session_state.get("trackiq_unified_compare_b")
            if not isinstance(result_a, TrackiqResult) or not isinstance(result_b, TrackiqResult):
                st.info("Load two result files in the sidebar to view comparison graphs.")
                return 0

            dashboard = CompareDashboard(
                result_a=result_a,
                result_b=result_b,
                label_a=str(st.session_state.get("trackiq_unified_compare_label_a", label_a)),
                label_b=str(st.session_state.get("trackiq_unified_compare_label_b", label_b)),
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
