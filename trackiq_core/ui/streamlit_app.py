"""Interactive Streamlit app for generic TrackIQ Core result exploration."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

from trackiq_core.schema import Metrics, PlatformInfo, RegressionInfo, TrackiqResult, WorkloadInfo
from trackiq_core.serializer import load_trackiq_result
from trackiq_core.ui import DARK_THEME, LIGHT_THEME, MetricTable, PowerGauge, RegressionBadge, ResultBrowser
from trackiq_core.ui.dashboard import TrackiqDashboard


def _apply_ui_style() -> None:
    """Apply visual polish for the TrackIQ Core app."""
    st.markdown(
        """
        <style>
        .core-hero {
            border: 1px solid rgba(59,130,246,0.25);
            background: linear-gradient(135deg, rgba(59,130,246,0.12), rgba(16,185,129,0.10));
            border-radius: 14px;
            padding: 14px 16px;
            margin-bottom: 14px;
        }
        .core-hero h2 {
            margin: 0 0 4px 0;
            font-size: 1.22rem;
        }
        .core-hero p {
            margin: 0;
            color: #4b5563;
            font-size: 0.95rem;
        }
        [data-testid="stMetric"] {
            border: 1px solid rgba(148,163,184,0.22);
            border-radius: 12px;
            padding: 8px 10px;
            background: rgba(15,23,42,0.02);
        }
        button[kind="primary"] {
            border-radius: 10px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_page_intro() -> None:
    """Render top-level orientation for first-time users."""
    st.markdown(
        """
        <div class="core-hero">
          <h2>TrackIQ Core Explorer</h2>
          <p>Load any canonical TrackiqResult JSON and inspect metrics, payloads, trends, and exports.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Quick Start", expanded=False):
        st.markdown(
            "1. Choose an input source in the sidebar (`Browse results` or `Manual path`).\n"
            "2. Click `Load Result`.\n"
            "3. Use tabs to inspect overview, power/KV cache, payload details, trends, and downloads."
        )


def _build_demo_result() -> TrackiqResult:
    """Build a fallback TrackiqResult so the app never renders blank."""
    return TrackiqResult(
        tool_name="trackiq_core_demo",
        tool_version="1.0.0",
        timestamp=datetime.now(timezone.utc),
        platform=PlatformInfo(
            hardware_name="Demo Hardware",
            os="Demo OS",
            framework="pytorch",
            framework_version="2.x",
        ),
        workload=WorkloadInfo(
            name="demo_workload",
            workload_type="inference",
            batch_size=4,
            steps=100,
        ),
        metrics=Metrics(
            throughput_samples_per_sec=100.0,
            latency_p50_ms=10.0,
            latency_p95_ms=12.5,
            latency_p99_ms=15.0,
            memory_utilization_percent=42.0,
            communication_overhead_percent=2.0,
            power_consumption_watts=120.0,
            energy_per_step_joules=1.2,
            performance_per_watt=0.83,
            temperature_celsius=58.0,
            scaling_efficiency_pct=92.0,
        ),
        regression=RegressionInfo(
            baseline_id=None,
            delta_percent=0.0,
            status="pass",
            failed_metrics=[],
        ),
        tool_payload={"note": "Demo payload shown because no result was loaded yet."},
    )


def _discover_result_rows() -> list[dict[str, Any]]:
    """Return discovered TrackiqResult metadata rows from common folders."""
    try:
        rows = ResultBrowser().to_dict()
        return rows if isinstance(rows, list) else []
    except Exception:
        return []


def _try_load_result(path: str) -> TrackiqResult | None:
    """Best-effort load TrackiqResult; returns None with UI feedback on failure."""
    try:
        return load_trackiq_result(path)
    except Exception as exc:  # pragma: no cover - UI feedback path
        st.error(f"Failed to load result file '{path}': {exc}")
        return None


class CoreDashboard(TrackiqDashboard):
    """Generic tabbed dashboard for canonical TrackIQ result objects."""

    def __init__(self, result: TrackiqResult) -> None:
        super().__init__(result=result, theme=DARK_THEME, title="TrackIQ Core Dashboard")

    def render_body(self) -> None:
        """Render generic TrackIQ Core tabs."""
        result = self._primary_result()
        payload = result.tool_payload if isinstance(result.tool_payload, dict) else {}

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tool", result.tool_name)
        with col2:
            st.metric("Throughput", f"{float(result.metrics.throughput_samples_per_sec):.2f} samples/s")
        with col3:
            p99 = result.metrics.latency_p99_ms
            st.metric("P99 Latency", f"{float(p99):.2f} ms" if isinstance(p99, (int, float)) else "N/A")
        with col4:
            power = result.metrics.power_consumption_watts
            st.metric("Power", f"{float(power):.2f} W" if isinstance(power, (int, float)) else "N/A")

        tab_overview, tab_power, tab_kv, tab_payload, tab_trends, tab_downloads = st.tabs(
            ["Overview", "Power & Thermal", "KV Cache", "Payload", "Trends", "Downloads"]
        )

        with tab_overview:
            RegressionBadge(result.regression, theme=self.theme).render()
            MetricTable(result=result, mode="single", theme=self.theme).render()
            st.markdown("### Run Context")
            st.table(
                [
                    {"Field": "Hardware", "Value": result.platform.hardware_name},
                    {"Field": "OS", "Value": result.platform.os},
                    {"Field": "Framework", "Value": f"{result.platform.framework} {result.platform.framework_version}"},
                    {"Field": "Workload", "Value": f"{result.workload.name} ({result.workload.workload_type})"},
                    {"Field": "Batch Size", "Value": result.workload.batch_size},
                    {"Field": "Steps", "Value": result.workload.steps},
                ]
            )

        with tab_power:
            PowerGauge(metrics=result.metrics, tool_payload=payload, theme=self.theme).render()
            temp = result.metrics.temperature_celsius
            if isinstance(temp, (int, float)):
                st.caption(f"Peak temperature reported: {float(temp):.2f} C")

        with tab_kv:
            self.render_kv_cache_section()
            if result.kv_cache is None and not (
                isinstance(payload, dict) and isinstance(payload.get("kv_cache"), dict)
            ):
                st.info("No KV cache payload present in this result.")

        with tab_payload:
            st.markdown("### Tool Payload")
            if payload:
                st.json(payload)
            else:
                st.info("No tool payload data available.")
            with st.expander("Canonical Result JSON", expanded=False):
                st.json(result.to_dict())

        with tab_trends:
            history_dir = st.text_input(
                "History directory",
                value="output",
                help="Directory with TrackiqResult JSON files for trend analysis.",
                key="trackiq_core_history_dir",
            )
            self.render_trend_section(history_dir=history_dir)
            st.caption("Trend charts appear when at least two valid result files are found.")

        with tab_downloads:
            self.render_download_section()


def main() -> None:
    """Render interactive TrackIQ Core dashboard."""
    st.set_page_config(
        page_title="TrackIQ Core Interactive Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_ui_style()
    st.title("TrackIQ Core Interactive Dashboard")
    _render_page_intro()

    with st.sidebar:
        st.subheader("Result Source")
        st.caption("Load any canonical TrackiqResult JSON.")
        input_mode = st.radio(
            "Input Mode",
            options=["Browse results", "Manual path"],
            index=0,
            key="trackiq_core_input_mode",
        )
        rows = _discover_result_rows()
        result_path = ""
        if input_mode == "Browse results":
            if not rows:
                st.info("No results discovered. Switch to manual mode or run a benchmark first.")
            labels = [
                f"{idx + 1}. {row.get('tool_name', '?')} | {row.get('workload_name', '?')} | {row.get('timestamp', '?')}"
                for idx, row in enumerate(rows)
            ]
            if labels:
                idx_sel = st.selectbox(
                    "Discovered Results",
                    options=list(range(len(labels))),
                    format_func=lambda i: labels[i],
                    key="trackiq_core_result_select",
                )
                result_path = str(rows[idx_sel]["path"])
        else:
            result_path = st.text_input("Result JSON path", value="output/autoperf_result.json")
        load_clicked = st.button("Load Result", use_container_width=True, type="primary")

    if "trackiq_core_result" not in st.session_state and rows:
        auto_loaded = _try_load_result(str(rows[0]["path"]))
        if auto_loaded is not None:
            st.session_state["trackiq_core_result"] = auto_loaded
            st.session_state["trackiq_core_result_path"] = str(rows[0]["path"])

    if load_clicked:
        if not result_path or not Path(result_path).exists():
            st.error(f"Result file not found: {result_path}")
        else:
            loaded = _try_load_result(result_path)
            if loaded is not None:
                st.session_state["trackiq_core_result"] = loaded
                st.session_state["trackiq_core_result_path"] = result_path

    result = st.session_state.get("trackiq_core_result")
    if result is None:
        result = _build_demo_result()
        st.info("Showing demo result. Load a real result from the sidebar.")

    active_theme = LIGHT_THEME if st.session_state.get("theme") == LIGHT_THEME.name else DARK_THEME
    dashboard = CoreDashboard(result=result)
    dashboard.theme = active_theme
    dashboard.apply_theme(active_theme)
    dashboard.render_header()
    dashboard.render_sidebar()
    dashboard.render_body()
    dashboard.render_footer()


if __name__ == "__main__":
    main()
