"""Interactive Streamlit app for running MiniCluster from the browser."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from minicluster.runner import RunConfig, run_distributed, save_metrics
from minicluster.ui.dashboard import MiniClusterDashboard
from trackiq_core.schema import Metrics, PlatformInfo, RegressionInfo, TrackiqResult, WorkloadInfo
from trackiq_core.serializer import load_trackiq_result

UI_THEME_OPTIONS = ["System", "Light", "Dark"]


def _apply_ui_style(theme: str = "System") -> None:
    """Apply consistent visual polish for MiniCluster app."""
    prefers_dark = theme == "Dark"
    card_bg = "rgba(255,255,255,0.06)" if prefers_dark else "rgba(15,23,42,0.02)"
    card_border = "rgba(148,163,184,0.35)" if prefers_dark else "rgba(148,163,184,0.22)"
    hero_text = "#d1d5db" if prefers_dark else "#4b5563"
    st.markdown(
        f"""
        <style>
        .mc-hero {
            border: 1px solid rgba(20,184,166,0.25);
            background: linear-gradient(135deg, rgba(20,184,166,0.14), rgba(59,130,246,0.10));
            border-radius: 14px;
            padding: 14px 16px;
            margin-bottom: 14px;
        }
        .mc-hero h2 {
            margin: 0 0 4px 0;
            font-size: 1.26rem;
        }
        .mc-hero p {
            margin: 0;
            color: {hero_text};
            font-size: 0.95rem;
        }
        [data-testid="stMetric"] {
            border: 1px solid {card_border};
            border-radius: 12px;
            padding: 8px 10px;
            background: {card_bg};
        }
        button[kind="primary"] {
            border-radius: 10px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_page_intro() -> None:
    """Render quick orientation content."""
    st.markdown(
        """
        <div class="mc-hero">
          <h2>MiniCluster Validation Console</h2>
          <p>Configure a run in the sidebar, execute, then inspect cluster health, timing and fault signals.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Quick Start", expanded=False):
        st.markdown(
            "1. Set worker count, backend, workload, and steps.\n"
            "2. Click `Run MiniCluster` (or `Quick Smoke Run`).\n"
            "3. Review `Cluster Health Summary`, `Training Graphs`, and fault sections."
        )


def _run_and_load_result(config: RunConfig) -> object | None:
    """Run MiniCluster once and load canonical TrackiqResult output."""
    metrics = run_distributed(config)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        output_path = handle.name
    try:
        save_metrics(metrics, output_path)
        return load_trackiq_result(output_path)
    finally:
        try:
            os.unlink(output_path)
        except OSError:
            pass


def _latest_result_path() -> str | None:
    """Return latest known MiniCluster result file path if available."""
    candidates = [
        Path("minicluster_results") / "run_metrics.json",
        Path("output") / "minicluster_result.json",
    ]
    for path in candidates:
        if path.exists():
            return str(path)

    directory = Path("minicluster_results")
    if not directory.exists():
        return None
    json_files = sorted(
        directory.glob("*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not json_files:
        return None
    return str(json_files[0])


def _build_demo_result() -> TrackiqResult:
    """Build a demo MiniCluster canonical result for first-time users."""
    return TrackiqResult(
        tool_name="minicluster",
        tool_version="1.0.0",
        timestamp=datetime.now(timezone.utc),
        platform=PlatformInfo(
            hardware_name="Demo Cluster",
            os="Demo OS",
            framework="pytorch",
            framework_version="2.x",
        ),
        workload=WorkloadInfo(
            name="synthetic_training",
            workload_type="training",
            batch_size=32,
            steps=100,
        ),
        metrics=Metrics(
            throughput_samples_per_sec=128.0,
            latency_p50_ms=6.0,
            latency_p95_ms=8.0,
            latency_p99_ms=9.5,
            memory_utilization_percent=58.0,
            communication_overhead_percent=14.0,
            power_consumption_watts=280.0,
            energy_per_step_joules=2.2,
            performance_per_watt=0.46,
            temperature_celsius=71.0,
            scaling_efficiency_pct=92.0,
        ),
        regression=RegressionInfo(
            baseline_id=None,
            delta_percent=0.0,
            status="pass",
            failed_metrics=[],
        ),
        tool_payload={
            "config": {
                "num_processes": 4,
                "num_steps": 100,
                "batch_size": 32,
                "learning_rate": 0.01,
                "collective_backend": "nccl",
                "workload": "transformer",
                "baseline_throughput": 35.0,
                "seed": 42,
                "tdp_watts": 300.0,
            },
            "average_throughput_samples_per_sec": 128.0,
            "p95_allreduce_ms": 2.8,
            "p99_allreduce_ms": 3.4,
            "allreduce_stdev_ms": 0.22,
            "workers": [
                {"worker_id": "0", "throughput": 130.0, "allreduce_time_ms": 3.1, "status": "healthy", "loss": 1.2},
                {"worker_id": "1", "throughput": 127.0, "allreduce_time_ms": 3.4, "status": "healthy", "loss": 1.2},
                {"worker_id": "2", "throughput": 129.0, "allreduce_time_ms": 3.2, "status": "healthy", "loss": 1.2},
                {"worker_id": "3", "throughput": 126.0, "allreduce_time_ms": 3.3, "status": "healthy", "loss": 1.3},
            ],
            "steps": [
                {
                    "step": i,
                    "loss": round(1.8 / (1.0 + i / 15.0), 4),
                    "throughput_samples_per_sec": round(122.0 + (i % 8), 3),
                    "allreduce_time_ms": round(2.6 + (i % 5) * 0.2, 3),
                    "compute_time_ms": round(7.5 + (i % 4) * 0.25, 3),
                }
                for i in range(24)
            ],
            "faults_detected": {
                "num_faults": 3,
                "num_detected": 2,
            },
        },
    )


def _build_result_summary_text(result: object) -> str:
    """Build compact summary text for sidebar feedback."""
    if not isinstance(result, TrackiqResult):
        return "Run completed"
    p99 = result.metrics.latency_p99_ms
    throughput = result.metrics.throughput_samples_per_sec
    parts: list[str] = []
    if isinstance(p99, (int, float)):
        parts.append(f"P99 {float(p99):.2f} ms")
    if isinstance(throughput, (int, float)):
        parts.append(f"{float(throughput):.1f} samples/s")
    return ", ".join(parts) if parts else "Run completed"


def main() -> None:
    """Render interactive MiniCluster runner + dashboard view."""
    st.set_page_config(
        page_title="MiniCluster Interactive Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    selected_theme = st.session_state.get("minicluster_ui_theme", "System")
    if selected_theme not in UI_THEME_OPTIONS:
        selected_theme = "System"
    _apply_ui_style(selected_theme)
    st.markdown("<div id='minicluster-dashboard-top'></div>", unsafe_allow_html=True)
    st.title("MiniCluster Interactive Dashboard")
    _render_page_intro()

    with st.sidebar:
        st.subheader("Run Settings")
        st.caption("Set core run parameters, then launch from below.")
        workers = st.number_input("Workers", min_value=1, max_value=16, value=1, step=1)
        backend = st.selectbox("Collective Backend", options=["gloo", "nccl"], index=0)
        workload = st.selectbox("Workload", options=["mlp", "transformer", "embedding"], index=0)
        steps = st.number_input("Steps", min_value=1, max_value=5000, value=100, step=1)
        batch_size = st.number_input("Batch Size", min_value=1, max_value=4096, value=32, step=1)
        learning_rate = st.number_input(
            "Learning Rate", min_value=0.0001, max_value=1.0, value=0.01, step=0.0001, format="%.4f"
        )
        seed = st.number_input("Seed", min_value=0, max_value=2_147_483_647, value=42, step=1)
        tdp_watts = st.number_input("TDP Watts", min_value=10.0, max_value=1000.0, value=150.0, step=1.0)
        use_baseline = st.checkbox("Use Baseline Throughput", value=False)
        baseline_throughput = st.number_input(
            "Baseline Throughput (samples/s)",
            min_value=0.0,
            max_value=1_000_000.0,
            value=100.0,
            step=1.0,
            disabled=not use_baseline,
        )
        with st.expander("What these settings mean", expanded=False):
            st.markdown(
                "- `Collective Backend`: `gloo` for CPU/local CI, `nccl` for GPU clusters.\n"
                "- `Workload`: synthetic model shape for communication/computation behavior.\n"
                "- `Baseline Throughput`: enables scaling efficiency in output metrics."
            )
        run_clicked = st.button("Run MiniCluster", use_container_width=True, type="primary")
        quick_clicked = st.button("Quick Smoke Run (20 steps)", use_container_width=True)
        demo_clicked = st.button("Load Demo Result", use_container_width=True)
        if st.session_state.get("minicluster_last_summary"):
            st.caption(f"Last run summary: {st.session_state['minicluster_last_summary']}")
        st.markdown("---")
        st.subheader("Load Existing Result")
        latest_path = _latest_result_path()
        default_path = latest_path or "minicluster_results/run_metrics.json"
        result_path = st.text_input("Result Path", value=default_path)
        load_clicked = st.button("Load Result", use_container_width=True)
        st.markdown("---")
        st.subheader("View Options")
        st.selectbox(
            "Theme",
            options=UI_THEME_OPTIONS,
            index=UI_THEME_OPTIONS.index(selected_theme),
            key="minicluster_ui_theme",
        )
        st.markdown("[Back to top](#minicluster-dashboard-top)")

    if "minicluster_result" not in st.session_state:
        latest = _latest_result_path()
        if latest:
            try:
                st.session_state["minicluster_result"] = load_trackiq_result(latest)
                st.session_state["minicluster_result_path"] = latest
            except Exception:
                pass

    if run_clicked:
        cfg = RunConfig(
            num_steps=int(steps),
            num_processes=int(workers),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            seed=int(seed),
            tdp_watts=float(tdp_watts),
            collective_backend=str(backend),
            workload=str(workload),
            baseline_throughput=float(baseline_throughput) if use_baseline else None,
        )
        try:
            with st.spinner("Running MiniCluster..."):
                st.session_state["minicluster_result"] = _run_and_load_result(cfg)
                st.session_state["minicluster_result_path"] = "generated-now"
            st.session_state["minicluster_last_summary"] = _build_result_summary_text(
                st.session_state.get("minicluster_result")
            )
        except Exception as exc:
            st.error(f"MiniCluster run failed: {exc}")
            st.info("Try `Quick Smoke Run` or `Load Demo Result` to continue exploring the dashboard.")

    if quick_clicked:
        cfg = RunConfig(
            num_steps=20,
            num_processes=1,
            batch_size=16,
            learning_rate=0.01,
            seed=42,
            tdp_watts=float(tdp_watts),
            collective_backend=str(backend),
            workload=str(workload),
            baseline_throughput=float(baseline_throughput) if use_baseline else None,
        )
        try:
            with st.spinner("Running quick MiniCluster smoke run..."):
                st.session_state["minicluster_result"] = _run_and_load_result(cfg)
                st.session_state["minicluster_result_path"] = "generated-quick-smoke"
            st.session_state["minicluster_last_summary"] = _build_result_summary_text(
                st.session_state.get("minicluster_result")
            )
        except Exception as exc:
            st.error(f"Quick smoke run failed: {exc}")
            st.info("Load an existing result JSON or use `Load Demo Result`.")

    if demo_clicked:
        st.session_state["minicluster_result"] = _build_demo_result()
        st.session_state["minicluster_result_path"] = "demo-result"
        st.session_state["minicluster_last_summary"] = _build_result_summary_text(
            st.session_state.get("minicluster_result")
        )

    if load_clicked:
        try:
            st.session_state["minicluster_result"] = load_trackiq_result(result_path)
            st.session_state["minicluster_result_path"] = result_path
            st.session_state["minicluster_last_summary"] = _build_result_summary_text(
                st.session_state.get("minicluster_result")
            )
        except Exception as exc:
            st.error(f"Failed to load result file: {exc}")
            st.info("Use `Load Demo Result` if you want to inspect dashboard behavior immediately.")

    result = st.session_state.get("minicluster_result")
    if result is None:
        st.markdown("### Run Configuration Preview")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Workers", int(workers))
            st.metric("Steps", int(steps))
        with c2:
            st.metric("Batch Size", int(batch_size))
            st.metric("Learning Rate", float(learning_rate))
        with c3:
            st.metric("Seed", int(seed))
            st.metric("TDP (W)", float(tdp_watts))
        c4, c5 = st.columns(2)
        with c4:
            st.metric("Backend", str(backend))
        with c5:
            st.metric("Workload", str(workload))
        if use_baseline:
            st.metric("Baseline Throughput (samples/s)", float(baseline_throughput))
        st.info(
            "Configure run settings and click 'Run MiniCluster', load an existing result JSON, "
            "or click 'Load Demo Result' to explore immediately."
        )
        return
    st.success(f"Active result source: {st.session_state.get('minicluster_result_path', 'loaded')}")

    dashboard = MiniClusterDashboard(result=result)
    dashboard.apply_theme(dashboard.theme)
    dashboard.render_sidebar()
    dashboard.render_body()
    dashboard.render_footer()


if __name__ == "__main__":
    main()
