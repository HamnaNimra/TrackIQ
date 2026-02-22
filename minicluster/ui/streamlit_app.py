"""Interactive Streamlit app for running MiniCluster from the browser."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from minicluster.runner import RunConfig, run_distributed, save_metrics
from minicluster.ui.dashboard import MiniClusterDashboard
from trackiq_core.serializer import load_trackiq_result


def _apply_ui_style() -> None:
    """Apply consistent visual polish for MiniCluster app."""
    st.markdown(
        """
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


def main() -> None:
    """Render interactive MiniCluster runner + dashboard view."""
    st.set_page_config(
        page_title="MiniCluster Interactive Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_ui_style()
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
        st.markdown("---")
        st.subheader("Load Existing Result")
        latest_path = _latest_result_path()
        default_path = latest_path or "minicluster_results/run_metrics.json"
        result_path = st.text_input("Result Path", value=default_path)
        load_clicked = st.button("Load Result", use_container_width=True)

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
        with st.spinner("Running MiniCluster..."):
            st.session_state["minicluster_result"] = _run_and_load_result(cfg)
            st.session_state["minicluster_result_path"] = "generated-now"

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
        with st.spinner("Running quick MiniCluster smoke run..."):
            st.session_state["minicluster_result"] = _run_and_load_result(cfg)
            st.session_state["minicluster_result_path"] = "generated-quick-smoke"

    if load_clicked:
        try:
            st.session_state["minicluster_result"] = load_trackiq_result(result_path)
            st.session_state["minicluster_result_path"] = result_path
        except Exception as exc:
            st.error(f"Failed to load result file: {exc}")

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
            "Configure run settings and click 'Run MiniCluster', or load an existing " "result JSON from the sidebar."
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
