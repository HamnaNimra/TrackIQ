"""Interactive Streamlit app for running MiniCluster from the browser."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st

from minicluster.runner import RunConfig, run_distributed, save_metrics
from minicluster.ui.dashboard import MiniClusterDashboard
from trackiq_core.serializer import load_trackiq_result


def _run_and_load_result(config: RunConfig) -> Optional[object]:
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


def _latest_result_path() -> Optional[str]:
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
    st.title("MiniCluster Interactive Dashboard")
    st.caption("Run distributed training and inspect validation metrics live.")

    with st.sidebar:
        st.subheader("Run Settings")
        workers = st.number_input("Workers", min_value=1, max_value=16, value=1, step=1)
        steps = st.number_input("Steps", min_value=1, max_value=5000, value=100, step=1)
        batch_size = st.number_input("Batch Size", min_value=1, max_value=4096, value=32, step=1)
        learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=1.0, value=0.01, step=0.0001, format="%.4f")
        seed = st.number_input("Seed", min_value=0, max_value=2_147_483_647, value=42, step=1)
        tdp_watts = st.number_input("TDP Watts", min_value=10.0, max_value=1000.0, value=150.0, step=1.0)
        run_clicked = st.button("Run MiniCluster", use_container_width=True)
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
        )
        with st.spinner("Running MiniCluster..."):
            st.session_state["minicluster_result"] = _run_and_load_result(cfg)
            st.session_state["minicluster_result_path"] = "generated-now"

    if load_clicked:
        try:
            st.session_state["minicluster_result"] = load_trackiq_result(result_path)
            st.session_state["minicluster_result_path"] = result_path
        except Exception as exc:
            st.error(f"Failed to load result file: {exc}")

    result = st.session_state.get("minicluster_result")
    if result is None:
        st.info(
            "Configure run settings and click 'Run MiniCluster', or load an existing "
            "result JSON from the sidebar."
        )
        return

    dashboard = MiniClusterDashboard(result=result)
    dashboard.apply_theme(dashboard.theme)
    dashboard.render_sidebar()
    dashboard.render_body()
    dashboard.render_footer()
