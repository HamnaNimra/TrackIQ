"""MiniCluster UI launcher helpers."""

from __future__ import annotations

from pathlib import Path

from minicluster.ui import streamlit_app
from minicluster.ui.dashboard import MiniClusterDashboard
from trackiq_core.ui import run_dashboard


def launch_minicluster_dashboard(result_path: str | None = None) -> None:
    """Launch MiniCluster dashboard from result file or interactive app."""
    if result_path:
        if not Path(result_path).exists():
            raise SystemExit(f"--result does not exist: {result_path}")
        run_dashboard(MiniClusterDashboard, result_path=result_path)
        return
    streamlit_app.main()
