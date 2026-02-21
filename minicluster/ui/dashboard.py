"""MiniCluster dashboard built on TrackIQ shared UI components."""

from __future__ import annotations

import time
from typing import Any, Dict, List

from minicluster.monitor.health_reader import HealthReader
from trackiq_core.schema import TrackiqResult
from trackiq_core.ui import (
    DARK_THEME,
    LossChart,
    MetricTable,
    PowerGauge,
    RegressionBadge,
    TrackiqDashboard,
    TrackiqTheme,
    WorkerGrid,
)


class MiniClusterDashboard(TrackiqDashboard):
    """Dashboard for MiniCluster distributed training result files."""

    def __init__(
        self,
        result: TrackiqResult,
        theme: TrackiqTheme = DARK_THEME,
        title: str = "MiniCluster Dashboard",
    ) -> None:
        super().__init__(result=result, theme=theme, title=title)

    def expected_tool_names(self) -> list[str]:
        """MiniCluster dashboard should only load MiniCluster results."""
        return ["minicluster"]

    def _tool_payload(self) -> Dict[str, Any]:
        result = self._primary_result()
        return result.tool_payload if isinstance(result.tool_payload, dict) else {}

    def _payload_from_checkpoint(self, checkpoint: Any) -> Dict[str, Any]:
        workers = [
            {
                "worker_id": worker.worker_id,
                "throughput": worker.throughput_samples_per_sec,
                "allreduce_time_ms": worker.allreduce_time_ms,
                "status": worker.status,
                "loss": worker.loss,
            }
            for worker in checkpoint.workers
        ]
        snapshots = []
        for worker in checkpoint.workers:
            snapshots.append(
                {
                    "step": worker.step,
                    "loss": worker.loss,
                    "worker_id": worker.worker_id,
                }
            )
        return {
            "workers": workers,
            "steps": snapshots,
            "health_checkpoint_path": self._tool_payload().get("health_checkpoint_path"),
            "faults_detected": self._tool_payload().get("faults_detected"),
        }

    def _render_dynamic_sections(self, payload: Dict[str, Any]) -> None:
        """Render worker grid and loss chart from the provided payload."""
        import streamlit as st

        result = self._primary_result()
        workers = payload.get("workers", [])
        steps_data: List[Dict[str, Any]] = (
            payload.get("steps", []) if isinstance(payload.get("steps"), list) else []
        )
        steps = [int(item.get("step", idx)) for idx, item in enumerate(steps_data)]
        losses = [float(item.get("loss", 0.0)) for item in steps_data]

        left, right = st.columns(2)
        with left:
            MetricTable(result=result, mode="single", theme=self.theme).render()
        with right:
            if isinstance(workers, list) and workers:
                WorkerGrid(workers=workers, theme=self.theme).render()
            else:
                st.markdown(
                    "<div class='trackiq-card'>Worker data not available in tool payload.</div>",
                    unsafe_allow_html=True,
                )

        if steps_data:
            LossChart(
                steps=steps,
                loss_values=losses,
                baseline_values=None,
                theme=self.theme,
            ).render()
        else:
            st.info("Loss curve unavailable in tool payload.")

    def render_sidebar(self) -> None:
        """Render shared sidebar plus MiniCluster auto-refresh controls."""
        import streamlit as st

        super().render_sidebar()
        with st.sidebar:
            st.markdown("---")
            st.subheader("Live Refresh")
            current = st.session_state.get("minicluster_auto_refresh", False)
            auto_refresh = st.toggle(
                "Auto-refresh (2s)",
                value=bool(current),
                key="minicluster_auto_refresh",
            )
            st.session_state["trackiq_live_indicator"] = bool(auto_refresh)
            if st.button("Refresh Now", key="minicluster_refresh_now"):
                st.session_state["minicluster_force_refresh"] = True
                if hasattr(st, "rerun"):
                    st.rerun()
                else:  # pragma: no cover
                    st.experimental_rerun()

    def build_components(self) -> Dict[str, object]:
        """Build component instances for testable, reusable rendering."""
        result = self._primary_result()
        payload = self._tool_payload()
        steps_data: List[Dict[str, Any]] = (
            payload.get("steps", []) if isinstance(payload.get("steps"), list) else []
        )
        steps = [int(item.get("step", idx)) for idx, item in enumerate(steps_data)]
        losses = [float(item.get("loss", 0.0)) for item in steps_data]
        workers = payload.get("workers", [])

        return {
            "regression_badge": RegressionBadge(result.regression, theme=self.theme),
            "metric_table": MetricTable(result=result, mode="single", theme=self.theme),
            "worker_grid": WorkerGrid(workers=workers, theme=self.theme),
            "loss_chart": LossChart(
                steps=steps,
                loss_values=losses,
                baseline_values=None,
                theme=self.theme,
            ),
            "power_gauge": PowerGauge(
                metrics=result.metrics,
                tool_payload=result.tool_payload,
                theme=self.theme,
            ),
        }

    def render_body(self) -> None:
        """Render MiniCluster-specific dashboard content."""
        import streamlit as st

        components = self.build_components()
        payload = self._tool_payload()

        components["regression_badge"].render()
        dynamic_container = st.empty()
        checkpoint_path = payload.get("health_checkpoint_path")
        force_refresh = bool(st.session_state.pop("minicluster_force_refresh", False))
        auto_refresh = bool(st.session_state.get("minicluster_auto_refresh", False))
        local_payload = payload
        if (auto_refresh or force_refresh) and checkpoint_path:
            reader = HealthReader(str(checkpoint_path), timeout_seconds=2.0)
            checkpoint = reader.read()
            if checkpoint is not None:
                local_payload = self._payload_from_checkpoint(checkpoint)

        with dynamic_container.container():
            self._render_dynamic_sections(local_payload)

        if auto_refresh and checkpoint_path:
            reader = HealthReader(str(checkpoint_path), timeout_seconds=2.0)
            while True:
                checkpoint = reader.read()
                if checkpoint is not None:
                    local_payload = self._payload_from_checkpoint(checkpoint)
                with dynamic_container.container():
                    self._render_dynamic_sections(local_payload)
                if checkpoint is not None and checkpoint.is_complete:
                    st.success("Live run complete. Auto-refresh stopped.")
                    st.session_state["minicluster_auto_refresh"] = False
                    st.session_state["trackiq_live_indicator"] = False
                    break
                time.sleep(2)

        components["power_gauge"].render()
        self.render_kv_cache_section()

        with st.expander("Fault Detection Report", expanded=False):
            faults = local_payload.get("faults_detected")
            if faults is None:
                st.write("No fault detection data available.")
            else:
                st.json(faults)
        self.render_download_section()
