"""MiniCluster dashboard built on TrackIQ shared UI components."""

from __future__ import annotations

from typing import Any, Dict, List

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

    def _tool_payload(self) -> Dict[str, Any]:
        result = self._primary_result()
        return result.tool_payload if isinstance(result.tool_payload, dict) else {}

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

        left, right = st.columns(2)
        with left:
            components["metric_table"].render()
        with right:
            workers = payload.get("workers", [])
            if isinstance(workers, list) and workers:
                components["worker_grid"].render()
            else:
                st.markdown(
                    "<div class='trackiq-card'>Worker data not available in tool payload.</div>",
                    unsafe_allow_html=True,
                )

        steps_data = payload.get("steps", [])
        if isinstance(steps_data, list) and steps_data:
            components["loss_chart"].render()
        else:
            st.info("Loss curve unavailable in tool payload.")

        components["power_gauge"].render()

        with st.expander("Fault Detection Report", expanded=False):
            faults = payload.get("faults_detected")
            if faults is None:
                st.write("No fault detection data available.")
            else:
                st.json(faults)

