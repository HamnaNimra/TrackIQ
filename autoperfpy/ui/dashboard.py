"""AutoPerfPy dashboard built on the shared TrackIQ UI layer."""

from __future__ import annotations

import json
from typing import Dict

from trackiq_core.schema import TrackiqResult
from trackiq_core.ui import (
    DARK_THEME,
    MetricTable,
    PowerGauge,
    RegressionBadge,
    TrackiqDashboard,
    TrackiqTheme,
)


class AutoPerfDashboard(TrackiqDashboard):
    """Dashboard for single AutoPerfPy inference results."""

    def __init__(
        self,
        result: TrackiqResult,
        theme: TrackiqTheme = DARK_THEME,
        title: str = "AutoPerfPy Dashboard",
    ) -> None:
        super().__init__(result=result, theme=theme, title=title)

    def expected_tool_names(self) -> list[str]:
        """AutoPerf dashboard should only load AutoPerfPy results."""
        return ["autoperfpy"]

    def build_components(self) -> Dict[str, object]:
        """Build component instances for testing and rendering."""
        result = self._primary_result()
        return {
            "regression_badge": RegressionBadge(result.regression, theme=self.theme),
            "metric_table": MetricTable(result=result, mode="single", theme=self.theme),
            "power_gauge": PowerGauge(
                metrics=result.metrics,
                tool_payload=result.tool_payload,
                theme=self.theme,
            ),
        }

    def render_body(self) -> None:
        """Render AutoPerfPy-specific dashboard body."""
        import streamlit as st

        components = self.build_components()
        components["regression_badge"].render()
        components["metric_table"].render()
        components["power_gauge"].render()
        self.render_kv_cache_section()

        result = self._primary_result()
        with st.expander("Raw Result", expanded=False):
            st.code(
                json.dumps(result.to_dict(), indent=2),
                language="json",
            )
        self.render_download_section()
