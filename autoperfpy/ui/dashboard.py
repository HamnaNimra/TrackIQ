"""AutoPerfPy dashboard built on the shared TrackIQ UI layer."""

from __future__ import annotations

import json
from typing import Any

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

    def build_components(self) -> dict[str, object]:
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

    def _tool_payload(self) -> dict[str, Any]:
        result = self._primary_result()
        return result.tool_payload if isinstance(result.tool_payload, dict) else {}

    def _render_performance_charts(self) -> None:
        """Render chart sections from canonical tool payload samples when available."""
        import streamlit as st

        payload = self._tool_payload()
        samples = payload.get("samples")
        if not isinstance(samples, list) or not samples:
            st.info("No sample timeline data available for charts in this result.")
            return

        try:
            from autoperfpy.reports import charts as shared_charts
        except Exception as exc:  # pragma: no cover - optional dependency path
            st.warning(f"Chart module unavailable: {exc}")
            return

        if not shared_charts.is_available():
            st.warning("Plotly/pandas are required to render charts.")
            return

        try:
            df = shared_charts.samples_to_dataframe(samples)
            if df.empty:
                st.info("Sample timeline data is empty.")
                return
            shared_charts.ensure_throughput_column(df)
            summary = payload.get("summary")
            if not isinstance(summary, dict) or not summary:
                summary = shared_charts.compute_summary_from_dataframe(df)
            sections = shared_charts.build_all_charts(df, summary)
        except Exception as exc:  # pragma: no cover - defensive UI path
            st.warning(f"Failed to build charts from result payload: {exc}")
            return

        if not sections:
            st.info("No supported chart metrics found in this result payload.")
            return

        st.markdown("### Performance Charts")
        for idx, (section, charts) in enumerate(sections.items()):
            with st.expander(section, expanded=(idx == 0)):
                for caption, fig in charts:
                    st.markdown(f"**{caption}**")
                    st.plotly_chart(fig, use_container_width=True)

    def render_body(self) -> None:
        """Render AutoPerfPy-specific dashboard body."""
        import streamlit as st

        components = self.build_components()
        components["regression_badge"].render()
        components["metric_table"].render()
        components["power_gauge"].render()
        self._render_performance_charts()
        self.render_kv_cache_section()

        result = self._primary_result()
        with st.expander("Raw Result", expanded=False):
            st.code(
                json.dumps(result.to_dict(), indent=2),
                language="json",
            )
        self.render_download_section()
