"""Comparison dashboard built on the shared TrackIQ UI layer."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from trackiq_core.schema import TrackiqResult
from trackiq_core.ui import (
    ComparisonTable,
    DARK_THEME,
    RegressionBadge,
    TrackiqDashboard,
    TrackiqTheme,
)
from trackiq_compare.comparator import MetricComparator, SummaryGenerator
from trackiq_compare.reporters import HtmlReporter


class CompareDashboard(TrackiqDashboard):
    """Dashboard for side-by-side TrackiqResult comparisons."""

    def __init__(
        self,
        result_a: TrackiqResult,
        result_b: Optional[TrackiqResult] = None,
        *,
        label_a: Optional[str] = None,
        label_b: Optional[str] = None,
        result: Optional[List[TrackiqResult]] = None,
        theme: TrackiqTheme = DARK_THEME,
        title: str = "TrackIQ Compare Dashboard",
    ) -> None:
        if result is not None:
            if len(result) != 2:
                raise ValueError("CompareDashboard requires exactly two results.")
            left, right = result
        else:
            if result_b is None:
                raise ValueError("CompareDashboard requires result_a and result_b.")
            left, right = result_a, result_b

        self.result_a = left
        self.result_b = right
        self.label_a = label_a or left.tool_name
        self.label_b = label_b or right.tool_name
        super().__init__(result=[left, right], theme=theme, title=title)

    def build_components(self) -> Dict[str, object]:
        """Build testable comparison dashboard components."""
        return {
            "comparison_table": ComparisonTable(
                self.result_a,
                self.result_b,
                label_a=self.label_a,
                label_b=self.label_b,
                theme=self.theme,
            ),
            "regression_badge_a": RegressionBadge(self.result_a.regression, theme=self.theme),
            "regression_badge_b": RegressionBadge(self.result_b.regression, theme=self.theme),
        }

    def render_body(self) -> None:
        """Render compare dashboard content."""
        import streamlit as st

        components = self.build_components()
        components["comparison_table"].render()

        col_a, col_b = st.columns(2)
        with col_a:
            st.caption(f"Regression: {self.label_a}")
            components["regression_badge_a"].render()
        with col_b:
            st.caption(f"Regression: {self.label_b}")
            components["regression_badge_b"].render()

        comparator = MetricComparator(label_a=self.label_a, label_b=self.label_b)
        comparison = comparator.compare(self.result_a, self.result_b)
        summary = SummaryGenerator().generate(comparison)

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = str(Path(tmpdir) / "trackiq_compare_report.html")
            HtmlReporter().generate(report_path, comparison, summary, self.result_a, self.result_b)
            html = Path(report_path).read_text(encoding="utf-8")
        st.download_button(
            "Download Report",
            data=html,
            file_name="trackiq_compare_report.html",
            mime="text/html",
        )

