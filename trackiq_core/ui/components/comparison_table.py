"""Result comparison component."""

from __future__ import annotations

from typing import Any

from trackiq_core.schema import TrackiqResult
from trackiq_core.ui.components.metric_table import MetricTable
from trackiq_core.ui.theme import DARK_THEME, TrackiqTheme


class ComparisonTable:
    """Render and serialize pairwise TrackiqResult comparison data."""

    def __init__(
        self,
        result_a: TrackiqResult,
        result_b: TrackiqResult,
        label_a: str | None = None,
        label_b: str | None = None,
        theme: TrackiqTheme = DARK_THEME,
        regression_threshold_percent: float = 5.0,
    ) -> None:
        self.result_a = result_a
        self.result_b = result_b
        self.label_a = label_a or result_a.tool_name
        self.label_b = label_b or result_b.tool_name
        self.theme = theme
        self.regression_threshold_percent = regression_threshold_percent

    def _platform_diff(self) -> dict[str, tuple[str, str]]:
        a = self.result_a.platform
        b = self.result_b.platform
        diff: dict[str, tuple[str, str]] = {}
        if a.hardware_name != b.hardware_name:
            diff["hardware_name"] = (a.hardware_name, b.hardware_name)
        if a.os != b.os:
            diff["os"] = (a.os, b.os)
        if a.framework != b.framework:
            diff["framework"] = (a.framework, b.framework)
        if a.framework_version != b.framework_version:
            diff["framework_version"] = (a.framework_version, b.framework_version)
        return diff

    def _summary(self, rows: list[dict[str, Any]]) -> str:
        comparable = [row for row in rows if row.get("winner") in ("A", "B")]
        wins_a = sum(1 for row in comparable if row["winner"] == "A")
        wins_b = sum(1 for row in comparable if row["winner"] == "B")
        overall = self.label_a if wins_a > wins_b else self.label_b if wins_b > wins_a else "tie"

        largest = sorted(
            [row for row in rows if isinstance(row.get("delta_percent"), (int, float))],
            key=lambda x: abs(float(x["delta_percent"])),
            reverse=True,
        )[:3]
        largest_text = (
            ", ".join(f"{row['metric']} ({row['delta_percent']:.2f}%)" for row in largest) if largest else "none"
        )

        regressions = [
            row["metric"]
            for row in rows
            if isinstance(row.get("delta_percent"), (int, float))
            and abs(float(row["delta_percent"])) > self.regression_threshold_percent
        ]
        regression_text = ", ".join(regressions) if regressions else "none"
        return (
            f"Overall winner: {overall}. "
            f"Largest deltas: {largest_text}. "
            f"Metrics exceeding {self.regression_threshold_percent:.1f}%: {regression_text}."
        )

    def to_dict(self) -> dict[str, Any]:
        """Return platform diff, metric table payload, and plain-English summary."""
        metric_payload = MetricTable(
            [self.result_a, self.result_b],
            mode="comparison",
            theme=self.theme,
        ).to_dict()
        rows = metric_payload.get("metrics", [])
        return {
            "labels": {"a": self.label_a, "b": self.label_b},
            "platform_diff": self._platform_diff(),
            "metric_comparison": metric_payload,
            "summary": self._summary(rows),
        }

    def render(self) -> None:
        """Render full comparison view."""
        import streamlit as st

        data = self.to_dict()
        st.markdown(
            f"<div style='font-weight:700;color:{self.theme.text_color};'>Platform Comparison</div>",
            unsafe_allow_html=True,
        )
        if data["platform_diff"]:
            st.json(data["platform_diff"])
        else:
            st.info("No platform differences detected.")

        st.markdown(
            f"<div style='font-weight:700;color:{self.theme.text_color};margin-top:8px;'>Metric Comparison</div>",
            unsafe_allow_html=True,
        )
        MetricTable(
            [self.result_a, self.result_b],
            mode="comparison",
            theme=self.theme,
        ).render()

        st.markdown(
            f"<div style='font-weight:700;color:{self.theme.text_color};margin-top:8px;'>Summary</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            (
                f"<div style='background:{self.theme.surface_color};padding:10px;"
                f"border-radius:{self.theme.border_radius};'>{data['summary']}</div>"
            ),
            unsafe_allow_html=True,
        )
