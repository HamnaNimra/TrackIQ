"""Comparison dashboard built on the shared TrackIQ UI layer."""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trackiq_compare.comparator import MetricComparator, SummaryGenerator
from trackiq_compare.reporters import HtmlReporter
from trackiq_core.schema import TrackiqResult
from trackiq_core.ui import (
    DARK_THEME,
    ComparisonTable,
    RegressionBadge,
    TrackiqDashboard,
    TrackiqTheme,
)

VENDOR_COLORS: dict[str, str] = {
    "AMD": "#e53935",
    "NVIDIA": "#76b900",
    "Intel": "#0071c5",
    "Apple": "#555555",
    "Qualcomm": "#3253dc",
    "CPU": "#80868b",
    "Unknown": "#80868b",
}


def detect_platform_vendor(hardware_name: str) -> str:
    """Detect normalized platform vendor from a hardware name string."""
    normalized = hardware_name.lower()
    if any(token in normalized for token in ["amd", "radeon", "mi300", "mi250", "cdna", "rdna", "rocm"]):
        return "AMD"
    if any(
        token in normalized
        for token in [
            "nvidia",
            "geforce",
            "rtx",
            "a100",
            "h100",
            "v100",
            "tesla",
            "quadro",
            "cuda",
        ]
    ):
        return "NVIDIA"
    if any(token in normalized for token in ["intel arc", "iris", "xe", "oneapi"]):
        return "Intel"
    if any(token in normalized for token in ["apple", "m1", "m2", "m3", "m4", "metal"]):
        return "Apple"
    if any(token in normalized for token in ["qualcomm", "snapdragon", "hexagon", "adreno"]):
        return "Qualcomm"
    if "cpu" in normalized or "ryzen" in normalized or "intel core" in normalized or "apple m" in normalized:
        return "CPU"
    if "intel" in normalized:
        return "Intel"
    return "Unknown"


class CompareDashboard(TrackiqDashboard):
    """Dashboard for side-by-side TrackiqResult comparisons."""

    def __init__(
        self,
        result_a: TrackiqResult,
        result_b: TrackiqResult | None = None,
        *,
        label_a: str | None = None,
        label_b: str | None = None,
        result: list[TrackiqResult] | None = None,
        theme: TrackiqTheme = DARK_THEME,
        title: str = "TrackIQ Compare Dashboard",
        regression_threshold_percent: float = 5.0,
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
        self.regression_threshold_percent = float(regression_threshold_percent)
        super().__init__(result=[left, right], theme=theme, title=title)

    def _vendors(self) -> tuple[str, str]:
        return (
            detect_platform_vendor(self.result_a.platform.hardware_name),
            detect_platform_vendor(self.result_b.platform.hardware_name),
        )

    @staticmethod
    def platform_export_filename(vendor_a: str, vendor_b: str, timestamp: str) -> str:
        """Build platform comparison export filename."""
        return f"{vendor_a.lower()}_vs_{vendor_b.lower()}_comparison_{timestamp}.html"

    def is_platform_comparison_mode(self) -> bool:
        """Return True when both results are from different detected vendors."""
        vendor_a, vendor_b = self._vendors()
        return vendor_a != vendor_b

    def _competitive_metric_rows(self) -> list[dict[str, Any]]:
        metrics_a = self.result_a.metrics
        metrics_b = self.result_b.metrics
        rows: list[dict[str, Any]] = []
        candidates = [
            ("performance_per_watt", metrics_a.performance_per_watt, metrics_b.performance_per_watt, "high", 2),
            (
                "throughput_samples_per_sec",
                metrics_a.throughput_samples_per_sec,
                metrics_b.throughput_samples_per_sec,
                "high",
                1,
            ),
            ("latency_p99_ms", metrics_a.latency_p99_ms, metrics_b.latency_p99_ms, "low", 1),
            (
                "memory_utilization_percent",
                metrics_a.memory_utilization_percent,
                metrics_b.memory_utilization_percent,
                "context",
                0,
            ),
            (
                "communication_overhead_percent",
                metrics_a.communication_overhead_percent,
                metrics_b.communication_overhead_percent,
                "low",
                1,
            ),
        ]
        for name, a_val, b_val, direction, weight in candidates:
            if a_val is None or b_val is None:
                continue
            if a_val == 0:
                delta_pct = 0.0 if b_val == 0 else float("inf")
            else:
                delta_pct = ((b_val - a_val) / a_val) * 100.0

            if direction == "high":
                winner = self.label_b if b_val > a_val else self.label_a if a_val > b_val else "tie"
            elif direction == "low":
                winner = self.label_b if b_val < a_val else self.label_a if a_val < b_val else "tie"
            else:
                winner = "context"

            rows.append(
                {
                    "metric": name,
                    "a": a_val,
                    "b": b_val,
                    "delta_percent": delta_pct,
                    "winner": winner,
                    "weight": weight,
                    "direction": direction,
                }
            )
        return rows

    def _competitive_verdict(self, rows: list[dict[str, Any]]) -> str:
        score_a = 0
        score_b = 0
        details: list[str] = []
        for row in rows:
            if row["winner"] == self.label_a:
                score_a += int(row["weight"])
            elif row["winner"] == self.label_b:
                score_b += int(row["weight"])
            if row["winner"] in (self.label_a, self.label_b) and row["delta_percent"] != float("inf"):
                details.append(f"{row['metric']}: {row['winner']} by {abs(row['delta_percent']):.2f}%")
            elif row["winner"] == "context":
                details.append(
                    f"{row['metric']}: context metric ({self.label_a}={row['a']:.2f}, {self.label_b}={row['b']:.2f})"
                )

        if score_a > score_b:
            overall = self.label_a
        elif score_b > score_a:
            overall = self.label_b
        else:
            overall = "tie"
        return (
            f"Overall winner: {overall}. "
            f"Weighted score ({self.label_a}={score_a}, {self.label_b}={score_b}), "
            f"with performance_per_watt weighted 2x. " + " | ".join(details)
        )

    def _render_platform_comparison_mode(self) -> None:
        import streamlit as st

        vendor_a, vendor_b = self._vendors()
        badge_a = VENDOR_COLORS.get(vendor_a, self.theme.warning_color)
        badge_b = VENDOR_COLORS.get(vendor_b, self.theme.warning_color)

        st.markdown(
            f"""
            <div style="padding:12px;border-radius:{self.theme.border_radius};background:{self.theme.surface_color};">
              <div style="font-size:22px;font-weight:800;color:{self.theme.accent_color};">
                {vendor_a} vs {vendor_b} Performance Comparison
              </div>
              <div style="margin-top:8px;">
                <span style="background:{badge_a};color:#fff;padding:4px 8px;border-radius:999px;margin-right:8px;">{vendor_a}</span>
                <span style="background:{badge_b};color:#fff;padding:4px 8px;border-radius:999px;">{vendor_b}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        rows = self._competitive_metric_rows()
        if rows:
            st.subheader("Competitive Metrics")
            st.table(
                [
                    {
                        "Metric": row["metric"],
                        self.label_a: row["a"],
                        self.label_b: row["b"],
                        "Delta %": row["delta_percent"],
                        "Winner": row["winner"],
                    }
                    for row in rows
                ]
            )
            st.subheader("Competitive Verdict")
            st.write(self._competitive_verdict(rows))

            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            filename = self.platform_export_filename(vendor_a, vendor_b, timestamp)
            html = (
                "<!doctype html><html><head><meta charset='utf-8'><title>Platform Comparison</title></head><body>"
                f"<h1>{vendor_a} vs {vendor_b} Performance Comparison</h1>"
                f"<p><b>{self.label_a}</b> ({vendor_a}) vs <b>{self.label_b}</b> ({vendor_b})</p>"
                f"<pre>{json.dumps(rows, indent=2)}</pre>"
                f"<p>{self._competitive_verdict(rows)}</p></body></html>"
            )
            st.download_button(
                "Export Comparison Report",
                data=html,
                file_name=filename,
                mime="text/html",
                key="platform_comparison_export",
            )

    def _render_input_context(self) -> None:
        """Render comparison input/configuration context for both sides."""
        import streamlit as st

        st.markdown("### Comparison Configuration")
        left, right = st.columns(2)
        with left:
            st.markdown(f"**{self.label_a}**")
            st.markdown(f"- Tool: `{self.result_a.tool_name} {self.result_a.tool_version}`")
            st.markdown(f"- Hardware: `{self.result_a.platform.hardware_name}`")
            st.markdown(f"- Framework: `{self.result_a.platform.framework} {self.result_a.platform.framework_version}`")
            st.markdown(f"- Workload: `{self.result_a.workload.name}` ({self.result_a.workload.workload_type})")
            st.markdown(f"- Batch Size: `{self.result_a.workload.batch_size}`")
            st.markdown(f"- Steps: `{self.result_a.workload.steps}`")
        with right:
            st.markdown(f"**{self.label_b}**")
            st.markdown(f"- Tool: `{self.result_b.tool_name} {self.result_b.tool_version}`")
            st.markdown(f"- Hardware: `{self.result_b.platform.hardware_name}`")
            st.markdown(f"- Framework: `{self.result_b.platform.framework} {self.result_b.platform.framework_version}`")
            st.markdown(f"- Workload: `{self.result_b.workload.name}` ({self.result_b.workload.workload_type})")
            st.markdown(f"- Batch Size: `{self.result_b.workload.batch_size}`")
            st.markdown(f"- Steps: `{self.result_b.workload.steps}`")

    def _render_metric_graphs(self, rows: list[dict[str, Any]]) -> None:
        """Render chart-based metric comparison visuals."""
        import streamlit as st

        if not rows:
            st.info("No comparable metric rows available for graph rendering.")
            return

        plot_rows = [
            row for row in rows if isinstance(row.get("a"), (int, float)) and isinstance(row.get("b"), (int, float))
        ]
        if not plot_rows:
            st.info("No numeric metrics available for graph rendering.")
            return

        st.markdown("### Comparison Graphs")
        try:
            import pandas as pd
            import plotly.express as px
            import plotly.graph_objects as go
        except Exception:
            st.bar_chart(
                {
                    self.label_a: [float(row["a"]) for row in plot_rows],
                    self.label_b: [float(row["b"]) for row in plot_rows],
                }
            )
            return

        df = pd.DataFrame(
            {
                "metric": [str(row["metric"]) for row in plot_rows],
                "a": [float(row["a"]) for row in plot_rows],
                "b": [float(row["b"]) for row in plot_rows],
                "delta_percent": [float(row.get("delta_percent", 0.0)) for row in plot_rows],
            }
        )
        long_df = df.melt(
            id_vars=["metric", "delta_percent"],
            value_vars=["a", "b"],
            var_name="side",
            value_name="value",
        )
        long_df["side"] = long_df["side"].map({"a": self.label_a, "b": self.label_b})

        col_a, col_b = st.columns(2)
        with col_a:
            fig_values = px.bar(
                long_df,
                x="metric",
                y="value",
                color="side",
                barmode="group",
                title="Metric Values by Side",
                labels={"metric": "Metric", "value": "Value", "side": "Result"},
            )
            fig_values.update_layout(xaxis_tickangle=-25)
            st.plotly_chart(fig_values, use_container_width=True)
        with col_b:
            fig_delta = px.bar(
                df,
                x="metric",
                y="delta_percent",
                title="Percent Delta (B vs A)",
                labels={"metric": "Metric", "delta_percent": "Delta (%)"},
                color="delta_percent",
                color_continuous_scale="RdYlGn",
            )
            fig_delta.update_layout(xaxis_tickangle=-25)
            fig_delta.add_hline(y=0, line_dash="dash", line_color="#6b7280")
            st.plotly_chart(fig_delta, use_container_width=True)

        weighted: list[tuple[str, int]] = []
        for row in plot_rows:
            winner = str(row.get("winner", "tie"))
            weight = int(row.get("weight", 0))
            if winner == self.label_a:
                weighted.append((self.label_a, weight))
            elif winner == self.label_b:
                weighted.append((self.label_b, weight))
        score_a = sum(weight for label, weight in weighted if label == self.label_a)
        score_b = sum(weight for label, weight in weighted if label == self.label_b)
        fig_score = go.Figure(
            data=[go.Bar(x=[self.label_a, self.label_b], y=[score_a, score_b], marker_color=["#3b82f6", "#22c55e"])]
        )
        fig_score.update_layout(
            title="Weighted Winner Score",
            xaxis_title="Result",
            yaxis_title="Score",
        )
        st.plotly_chart(fig_score, use_container_width=True)

    def build_components(self) -> dict[str, object]:
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
        self._render_input_context()
        if self.is_platform_comparison_mode():
            self._render_platform_comparison_mode()
        rows = self._competitive_metric_rows()
        self._render_metric_graphs(rows)
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
        summary = SummaryGenerator(regression_threshold_percent=self.regression_threshold_percent).generate(comparison)

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
        self.render_kv_cache_section()
        self.render_download_section()
