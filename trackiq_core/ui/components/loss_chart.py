"""Loss chart component for TrackIQ dashboards."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from trackiq_core.ui.theme import DARK_THEME, TrackiqTheme


class LossChart:
    """Render training loss evolution with optional baseline overlay."""

    def __init__(
        self,
        steps: List[int],
        loss_values: List[float],
        baseline_values: Optional[List[float]] = None,
        tolerance: float = 0.05,
        theme: TrackiqTheme = DARK_THEME,
    ) -> None:
        self.steps = steps
        self.loss_values = loss_values
        self.baseline_values = baseline_values
        self.tolerance = tolerance
        self.theme = theme

    def to_dict(self) -> Dict[str, Any]:
        """Return serializable chart data."""
        return {
            "steps": self.steps,
            "loss_values": self.loss_values,
            "baseline_values": self.baseline_values,
            "tolerance": self.tolerance,
        }

    def render(self) -> None:
        """Render line chart in Streamlit."""
        import streamlit as st

        st.markdown(
            f"<div style='font-weight:700;color:{self.theme.text_color};'>Loss Trend</div>",
            unsafe_allow_html=True,
        )
        if not self.steps or not self.loss_values:
            st.info("No loss data available.")
            return

        try:
            import plotly.graph_objects as go
        except Exception:
            if self.baseline_values and len(self.baseline_values) == len(self.steps):
                st.line_chart(
                    {
                        "loss": self.loss_values,
                        "baseline": self.baseline_values,
                    }
                )
                return
            st.line_chart({"loss": self.loss_values})
            return

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.steps,
                y=self.loss_values,
                mode="lines",
                name="loss",
                line={"color": self.theme.chart_colors[0]},
            )
        )
        if self.baseline_values and len(self.baseline_values) == len(self.steps):
            upper = [v * (1.0 + self.tolerance) for v in self.baseline_values]
            lower = [v * (1.0 - self.tolerance) for v in self.baseline_values]
            fig.add_trace(
                go.Scatter(
                    x=self.steps,
                    y=self.baseline_values,
                    mode="lines",
                    name="baseline",
                    line={"color": self.theme.chart_colors[1]},
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=self.steps + list(reversed(self.steps)),
                    y=upper + list(reversed(lower)),
                    fill="toself",
                    fillcolor="rgba(200,200,200,0.2)",
                    line={"color": "rgba(255,255,255,0)"},
                    name="tolerance_band",
                )
            )
        st.plotly_chart(fig, use_container_width=True)
