"""Regression badge component."""

from typing import Any

from trackiq_core.schema import RegressionInfo
from trackiq_core.ui.theme import DARK_THEME, TrackiqTheme


class RegressionBadge:
    """Render regression status as a dominant visual badge."""

    def __init__(self, regression: RegressionInfo, theme: TrackiqTheme = DARK_THEME) -> None:
        self.regression = regression
        self.theme = theme

    def to_dict(self) -> dict[str, Any]:
        """Return regression payload for testing and transport."""
        return {
            "status": self.regression.status,
            "delta_percent": self.regression.delta_percent,
            "failed_metrics": list(self.regression.failed_metrics),
            "baseline_id": self.regression.baseline_id,
        }

    def render(self) -> None:
        """Render regression badge and delta metric."""
        import streamlit as st

        is_pass = self.regression.status == "pass"
        color = self.theme.pass_color if is_pass else self.theme.fail_color
        label = "PASS" if is_pass else "FAIL"

        st.markdown(
            f"""
            <div style="
                background:{color};
                color:white;
                font-size:40px;
                font-weight:700;
                border-radius:{self.theme.border_radius};
                padding:18px;
                text-align:center;">
                {label}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.metric(
            label="Delta Percent",
            value=f"{self.regression.delta_percent:.2f}%",
            delta=f"{self.regression.delta_percent:.2f}%",
            delta_color="inverse",
        )
