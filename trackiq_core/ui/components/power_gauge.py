"""Power gauge component."""

from typing import Any, Dict, Optional

from trackiq_core.schema import Metrics
from trackiq_core.ui.theme import DARK_THEME, TrackiqTheme


class PowerGauge:
    """Render power and efficiency metrics."""

    def __init__(
        self,
        metrics: Metrics,
        tool_payload: Optional[Dict[str, Any]] = None,
        theme: TrackiqTheme = DARK_THEME,
    ) -> None:
        self.metrics = metrics
        self.tool_payload = tool_payload or {}
        self.theme = theme

    def to_dict(self) -> Dict[str, Any]:
        """Return power/efficiency payload."""
        peak = (
            self.tool_payload.get("power_profile", {})
            .get("summary", {})
            .get("peak_power_watts")
        )
        payload = {
            "power_consumption_watts": self.metrics.power_consumption_watts,
            "peak_power_watts": peak,
            "performance_per_watt": self.metrics.performance_per_watt,
            "energy_per_step_joules": self.metrics.energy_per_step_joules,
            "temperature_celsius": self.metrics.temperature_celsius,
        }
        if (
            payload["power_consumption_watts"] is None
            and payload["peak_power_watts"] is None
            and payload["performance_per_watt"] is None
            and payload["energy_per_step_joules"] is None
        ):
            payload["placeholder"] = "Power profiling not available in this environment."
        return payload

    def render(self) -> None:
        """Render power metrics as Streamlit metric widgets."""
        import streamlit as st

        data = self.to_dict()
        st.subheader("Power Profile")
        if "placeholder" in data:
            st.markdown(
                f"<div class='trackiq-card'>{data['placeholder']}</div>",
                unsafe_allow_html=True,
            )
            return

        cols = st.columns(4)
        cols[0].metric("Mean Power (W)", data["power_consumption_watts"])
        cols[1].metric("Peak Power (W)", data["peak_power_watts"] or "N/A")
        cols[2].metric("Perf / Watt", data["performance_per_watt"] or "N/A")
        cols[3].metric("Energy / Step (J)", data["energy_per_step_joules"] or "N/A")

