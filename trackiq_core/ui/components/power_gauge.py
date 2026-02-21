"""Power gauge component."""

from typing import Any

from trackiq_core.hardware.devices import DEVICE_TYPE_NVIDIA_GPU, DeviceProfile
from trackiq_core.hardware.gpu import get_performance_metrics
from trackiq_core.schema import Metrics
from trackiq_core.ui.theme import DARK_THEME, TrackiqTheme


class PowerGauge:
    """Render power and efficiency metrics."""

    def __init__(
        self,
        metrics: Metrics,
        tool_payload: dict[str, Any] | None = None,
        live_device: DeviceProfile | None = None,
        theme: TrackiqTheme = DARK_THEME,
    ) -> None:
        self.metrics = metrics
        self.tool_payload = tool_payload or {}
        self.live_device = live_device
        self.theme = theme

    def to_dict(self) -> dict[str, Any]:
        """Return power/efficiency payload."""
        peak = self.tool_payload.get("power_profile", {}).get("summary", {}).get("peak_power_watts")
        payload: dict[str, Any] = {
            "power_consumption_watts": self.metrics.power_consumption_watts,
            "peak_power_watts": peak,
            "performance_per_watt": self.metrics.performance_per_watt,
            "energy_per_step_joules": self.metrics.energy_per_step_joules,
            "temperature_celsius": self.metrics.temperature_celsius,
        }
        if self.live_device is not None and self.live_device.device_type == DEVICE_TYPE_NVIDIA_GPU:
            perf = get_performance_metrics()
            payload["live_power_watts"] = float(perf.get("power")) if perf and perf.get("power") is not None else None
        if (
            payload["power_consumption_watts"] is None
            and payload["peak_power_watts"] is None
            and payload["performance_per_watt"] is None
            and payload["energy_per_step_joules"] is None
            and payload.get("live_power_watts") is None
        ):
            payload["placeholder"] = "Power profiling not available in this environment."
        return payload

    def render(self) -> None:
        """Render power metrics as Streamlit metric widgets."""
        import streamlit as st

        data = self.to_dict()
        st.markdown(
            f"<div style='font-weight:700;color:{self.theme.text_color};'>Power Profile</div>",
            unsafe_allow_html=True,
        )
        if "placeholder" in data:
            st.markdown(
                (
                    f"<div class='trackiq-card' style='background:{self.theme.surface_color};"
                    f"color:{self.theme.warning_color};'>{data['placeholder']}</div>"
                ),
                unsafe_allow_html=True,
            )
            return

        cols = st.columns(4)
        if data.get("live_power_watts") is not None:
            cols[0].metric("Benchmark Power (W)", data["power_consumption_watts"] or "N/A")
            cols[1].metric("Live Power (W)", data["live_power_watts"])
            cols[2].metric("Perf / Watt", data["performance_per_watt"] or "N/A")
            cols[3].metric("Energy / Step (J)", data["energy_per_step_joules"] or "N/A")
            return

        cols[0].metric("Mean Power (W)", data["power_consumption_watts"] or "N/A")
        cols[1].metric("Peak Power (W)", data["peak_power_watts"] or "N/A")
        cols[2].metric("Perf / Watt", data["performance_per_watt"] or "N/A")
        cols[3].metric("Energy / Step (J)", data["energy_per_step_joules"] or "N/A")
