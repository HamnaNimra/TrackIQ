"""Device panel component for TrackIQ dashboards."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from trackiq_core.hardware.devices import (
    DEVICE_TYPE_AMD_GPU,
    DEVICE_TYPE_APPLE_SILICON,
    DEVICE_TYPE_CPU,
    DEVICE_TYPE_INTEL_GPU,
    DEVICE_TYPE_NVIDIA_DRIVE,
    DEVICE_TYPE_NVIDIA_GPU,
    DEVICE_TYPE_NVIDIA_JETSON,
    DeviceProfile,
    get_platform_metadata_for_device,
)
from trackiq_core.hardware.gpu import (
    get_amd_gpu_metrics,
    get_apple_silicon_metrics,
    get_cpu_metrics,
    get_intel_gpu_metrics,
    get_memory_metrics,
    get_performance_metrics,
)
from trackiq_core.ui.theme import DARK_THEME, TrackiqTheme


def _nvidia_metrics(_: int) -> dict[str, Any] | None:
    mem = get_memory_metrics()
    perf = get_performance_metrics()
    if mem is None and perf is None:
        return None
    return {**(mem or {}), **(perf or {})}


METRICS_DISPATCH: dict[str, Callable[[int], dict[str, Any] | None]] = {
    DEVICE_TYPE_NVIDIA_GPU: _nvidia_metrics,
    DEVICE_TYPE_AMD_GPU: lambda idx: get_amd_gpu_metrics(idx),
    DEVICE_TYPE_INTEL_GPU: lambda idx: get_intel_gpu_metrics(),
    DEVICE_TYPE_APPLE_SILICON: lambda idx: get_apple_silicon_metrics(),
    DEVICE_TYPE_CPU: lambda idx: get_cpu_metrics(),
    DEVICE_TYPE_NVIDIA_JETSON: lambda idx: None,
    DEVICE_TYPE_NVIDIA_DRIVE: lambda idx: None,
}


class DevicePanel:
    """Render device details and optional live metrics."""

    def __init__(
        self,
        devices: list[DeviceProfile],
        show_live_metrics: bool = True,
        theme: TrackiqTheme = DARK_THEME,
    ) -> None:
        self.devices = devices
        self.show_live_metrics = show_live_metrics
        self.theme = theme
        self.selected_device_index = 0

    def _selected_device(self) -> DeviceProfile | None:
        if not self.devices:
            return None
        idx = max(0, min(self.selected_device_index, len(self.devices) - 1))
        return self.devices[idx]

    def _live_metrics(self, device: DeviceProfile | None) -> dict[str, Any] | None:
        if not self.show_live_metrics or device is None:
            return None
        handler = METRICS_DISPATCH.get(device.device_type)
        if handler is None:
            return None
        try:
            return handler(device.index)
        except Exception:
            return None

    def to_dict(self) -> dict[str, Any]:
        """Return serializable device panel payload."""
        selected = self._selected_device()
        return {
            "devices": [device.to_dict() for device in self.devices],
            "selected_device_index": (self.selected_device_index if self.devices else None),
            "live_metrics": self._live_metrics(selected),
        }

    def _temperature_state(self, temperature: float | None) -> str:
        if temperature is None:
            return "N/A"
        if temperature < 70:
            return "green"
        if temperature <= 85:
            return "amber"
        return "red"

    def _render_util_temp_power_memory(self, live: dict[str, Any], label_prefix: str) -> None:
        import streamlit as st

        util = live.get("utilization") or live.get("gpu_utilization") or live.get("cpu_utilization")
        temp = live.get("temperature")
        power = live.get("power")
        used = live.get("gpu_memory_used_mb")
        total = live.get("gpu_memory_total_mb")
        percent = live.get("gpu_memory_percent")
        if util is not None:
            st.metric(f"{label_prefix} Utilization", f"{float(util):.1f}%")
        if temp is not None:
            temp_state = self._temperature_state(float(temp))
            color = (
                self.theme.pass_color
                if temp_state == "green"
                else self.theme.warning_color if temp_state == "amber" else self.theme.fail_color
            )
            st.metric("Temperature (°C)", f"{float(temp):.1f}")
            st.markdown(
                f"<div style='color:{color};font-size:12px;'>Thermal State: {temp_state.upper()}</div>",
                unsafe_allow_html=True,
            )
        if power is not None:
            st.metric("Power Draw (W)", f"{float(power):.1f}")
        if used is not None and total is not None:
            pct = (
                float(percent)
                if percent is not None
                else (float(used) / float(total) * 100.0 if float(total) > 0 else 0.0)
            )
            st.caption(f"Memory Used: {float(used):.0f}/{float(total):.0f} MB")
            st.progress(max(0.0, min(1.0, pct / 100.0)))

    def render(self) -> None:
        """Render device selector and live metrics."""
        import streamlit as st

        st.markdown(
            f"<div style='font-weight:700;color:{self.theme.text_color};margin:6px 0;'>Devices</div>",
            unsafe_allow_html=True,
        )
        if not self.devices:
            st.info("No devices detected.")
            return

        if len(self.devices) > 1:
            selected = st.selectbox(
                "Device",
                options=list(range(len(self.devices))),
                format_func=lambda i: self.devices[i].device_name,
                index=min(self.selected_device_index, len(self.devices) - 1),
                key="trackiq_device_selector",
            )
            self.selected_device_index = int(selected)
        else:
            self.selected_device_index = 0

        device = self._selected_device()
        if device is None:
            st.info("No devices detected.")
            return

        metadata = get_platform_metadata_for_device(device)
        st.markdown(
            f"<div style='background:{self.theme.surface_color};padding:10px;border-radius:{self.theme.border_radius};'>"
            f"<div><b>Name:</b> {device.device_name}</div>"
            f"<div><b>Type:</b> {device.device_type}</div>"
            f"<div><b>GPU:</b> {device.gpu_model or 'N/A'}</div>"
            f"<div><b>SoC:</b> {device.soc or 'N/A'}</div>"
            f"<div><b>OS:</b> {metadata.get('os', 'N/A')}</div>"
            f"<div><b>Driver:</b> {metadata.get('driver_version', metadata.get('rocm_version', 'N/A'))}</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        if not self.show_live_metrics:
            return

        if device.device_type in (DEVICE_TYPE_NVIDIA_JETSON, DEVICE_TYPE_NVIDIA_DRIVE):
            st.markdown(
                (
                    f"<div style='background:{self.theme.surface_color};padding:10px;border-radius:{self.theme.border_radius};'>"
                    f"<div><b>{device.device_name}</b> ({device.soc or 'N/A'})</div>"
                    "<div>Connect tegrastats reader for live metrics.</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            return

        live = self._live_metrics(device)
        if live is None:
            st.markdown(
                f"<div style='color:{self.theme.warning_color};'>Live metrics not available for this device</div>",
                unsafe_allow_html=True,
            )
            return

        if device.device_type == DEVICE_TYPE_CPU:
            self._render_util_temp_power_memory(live, "CPU")
        elif device.device_type in (
            DEVICE_TYPE_NVIDIA_GPU,
            DEVICE_TYPE_AMD_GPU,
            DEVICE_TYPE_INTEL_GPU,
            DEVICE_TYPE_APPLE_SILICON,
        ):
            self._render_util_temp_power_memory(live, "GPU")
        else:
            st.markdown(
                "<div class='trackiq-card'>Live metrics not available for this device</div>",
                unsafe_allow_html=True,
            )
