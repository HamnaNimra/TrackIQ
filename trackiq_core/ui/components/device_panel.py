"""Device panel component for TrackIQ dashboards."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from trackiq_core.hardware.devices import (
    DEVICE_TYPE_CPU,
    DEVICE_TYPE_NVIDIA_DRIVE,
    DEVICE_TYPE_NVIDIA_GPU,
    DEVICE_TYPE_NVIDIA_JETSON,
    DeviceProfile,
    get_platform_metadata_for_device,
)
from trackiq_core.hardware.gpu import get_memory_metrics, get_performance_metrics
from trackiq_core.ui.theme import DARK_THEME, TrackiqTheme


class DevicePanel:
    """Render device details and optional live metrics."""

    def __init__(
        self,
        devices: List[DeviceProfile],
        show_live_metrics: bool = True,
        theme: TrackiqTheme = DARK_THEME,
    ) -> None:
        self.devices = devices
        self.show_live_metrics = show_live_metrics
        self.theme = theme
        self.selected_device_index = 0

    def _selected_device(self) -> Optional[DeviceProfile]:
        if not self.devices:
            return None
        idx = max(0, min(self.selected_device_index, len(self.devices) - 1))
        return self.devices[idx]

    def _live_metrics(self, device: Optional[DeviceProfile]) -> Optional[Dict[str, Any]]:
        if not self.show_live_metrics or device is None:
            return None
        if device.device_type == DEVICE_TYPE_NVIDIA_GPU:
            perf = get_performance_metrics()
            mem = get_memory_metrics()
            if perf is None and mem is None:
                return None
            return {"performance": perf, "memory": mem}
        if device.device_type == DEVICE_TYPE_CPU:
            try:
                import psutil

                return {
                    "cpu_percent": float(psutil.cpu_percent(interval=None)),
                    "memory_percent": float(psutil.virtual_memory().percent),
                }
            except Exception:
                return None
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Return serializable device panel payload."""
        selected = self._selected_device()
        return {
            "devices": [device.to_dict() for device in self.devices],
            "selected_device_index": (
                self.selected_device_index if self.devices else None
            ),
            "live_metrics": self._live_metrics(selected),
        }

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
            f"<div><b>Driver:</b> {metadata.get('driver_version', 'N/A')}</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        live = self._live_metrics(device)
        if not self.show_live_metrics:
            return

        if device.device_type == DEVICE_TYPE_NVIDIA_GPU:
            if not live:
                st.markdown(
                    f"<div style='color:{self.theme.warning_color};'>Live metrics not available for this device</div>",
                    unsafe_allow_html=True,
                )
                return
            perf = live.get("performance") or {}
            mem = live.get("memory") or {}
            col1, col2, col3 = st.columns(3)
            col1.metric("GPU Utilization", f"{perf.get('utilization', 0):.1f}%")
            col2.metric("Temperature", f"{perf.get('temperature', 0):.1f} C")
            col3.metric("Power Draw", f"{perf.get('power', 0):.1f} W")
            used = float(mem.get("gpu_memory_used_mb", 0.0) or 0.0)
            total = float(mem.get("gpu_memory_total_mb", 0.0) or 0.0)
            percent = float(mem.get("gpu_memory_percent", 0.0) or 0.0)
            st.caption(f"Memory: {used:.0f} / {total:.0f} MB")
            st.progress(max(0.0, min(1.0, percent / 100.0)))
            return

        if device.device_type == DEVICE_TYPE_CPU:
            if not live:
                st.markdown(
                    f"<div style='color:{self.theme.warning_color};'>Live metrics not available for this device</div>",
                    unsafe_allow_html=True,
                )
                return
            col1, col2 = st.columns(2)
            col1.metric("CPU Utilization", f"{live.get('cpu_percent', 0):.1f}%")
            col2.metric("Memory Utilization", f"{live.get('memory_percent', 0):.1f}%")
            return

        if device.device_type in (DEVICE_TYPE_NVIDIA_JETSON, DEVICE_TYPE_NVIDIA_DRIVE):
            st.info(
                f"Connect tegrastats for live metrics. Device: {device.device_name}, SoC: {device.soc or 'N/A'}"
            )
            return

        st.markdown(
            f"<div style='color:{self.theme.warning_color};'>Live metrics not available for this device</div>",
            unsafe_allow_html=True,
        )

