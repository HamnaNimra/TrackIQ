"""Tests for multi-platform hardware detection and live metrics."""

from __future__ import annotations

import sys

import pytest

from trackiq_core.hardware.devices import (
    DEVICE_TYPE_AMD_GPU,
    DEVICE_TYPE_APPLE_SILICON,
    DEVICE_TYPE_CPU,
    DEVICE_TYPE_INTEL_GPU,
    DEVICE_TYPE_NVIDIA_DRIVE,
    DEVICE_TYPE_NVIDIA_GPU,
    DEVICE_TYPE_NVIDIA_JETSON,
    DeviceProfile,
    detect_amd_gpus,
    detect_apple_silicon,
    get_all_devices,
)
from trackiq_core.hardware import gpu as gpu_mod
from trackiq_core.ui.components.device_panel import DevicePanel, METRICS_DISPATCH


def test_detect_amd_gpus_empty_when_rocm_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """detect_amd_gpus should return [] when rocm-smi is unavailable."""

    def _raise(*args, **kwargs):  # noqa: ANN001
        raise FileNotFoundError

    monkeypatch.setattr("trackiq_core.hardware.devices.subprocess.run", _raise)
    assert detect_amd_gpus() == []


def test_detect_apple_silicon_empty_on_non_apple(monkeypatch: pytest.MonkeyPatch) -> None:
    """detect_apple_silicon should return [] on non-darwin hosts."""
    monkeypatch.setattr(sys, "platform", "linux", raising=False)
    monkeypatch.setattr("trackiq_core.hardware.devices.platform.machine", lambda: "x86_64")
    assert detect_apple_silicon() == []


def test_detect_apple_silicon_profile_on_darwin_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    """detect_apple_silicon should return one profile on darwin arm64."""
    monkeypatch.setattr(sys, "platform", "darwin", raising=False)
    monkeypatch.setattr("trackiq_core.hardware.devices.platform.machine", lambda: "arm64")

    class _Result:
        stdout = "Apple M2 Pro\n"
        stderr = ""

    monkeypatch.setattr(
        "trackiq_core.hardware.devices.subprocess.run",
        lambda *args, **kwargs: _Result(),
    )
    devices = detect_apple_silicon()
    assert len(devices) == 1
    assert devices[0].device_type == DEVICE_TYPE_APPLE_SILICON


def test_get_amd_gpu_metrics_none_when_rocm_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_amd_gpu_metrics should return None when rocm-smi is unavailable."""
    monkeypatch.setattr(gpu_mod, "query_rocm_smi", lambda *args, **kwargs: None)
    assert gpu_mod.get_amd_gpu_metrics() is None


def test_get_intel_gpu_metrics_none_when_no_tools_or_sysfs(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_intel_gpu_metrics should return None without intel_gpu_top and sysfs."""
    monkeypatch.setattr("trackiq_core.hardware.gpu.shutil.which", lambda _: None)
    monkeypatch.setattr("trackiq_core.hardware.gpu.os.path.exists", lambda _: False)
    assert gpu_mod.get_intel_gpu_metrics() is None


def test_get_apple_silicon_metrics_none_when_not_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_apple_silicon_metrics should return None off macOS."""
    monkeypatch.setattr(sys, "platform", "linux", raising=False)
    assert gpu_mod.get_apple_silicon_metrics() is None


def test_get_cpu_metrics_always_returns_dict() -> None:
    """get_cpu_metrics should always return a dictionary with core keys."""
    data = gpu_mod.get_cpu_metrics()
    assert data is not None
    assert "cpu_utilization" in data
    assert "memory_percent" in data


def test_get_cpu_metrics_uses_windows_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """When psutil has no temps on Windows, fallback probe should populate temperature."""
    fake_psutil = type(
        "FakePsutil",
        (),
        {
            "cpu_percent": staticmethod(lambda interval=0.1: 33.0),
            "virtual_memory": staticmethod(lambda: type("VM", (), {"percent": 44.0})()),
            "sensors_temperatures": staticmethod(lambda: {}),
        },
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(gpu_mod.sys, "platform", "win32", raising=False)
    monkeypatch.setattr(gpu_mod, "_get_windows_cpu_temperature", lambda: 61.5)
    data = gpu_mod.get_cpu_metrics()
    assert data is not None
    assert data["temperature"] == 61.5


def test_get_cpu_metrics_uses_macos_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """When psutil has no temps on macOS, fallback probe should populate temperature."""
    fake_psutil = type(
        "FakePsutil",
        (),
        {
            "cpu_percent": staticmethod(lambda interval=0.1: 22.0),
            "virtual_memory": staticmethod(lambda: type("VM", (), {"percent": 55.0})()),
            "sensors_temperatures": staticmethod(lambda: {}),
        },
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(gpu_mod.sys, "platform", "darwin", raising=False)
    monkeypatch.setattr(gpu_mod, "_get_macos_cpu_temperature", lambda: 58.0)
    data = gpu_mod.get_cpu_metrics()
    assert data is not None
    assert data["temperature"] == 58.0


def test_get_all_devices_includes_amd_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_all_devices should include AMD devices when include_amd=True."""
    amd = DeviceProfile("amd_0", DEVICE_TYPE_AMD_GPU, "AMD MI300X")
    monkeypatch.setattr("trackiq_core.hardware.devices.detect_nvidia_gpus", lambda: [])
    monkeypatch.setattr("trackiq_core.hardware.devices.detect_amd_gpus", lambda: [amd])
    monkeypatch.setattr("trackiq_core.hardware.devices.detect_apple_silicon", lambda: [])
    monkeypatch.setattr("trackiq_core.hardware.devices.detect_tegrastats_platforms", lambda: [])
    monkeypatch.setattr("trackiq_core.hardware.devices.detect_intel_gpus", lambda: [])
    monkeypatch.setattr(
        "trackiq_core.hardware.devices.detect_cpu",
        lambda: DeviceProfile("cpu_0", DEVICE_TYPE_CPU, "CPU"),
    )
    devices = get_all_devices(include_amd=True)
    assert any(d.device_type == DEVICE_TYPE_AMD_GPU for d in devices)


def test_get_all_devices_skips_amd_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_all_devices should skip AMD devices when include_amd=False."""
    amd = DeviceProfile("amd_0", DEVICE_TYPE_AMD_GPU, "AMD MI300X")
    monkeypatch.setattr("trackiq_core.hardware.devices.detect_nvidia_gpus", lambda: [])
    monkeypatch.setattr("trackiq_core.hardware.devices.detect_amd_gpus", lambda: [amd])
    monkeypatch.setattr("trackiq_core.hardware.devices.detect_apple_silicon", lambda: [])
    monkeypatch.setattr("trackiq_core.hardware.devices.detect_tegrastats_platforms", lambda: [])
    monkeypatch.setattr("trackiq_core.hardware.devices.detect_intel_gpus", lambda: [])
    monkeypatch.setattr(
        "trackiq_core.hardware.devices.detect_cpu",
        lambda: DeviceProfile("cpu_0", DEVICE_TYPE_CPU, "CPU"),
    )
    devices = get_all_devices(include_amd=False)
    assert all(d.device_type != DEVICE_TYPE_AMD_GPU for d in devices)


def test_get_all_devices_always_includes_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU should always be present regardless of include flags."""
    monkeypatch.setattr("trackiq_core.hardware.devices.detect_nvidia_gpus", lambda: [])
    monkeypatch.setattr("trackiq_core.hardware.devices.detect_amd_gpus", lambda: [])
    monkeypatch.setattr("trackiq_core.hardware.devices.detect_apple_silicon", lambda: [])
    monkeypatch.setattr("trackiq_core.hardware.devices.detect_tegrastats_platforms", lambda: [])
    monkeypatch.setattr("trackiq_core.hardware.devices.detect_intel_gpus", lambda: [])
    monkeypatch.setattr(
        "trackiq_core.hardware.devices.detect_cpu",
        lambda: DeviceProfile("cpu_0", DEVICE_TYPE_CPU, "CPU"),
    )
    devices = get_all_devices(include_cpu=False)
    assert any(d.device_type == DEVICE_TYPE_CPU for d in devices)


def test_device_panel_to_dict_handles_all_device_types() -> None:
    """DevicePanel.to_dict should handle every device type constant."""
    devices = [
        DeviceProfile("nvidia_0", DEVICE_TYPE_NVIDIA_GPU, "NVIDIA A100"),
        DeviceProfile("amd_0", DEVICE_TYPE_AMD_GPU, "AMD MI300X"),
        DeviceProfile("intel_0", DEVICE_TYPE_INTEL_GPU, "Intel Arc"),
        DeviceProfile("apple_0", DEVICE_TYPE_APPLE_SILICON, "Apple M2"),
        DeviceProfile("cpu_0", DEVICE_TYPE_CPU, "CPU"),
        DeviceProfile("jetson_0", DEVICE_TYPE_NVIDIA_JETSON, "Jetson"),
        DeviceProfile("drive_0", DEVICE_TYPE_NVIDIA_DRIVE, "DRIVE"),
    ]
    panel = DevicePanel(devices=devices, show_live_metrics=False)
    payload = panel.to_dict()
    assert isinstance(payload["devices"], list)
    assert len(payload["devices"]) == 7


def test_metrics_dispatch_contains_all_device_types() -> None:
    """METRICS_DISPATCH should include handlers for all supported device constants."""
    expected = {
        DEVICE_TYPE_NVIDIA_GPU,
        DEVICE_TYPE_AMD_GPU,
        DEVICE_TYPE_INTEL_GPU,
        DEVICE_TYPE_APPLE_SILICON,
        DEVICE_TYPE_CPU,
        DEVICE_TYPE_NVIDIA_JETSON,
        DEVICE_TYPE_NVIDIA_DRIVE,
    }
    assert expected.issubset(set(METRICS_DISPATCH.keys()))
