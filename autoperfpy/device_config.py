"""Device and inference config for AutoPerfPy.

Re-exports generic inference config from trackiq_core and adds
automotive-specific device resolution.
"""

from trackiq_core.hardware.devices import DeviceProfile, get_all_devices

# Re-export from trackiq_core
from trackiq_core.inference import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_ITERATIONS,
    DEFAULT_STREAMS,
    DEFAULT_WARMUP_RUNS,
    PRECISION_BF16,
    PRECISION_FP16,
    PRECISION_FP32,
    PRECISION_INT4,
    PRECISION_INT8,
    PRECISION_MIXED,
    PRECISIONS,
    InferenceConfig,
    enumerate_inference_configs,
    get_supported_precisions_for_device,
    is_precision_supported,
    resolve_precision_for_device,
)


def resolve_device(
    device_id: str,
    devices: list[DeviceProfile] | None = None,
) -> DeviceProfile | None:
    """Resolve device ID (e.g. nvidia_0, cpu_0, 0) to a DeviceProfile.

    Args:
        device_id: ID or GPU index (e.g. "nvidia_0", "cpu_0", "0").
        devices: List of devices to search; if None, calls get_all_devices().

    Returns:
        Matching DeviceProfile or None if not found.
    """
    if devices is None:
        devices = get_all_devices()
    if not devices:
        return None
    device_id = (device_id or "").strip().lower()
    if device_id.isdigit():
        idx = int(device_id)
        for d in devices:
            if d.device_type == "nvidia_gpu" and d.index == idx:
                return d
        for d in devices:
            if d.index == idx:
                return d
    for d in devices:
        if d.device_id == device_id:
            return d
    return devices[0]


def get_devices_and_configs_auto(
    include_nvidia: bool = True,
    include_intel: bool = True,
    include_cpu: bool = True,
    include_tegrastats: bool = True,
    device_ids_filter: list[str] | None = None,
    precisions: list[str] | None = None,
    batch_sizes: list[int] | None = None,
    max_configs_per_device: int | None = 6,
) -> list[tuple[DeviceProfile, InferenceConfig]]:
    """Detect all devices and enumerate inference configs (auto mode).

    When device_ids_filter is set, only devices whose device_id is in the list
    are included (e.g. ["nvidia_0", "cpu_0"]).

    Args:
        include_nvidia: Include NVIDIA GPUs
        include_intel: Include Intel GPUs
        include_cpu: Include CPUs
        include_tegrastats: Include Jetson/DRIVE devices
        device_ids_filter: Optional list of device IDs to include
        precisions: List of precisions to test
        batch_sizes: List of batch sizes to test
        max_configs_per_device: Maximum configs per device

    Returns:
        List of (DeviceProfile, InferenceConfig) tuples
    """
    devices = get_all_devices(
        include_nvidia=include_nvidia,
        include_intel=include_intel,
        include_cpu=include_cpu,
        include_tegrastats=include_tegrastats,
    )
    if device_ids_filter:
        allowed = {s.strip().lower() for s in device_ids_filter if s and s.strip()}
        devices = [d for d in devices if d.device_id.lower() in allowed]
    return enumerate_inference_configs(
        devices,
        precisions=precisions,
        batch_sizes=batch_sizes,
        max_configs_per_device=max_configs_per_device,
    )


__all__ = [
    # Re-exports from trackiq_core
    "InferenceConfig",
    "enumerate_inference_configs",
    "PRECISION_FP32",
    "PRECISION_FP16",
    "PRECISION_BF16",
    "PRECISION_INT8",
    "PRECISION_INT4",
    "PRECISION_MIXED",
    "PRECISIONS",
    "DEFAULT_BATCH_SIZES",
    "DEFAULT_WARMUP_RUNS",
    "DEFAULT_ITERATIONS",
    "DEFAULT_STREAMS",
    "get_supported_precisions_for_device",
    "is_precision_supported",
    "resolve_precision_for_device",
    # AutoPerfPy-specific
    "resolve_device",
    "get_devices_and_configs_auto",
]
