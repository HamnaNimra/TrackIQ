"""Platform and hardware detection for TrackIQ."""

from .gpu import (
    query_nvidia_smi,
    parse_gpu_metrics,
    get_memory_metrics,
    get_performance_metrics,
    DEFAULT_NVIDIA_SMI_TIMEOUT,
)
from .devices import (
    DeviceProfile,
    get_all_devices,
    detect_nvidia_gpus,
    detect_cpu,
    detect_intel_gpus,
    get_platform_metadata_for_device,
    DEVICE_TYPE_NVIDIA_GPU,
    DEVICE_TYPE_INTEL_GPU,
    DEVICE_TYPE_CPU,
)

__all__ = [
    "query_nvidia_smi",
    "parse_gpu_metrics",
    "get_memory_metrics",
    "get_performance_metrics",
    "DEFAULT_NVIDIA_SMI_TIMEOUT",
    "DeviceProfile",
    "get_all_devices",
    "detect_nvidia_gpus",
    "detect_cpu",
    "detect_intel_gpus",
    "get_platform_metadata_for_device",
    "DEVICE_TYPE_NVIDIA_GPU",
    "DEVICE_TYPE_INTEL_GPU",
    "DEVICE_TYPE_CPU",
]
