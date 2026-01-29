"""Platform and hardware detection for TrackIQ.

Generic detection for edge AI and embedded: NVIDIA GPUs, Intel GPUs, CPUs,
and tegrastats-capable platforms (NVIDIA Jetson, DRIVE). Applications map
device_type to collectors; trackiq does not depend on collectors.
"""

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
    detect_tegrastats_platforms,
    get_platform_metadata_for_device,
    DEVICE_TYPE_NVIDIA_GPU,
    DEVICE_TYPE_INTEL_GPU,
    DEVICE_TYPE_CPU,
    DEVICE_TYPE_NVIDIA_JETSON,
    DEVICE_TYPE_NVIDIA_DRIVE,
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
    "detect_tegrastats_platforms",
    "get_platform_metadata_for_device",
    "DEVICE_TYPE_NVIDIA_GPU",
    "DEVICE_TYPE_INTEL_GPU",
    "DEVICE_TYPE_CPU",
    "DEVICE_TYPE_NVIDIA_JETSON",
    "DEVICE_TYPE_NVIDIA_DRIVE",
]
