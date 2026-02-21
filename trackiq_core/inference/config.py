"""Inference configuration for TrackIQ.

Defines InferenceConfig dataclass and utilities for enumerating capability-aware
inference configurations across devices.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from trackiq_core.hardware.devices import (
    DEVICE_TYPE_AMD_GPU,
    DEVICE_TYPE_APPLE_SILICON,
    DEVICE_TYPE_CPU,
    DEVICE_TYPE_INTEL_GPU,
    DEVICE_TYPE_NVIDIA_DRIVE,
    DEVICE_TYPE_NVIDIA_GPU,
    DEVICE_TYPE_NVIDIA_JETSON,
    DeviceProfile,
)

# Supported precisions
PRECISION_FP32 = "fp32"
PRECISION_FP16 = "fp16"
PRECISION_BF16 = "bf16"
PRECISION_INT8 = "int8"
PRECISION_INT4 = "int4"
PRECISION_MIXED = "mixed"
PRECISIONS = [
    PRECISION_FP32,
    PRECISION_FP16,
    PRECISION_BF16,
    PRECISION_INT8,
    PRECISION_INT4,
    PRECISION_MIXED,
]

# Conservative default capability map by accelerator category.
DEFAULT_DEVICE_PRECISION_CAPABILITIES: Dict[str, List[str]] = {
    DEVICE_TYPE_NVIDIA_GPU: [
        PRECISION_FP32,
        PRECISION_FP16,
        PRECISION_BF16,
        PRECISION_INT8,
        PRECISION_INT4,
        PRECISION_MIXED,
    ],
    DEVICE_TYPE_NVIDIA_JETSON: [
        PRECISION_FP32,
        PRECISION_FP16,
        PRECISION_BF16,
        PRECISION_INT8,
        PRECISION_MIXED,
    ],
    DEVICE_TYPE_NVIDIA_DRIVE: [
        PRECISION_FP32,
        PRECISION_FP16,
        PRECISION_BF16,
        PRECISION_INT8,
        PRECISION_MIXED,
    ],
    DEVICE_TYPE_AMD_GPU: [
        PRECISION_FP32,
        PRECISION_FP16,
        PRECISION_BF16,
        PRECISION_INT8,
        PRECISION_MIXED,
    ],
    DEVICE_TYPE_INTEL_GPU: [
        PRECISION_FP32,
        PRECISION_FP16,
        PRECISION_BF16,
        PRECISION_INT8,
        PRECISION_MIXED,
    ],
    DEVICE_TYPE_APPLE_SILICON: [
        PRECISION_FP32,
        PRECISION_FP16,
        PRECISION_MIXED,
    ],
    DEVICE_TYPE_CPU: [
        PRECISION_FP32,
        PRECISION_BF16,
        PRECISION_INT8,
    ],
}

# Default batch sizes and steps for auto enumeration
DEFAULT_BATCH_SIZES = [1, 4, 8]
DEFAULT_WARMUP_RUNS = 5
DEFAULT_ITERATIONS = 100
DEFAULT_STREAMS = 1


@dataclass
class InferenceConfig:
    """Inference run configuration.

    Attributes:
        precision: fp32, fp16, bf16, int8, int4, or mixed
        batch_size: Batch size
        accelerator: Device id (e.g. nvidia_0, cpu_0)
        streams: Number of streams (default 1)
        warmup_runs: Warmup iterations before timed run
        iterations: Timed iterations
    """

    precision: str = PRECISION_FP32
    batch_size: int = 1
    accelerator: str = "cpu_0"
    streams: int = 1
    warmup_runs: int = DEFAULT_WARMUP_RUNS
    iterations: int = DEFAULT_ITERATIONS

    def to_dict(self) -> Dict[str, Any]:
        """Export for JSON and UI."""
        return {
            "precision": self.precision,
            "batch_size": self.batch_size,
            "accelerator": self.accelerator,
            "streams": self.streams,
            "warmup_runs": self.warmup_runs,
            "iterations": self.iterations,
        }


def normalize_precision_label(precision: str) -> str:
    """Normalize precision label to lowercase canonical value."""
    return str(precision or "").strip().lower()


def normalize_precision_list(precisions: Optional[List[str]]) -> List[str]:
    """Normalize, de-duplicate, and filter precision labels to supported values."""
    values = precisions or PRECISIONS
    normalized: List[str] = []
    for value in values:
        item = normalize_precision_label(value)
        if item in PRECISIONS and item not in normalized:
            normalized.append(item)
    return normalized


def get_supported_precisions_for_device(device: DeviceProfile) -> List[str]:
    """Return supported precisions for a device profile.

    Device metadata may override defaults using `metadata["supported_precisions"]`.
    """
    metadata = device.metadata if isinstance(device.metadata, dict) else {}
    override = metadata.get("supported_precisions")
    if isinstance(override, list):
        normalized_override = normalize_precision_list([str(item) for item in override])
        if normalized_override:
            return normalized_override
    return list(
        DEFAULT_DEVICE_PRECISION_CAPABILITIES.get(
            device.device_type, [PRECISION_FP32]
        )
    )


def is_precision_supported(device: DeviceProfile, precision: str) -> bool:
    """Return True when precision is supported for the device."""
    label = normalize_precision_label(precision)
    return label in get_supported_precisions_for_device(device)


def resolve_precision_for_device(
    device: DeviceProfile,
    requested_precision: str,
    fallback_precision: str = PRECISION_FP32,
) -> str:
    """Resolve precision to a supported value, falling back when needed."""
    requested = normalize_precision_label(requested_precision)
    fallback = normalize_precision_label(fallback_precision)
    supported = get_supported_precisions_for_device(device)
    if requested in supported:
        return requested
    if fallback in supported:
        return fallback
    return supported[0] if supported else PRECISION_FP32


def enumerate_inference_configs(
    devices: List[DeviceProfile],
    precisions: Optional[List[str]] = None,
    batch_sizes: Optional[List[int]] = None,
    streams: int = 1,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    iterations: int = DEFAULT_ITERATIONS,
    max_configs_per_device: Optional[int] = 6,
    fallback_precision: str = PRECISION_FP32,
) -> List[Tuple[DeviceProfile, InferenceConfig]]:
    """Enumerate (device, inference_config) for all devices and selected options.

    If max_configs_per_device is set, limits combinations per device
    (e.g. 6 = 3 precisions x 2 batch sizes).

    Args:
        devices: List of device profiles to enumerate
        precisions: List of precisions to test (default: all)
        batch_sizes: List of batch sizes to test (default: [1, 4, 8])
        streams: Number of streams (default: 1)
        warmup_runs: Warmup iterations (default: 5)
        iterations: Timed iterations (default: 100)
        max_configs_per_device: Maximum configs per device (default: 6)

    Returns:
        List of (DeviceProfile, InferenceConfig) tuples
    """
    normalized_precisions = normalize_precision_list(precisions)
    if not normalized_precisions:
        normalized_precisions = [normalize_precision_label(fallback_precision)]
    batch_sizes = batch_sizes or DEFAULT_BATCH_SIZES
    result: List[Tuple[DeviceProfile, InferenceConfig]] = []
    for device in devices:
        supported_precisions = get_supported_precisions_for_device(device)
        selected_precisions = [
            item for item in normalized_precisions if item in supported_precisions
        ]
        if not selected_precisions:
            selected_precisions = [
                resolve_precision_for_device(
                    device,
                    requested_precision=normalize_precision_label(fallback_precision),
                    fallback_precision=PRECISION_FP32,
                )
            ]
        count = 0
        for prec in selected_precisions:
            for bs in batch_sizes:
                if (
                    max_configs_per_device is not None
                    and count >= max_configs_per_device
                ):
                    break
                result.append(
                    (
                        device,
                        InferenceConfig(
                            precision=prec,
                            batch_size=bs,
                            accelerator=device.device_id,
                            streams=streams,
                            warmup_runs=warmup_runs,
                            iterations=iterations,
                        ),
                    )
                )
                count += 1
            if max_configs_per_device is not None and count >= max_configs_per_device:
                break
    return result
