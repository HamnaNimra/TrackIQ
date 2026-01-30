"""Inference configuration for TrackIQ.

Defines InferenceConfig dataclass and utilities for enumerating inference
configurations across devices.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from trackiq_core.hardware.devices import DeviceProfile

# TODO: Add more precisions as supported by hardware 
# FP4, FP8, FP16, BF16, TF32, FP32, FP64 either using pytorch or onnxruntime
# or tensorflow
# Supported precisions
PRECISION_FP32 = "fp32"
PRECISION_FP16 = "fp16"
PRECISION_INT8 = "int8"
PRECISIONS = [PRECISION_FP32, PRECISION_FP16, PRECISION_INT8]

# Default batch sizes and steps for auto enumeration
DEFAULT_BATCH_SIZES = [1, 4, 8]
DEFAULT_WARMUP_RUNS = 5
DEFAULT_ITERATIONS = 100
DEFAULT_STREAMS = 1


@dataclass
class InferenceConfig:
    """Inference run configuration.

    Attributes:
        precision: fp32, fp16, or int8
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


def enumerate_inference_configs(
    devices: List[DeviceProfile],
    precisions: Optional[List[str]] = None,
    batch_sizes: Optional[List[int]] = None,
    streams: int = 1,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    iterations: int = DEFAULT_ITERATIONS,
    max_configs_per_device: Optional[int] = 6,
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
    precisions = precisions or PRECISIONS
    batch_sizes = batch_sizes or DEFAULT_BATCH_SIZES
    result: List[Tuple[DeviceProfile, InferenceConfig]] = []
    for device in devices:
        count = 0
        for prec in precisions:
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
