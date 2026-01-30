"""Device and inference config for AutoPerfPy Phase 5.

Inference config: precision, batch_size, accelerator, streams, warmup_runs, iterations.
Enumerates configs for all detected devices. No remote execution or networking.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from trackiq_core.hardware.devices import DeviceProfile, get_all_devices

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

    If max_configs_per_device is set, limits combinations per device (e.g. 6 = 3 precisions x 2 batch sizes).
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


def resolve_device(
    device_id: str,
    devices: Optional[List[DeviceProfile]] = None,
) -> Optional[DeviceProfile]:
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
    device_ids_filter: Optional[List[str]] = None,
    precisions: Optional[List[str]] = None,
    batch_sizes: Optional[List[int]] = None,
    max_configs_per_device: Optional[int] = 6,
) -> List[Tuple[DeviceProfile, InferenceConfig]]:
    """Detect all devices and enumerate inference configs (auto mode).

    When device_ids_filter is set, only devices whose device_id is in the list
    are included (e.g. ["nvidia_0", "cpu_0"]).
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
