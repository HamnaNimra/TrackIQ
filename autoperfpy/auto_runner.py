"""Auto benchmark runner for AutoPerfPy Phase 5.

Runs benchmarks sequentially for each (device, inference_config). Uses the same
BenchmarkRunner for both automatic and manual modes. Saves each result with
platform_metadata, inference_config, and auto-generated run_label. No remote execution.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from trackiq.platform.devices import (
    DeviceProfile,
    get_platform_metadata_for_device,
    DEVICE_TYPE_NVIDIA_GPU,
    DEVICE_TYPE_CPU,
    DEVICE_TYPE_INTEL_GPU,
)
from trackiq.runner import BenchmarkRunner
from trackiq.collectors import SyntheticCollector

from .device_config import InferenceConfig


def _make_run_label(device: DeviceProfile, config: InferenceConfig) -> str:
    """Generate a short label for this run (e.g. nvidia_0_fp16_bs4)."""
    return f"{device.device_id}_{config.precision}_bs{config.batch_size}"


def _collector_for_device(
    device: DeviceProfile,
    config: InferenceConfig,
) -> Any:
    """Build collector for the given device. Uses NVML for NVIDIA, Psutil for CPU/Intel, else Synthetic."""
    cfg = {
        "warmup_samples": config.warmup_runs,
        "batch_sizes": [config.batch_size],
    }
    if device.device_type == DEVICE_TYPE_NVIDIA_GPU:
        try:
            from trackiq.collectors import NVMLCollector
            return NVMLCollector(device_index=device.index, config=cfg)
        except (ImportError, Exception):
            pass
    if device.device_type in (DEVICE_TYPE_CPU, DEVICE_TYPE_INTEL_GPU):
        try:
            from trackiq.collectors import PsutilCollector
            return PsutilCollector(config=cfg)
        except (ImportError, Exception):
            pass
    return SyntheticCollector(config={
        "warmup_samples": config.warmup_runs,
        "batch_sizes": [config.batch_size],
    })


def run_single_benchmark(
    device: DeviceProfile,
    config: InferenceConfig,
    duration_seconds: float = 10.0,
    sample_interval_seconds: float = 0.2,
    quiet: bool = True,
) -> Dict[str, Any]:
    """Run one benchmark for (device, config) and return result dict with metadata and run_label."""
    collector = _collector_for_device(device, config)
    runner = BenchmarkRunner(
        collector,
        duration_seconds=duration_seconds,
        sample_interval_seconds=sample_interval_seconds,
        quiet=quiet,
    )
    export = runner.run()
    result = export.to_dict()
    result["platform_metadata"] = get_platform_metadata_for_device(device)
    result["inference_config"] = config.to_dict()
    result["run_label"] = _make_run_label(device, config)
    return result


def run_auto_benchmarks(
    device_config_pairs: List[Tuple[DeviceProfile, InferenceConfig]],
    duration_seconds: float = 10.0,
    sample_interval_seconds: float = 0.2,
    quiet: bool = True,
    progress_callback: Optional[callable] = None,
) -> List[Dict[str, Any]]:
    """Run benchmarks sequentially for each (device, config). Same runner for auto/manual."""
    results: List[Dict[str, Any]] = []
    total = len(device_config_pairs)
    for i, (device, config) in enumerate(device_config_pairs):
        if progress_callback:
            progress_callback(i + 1, total, device, config)
        try:
            res = run_single_benchmark(
                device,
                config,
                duration_seconds=duration_seconds,
                sample_interval_seconds=sample_interval_seconds,
                quiet=quiet,
            )
            results.append(res)
        except Exception as e:
            results.append({
                "run_label": _make_run_label(device, config),
                "platform_metadata": get_platform_metadata_for_device(device),
                "inference_config": config.to_dict(),
                "error": str(e),
                "samples": [],
                "summary": {},
            })
    return results
