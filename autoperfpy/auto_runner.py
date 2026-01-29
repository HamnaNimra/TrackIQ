"""Auto benchmark runner for AutoPerfPy Phase 5.

Runs benchmarks sequentially for each (device, inference_config). Uses a
collector factory (device_type -> collector instance) so adding new hardware
is a single registration. Same runner for automatic and manual modes. No remote
execution. Automotive-focused: wires TegrastatsCollector for Jetson/DRIVE;
trackiq provides generic device detection only.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from trackiq.platform.devices import (
    DeviceProfile,
    get_platform_metadata_for_device,
    DEVICE_TYPE_NVIDIA_GPU,
    DEVICE_TYPE_CPU,
    DEVICE_TYPE_INTEL_GPU,
    DEVICE_TYPE_NVIDIA_JETSON,
    DEVICE_TYPE_NVIDIA_DRIVE,
)
from trackiq.runner import BenchmarkRunner
from trackiq.collectors import SyntheticCollector

from .device_config import InferenceConfig


def _make_run_label(device: DeviceProfile, config: InferenceConfig) -> str:
    """Generate a short label for this run (e.g. nvidia_0_fp16_bs4)."""
    return f"{device.device_id}_{config.precision}_bs{config.batch_size}"


def _base_config(config: InferenceConfig) -> Dict[str, Any]:
    """Base collector config from inference config."""
    return {
        "warmup_samples": config.warmup_runs,
        "batch_sizes": [config.batch_size],
    }


def _collector_nvidia_gpu(
    device: DeviceProfile, config: InferenceConfig
) -> Optional[Any]:
    """Build NVML collector for discrete NVIDIA GPU."""
    try:
        from trackiq.collectors import NVMLCollector

        return NVMLCollector(device_index=device.index, config=_base_config(config))
    except (ImportError, Exception):
        return None


def _collector_psutil(device: DeviceProfile, config: InferenceConfig) -> Optional[Any]:
    """Build Psutil collector for CPU or Intel GPU."""
    try:
        from trackiq.collectors import PsutilCollector

        return PsutilCollector(config=_base_config(config))
    except (ImportError, Exception):
        return None


def _collector_tegrastats(
    device: DeviceProfile, config: InferenceConfig
) -> Optional[Any]:
    """Build Tegrastats collector for Jetson/DRIVE (automotive / edge AI)."""
    try:
        from autoperfpy.collectors import TegrastatsCollector

        cfg = _base_config(config)
        return TegrastatsCollector(mode="live", config=cfg)
    except (ImportError, Exception):
        return None


# Registry: device_type -> (device, config) -> collector or None. Add new hardware here.
COLLECTOR_FACTORY: Dict[
    str, Callable[[DeviceProfile, InferenceConfig], Optional[Any]]
] = {
    DEVICE_TYPE_NVIDIA_GPU: _collector_nvidia_gpu,
    DEVICE_TYPE_CPU: _collector_psutil,
    DEVICE_TYPE_INTEL_GPU: _collector_psutil,
    DEVICE_TYPE_NVIDIA_JETSON: _collector_tegrastats,
    DEVICE_TYPE_NVIDIA_DRIVE: _collector_tegrastats,
}


def _collector_for_device(device: DeviceProfile, config: InferenceConfig) -> Any:
    """Build collector for the given device via registry; fallback Synthetic."""
    builder = COLLECTOR_FACTORY.get(device.device_type)
    if builder:
        collector = builder(device, config)
        if collector is not None:
            return collector
    return SyntheticCollector(config=_base_config(config))


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
    progress_callback: Optional[Callable[..., None]] = None,
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
            results.append(
                {
                    "run_label": _make_run_label(device, config),
                    "platform_metadata": get_platform_metadata_for_device(device),
                    "inference_config": config.to_dict(),
                    "error": str(e),
                    "samples": [],
                    "summary": {},
                }
            )
    return results
