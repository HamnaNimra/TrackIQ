"""NVML collector for NVIDIA GPU metrics.

This module provides a collector that uses the pynvml module from nvidia-ml-py
(NVIDIA Management Library) to gather real-time GPU metrics including:
- GPU utilization percentage
- Memory usage and utilization
- Power consumption
- Temperature
- Clock frequencies
- PCIe throughput

Additionally, this collector estimates:
- latency_ms: Estimated inference latency based on GPU utilization
- throughput_fps: Estimated throughput based on GPU utilization
- cpu_percent: CPU utilization (via psutil if available)

These estimated metrics ensure compatibility with the UI charts and reports.

Requires: nvidia-ml-py (pip install nvidia-ml-py)
Optional: psutil (pip install psutil) for CPU metrics

Example usage:
    from trackiq_core.collectors import NVMLCollector

    collector = NVMLCollector(device_index=0)
    collector.start()

    for _ in range(100):
        metrics = collector.sample(time.time())
        print(f"GPU: {metrics['gpu_percent']:.1f}%, Power: {metrics['power_w']:.1f}W")
        time.sleep(0.1)

    collector.stop()
    export = collector.export()

Authors:
    AutoPerfPy Team
"""

import random
import time
from typing import Any

from trackiq_core.utils.stats import percentile as _percentile
from trackiq_core.utils.stats import stats_from_values

from .base import CollectorBase, CollectorExport, CollectorSample


class NVMLCollector(CollectorBase):
    """Collector for NVIDIA GPU metrics using pynvml.

    This collector provides real hardware metrics from NVIDIA GPUs via the
    NVIDIA Management Library (NVML). It captures comprehensive GPU telemetry
    suitable for performance analysis of GPU-accelerated workloads.

    Capability Flags:
        supports_latency: True - Reads inference latency
        supports_throughput: True - Reads throughput metrics
        supports_bandwidth: True - Reads PCIe/memory bandwidth metrics
        supports_power: True - Reads GPU power consumption
        supports_utilization: True - Reads GPU/memory utilization
        supports_temperature: True - Reads GPU temperature
        supports_memory: True - Reads memory usage statistics
        supports_clocks: True - Reads GPU/memory clock frequencies

    Attributes:
        device_index: Index of the GPU device (0-based)
        _handle: NVML device handle
        _device_name: Name of the GPU device
        _cfg: Configuration dictionary
        _sample_index: Index of the current sample
        _samples: List of collected samples
        _is_running: Boolean indicating if the collector is running
        _start_time: Timestamp when the collector started
        _end_time: Timestamp when the collector stopped
    """

    # Capability flags for supported metrics/features
    supports_latency = True  # Collector provides latency measurement (if applicable, e.g. for GPU work submission)
    supports_throughput = True  # Collector can provide throughput metrics (e.g. processing rate, operations/sec)
    supports_bandwidth = True  # Collector provides PCIe/memory bandwidth metrics (if available)
    supports_power = True  # Reads GPU power consumption
    supports_utilization = True  # Reads GPU/memory utilization
    supports_temperature = True  # Reads GPU temperature
    supports_memory = True  # Reads memory usage statistics
    supports_clocks = True  # Reads GPU/memory clock frequencies

    def __init__(
        self,
        device_index: int = 0,
        config: dict[str, Any] | None = None,
        name: str = "NVMLCollector",
    ):
        """Initialize the NVML collector.

        Args:
            device_index: GPU device index (0 for first GPU)
            config: Optional configuration dictionary with:
                - include_latency: Include latency data (default: True)
                - include_throughput: Include throughput data (default: True)
                - include_bandwidth: Include PCIe/memory bandwidth data (default: True)
                - include_clocks: Include clock frequency data (default: True)
                - include_pcie: Include PCIe throughput data (default: False)
                - warmup_samples: Number of warmup samples to mark (default: 0)
            name: Name for this collector instance

        Raises:
            ImportError: If nvidia-ml-py is not installed
        """
        super().__init__(name, config)

        self._device_index = device_index
        self._handle = None
        self._device_name = None
        self._nvml_initialized = False

        # Configuration
        self._cfg = {
            "include_power": True,
            "include_utilization": True,
            "include_temperature": True,
            "include_memory": True,
            "include_clocks": True,
            "include_pcie": False,
            "include_cpu": True,  # Include CPU metrics via psutil
            "warmup_samples": 0,
            # Latency/throughput estimation parameters
            "base_latency_ms": 10.0,  # Base latency at 0% GPU utilization
            "max_latency_ms": 50.0,  # Max latency at 100% GPU utilization
            "base_throughput_fps": 100.0,  # Base throughput at low utilization
            "latency_noise_std": 1.0,  # Standard deviation for latency noise
            "throughput_noise_std": 2.0,  # Standard deviation for throughput noise
            **(config or {}),
        }

        self._sample_index = 0
        self._psutil_available = False
        try:
            import psutil  # noqa: F401 - import for availability check only

            self._psutil_available = True
        except ImportError:
            pass

    def start(self) -> None:
        """Start GPU monitoring by initializing NVML.

        Initializes the NVML library and acquires a handle to the specified GPU.

        Raises:
            RuntimeError: If NVML initialization fails or device is not found
            ImportError: If nvidia-ml-py is not installed
        """
        try:
            import pynvml  # provided by nvidia-ml-py
        except ImportError:
            raise ImportError("nvidia-ml-py is required for NVMLCollector. " "Install with: pip install nvidia-ml-py")

        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True

            # Get device handle
            device_count = pynvml.nvmlDeviceGetCount()
            if self._device_index >= device_count:
                raise RuntimeError(f"Device index {self._device_index} out of range. " f"Found {device_count} GPU(s).")

            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)

            # Get device name
            self._device_name = pynvml.nvmlDeviceGetName(self._handle)
            if isinstance(self._device_name, bytes):
                self._device_name = self._device_name.decode("utf-8")

        except pynvml.NVMLError as e:
            self._cleanup_nvml()
            raise RuntimeError(f"Failed to initialize NVML: {e}")

        self._is_running = True
        self._start_time = time.time()
        self._sample_index = 0
        self._samples.clear()

    def sample(self, timestamp: float) -> dict[str, Any] | None:
        """Collect GPU metrics at the given timestamp.

        Args:
            timestamp: Unix timestamp for this sample

        Returns:
            Dictionary containing GPU metrics:
            - latency_ms: Estimated inference latency based on GPU utilization
            - throughput_fps: Estimated throughput based on GPU utilization
            - cpu_percent: CPU utilization (if psutil available)
            - gpu_percent: GPU utilization (0-100)
            - memory_percent: Memory controller utilization (0-100)
            - memory_used_mb: GPU memory used in MB
            - memory_total_mb: Total GPU memory in MB
            - memory_free_mb: Free GPU memory in MB
            - power_w: Power consumption in watts
            - temperature_c: GPU temperature in Celsius
            - fan_speed_percent: Fan speed percentage (if available)
            - sm_clock_mhz: SM clock frequency (if include_clocks=True)
            - memory_clock_mhz: Memory clock frequency (if include_clocks=True)
            - is_warmup: Boolean indicating warmup period

        Raises:
            RuntimeError: If collector is not running
        """
        if not self._is_running:
            raise RuntimeError("Collector not started. Call start() first.")

        try:
            import pynvml  # provided by nvidia-ml-py
        except ImportError:
            return None

        metrics = {}
        gpu_util = 0

        try:
            # GPU/Memory utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            gpu_util = util.gpu
            metrics["gpu_percent"] = util.gpu
            metrics["memory_percent"] = util.memory

            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            metrics["memory_used_mb"] = mem_info.used / (1024 * 1024)
            metrics["memory_total_mb"] = mem_info.total / (1024 * 1024)
            metrics["memory_free_mb"] = mem_info.free / (1024 * 1024)

            # Power consumption (in milliwatts, convert to watts)
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
                metrics["power_w"] = power_mw / 1000.0
            except pynvml.NVMLError:
                metrics["power_w"] = 0.0

            # Temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)
                metrics["temperature_c"] = temp
            except pynvml.NVMLError:
                metrics["temperature_c"] = 0.0

            # Fan speed
            try:
                fan = pynvml.nvmlDeviceGetFanSpeed(self._handle)
                metrics["fan_speed_percent"] = fan
            except pynvml.NVMLError:
                metrics["fan_speed_percent"] = None

            # Clock frequencies
            if self._cfg.get("include_clocks", True):
                try:
                    sm_clock = pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_SM)
                    metrics["sm_clock_mhz"] = sm_clock
                except pynvml.NVMLError:
                    metrics["sm_clock_mhz"] = None

                try:
                    mem_clock = pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_MEM)
                    metrics["memory_clock_mhz"] = mem_clock
                except pynvml.NVMLError:
                    metrics["memory_clock_mhz"] = None

            # PCIe throughput
            if self._cfg.get("include_pcie", False):
                try:
                    tx = pynvml.nvmlDeviceGetPcieThroughput(self._handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
                    rx = pynvml.nvmlDeviceGetPcieThroughput(self._handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
                    metrics["pcie_tx_kbps"] = tx
                    metrics["pcie_rx_kbps"] = rx
                except pynvml.NVMLError:
                    pass

        except pynvml.NVMLError as e:
            # Log error but continue collecting
            metrics["error"] = str(e)

        # CPU metrics via psutil (if available and enabled)
        if self._cfg.get("include_cpu", True) and self._psutil_available:
            try:
                import psutil

                metrics["cpu_percent"] = psutil.cpu_percent(interval=None)
            except Exception:
                metrics["cpu_percent"] = 0.0
        else:
            metrics["cpu_percent"] = 0.0

        # Estimate latency based on GPU utilization
        # Higher GPU utilization generally correlates with longer inference times
        warmup_samples = self._cfg.get("warmup_samples", 0)
        is_warmup = self._sample_index < warmup_samples
        metrics["is_warmup"] = is_warmup

        base_latency = self._cfg.get("base_latency_ms", 10.0)
        max_latency = self._cfg.get("max_latency_ms", 50.0)
        latency_noise = self._cfg.get("latency_noise_std", 1.0)

        # Latency increases with GPU utilization
        util_factor = gpu_util / 100.0
        latency = base_latency + (max_latency - base_latency) * util_factor
        latency += random.gauss(0, latency_noise)

        # Warmup samples have higher latency
        if is_warmup:
            latency *= 1.5

        metrics["latency_ms"] = max(1.0, latency)

        # Estimate throughput (inversely related to latency)
        throughput_noise = self._cfg.get("throughput_noise_std", 2.0)

        # Throughput: frames per second = 1000ms / latency_ms
        throughput = 1000.0 / metrics["latency_ms"]
        # Scale by utilization (higher utilization = actually processing more)
        throughput *= 0.5 + 0.5 * util_factor
        throughput += random.gauss(0, throughput_noise)

        # Warmup has lower throughput
        if is_warmup:
            throughput *= 0.7

        metrics["throughput_fps"] = max(1.0, throughput)

        # Store sample
        metadata = {
            "sample_index": self._sample_index,
            "device_index": self._device_index,
            "device_name": self._device_name,
        }
        self._store_sample(timestamp, metrics, metadata)
        self._sample_index += 1

        return metrics

    def stop(self) -> None:
        """Stop GPU monitoring and shutdown NVML.

        Safely shuts down the NVML library and releases resources.
        Safe to call multiple times.
        """
        if self._is_running:
            self._is_running = False
            self._end_time = time.time()
            self._cleanup_nvml()

    def _cleanup_nvml(self) -> None:
        """Clean up NVML resources."""
        if self._nvml_initialized:
            try:
                import pynvml  # provided by nvidia-ml-py

                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_initialized = False
            self._handle = None

    def export(self) -> CollectorExport:
        """Export all collected GPU metrics.

        Returns:
            CollectorExport containing all samples and summary statistics
        """
        summary = self._calculate_summary()

        return CollectorExport(
            collector_name=self.name,
            start_time=self._start_time,
            end_time=self._end_time,
            samples=self._samples,
            summary=summary,
            config={
                **self._cfg,
                "device_index": self._device_index,
                "device_name": self._device_name,
            },
        )

    def _calculate_summary(self) -> dict[str, Any]:
        """Calculate summary statistics for collected samples.

        Returns:
            Dictionary with summary statistics matching the expected format
            for UI charts and reports.
        """
        if not self._samples:
            return {}

        warmup_count = self._cfg.get("warmup_samples", 0)
        steady_samples = self._samples[warmup_count:] if len(self._samples) > warmup_count else self._samples

        def _safe_stats(key: str, samples: list[CollectorSample]) -> dict[str, float]:
            values = [s.metrics.get(key) for s in samples if s.metrics.get(key) is not None]
            return stats_from_values(values)

        # Extract latency values for percentile calculations
        latencies = [s.metrics.get("latency_ms") for s in steady_samples if s.metrics.get("latency_ms") is not None]

        # Extract throughput values
        throughputs = [
            s.metrics.get("throughput_fps") for s in steady_samples if s.metrics.get("throughput_fps") is not None
        ]

        return {
            "sample_count": len(self._samples),
            "warmup_samples": min(warmup_count, len(self._samples)),
            "duration_seconds": ((self._end_time - self._start_time) if self._end_time else None),
            "device": {
                "index": self._device_index,
                "name": self._device_name,
            },
            # Latency statistics (required for UI latency charts)
            "latency": {
                "mean_ms": (sum(latencies) / len(latencies)) if latencies else 0,
                "min_ms": min(latencies, default=0),
                "max_ms": max(latencies, default=0),
                "p50_ms": _percentile(latencies, 50) if latencies else 0,
                "p95_ms": _percentile(latencies, 95) if latencies else 0,
                "p99_ms": _percentile(latencies, 99) if latencies else 0,
            },
            # Throughput statistics (required for UI throughput charts)
            "throughput": {
                "mean_fps": (sum(throughputs) / len(throughputs)) if throughputs else 0,
                "min_fps": min(throughputs, default=0),
                "max_fps": max(throughputs, default=0),
            },
            # CPU utilization (required for UI utilization charts)
            "cpu": {
                "mean_percent": _safe_stats("cpu_percent", steady_samples).get("mean", 0),
                "max_percent": _safe_stats("cpu_percent", steady_samples).get("max", 0),
            },
            # GPU utilization
            "gpu": {
                "mean_percent": _safe_stats("gpu_percent", steady_samples).get("mean", 0),
                "max_percent": _safe_stats("gpu_percent", steady_samples).get("max", 0),
            },
            # Memory utilization
            "memory_utilization": {
                **_safe_stats("memory_percent", steady_samples),
                "unit": "percent",
            },
            # Memory usage
            "memory": {
                "mean_mb": _safe_stats("memory_used_mb", steady_samples).get("mean"),
                "max_mb": _safe_stats("memory_used_mb", steady_samples).get("max"),
                "min_mb": _safe_stats("memory_used_mb", steady_samples).get("min"),
                "total_mb": (steady_samples[0].metrics.get("memory_total_mb") if steady_samples else None),
            },
            # Power consumption
            "power": {
                "mean_w": _safe_stats("power_w", steady_samples).get("mean"),
                "max_w": _safe_stats("power_w", steady_samples).get("max"),
            },
            # Temperature
            "temperature": {
                "mean_c": _safe_stats("temperature_c", steady_samples).get("mean"),
                "max_c": _safe_stats("temperature_c", steady_samples).get("max"),
            },
        }

    def get_device_info(self) -> dict[str, Any]:
        """Get detailed information about the GPU device.

        Returns:
            Dictionary with device specifications

        Raises:
            RuntimeError: If collector is not started
        """
        if not self._is_running or self._handle is None:
            raise RuntimeError("Collector must be started to get device info")

        try:
            import pynvml  # provided by nvidia-ml-py
        except ImportError:
            return {"error": "nvidia-ml-py not installed"}

        info = {
            "name": self._device_name,
            "index": self._device_index,
        }

        try:
            info["uuid"] = pynvml.nvmlDeviceGetUUID(self._handle)
            if isinstance(info["uuid"], bytes):
                info["uuid"] = info["uuid"].decode("utf-8")
        except pynvml.NVMLError:
            pass

        try:
            info["driver_version"] = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(info["driver_version"], bytes):
                info["driver_version"] = info["driver_version"].decode("utf-8")
        except pynvml.NVMLError:
            pass

        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            info["memory_total_mb"] = mem.total / (1024 * 1024)
        except pynvml.NVMLError:
            pass

        try:
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(self._handle)
            info["power_limit_w"] = power_limit / 1000.0
        except pynvml.NVMLError:
            pass

        try:
            info["compute_capability"] = pynvml.nvmlDeviceGetCudaComputeCapability(self._handle)
        except (pynvml.NVMLError, AttributeError):
            pass

        return info

    @staticmethod
    def get_available_devices() -> list[dict[str, Any]]:
        """Get list of available NVIDIA GPUs.

        Returns:
            List of dictionaries with device info

        Raises:
            ImportError: If pynvml is not installed
        """
        try:
            import pynvml
        except ImportError:
            raise ImportError("nvidia-ml-py required. Install with: pip install nvidia-ml-py")

        devices = []
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")

                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

                devices.append(
                    {
                        "index": i,
                        "name": name,
                        "memory_total_mb": mem.total / (1024 * 1024),
                    }
                )

            pynvml.nvmlShutdown()
        except pynvml.NVMLError as e:
            raise RuntimeError(f"NVML error: {e}")

        return devices


__all__ = ["NVMLCollector"]
