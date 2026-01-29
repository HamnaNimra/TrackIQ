"""NVML collector for NVIDIA GPU metrics.

This module provides a collector that uses pynvml (NVIDIA Management Library)
to gather real-time GPU metrics including:
- GPU utilization percentage
- Memory usage and utilization
- Power consumption
- Temperature
- Clock frequencies
- PCIe throughput

Requires: pynvml (pip install pynvml)

Example usage:
    from trackiq.collectors import NVMLCollector

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

import time
from typing import Any, Dict, List, Optional

from .base import CollectorBase, CollectorExport, CollectorSample


class NVMLCollector(CollectorBase):
    """Collector for NVIDIA GPU metrics using pynvml.

    This collector provides real hardware metrics from NVIDIA GPUs via the
    NVIDIA Management Library (NVML). It captures comprehensive GPU telemetry
    suitable for performance analysis of GPU-accelerated workloads.

    Capability Flags:
        supports_power: True - Reads GPU power consumption
        supports_utilization: True - Reads GPU/memory utilization
        supports_temperature: True - Reads GPU temperature
        supports_memory: True - Reads memory usage statistics
        supports_clocks: True - Reads GPU/memory clock frequencies

    Attributes:
        device_index: Index of the GPU device (0-based)
        _handle: NVML device handle
        _device_name: Name of the GPU device
    """

    # Capability flags
    supports_power = True
    supports_utilization = True
    supports_temperature = True
    supports_memory = True
    supports_clocks = True

    def __init__(
        self,
        device_index: int = 0,
        config: Optional[Dict[str, Any]] = None,
        name: str = "NVMLCollector",
    ):
        """Initialize the NVML collector.

        Args:
            device_index: GPU device index (0 for first GPU)
            config: Optional configuration dictionary with:
                - include_clocks: Include clock frequency data (default: True)
                - include_pcie: Include PCIe throughput data (default: False)
                - warmup_samples: Number of warmup samples to mark (default: 0)
            name: Name for this collector instance

        Raises:
            ImportError: If pynvml is not installed
        """
        super().__init__(name, config)

        self._device_index = device_index
        self._handle = None
        self._device_name = None
        self._nvml_initialized = False

        # Configuration
        self._cfg = {
            "include_clocks": True,
            "include_pcie": False,
            "warmup_samples": 0,
            **(config or {}),
        }

        self._sample_index = 0

    def start(self) -> None:
        """Start GPU monitoring by initializing NVML.

        Initializes the NVML library and acquires a handle to the specified GPU.

        Raises:
            RuntimeError: If NVML initialization fails or device is not found
            ImportError: If pynvml is not installed
        """
        try:
            import pynvml
        except ImportError:
            raise ImportError(
                "pynvml is required for NVMLCollector. "
                "Install with: pip install pynvml"
            )

        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True

            # Get device handle
            device_count = pynvml.nvmlDeviceGetCount()
            if self._device_index >= device_count:
                raise RuntimeError(
                    f"Device index {self._device_index} out of range. "
                    f"Found {device_count} GPU(s)."
                )

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

    def sample(self, timestamp: float) -> Optional[Dict[str, Any]]:
        """Collect GPU metrics at the given timestamp.

        Args:
            timestamp: Unix timestamp for this sample

        Returns:
            Dictionary containing GPU metrics:
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
            import pynvml
        except ImportError:
            return None

        metrics = {}

        try:
            # GPU/Memory utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
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
                metrics["power_w"] = None

            # Temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    self._handle, pynvml.NVML_TEMPERATURE_GPU
                )
                metrics["temperature_c"] = temp
            except pynvml.NVMLError:
                metrics["temperature_c"] = None

            # Fan speed
            try:
                fan = pynvml.nvmlDeviceGetFanSpeed(self._handle)
                metrics["fan_speed_percent"] = fan
            except pynvml.NVMLError:
                metrics["fan_speed_percent"] = None

            # Clock frequencies
            if self._cfg.get("include_clocks", True):
                try:
                    sm_clock = pynvml.nvmlDeviceGetClockInfo(
                        self._handle, pynvml.NVML_CLOCK_SM
                    )
                    metrics["sm_clock_mhz"] = sm_clock
                except pynvml.NVMLError:
                    metrics["sm_clock_mhz"] = None

                try:
                    mem_clock = pynvml.nvmlDeviceGetClockInfo(
                        self._handle, pynvml.NVML_CLOCK_MEM
                    )
                    metrics["memory_clock_mhz"] = mem_clock
                except pynvml.NVMLError:
                    metrics["memory_clock_mhz"] = None

            # PCIe throughput
            if self._cfg.get("include_pcie", False):
                try:
                    tx = pynvml.nvmlDeviceGetPcieThroughput(
                        self._handle, pynvml.NVML_PCIE_UTIL_TX_BYTES
                    )
                    rx = pynvml.nvmlDeviceGetPcieThroughput(
                        self._handle, pynvml.NVML_PCIE_UTIL_RX_BYTES
                    )
                    metrics["pcie_tx_kbps"] = tx
                    metrics["pcie_rx_kbps"] = rx
                except pynvml.NVMLError:
                    pass

            # Warmup marker
            warmup_samples = self._cfg.get("warmup_samples", 0)
            metrics["is_warmup"] = self._sample_index < warmup_samples

        except pynvml.NVMLError as e:
            # Log error but continue collecting
            metrics["error"] = str(e)

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
                import pynvml
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

    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics for collected samples.

        Returns:
            Dictionary with summary statistics
        """
        if not self._samples:
            return {}

        warmup_count = self._cfg.get("warmup_samples", 0)
        steady_samples = self._samples[warmup_count:] if len(self._samples) > warmup_count else self._samples

        def safe_stats(key: str, samples: List[CollectorSample]) -> Dict[str, float]:
            """Calculate stats for a metric, handling None values."""
            values = [s.metrics.get(key) for s in samples if s.metrics.get(key) is not None]
            if not values:
                return {}
            return {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }

        return {
            "sample_count": len(self._samples),
            "warmup_samples": min(warmup_count, len(self._samples)),
            "duration_seconds": (self._end_time - self._start_time) if self._end_time else None,
            "device": {
                "index": self._device_index,
                "name": self._device_name,
            },
            "gpu": {
                **safe_stats("gpu_percent", steady_samples),
                "unit": "percent",
            },
            "memory_utilization": {
                **safe_stats("memory_percent", steady_samples),
                "unit": "percent",
            },
            "memory": {
                "mean_mb": safe_stats("memory_used_mb", steady_samples).get("mean"),
                "max_mb": safe_stats("memory_used_mb", steady_samples).get("max"),
                "min_mb": safe_stats("memory_used_mb", steady_samples).get("min"),
                "total_mb": steady_samples[0].metrics.get("memory_total_mb") if steady_samples else None,
            },
            "power": {
                "mean_w": safe_stats("power_w", steady_samples).get("mean"),
                "max_w": safe_stats("power_w", steady_samples).get("max"),
            },
            "temperature": {
                "mean_c": safe_stats("temperature_c", steady_samples).get("mean"),
                "max_c": safe_stats("temperature_c", steady_samples).get("max"),
            },
        }

    def get_device_info(self) -> Dict[str, Any]:
        """Get detailed information about the GPU device.

        Returns:
            Dictionary with device specifications

        Raises:
            RuntimeError: If collector is not started
        """
        if not self._is_running or self._handle is None:
            raise RuntimeError("Collector must be started to get device info")

        try:
            import pynvml
        except ImportError:
            return {"error": "pynvml not installed"}

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
    def get_available_devices() -> List[Dict[str, Any]]:
        """Get list of available NVIDIA GPUs.

        Returns:
            List of dictionaries with device info

        Raises:
            ImportError: If pynvml is not installed
        """
        try:
            import pynvml
        except ImportError:
            raise ImportError("pynvml required. Install with: pip install pynvml")

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

                devices.append({
                    "index": i,
                    "name": name,
                    "memory_total_mb": mem.total / (1024 * 1024),
                })

            pynvml.nvmlShutdown()
        except pynvml.NVMLError as e:
            raise RuntimeError(f"NVML error: {e}")

        return devices


__all__ = ["NVMLCollector"]
