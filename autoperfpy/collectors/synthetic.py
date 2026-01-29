"""Synthetic data collector for AutoPerfPy.

This module provides a synthetic data collector that generates realistic
time-series performance data for testing, development, and demonstration
purposes without requiring actual hardware.

The SyntheticCollector simulates:
- CPU/GPU utilization patterns with realistic noise
- Memory usage with gradual changes and occasional spikes
- Power consumption correlated with utilization
- Inference latency with jitter and warmup effects
- Temperature variations based on load

Example usage:
    from autoperfpy.collectors import SyntheticCollector

    # Create collector with custom config
    collector = SyntheticCollector(config={
        "warmup_samples": 10,
        "base_latency_ms": 25.0,
        "latency_jitter_percent": 15.0,
    })

    # Run collection
    collector.start()
    for i in range(100):
        metrics = collector.sample(time.time())
        print(f"Latency: {metrics['latency_ms']:.2f}ms")
        time.sleep(0.1)
    collector.stop()

    # Export results
    export = collector.export()
    print(f"Collected {len(export.samples)} samples")
"""

import math
import random
import time
from typing import Any, Dict, List, Optional

from .base import CollectorBase, CollectorExport, CollectorSample


class SyntheticCollector(CollectorBase):
    """Synthetic data collector generating realistic performance metrics.

    Generates time-series data that mimics real hardware behavior including:
    - Warmup period with gradually improving latency
    - Random jitter in latency measurements
    - Correlated CPU/GPU utilization patterns
    - Memory usage with gradual drift and occasional spikes
    - Power consumption based on utilization
    - Temperature variations with thermal inertia

    This collector is useful for:
    - Testing analysis pipelines without hardware
    - Demonstrating AutoPerfPy capabilities
    - CI/CD smoke tests
    - Developing new visualization features

    Attributes:
        config: Configuration dictionary controlling data generation
    """

    # Default configuration values
    DEFAULT_CONFIG = {
        # Warmup settings
        "warmup_samples": 10,           # Number of samples in warmup period
        "warmup_latency_factor": 2.0,   # Latency multiplier during warmup

        # Latency settings
        "base_latency_ms": 25.0,        # Base inference latency
        "latency_jitter_percent": 10.0, # Random jitter as percentage
        "latency_spike_prob": 0.02,     # Probability of latency spike
        "latency_spike_factor": 3.0,    # Spike multiplier

        # CPU settings
        "base_cpu_percent": 45.0,       # Base CPU utilization
        "cpu_noise_std": 8.0,           # Standard deviation of CPU noise
        "cpu_correlation": 0.7,         # Correlation with previous sample

        # GPU settings
        "base_gpu_percent": 75.0,       # Base GPU utilization
        "gpu_noise_std": 5.0,           # Standard deviation of GPU noise
        "gpu_correlation": 0.8,         # Correlation with previous sample

        # Memory settings
        "base_memory_mb": 4096.0,       # Base memory usage in MB
        "total_memory_mb": 16384.0,     # Total available memory
        "memory_drift_rate": 0.5,       # MB per sample drift
        "memory_spike_prob": 0.01,      # Probability of memory spike
        "memory_spike_mb": 512.0,       # Size of memory spikes

        # Power settings
        "idle_power_w": 15.0,           # Idle power consumption
        "max_power_w": 150.0,           # Maximum power consumption
        "power_noise_std": 3.0,         # Power measurement noise

        # Temperature settings
        "base_temperature_c": 45.0,     # Base temperature
        "max_temperature_c": 85.0,      # Maximum temperature
        "thermal_inertia": 0.9,         # Temperature change smoothing

        # Throughput settings
        "base_throughput_fps": 40.0,    # Base throughput in FPS
        "throughput_noise_std": 2.0,    # Throughput variation

        # Workload simulation
        "workload_pattern": "steady",   # steady, cyclic, ramp, burst
        "cycle_period_samples": 50,     # Period for cyclic pattern
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None, name: str = "SyntheticCollector"):
        """Initialize the synthetic collector.

        Args:
            config: Configuration dictionary. Keys not provided will use defaults.
                   See DEFAULT_CONFIG for available options.
            name: Name for this collector instance
        """
        super().__init__(name, config)

        # Merge provided config with defaults
        self._cfg = {**self.DEFAULT_CONFIG, **(config or {})}

        # Internal state for correlated noise generation
        self._prev_cpu = self._cfg["base_cpu_percent"]
        self._prev_gpu = self._cfg["base_gpu_percent"]
        self._prev_temp = self._cfg["base_temperature_c"]
        self._current_memory = self._cfg["base_memory_mb"]
        self._sample_index = 0

        # Random seed for reproducibility (optional)
        self._seed = config.get("seed") if config else None
        if self._seed is not None:
            random.seed(self._seed)

    def start(self) -> None:
        """Start the synthetic data collection.

        Initializes internal state and marks the collector as running.
        Resets sample counter for warmup period calculation.
        """
        self._is_running = True
        self._start_time = time.time()
        self._sample_index = 0
        self._samples.clear()

        # Reset state variables
        self._prev_cpu = self._cfg["base_cpu_percent"]
        self._prev_gpu = self._cfg["base_gpu_percent"]
        self._prev_temp = self._cfg["base_temperature_c"]
        self._current_memory = self._cfg["base_memory_mb"]

    def sample(self, timestamp: float) -> Optional[Dict[str, Any]]:
        """Generate a synthetic sample at the given timestamp.

        Produces realistic metrics including latency, CPU/GPU utilization,
        memory usage, power consumption, temperature, and throughput.

        Args:
            timestamp: Unix timestamp for this sample

        Returns:
            Dictionary containing:
            - latency_ms: Inference latency with jitter and warmup effects
            - cpu_percent: CPU utilization (0-100)
            - gpu_percent: GPU utilization (0-100)
            - memory_used_mb: Memory usage in MB
            - memory_total_mb: Total memory available
            - memory_percent: Memory usage percentage
            - power_w: Power consumption in watts
            - temperature_c: Temperature in Celsius
            - throughput_fps: Throughput in frames per second
            - is_warmup: Boolean indicating if in warmup period

        Raises:
            RuntimeError: If collector has not been started
        """
        if not self._is_running:
            raise RuntimeError("Collector not started. Call start() first.")

        # Calculate workload factor based on pattern
        workload_factor = self._get_workload_factor()

        # Generate metrics
        metrics = {
            "latency_ms": self._generate_latency(workload_factor),
            "cpu_percent": self._generate_cpu(workload_factor),
            "gpu_percent": self._generate_gpu(workload_factor),
            "memory_used_mb": self._generate_memory(),
            "memory_total_mb": self._cfg["total_memory_mb"],
            "memory_percent": (self._current_memory / self._cfg["total_memory_mb"]) * 100,
            "power_w": self._generate_power(),
            "temperature_c": self._generate_temperature(),
            "throughput_fps": self._generate_throughput(workload_factor),
            "is_warmup": self._sample_index < self._cfg["warmup_samples"],
        }

        # Store the sample
        self._store_sample(timestamp, metrics, {"sample_index": self._sample_index})
        self._sample_index += 1

        return metrics

    def stop(self) -> None:
        """Stop the synthetic data collection.

        Marks the collector as stopped and records the end time.
        """
        if self._is_running:
            self._is_running = False
            self._end_time = time.time()

    def export(self) -> CollectorExport:
        """Export all collected synthetic data.

        Returns:
            CollectorExport containing all samples and computed summary statistics
        """
        summary = self._calculate_summary()

        return CollectorExport(
            collector_name=self.name,
            start_time=self._start_time,
            end_time=self._end_time,
            samples=self._samples,
            summary=summary,
            config=self._cfg,
        )

    def _get_workload_factor(self) -> float:
        """Calculate workload intensity factor based on configured pattern.

        Returns:
            Factor between 0.5 and 1.5 representing workload intensity
        """
        pattern = self._cfg["workload_pattern"]

        if pattern == "steady":
            return 1.0

        elif pattern == "cyclic":
            # Sinusoidal pattern
            period = self._cfg["cycle_period_samples"]
            phase = (self._sample_index % period) / period * 2 * math.pi
            return 1.0 + 0.3 * math.sin(phase)

        elif pattern == "ramp":
            # Gradually increasing load
            max_samples = 200
            progress = min(self._sample_index / max_samples, 1.0)
            return 0.7 + 0.6 * progress

        elif pattern == "burst":
            # Random bursts of activity
            if random.random() < 0.1:  # 10% chance of burst
                return 1.5
            return 0.8

        return 1.0

    def _generate_latency(self, workload_factor: float) -> float:
        """Generate latency with jitter and warmup effects.

        Args:
            workload_factor: Current workload intensity

        Returns:
            Latency in milliseconds
        """
        base = self._cfg["base_latency_ms"] * workload_factor

        # Warmup effect - higher latency during initial samples
        if self._sample_index < self._cfg["warmup_samples"]:
            warmup_progress = self._sample_index / self._cfg["warmup_samples"]
            warmup_multiplier = self._cfg["warmup_latency_factor"] - (
                (self._cfg["warmup_latency_factor"] - 1.0) * warmup_progress
            )
            base *= warmup_multiplier

        # Add random jitter
        jitter_range = base * (self._cfg["latency_jitter_percent"] / 100.0)
        jitter = random.gauss(0, jitter_range / 2)

        # Occasional spikes
        if random.random() < self._cfg["latency_spike_prob"]:
            base *= self._cfg["latency_spike_factor"]

        return max(1.0, base + jitter)  # Minimum 1ms

    def _generate_cpu(self, workload_factor: float) -> float:
        """Generate CPU utilization with correlated noise.

        Args:
            workload_factor: Current workload intensity

        Returns:
            CPU utilization percentage (0-100)
        """
        target = self._cfg["base_cpu_percent"] * workload_factor
        noise = random.gauss(0, self._cfg["cpu_noise_std"])

        # Apply correlation with previous value for smoother transitions
        alpha = self._cfg["cpu_correlation"]
        self._prev_cpu = alpha * self._prev_cpu + (1 - alpha) * (target + noise)

        return max(0.0, min(100.0, self._prev_cpu))

    def _generate_gpu(self, workload_factor: float) -> float:
        """Generate GPU utilization with correlated noise.

        Args:
            workload_factor: Current workload intensity

        Returns:
            GPU utilization percentage (0-100)
        """
        target = self._cfg["base_gpu_percent"] * workload_factor
        noise = random.gauss(0, self._cfg["gpu_noise_std"])

        # Apply correlation for smooth transitions
        alpha = self._cfg["gpu_correlation"]
        self._prev_gpu = alpha * self._prev_gpu + (1 - alpha) * (target + noise)

        return max(0.0, min(100.0, self._prev_gpu))

    def _generate_memory(self) -> float:
        """Generate memory usage with drift and occasional spikes.

        Returns:
            Memory usage in megabytes
        """
        # Gradual drift
        drift = random.gauss(0, self._cfg["memory_drift_rate"])
        self._current_memory += drift

        # Occasional spikes
        if random.random() < self._cfg["memory_spike_prob"]:
            spike = random.choice([-1, 1]) * self._cfg["memory_spike_mb"]
            self._current_memory += spike

        # Clamp to valid range
        min_mem = self._cfg["base_memory_mb"] * 0.5
        max_mem = self._cfg["total_memory_mb"] * 0.95
        self._current_memory = max(min_mem, min(max_mem, self._current_memory))

        return self._current_memory

    def _generate_power(self) -> float:
        """Generate power consumption based on GPU utilization.

        Returns:
            Power consumption in watts
        """
        # Power scales with GPU utilization
        gpu_factor = self._prev_gpu / 100.0
        power_range = self._cfg["max_power_w"] - self._cfg["idle_power_w"]

        base_power = self._cfg["idle_power_w"] + power_range * gpu_factor
        noise = random.gauss(0, self._cfg["power_noise_std"])

        return max(self._cfg["idle_power_w"], base_power + noise)

    def _generate_temperature(self) -> float:
        """Generate temperature with thermal inertia.

        Returns:
            Temperature in Celsius
        """
        # Temperature correlates with power/utilization
        gpu_factor = self._prev_gpu / 100.0
        temp_range = self._cfg["max_temperature_c"] - self._cfg["base_temperature_c"]

        target_temp = self._cfg["base_temperature_c"] + temp_range * gpu_factor * 0.7
        noise = random.gauss(0, 1.0)

        # Apply thermal inertia (slow temperature changes)
        alpha = self._cfg["thermal_inertia"]
        self._prev_temp = alpha * self._prev_temp + (1 - alpha) * (target_temp + noise)

        return max(self._cfg["base_temperature_c"], min(self._cfg["max_temperature_c"], self._prev_temp))

    def _generate_throughput(self, workload_factor: float) -> float:
        """Generate throughput inversely related to latency.

        Args:
            workload_factor: Current workload intensity

        Returns:
            Throughput in frames per second
        """
        base = self._cfg["base_throughput_fps"] / workload_factor
        noise = random.gauss(0, self._cfg["throughput_noise_std"])

        # Lower throughput during warmup
        if self._sample_index < self._cfg["warmup_samples"]:
            base *= 0.7

        return max(1.0, base + noise)

    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics for all collected samples.

        Returns:
            Dictionary with summary statistics for each metric
        """
        if not self._samples:
            return {}

        # Extract metric arrays
        latencies = [s.metrics["latency_ms"] for s in self._samples]
        cpu_values = [s.metrics["cpu_percent"] for s in self._samples]
        gpu_values = [s.metrics["gpu_percent"] for s in self._samples]
        memory_values = [s.metrics["memory_used_mb"] for s in self._samples]
        power_values = [s.metrics["power_w"] for s in self._samples]
        temp_values = [s.metrics["temperature_c"] for s in self._samples]
        throughput_values = [s.metrics["throughput_fps"] for s in self._samples]

        # Exclude warmup samples for latency stats
        warmup_count = self._cfg["warmup_samples"]
        steady_latencies = latencies[warmup_count:] if len(latencies) > warmup_count else latencies

        return {
            "sample_count": len(self._samples),
            "warmup_samples": min(warmup_count, len(self._samples)),
            "duration_seconds": (self._end_time - self._start_time) if self._end_time else None,
            "latency": {
                "mean_ms": sum(steady_latencies) / len(steady_latencies) if steady_latencies else 0,
                "min_ms": min(steady_latencies) if steady_latencies else 0,
                "max_ms": max(steady_latencies) if steady_latencies else 0,
                "p50_ms": self._percentile(steady_latencies, 50),
                "p95_ms": self._percentile(steady_latencies, 95),
                "p99_ms": self._percentile(steady_latencies, 99),
            },
            "cpu": {
                "mean_percent": sum(cpu_values) / len(cpu_values),
                "max_percent": max(cpu_values),
            },
            "gpu": {
                "mean_percent": sum(gpu_values) / len(gpu_values),
                "max_percent": max(gpu_values),
            },
            "memory": {
                "mean_mb": sum(memory_values) / len(memory_values),
                "max_mb": max(memory_values),
                "min_mb": min(memory_values),
            },
            "power": {
                "mean_w": sum(power_values) / len(power_values),
                "max_w": max(power_values),
            },
            "temperature": {
                "mean_c": sum(temp_values) / len(temp_values),
                "max_c": max(temp_values),
            },
            "throughput": {
                "mean_fps": sum(throughput_values) / len(throughput_values),
                "min_fps": min(throughput_values),
            },
        }

    @staticmethod
    def _percentile(data: List[float], percentile: float) -> float:
        """Calculate percentile of a list of values.

        Args:
            data: List of numeric values
            percentile: Percentile to calculate (0-100)

        Returns:
            Percentile value
        """
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = (percentile / 100.0) * (len(sorted_data) - 1)
        lower = int(index)
        upper = min(lower + 1, len(sorted_data) - 1)
        weight = index - lower
        return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight


# TODO: Implement NVMLCollector using pynvml for real NVIDIA GPU metrics
# Example structure:
# class NVMLCollector(CollectorBase):
#     def __init__(self, device_index=0, config=None):
#         super().__init__("NVMLCollector", config)
#         self._device_index = device_index
#         self._handle = None
#
#     def start(self):
#         import pynvml
#         pynvml.nvmlInit()
#         self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
#         self._is_running = True
#         self._start_time = time.time()
#
#     def sample(self, timestamp):
#         import pynvml
#         memory = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
#         utilization = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
#         power = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000  # mW to W
#         temp = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)
#         metrics = {
#             "gpu_memory_used_mb": memory.used / (1024 * 1024),
#             "gpu_memory_total_mb": memory.total / (1024 * 1024),
#             "gpu_percent": utilization.gpu,
#             "memory_percent": utilization.memory,
#             "power_w": power,
#             "temperature_c": temp,
#         }
#         self._store_sample(timestamp, metrics)
#         return metrics

# TODO: Implement TegrastatsCollector for NVIDIA Jetson platforms
# Parse tegrastats output for comprehensive system metrics

# TODO: Implement PsutilCollector for cross-platform system monitoring
# Example structure:
# class PsutilCollector(CollectorBase):
#     def sample(self, timestamp):
#         import psutil
#         metrics = {
#             "cpu_percent": psutil.cpu_percent(),
#             "memory_used_mb": psutil.virtual_memory().used / (1024 * 1024),
#             "memory_percent": psutil.virtual_memory().percent,
#             "disk_percent": psutil.disk_usage('/').percent,
#         }
#         self._store_sample(timestamp, metrics)
#         return metrics


__all__ = ["SyntheticCollector"]
