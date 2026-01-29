"""Base collector interface for AutoPerfPy.

This module defines the abstract base class for all data collectors in AutoPerfPy.
Collectors are responsible for gathering time-series performance metrics from
various sources (synthetic, hardware monitors, profilers, etc.).

Example usage:
    class MyCollector(CollectorBase):
        def start(self):
            # Initialize hardware/connection
            pass

        def sample(self, timestamp):
            # Collect a single data point
            return {"cpu_percent": 50.0, "memory_mb": 1024}

        def stop(self):
            # Cleanup resources
            pass

        def export(self):
            # Return all collected data
            return self._samples
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class CollectorSample:
    """A single sample collected by a collector.

    Attributes:
        timestamp: Unix timestamp when the sample was collected
        metrics: Dictionary of metric names to values
        metadata: Optional metadata about the sample (e.g., source, tags)
    """

    timestamp: float
    metrics: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dictionary format.

        Returns:
            Dictionary representation of the sample
        """
        return {
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }


@dataclass
class CollectorExport:
    """Export format for collected data.

    Attributes:
        collector_name: Name/type of the collector
        start_time: When collection started
        end_time: When collection ended
        samples: List of collected samples
        summary: Aggregated summary statistics
        config: Configuration used during collection
    """

    collector_name: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    samples: List[CollectorSample] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert export to dictionary format.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "collector_name": self.collector_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "sample_count": len(self.samples),
            "samples": [s.to_dict() for s in self.samples],
            "summary": self.summary,
            "config": self.config,
        }


class CollectorBase(ABC):
    """Abstract base class for all performance data collectors.

    Collectors follow a lifecycle pattern:
    1. Initialize with configuration
    2. start() - Begin collection, initialize resources
    3. sample(timestamp) - Collect data points (called repeatedly)
    4. stop() - End collection, cleanup resources
    5. export() - Retrieve all collected data

    Subclasses must implement all abstract methods. The base class provides
    common functionality for sample storage and metadata tracking.

    Attributes:
        name: Human-readable name for the collector
        _samples: Internal storage for collected samples
        _is_running: Flag indicating if collector is active
        _start_time: Timestamp when collection started
        _config: Configuration dictionary
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize the collector.

        Args:
            name: Human-readable name for this collector instance
            config: Optional configuration dictionary
        """
        self.name = name
        self._samples: List[CollectorSample] = []
        self._is_running: bool = False
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._config = config or {}

    @abstractmethod
    def start(self) -> None:
        """Start the data collection process.

        This method should:
        - Initialize any hardware connections or resources
        - Set up monitoring threads if needed
        - Prepare internal state for sampling
        - Set _is_running to True

        Raises:
            RuntimeError: If collector cannot be started (e.g., hardware unavailable)
            ConnectionError: If unable to connect to data source

        Example:
            def start(self):
                self._init_hardware()
                self._is_running = True
                self._start_time = time.time()
        """
        pass

    @abstractmethod
    def sample(self, timestamp: float) -> Optional[Dict[str, Any]]:
        """Collect a single sample at the given timestamp.

        This method should be called repeatedly during the collection period
        to gather time-series data points. Each call should return the current
        metrics from the data source.

        Args:
            timestamp: Unix timestamp for this sample (seconds since epoch).
                       Use time.time() for current time.

        Returns:
            Dictionary of metric names to values for this sample, or None
            if sampling failed. Common metrics include:
            - cpu_percent: CPU utilization percentage (0-100)
            - gpu_percent: GPU utilization percentage (0-100)
            - memory_used_mb: Memory usage in megabytes
            - power_w: Power consumption in watts
            - latency_ms: Inference/operation latency in milliseconds
            - temperature_c: Temperature in Celsius

        Raises:
            RuntimeError: If collector is not running (start() not called)

        Example:
            def sample(self, timestamp):
                if not self._is_running:
                    raise RuntimeError("Collector not started")
                metrics = self._read_hardware_metrics()
                self._store_sample(timestamp, metrics)
                return metrics
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the data collection process.

        This method should:
        - Clean up any hardware connections or resources
        - Stop monitoring threads gracefully
        - Set _is_running to False
        - Record the end time

        Should be safe to call multiple times (idempotent).

        Example:
            def stop(self):
                if self._is_running:
                    self._cleanup_hardware()
                    self._is_running = False
                    self._end_time = time.time()
        """
        pass

    @abstractmethod
    def export(self) -> CollectorExport:
        """Export all collected data.

        Returns all samples collected between start() and stop() calls,
        along with summary statistics and metadata.

        Returns:
            CollectorExport containing:
            - All collected samples with timestamps
            - Summary statistics (min, max, mean, percentiles)
            - Collection metadata (duration, sample count, config)

        Example:
            def export(self):
                summary = self._calculate_summary()
                return CollectorExport(
                    collector_name=self.name,
                    start_time=self._start_time,
                    end_time=self._end_time,
                    samples=self._samples,
                    summary=summary,
                    config=self._config
                )
        """
        pass

    def _store_sample(self, timestamp: float, metrics: Dict[str, Any],
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store a sample in the internal buffer.

        Helper method for subclasses to store collected samples.

        Args:
            timestamp: Unix timestamp for the sample
            metrics: Dictionary of metric values
            metadata: Optional metadata for the sample
        """
        sample = CollectorSample(
            timestamp=timestamp,
            metrics=metrics,
            metadata=metadata or {}
        )
        self._samples.append(sample)

    def get_sample_count(self) -> int:
        """Get the number of samples collected.

        Returns:
            Number of samples in the buffer
        """
        return len(self._samples)

    def is_running(self) -> bool:
        """Check if the collector is currently running.

        Returns:
            True if collector is active, False otherwise
        """
        return self._is_running

    def clear(self) -> None:
        """Clear all collected samples.

        Useful for resetting the collector without full stop/start cycle.
        """
        self._samples.clear()


# TODO: Implement NVMLCollector for NVIDIA GPU metrics via pynvml
# - GPU utilization, memory usage, temperature, power draw
# - See: https://pypi.org/project/pynvml/

# TODO: Implement TegrastatsCollector for Jetson/DriveOS platforms
# - Parse tegrastats output for CPU, GPU, EMC, thermal metrics
# - Support both file-based and live collection modes

# TODO: Implement PsutilCollector for cross-platform system metrics
# - CPU, memory, disk, network via psutil library
# - See: https://psutil.readthedocs.io/

# TODO: Implement TensorRTCollector for inference profiling
# - Layer-level timing, memory allocation, throughput
# - Integrate with TensorRT Python API


__all__ = ["CollectorBase", "CollectorSample", "CollectorExport"]
