"""Base collector interface for TrackIQ.

This module defines the abstract base class for all data collectors in TrackIQ.
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
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CollectorSample:
    """A single sample collected by a collector.

    Attributes:
        timestamp: Unix timestamp when the sample was collected
        metrics: Dictionary of metric names to values
        metadata: Optional metadata about the sample (e.g., source, tags)
    """

    timestamp: float
    metrics: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
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
    start_time: float | None = None
    end_time: float | None = None
    samples: list[CollectorSample] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
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

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        """Initialize the collector.

        Args:
            name: Human-readable name for this collector instance
            config: Optional configuration dictionary
        """
        self.name = name
        self._samples: list[CollectorSample] = []
        self._is_running: bool = False
        self._start_time: float | None = None
        self._end_time: float | None = None
        self._config = config or {}

    @abstractmethod
    def start(self) -> None:
        """Start the data collection process."""
        pass

    @abstractmethod
    def sample(self, timestamp: float) -> dict[str, Any] | None:
        """Collect a single sample at the given timestamp."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the data collection process."""
        pass

    @abstractmethod
    def export(self) -> CollectorExport:
        """Export all collected data."""
        pass

    def _store_sample(self, timestamp: float, metrics: dict[str, Any], metadata: dict[str, Any] | None = None) -> None:
        """Store a sample in the internal buffer."""
        sample = CollectorSample(timestamp=timestamp, metrics=metrics, metadata=metadata or {})
        self._samples.append(sample)

    def get_sample_count(self) -> int:
        """Get the number of samples collected."""
        return len(self._samples)

    def is_running(self) -> bool:
        """Check if the collector is currently running."""
        return self._is_running

    def clear(self) -> None:
        """Clear all collected samples."""
        self._samples.clear()


__all__ = ["CollectorBase", "CollectorSample", "CollectorExport"]
