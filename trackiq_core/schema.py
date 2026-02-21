"""Canonical TrackIQ result schema.

This module defines the standard result shape shared by AutoPerfPy and
MiniCluster, and consumed by comparison tooling.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional


WorkloadType = Literal["inference", "training"]
RegressionStatus = Literal["pass", "fail"]


@dataclass
class PlatformInfo:
    """Platform metadata for a benchmark/validation run."""

    hardware_name: str
    os: str
    framework: str
    framework_version: str


@dataclass
class WorkloadInfo:
    """Workload metadata for a benchmark/validation run."""

    name: str
    workload_type: WorkloadType
    batch_size: int
    steps: int


@dataclass
class Metrics:
    """Core metrics captured by TrackIQ tools, including optional power/thermal fields."""

    throughput_samples_per_sec: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    memory_utilization_percent: float
    communication_overhead_percent: Optional[float]
    power_consumption_watts: Optional[float]
    energy_per_step_joules: Optional[float] = None
    performance_per_watt: Optional[float] = None
    temperature_celsius: Optional[float] = None


@dataclass
class RegressionInfo:
    """Regression comparison metadata."""

    baseline_id: Optional[str]
    delta_percent: float
    status: RegressionStatus
    failed_metrics: List[str] = field(default_factory=list)


@dataclass
class TrackiqResult:
    """
    Canonical result object for TrackIQ tools.

    Fields above tool_payload are canonical and compared across tools.
    tool_payload is an optional extension field for tool-specific
    metadata that does not belong in the canonical schema — for example,
    per-step power readings, per-worker throughput breakdowns, or
    layer-by-layer inference timing. Comparison tooling ignores this
    field. It is preserved in serialization for debugging and analysis.
    """

    tool_name: str
    tool_version: str
    timestamp: datetime
    platform: PlatformInfo
    workload: WorkloadInfo
    metrics: Metrics
    regression: RegressionInfo
    tool_payload: Optional[Dict[str, Any]] = None
    schema_version: str = "1.1.0"

    def to_dict(self) -> Dict[str, object]:
        """Convert to JSON-serializable dictionary."""
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "TrackiqResult":
        """Build a TrackiqResult from dictionary data."""
        return cls(
            tool_name=str(payload["tool_name"]),
            tool_version=str(payload["tool_version"]),
            timestamp=datetime.fromisoformat(str(payload["timestamp"])),
            platform=PlatformInfo(**payload["platform"]),  # type: ignore[arg-type]
            workload=WorkloadInfo(**payload["workload"]),  # type: ignore[arg-type]
            metrics=Metrics(**payload["metrics"]),  # type: ignore[arg-type]
            regression=RegressionInfo(**payload["regression"]),  # type: ignore[arg-type]
            tool_payload=payload.get("tool_payload"),
            schema_version=str(payload.get("schema_version", "1.0.0")),
        )
