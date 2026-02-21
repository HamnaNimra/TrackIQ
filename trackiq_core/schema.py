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
    ttft_ms: Optional[float] = None
    tokens_per_sec: Optional[float] = None
    decode_tpt_ms: Optional[float] = None


@dataclass
class RegressionInfo:
    """Regression comparison metadata."""

    baseline_id: Optional[str]
    delta_percent: float
    status: RegressionStatus
    failed_metrics: List[str] = field(default_factory=list)


@dataclass
class KVCacheInfo:
    """Optional LLM KV cache telemetry captured during inference monitoring."""

    estimated_size_mb: float
    max_sequence_length: int
    batch_size: int
    num_layers: int
    num_heads: int
    head_size: int
    precision: str
    samples: List[Dict[str, Any]] = field(default_factory=list)


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
    kv_cache: Optional[KVCacheInfo] = None
    tool_payload: Optional[Dict[str, Any]] = None
    schema_version: str = "1.1.0"

    def __post_init__(self) -> None:
        """Backfill LLM metrics from tool payload when canonical fields are absent."""
        if not isinstance(self.tool_payload, dict):
            return
        payload = self.tool_payload

        if self.metrics.ttft_ms is None:
            self.metrics.ttft_ms = _coerce_optional_float(
                payload.get("ttft_ms", payload.get("ttft_p50"))
            )
        if self.metrics.tokens_per_sec is None:
            self.metrics.tokens_per_sec = _coerce_optional_float(
                payload.get("tokens_per_sec", payload.get("throughput_tokens_per_sec"))
            )
        if self.metrics.decode_tpt_ms is None:
            self.metrics.decode_tpt_ms = _coerce_optional_float(
                payload.get("decode_tpt_ms", payload.get("tpt_p50"))
            )

    def to_dict(self) -> Dict[str, object]:
        """Convert to JSON-serializable dictionary."""
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "TrackiqResult":
        """Build a TrackiqResult from dictionary data."""
        metrics_payload = (
            dict(payload.get("metrics", {}))
            if isinstance(payload.get("metrics"), dict)
            else {}
        )
        # Backward compatibility: older payloads may omit newer/nullable metric keys.
        metrics_payload.setdefault("communication_overhead_percent", None)
        metrics_payload.setdefault("power_consumption_watts", None)
        metrics_payload.setdefault("energy_per_step_joules", None)
        metrics_payload.setdefault("performance_per_watt", None)
        metrics_payload.setdefault("temperature_celsius", None)
        metrics_payload.setdefault("ttft_ms", None)
        metrics_payload.setdefault("tokens_per_sec", None)
        metrics_payload.setdefault("decode_tpt_ms", None)

        kv_cache_payload = payload.get("kv_cache")
        if kv_cache_payload is None and isinstance(payload.get("tool_payload"), dict):
            kv_cache_payload = payload["tool_payload"].get("kv_cache")  # type: ignore[index]
        return cls(
            tool_name=str(payload["tool_name"]),
            tool_version=str(payload["tool_version"]),
            timestamp=datetime.fromisoformat(str(payload["timestamp"])),
            platform=PlatformInfo(**payload["platform"]),  # type: ignore[arg-type]
            workload=WorkloadInfo(**payload["workload"]),  # type: ignore[arg-type]
            metrics=Metrics(**metrics_payload),
            regression=RegressionInfo(**payload["regression"]),  # type: ignore[arg-type]
            kv_cache=KVCacheInfo(**kv_cache_payload)  # type: ignore[arg-type]
            if isinstance(kv_cache_payload, dict)
            else None,
            tool_payload=payload.get("tool_payload"),
            schema_version=str(payload.get("schema_version", "1.0.0")),
        )


def _coerce_optional_float(value: object) -> Optional[float]:
    """Return float(value) when possible, otherwise None."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
