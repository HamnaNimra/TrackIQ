"""Profile definitions for TrackIQ.

This module defines the profile dataclass and registry for performance
testing profiles. Applications (e.g. AutoPerfPy) register their own
predefined profiles via register_profile().
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from trackiq_core.inference import PRECISIONS
from trackiq_core.utils.errors import ProfileValidationError


class CollectorType(Enum):
    """Types of collectors that can be used with profiles."""

    SYNTHETIC = "synthetic"
    NVML = "nvml"
    TEGRASTATS = "tegrastats"
    PSUTIL = "psutil"
    TENSORRT = "tensorrt"


@dataclass
class Profile:
    """Performance testing profile configuration."""

    name: str
    description: str

    latency_threshold_ms: float = 100.0
    latency_target_ms: float = 50.0
    latency_percentiles: list[int] = field(default_factory=lambda: [50, 95, 99])

    throughput_min_fps: float = 10.0
    throughput_target_fps: float = 30.0

    power_budget_w: float | None = None
    thermal_limit_c: float = 85.0

    memory_limit_mb: float | None = None
    memory_headroom_percent: float = 20.0

    batch_sizes: list[int] = field(default_factory=lambda: [1, 4, 8, 16])
    warmup_iterations: int = 10
    test_iterations: int = 100
    num_runs: int = 3

    sample_interval_ms: int = 100
    duration_seconds: int = 60
    enable_continuous_monitoring: bool = True

    supported_collectors: list[CollectorType] = field(default_factory=lambda: [CollectorType.SYNTHETIC])
    collector_config: dict[str, Any] = field(default_factory=dict)
    analysis_config: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    supported_precisions: list[str] = field(default_factory=lambda: list(PRECISIONS))

    def to_dict(self) -> dict[str, Any]:
        """Convert profile to dictionary format."""
        return {
            "name": self.name,
            "description": self.description,
            "latency": {
                "threshold_ms": self.latency_threshold_ms,
                "target_ms": self.latency_target_ms,
                "percentiles": self.latency_percentiles,
            },
            "throughput": {
                "min_fps": self.throughput_min_fps,
                "target_fps": self.throughput_target_fps,
            },
            "constraints": {
                "power_budget_w": self.power_budget_w,
                "thermal_limit_c": self.thermal_limit_c,
                "memory_limit_mb": self.memory_limit_mb,
                "memory_headroom_percent": self.memory_headroom_percent,
            },
            "benchmark": {
                "batch_sizes": self.batch_sizes,
                "warmup_iterations": self.warmup_iterations,
                "test_iterations": self.test_iterations,
                "num_runs": self.num_runs,
            },
            "monitoring": {
                "sample_interval_ms": self.sample_interval_ms,
                "duration_seconds": self.duration_seconds,
                "enable_continuous_monitoring": self.enable_continuous_monitoring,
            },
            "supported_collectors": [c.value for c in self.supported_collectors],
            "collector_config": self.collector_config,
            "analysis_config": self.analysis_config,
            "tags": self.tags,
            "supported_precisions": list(self.supported_precisions),
        }

    def validate_collector(self, collector_type: CollectorType) -> bool:
        """Check if a collector type is supported by this profile."""
        return collector_type in self.supported_collectors

    def validate_precision(self, precision: str) -> bool:
        """Check if requested precision is supported by this profile."""
        normalized = str(precision or "").strip().lower()
        return normalized in {str(item).strip().lower() for item in self.supported_precisions}

    def get_synthetic_config(self) -> dict[str, Any]:
        """Get configuration for SyntheticCollector based on profile."""
        return {
            "warmup_samples": self.warmup_iterations,
            "base_latency_ms": self.latency_target_ms,
            "latency_jitter_percent": 10.0,
            "base_gpu_percent": 75.0 if self.power_budget_w is None else 50.0,
            **self.collector_config.get("synthetic", {}),
        }


# Profile registry (applications register their profiles via register_profile)
_PROFILE_REGISTRY: dict[str, Profile] = {}


def get_profile(name: str) -> Profile:
    """Get a profile by name."""
    if name not in _PROFILE_REGISTRY:
        available = ", ".join(_PROFILE_REGISTRY.keys())
        raise ValueError(f"Unknown profile '{name}'. Available profiles: {available}")
    return _PROFILE_REGISTRY[name]


def list_profiles() -> list[str]:
    """Get list of all available profile names."""
    return list(_PROFILE_REGISTRY.keys())


def register_profile(profile: Profile) -> None:
    """Register a custom profile."""
    if profile.name in _PROFILE_REGISTRY:
        raise ValueError(f"Profile '{profile.name}' already exists")
    _PROFILE_REGISTRY[profile.name] = profile


def get_profile_info() -> dict[str, dict[str, Any]]:
    """Get summary information about all profiles."""
    return {
        name: {
            "description": p.description,
            "latency_threshold_ms": p.latency_threshold_ms,
            "throughput_target_fps": p.throughput_target_fps,
            "power_budget_w": p.power_budget_w,
            "tags": p.tags,
        }
        for name, p in _PROFILE_REGISTRY.items()
    }


def validate_profile_collector(profile: Profile, collector_type: CollectorType) -> None:
    """Validate that a collector is compatible with a profile."""
    if not profile.validate_collector(collector_type):
        supported = [c.value for c in profile.supported_collectors]
        raise ProfileValidationError(
            f"Collector '{collector_type.value}' is not supported by profile '{profile.name}'. "
            f"Supported collectors: {supported}"
        )


def validate_profile_precision(profile: Profile, precision: str) -> None:
    """Validate that a precision is compatible with a profile."""
    normalized = str(precision or "").strip().lower()
    if not profile.validate_precision(normalized):
        supported = [str(item).strip().lower() for item in profile.supported_precisions]
        raise ProfileValidationError(
            f"Precision '{normalized}' is not supported by profile '{profile.name}'. "
            f"Supported precisions: {supported}"
        )
