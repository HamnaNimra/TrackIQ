"""Profile definitions for AutoPerfPy.

This module defines performance testing profiles for different use cases.
Profiles encapsulate configuration presets for benchmarking, monitoring,
and analysis tailored to specific deployment scenarios.

Available Profiles:
    - automotive_safety: Strict latency requirements for ADAS/autonomous driving
    - edge_max_perf: Maximum performance for edge inference (GPU-intensive)
    - edge_low_power: Power-optimized for battery/thermal constrained devices
    - ci_smoke: Quick validation tests for CI/CD pipelines

Example usage:
    from autoperfpy.profiles import get_profile, list_profiles

    # Get a specific profile
    profile = get_profile("automotive_safety")
    print(f"Max latency: {profile.latency_threshold_ms}ms")

    # List all available profiles
    for name in list_profiles():
        print(f"  - {name}")
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class CollectorType(Enum):
    """Types of collectors that can be used with profiles."""
    SYNTHETIC = "synthetic"
    NVML = "nvml"
    TEGRASTATS = "tegrastats"
    PSUTIL = "psutil"
    TENSORRT = "tensorrt"


@dataclass
class Profile:
    """Performance testing profile configuration.

    Encapsulates all settings needed for a specific testing scenario,
    including thresholds, collector settings, and benchmark parameters.

    Attributes:
        name: Unique identifier for the profile
        description: Human-readable description
        latency_threshold_ms: Maximum acceptable latency (P99)
        latency_target_ms: Target latency for optimization
        throughput_min_fps: Minimum acceptable throughput
        power_budget_w: Maximum power consumption allowed
        thermal_limit_c: Maximum temperature threshold
        memory_limit_mb: Maximum memory usage allowed
        batch_sizes: Batch sizes to test
        warmup_iterations: Number of warmup iterations
        test_iterations: Number of test iterations
        sample_interval_ms: Interval between samples in monitoring
        duration_seconds: Total test duration
        supported_collectors: List of compatible collector types
        collector_config: Collector-specific configuration
        analysis_config: Analysis-specific settings
        tags: Metadata tags for filtering/grouping
    """

    name: str
    description: str

    # Latency requirements
    latency_threshold_ms: float = 100.0      # P99 must be under this
    latency_target_ms: float = 50.0          # Optimization target
    latency_percentiles: List[int] = field(default_factory=lambda: [50, 95, 99])

    # Throughput requirements
    throughput_min_fps: float = 10.0
    throughput_target_fps: float = 30.0

    # Power/thermal constraints
    power_budget_w: Optional[float] = None
    thermal_limit_c: float = 85.0

    # Memory constraints
    memory_limit_mb: Optional[float] = None
    memory_headroom_percent: float = 20.0    # Reserve this % of total memory

    # Benchmark settings
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16])
    warmup_iterations: int = 10
    test_iterations: int = 100
    num_runs: int = 3                        # Repetitions for statistical significance

    # Monitoring settings
    sample_interval_ms: int = 100
    duration_seconds: int = 60
    enable_continuous_monitoring: bool = True

    # Collector configuration
    supported_collectors: List[CollectorType] = field(
        default_factory=lambda: [CollectorType.SYNTHETIC]
    )
    collector_config: Dict[str, Any] = field(default_factory=dict)

    # Analysis configuration
    analysis_config: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary format.

        Returns:
            Dictionary representation suitable for serialization
        """
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
        }

    def validate_collector(self, collector_type: CollectorType) -> bool:
        """Check if a collector type is supported by this profile.

        Args:
            collector_type: The collector type to validate

        Returns:
            True if collector is supported, False otherwise
        """
        return collector_type in self.supported_collectors

    def get_synthetic_config(self) -> Dict[str, Any]:
        """Get configuration for SyntheticCollector based on profile.

        Returns:
            Configuration dictionary for SyntheticCollector
        """
        return {
            "warmup_samples": self.warmup_iterations,
            "base_latency_ms": self.latency_target_ms,
            "latency_jitter_percent": 10.0,
            "base_gpu_percent": 75.0 if self.power_budget_w is None else 50.0,
            **self.collector_config.get("synthetic", {}),
        }


# ============================================================================
# Predefined Profiles
# ============================================================================

AUTOMOTIVE_SAFETY = Profile(
    name="automotive_safety",
    description="Strict latency requirements for ADAS and autonomous driving applications",

    # Tight latency constraints for real-time safety
    latency_threshold_ms=33.3,         # Must meet 30 FPS (33.3ms)
    latency_target_ms=20.0,            # Target for comfortable margin
    latency_percentiles=[50, 95, 99, 99.9],  # Include P99.9 for safety

    # High throughput for real-time processing
    throughput_min_fps=30.0,
    throughput_target_fps=60.0,

    # Conservative power/thermal for automotive environments
    power_budget_w=50.0,               # Typical automotive compute budget
    thermal_limit_c=80.0,              # Lower limit for reliability

    # Memory constraints for embedded systems
    memory_limit_mb=8192,              # 8GB typical for automotive
    memory_headroom_percent=30.0,      # Extra headroom for safety

    # Rigorous testing
    batch_sizes=[1, 2, 4],             # Small batches for low latency
    warmup_iterations=50,              # Thorough warmup
    test_iterations=1000,              # High sample count
    num_runs=5,                        # Multiple runs for confidence

    # Fine-grained monitoring
    sample_interval_ms=10,             # 100 Hz sampling
    duration_seconds=300,              # 5-minute test
    enable_continuous_monitoring=True,

    # Collectors suitable for automotive
    supported_collectors=[
        CollectorType.SYNTHETIC,
        CollectorType.TEGRASTATS,       # Jetson/DriveOS
        CollectorType.NVML,
    ],

    collector_config={
        "synthetic": {
            "latency_jitter_percent": 5.0,   # Lower jitter
            "latency_spike_prob": 0.001,     # Rare spikes
            "workload_pattern": "steady",
        },
    },

    analysis_config={
        "outlier_method": "zscore",
        "zscore_threshold": 4.0,        # Strict outlier detection
        "cv_threshold_high": 10.0,      # Tighter consistency requirement
    },

    tags=["automotive", "safety-critical", "real-time", "adas"],
)


EDGE_MAX_PERF = Profile(
    name="edge_max_perf",
    description="Maximum performance for edge inference workloads",

    # Relaxed latency for throughput optimization
    latency_threshold_ms=100.0,
    latency_target_ms=50.0,
    latency_percentiles=[50, 95, 99],

    # High throughput target
    throughput_min_fps=60.0,
    throughput_target_fps=120.0,

    # Higher power budget for max performance
    power_budget_w=150.0,
    thermal_limit_c=85.0,

    # Larger memory for bigger models/batches
    memory_limit_mb=16384,             # 16GB
    memory_headroom_percent=15.0,

    # Larger batches for throughput
    batch_sizes=[1, 4, 8, 16, 32, 64],
    warmup_iterations=20,
    test_iterations=500,
    num_runs=3,

    # Standard monitoring
    sample_interval_ms=50,
    duration_seconds=120,
    enable_continuous_monitoring=True,

    # All collectors supported
    supported_collectors=[
        CollectorType.SYNTHETIC,
        CollectorType.NVML,
        CollectorType.TEGRASTATS,
        CollectorType.PSUTIL,
        CollectorType.TENSORRT,
    ],

    collector_config={
        "synthetic": {
            "base_gpu_percent": 90.0,
            "workload_pattern": "steady",
        },
    },

    analysis_config={
        "include_pareto_analysis": True,
    },

    tags=["edge", "high-performance", "throughput-optimized"],
)


EDGE_LOW_POWER = Profile(
    name="edge_low_power",
    description="Power-optimized profile for battery/thermal constrained edge devices",

    # Relaxed latency for power efficiency
    latency_threshold_ms=200.0,
    latency_target_ms=100.0,
    latency_percentiles=[50, 95, 99],

    # Lower throughput acceptable
    throughput_min_fps=15.0,
    throughput_target_fps=30.0,

    # Strict power budget
    power_budget_w=15.0,               # Low power envelope
    thermal_limit_c=70.0,              # Conservative thermal

    # Conservative memory
    memory_limit_mb=4096,              # 4GB
    memory_headroom_percent=25.0,

    # Small batches to minimize power spikes
    batch_sizes=[1, 2, 4],
    warmup_iterations=10,
    test_iterations=200,
    num_runs=3,

    # Lower frequency monitoring to save power
    sample_interval_ms=200,
    duration_seconds=180,
    enable_continuous_monitoring=True,

    # Collectors for power monitoring
    supported_collectors=[
        CollectorType.SYNTHETIC,
        CollectorType.TEGRASTATS,
        CollectorType.PSUTIL,
    ],

    collector_config={
        "synthetic": {
            "base_gpu_percent": 50.0,
            "idle_power_w": 5.0,
            "max_power_w": 15.0,
            "workload_pattern": "cyclic",  # Simulate power cycling
        },
    },

    analysis_config={
        "power_column": "power_w",
        "include_pareto_analysis": True,
    },

    tags=["edge", "low-power", "battery", "thermal-constrained"],
)


CI_SMOKE = Profile(
    name="ci_smoke",
    description="Quick validation tests for CI/CD pipelines",

    # Relaxed thresholds for quick pass/fail
    latency_threshold_ms=500.0,
    latency_target_ms=100.0,
    latency_percentiles=[50, 99],

    # Basic throughput check
    throughput_min_fps=5.0,
    throughput_target_fps=30.0,

    # No strict power/thermal limits
    power_budget_w=None,
    thermal_limit_c=90.0,

    # No memory limit
    memory_limit_mb=None,
    memory_headroom_percent=10.0,

    # Minimal testing
    batch_sizes=[1, 4],
    warmup_iterations=5,
    test_iterations=50,
    num_runs=1,

    # Quick monitoring
    sample_interval_ms=500,
    duration_seconds=10,
    enable_continuous_monitoring=False,

    # Synthetic only for CI
    supported_collectors=[
        CollectorType.SYNTHETIC,
    ],

    collector_config={
        "synthetic": {
            "warmup_samples": 3,
            "workload_pattern": "steady",
        },
    },

    analysis_config={
        "outlier_method": "iqr",
    },

    tags=["ci", "smoke-test", "quick", "validation"],
)


# ============================================================================
# Profile Registry
# ============================================================================

_PROFILE_REGISTRY: Dict[str, Profile] = {
    "automotive_safety": AUTOMOTIVE_SAFETY,
    "edge_max_perf": EDGE_MAX_PERF,
    "edge_low_power": EDGE_LOW_POWER,
    "ci_smoke": CI_SMOKE,
}


def get_profile(name: str) -> Profile:
    """Get a profile by name.

    Args:
        name: Profile name (e.g., "automotive_safety")

    Returns:
        Profile instance

    Raises:
        ValueError: If profile name is not found
    """
    if name not in _PROFILE_REGISTRY:
        available = ", ".join(_PROFILE_REGISTRY.keys())
        raise ValueError(f"Unknown profile '{name}'. Available profiles: {available}")
    return _PROFILE_REGISTRY[name]


def list_profiles() -> List[str]:
    """Get list of all available profile names.

    Returns:
        List of profile name strings
    """
    return list(_PROFILE_REGISTRY.keys())


def register_profile(profile: Profile) -> None:
    """Register a custom profile.

    Args:
        profile: Profile instance to register

    Raises:
        ValueError: If profile with same name already exists
    """
    if profile.name in _PROFILE_REGISTRY:
        raise ValueError(f"Profile '{profile.name}' already exists")
    _PROFILE_REGISTRY[profile.name] = profile


def get_profile_info() -> Dict[str, Dict[str, Any]]:
    """Get summary information about all profiles.

    Returns:
        Dictionary mapping profile names to summary dicts
    """
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


class ProfileValidationError(Exception):
    """Raised when profile validation fails."""
    pass


def validate_profile_collector(profile: Profile, collector_type: CollectorType) -> None:
    """Validate that a collector is compatible with a profile.

    Args:
        profile: The profile to validate against
        collector_type: The collector type to check

    Raises:
        ProfileValidationError: If collector is not compatible
    """
    if not profile.validate_collector(collector_type):
        supported = [c.value for c in profile.supported_collectors]
        raise ProfileValidationError(
            f"Collector '{collector_type.value}' is not supported by profile '{profile.name}'. "
            f"Supported collectors: {supported}"
        )


__all__ = [
    "Profile",
    "CollectorType",
    "ProfileValidationError",
    "get_profile",
    "list_profiles",
    "register_profile",
    "get_profile_info",
    "validate_profile_collector",
    # Predefined profiles
    "AUTOMOTIVE_SAFETY",
    "EDGE_MAX_PERF",
    "EDGE_LOW_POWER",
    "CI_SMOKE",
]
