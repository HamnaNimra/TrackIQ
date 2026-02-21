"""Profile definitions for AutoPerfPy.

Predefined performance testing profiles. Uses TrackIQ's Profile and
registry; registers AutoPerfPy-specific profiles on import.
"""

from trackiq_core.configs.profiles import (
    CollectorType,
    Profile,
    ProfileValidationError,
    get_profile,
    get_profile_info,
    list_profiles,
    register_profile,
    validate_profile_collector,
    validate_profile_precision,
)

# ============================================================================
# Predefined Profiles (AutoPerfPy-specific)
# ============================================================================

AUTOMOTIVE_SAFETY = Profile(
    name="automotive_safety",
    description="Strict latency requirements for ADAS and autonomous driving applications",
    latency_threshold_ms=33.3,
    latency_target_ms=20.0,
    latency_percentiles=[50, 95, 99, 99.9],
    throughput_min_fps=30.0,
    throughput_target_fps=60.0,
    power_budget_w=50.0,
    thermal_limit_c=80.0,
    memory_limit_mb=8192,
    memory_headroom_percent=30.0,
    batch_sizes=[1, 2, 4],
    warmup_iterations=50,
    test_iterations=1000,
    num_runs=5,
    sample_interval_ms=10,
    duration_seconds=300,
    enable_continuous_monitoring=True,
    supported_collectors=[
        CollectorType.SYNTHETIC,
        CollectorType.TEGRASTATS,
        CollectorType.NVML,
    ],
    collector_config={
        "synthetic": {
            "latency_jitter_percent": 5.0,
            "latency_spike_prob": 0.001,
            "workload_pattern": "steady",
        },
    },
    analysis_config={
        "outlier_method": "zscore",
        "zscore_threshold": 4.0,
        "cv_threshold_high": 10.0,
    },
    tags=["automotive", "safety-critical", "real-time", "adas"],
)


EDGE_MAX_PERF = Profile(
    name="edge_max_perf",
    description="Maximum performance for edge inference workloads",
    latency_threshold_ms=100.0,
    latency_target_ms=50.0,
    latency_percentiles=[50, 95, 99],
    throughput_min_fps=60.0,
    throughput_target_fps=120.0,
    power_budget_w=150.0,
    thermal_limit_c=85.0,
    memory_limit_mb=16384,
    memory_headroom_percent=15.0,
    batch_sizes=[1, 4, 8, 16, 32, 64],
    warmup_iterations=20,
    test_iterations=500,
    num_runs=3,
    sample_interval_ms=50,
    duration_seconds=120,
    enable_continuous_monitoring=True,
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
    analysis_config={"include_pareto_analysis": True},
    tags=["edge", "high-performance", "throughput-optimized"],
)


EDGE_LOW_POWER = Profile(
    name="edge_low_power",
    description="Power-optimized profile for battery/thermal constrained edge devices",
    latency_threshold_ms=200.0,
    latency_target_ms=100.0,
    latency_percentiles=[50, 95, 99],
    throughput_min_fps=15.0,
    throughput_target_fps=30.0,
    power_budget_w=15.0,
    thermal_limit_c=70.0,
    memory_limit_mb=4096,
    memory_headroom_percent=25.0,
    batch_sizes=[1, 2, 4],
    warmup_iterations=10,
    test_iterations=200,
    num_runs=3,
    sample_interval_ms=200,
    duration_seconds=180,
    enable_continuous_monitoring=True,
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
            "workload_pattern": "cyclic",
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
    latency_threshold_ms=500.0,
    latency_target_ms=100.0,
    latency_percentiles=[50, 99],
    throughput_min_fps=5.0,
    throughput_target_fps=30.0,
    power_budget_w=None,
    thermal_limit_c=90.0,
    memory_limit_mb=None,
    memory_headroom_percent=10.0,
    batch_sizes=[1, 4],
    warmup_iterations=5,
    test_iterations=50,
    num_runs=1,
    sample_interval_ms=500,
    duration_seconds=10,
    enable_continuous_monitoring=False,
    supported_collectors=[CollectorType.SYNTHETIC],
    collector_config={
        "synthetic": {
            "warmup_samples": 3,
            "workload_pattern": "steady",
        },
    },
    analysis_config={"outlier_method": "iqr"},
    tags=["ci", "smoke-test", "quick", "validation"],
)


# Register predefined profiles with TrackIQ registry (idempotent on re-import)
for _profile in [AUTOMOTIVE_SAFETY, EDGE_MAX_PERF, EDGE_LOW_POWER, CI_SMOKE]:
    if _profile.name not in list_profiles():
        try:
            register_profile(_profile)
        except ValueError:
            pass

__all__ = [
    "Profile",
    "CollectorType",
    "ProfileValidationError",
    "get_profile",
    "list_profiles",
    "register_profile",
    "get_profile_info",
    "validate_profile_collector",
    "validate_profile_precision",
    "AUTOMOTIVE_SAFETY",
    "EDGE_MAX_PERF",
    "EDGE_LOW_POWER",
    "CI_SMOKE",
]
