"""Profiles module for AutoPerfPy.

This module provides predefined performance testing profiles for different
use cases and deployment scenarios.

Available Profiles:
    - automotive_safety: Strict latency for ADAS/autonomous driving
    - edge_max_perf: Maximum throughput for edge inference
    - edge_low_power: Power-optimized for constrained devices
    - ci_smoke: Quick validation for CI/CD pipelines

Example usage:
    from autoperfpy.profiles import get_profile, list_profiles

    # List available profiles
    print("Available profiles:", list_profiles())

    # Load a profile
    profile = get_profile("automotive_safety")
    print(f"Latency threshold: {profile.latency_threshold_ms}ms")

    # Use profile settings
    collector = SyntheticCollector(config=profile.get_synthetic_config())
"""

from .profiles import (
    Profile,
    CollectorType,
    ProfileValidationError,
    get_profile,
    list_profiles,
    register_profile,
    get_profile_info,
    validate_profile_collector,
    # Predefined profiles
    AUTOMOTIVE_SAFETY,
    EDGE_MAX_PERF,
    EDGE_LOW_POWER,
    CI_SMOKE,
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
    "AUTOMOTIVE_SAFETY",
    "EDGE_MAX_PERF",
    "EDGE_LOW_POWER",
    "CI_SMOKE",
]
