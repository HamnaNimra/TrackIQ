"""Custom exceptions for TrackIQ.

This module defines application-specific errors so callers can handle
hardware missing, config errors, and validation failures explicitly.
"""


class TrackIQError(Exception):
    """Base exception for all TrackIQ errors."""

    pass


class HardwareNotFoundError(TrackIQError):
    """Raised when required hardware or driver is not available (e.g. no GPU, no nvidia-smi)."""

    pass


class ConfigError(TrackIQError):
    """Raised when configuration is invalid or a required config file is missing."""

    pass


class DependencyError(TrackIQError):
    """Raised when an optional dependency (e.g. NVML, TensorRT) is required but not installed."""

    pass


class ProfileValidationError(TrackIQError):
    """Raised when a profile does not support the requested collector or option."""

    pass
