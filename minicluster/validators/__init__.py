"""Validators module for minicluster."""

from minicluster.validators.correctness_validator import (
    CorrectnessValidator,
    CorrectnessReport,
    StepComparison,
)

try:
    from minicluster.validators.fault_injector import (
        FaultInjector,
        FaultInjectionReport,
        FaultDetectionResult,
        FaultType,
        FaultInjectionConfig,
    )
except Exception as exc:  # pragma: no cover - optional dependency guard
    _FAULT_IMPORT_ERROR = exc

    class FaultInjector:  # type: ignore[no-redef]
        """Fallback that raises an actionable dependency error."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Fault injection requires optional ML dependencies. "
                "Install with: pip install -e \".[ml]\""
            ) from _FAULT_IMPORT_ERROR

    FaultInjectionReport = None  # type: ignore[assignment]
    FaultDetectionResult = None  # type: ignore[assignment]
    FaultType = None  # type: ignore[assignment]
    FaultInjectionConfig = None  # type: ignore[assignment]

__all__ = [
    "CorrectnessValidator",
    "CorrectnessReport",
    "StepComparison",
    "FaultInjector",
    "FaultInjectionReport",
    "FaultDetectionResult",
    "FaultType",
    "FaultInjectionConfig",
]
