"""Centralized TrackIQ-core dependencies for trackiq_compare.

This module loads only the required trackiq_core modules without importing
trackiq_core package-level initializers, which may pull optional heavy deps.
"""

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Dict


def _load_module(name: str, file_path: Path) -> ModuleType:
    """Load a module from a concrete file path."""
    spec = importlib.util.spec_from_file_location(name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ROOT = Path(__file__).resolve().parents[1]
_SCHEMA = _load_module("trackiq_core_schema", _ROOT / "trackiq_core" / "schema.py")
_REGRESSION = _load_module(
    "trackiq_core_regression",
    _ROOT / "trackiq_core" / "utils" / "compare" / "regression.py",
)

TrackiqResult = _SCHEMA.TrackiqResult
PlatformInfo = _SCHEMA.PlatformInfo
WorkloadInfo = _SCHEMA.WorkloadInfo
Metrics = _SCHEMA.Metrics
RegressionInfo = _SCHEMA.RegressionInfo

RegressionDetector = _REGRESSION.RegressionDetector
RegressionThreshold = _REGRESSION.RegressionThreshold


def ensure_parent_dir(path: str) -> None:
    """Create parent directory for a file path if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_trackiq_result(result: Any, path: str) -> None:
    """Save TrackiqResult object to JSON."""
    ensure_parent_dir(path)
    payload = result.to_dict() if hasattr(result, "to_dict") else result
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_trackiq_result(path: str):
    """Load TrackiqResult object from JSON file."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return TrackiqResult.from_dict(payload)


def validate_trackiq_result(payload: Dict[str, Any]) -> None:
    """Validate payload by attempting schema construction."""
    try:
        TrackiqResult.from_dict(payload)
    except Exception as exc:  # pragma: no cover - passthrough validation helper
        raise ValueError(f"Invalid TrackiqResult payload: {exc}") from exc


__all__ = [
    "TrackiqResult",
    "PlatformInfo",
    "WorkloadInfo",
    "Metrics",
    "RegressionInfo",
    "load_trackiq_result",
    "save_trackiq_result",
    "validate_trackiq_result",
    "RegressionDetector",
    "RegressionThreshold",
    "ensure_parent_dir",
]

