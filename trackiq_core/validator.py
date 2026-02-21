"""Validation helpers for canonical TrackIQ results."""

from collections.abc import Iterable
from typing import Any

from trackiq_core.schema import TrackiqResult


def _require_keys(container: dict[str, Any], keys: Iterable[str], prefix: str) -> None:
    for key in keys:
        if key not in container:
            raise ValueError(f"Missing required field: {prefix}{key}")


def validate_trackiq_result(data: dict[str, Any]) -> None:
    """Validate a loaded TrackiqResult payload.

    Raises:
        ValueError: missing required fields or invalid enum-like values.
        TypeError: wrong field types.
    """
    _require_keys(
        data,
        [
            "tool_name",
            "tool_version",
            "timestamp",
            "platform",
            "workload",
            "metrics",
            "regression",
        ],
        "",
    )
    if not isinstance(data["tool_name"], str):
        raise TypeError("Field 'tool_name' must be str")
    if not isinstance(data["tool_version"], str):
        raise TypeError("Field 'tool_version' must be str")
    if not isinstance(data["timestamp"], str):
        raise TypeError("Field 'timestamp' must be ISO datetime string")

    platform = data["platform"]
    if not isinstance(platform, dict):
        raise TypeError("Field 'platform' must be object")
    _require_keys(platform, ["hardware_name", "os", "framework", "framework_version"], "platform.")
    for key in ["hardware_name", "os", "framework", "framework_version"]:
        if not isinstance(platform[key], str):
            raise TypeError(f"Field 'platform.{key}' must be str")

    workload = data["workload"]
    if not isinstance(workload, dict):
        raise TypeError("Field 'workload' must be object")
    _require_keys(workload, ["name", "workload_type", "batch_size", "steps"], "workload.")
    if not isinstance(workload["name"], str):
        raise TypeError("Field 'workload.name' must be str")
    if workload["workload_type"] not in ("inference", "training"):
        raise ValueError("Field 'workload.workload_type' must be 'inference' or 'training'")
    if not isinstance(workload["batch_size"], int):
        raise TypeError("Field 'workload.batch_size' must be int")
    if not isinstance(workload["steps"], int):
        raise TypeError("Field 'workload.steps' must be int")

    metrics = data["metrics"]
    if not isinstance(metrics, dict):
        raise TypeError("Field 'metrics' must be object")
    _require_keys(
        metrics,
        [
            "throughput_samples_per_sec",
            "latency_p50_ms",
            "latency_p95_ms",
            "latency_p99_ms",
            "memory_utilization_percent",
            "communication_overhead_percent",
            "power_consumption_watts",
        ],
        "metrics.",
    )
    for key in [
        "throughput_samples_per_sec",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "memory_utilization_percent",
    ]:
        if not isinstance(metrics[key], (int, float)):
            raise TypeError(f"Field 'metrics.{key}' must be number")
    if metrics["communication_overhead_percent"] is not None and not isinstance(
        metrics["communication_overhead_percent"], (int, float)
    ):
        raise TypeError("Field 'metrics.communication_overhead_percent' must be number or null")
    if metrics["power_consumption_watts"] is not None and not isinstance(
        metrics["power_consumption_watts"], (int, float)
    ):
        raise TypeError("Field 'metrics.power_consumption_watts' must be number or null")
    for key in ["ttft_ms", "tokens_per_sec", "decode_tpt_ms"]:
        if key in metrics and metrics[key] is not None and not isinstance(metrics[key], (int, float)):
            raise TypeError(f"Field 'metrics.{key}' must be number or null")
    if (
        "scaling_efficiency_pct" in metrics
        and metrics["scaling_efficiency_pct"] is not None
        and not isinstance(metrics["scaling_efficiency_pct"], (int, float))
    ):
        raise TypeError("Field 'metrics.scaling_efficiency_pct' must be number or null")

    regression = data["regression"]
    if not isinstance(regression, dict):
        raise TypeError("Field 'regression' must be object")
    _require_keys(regression, ["baseline_id", "delta_percent", "status", "failed_metrics"], "regression.")
    if regression["baseline_id"] is not None and not isinstance(regression["baseline_id"], str):
        raise TypeError("Field 'regression.baseline_id' must be str or null")
    if not isinstance(regression["delta_percent"], (int, float)):
        raise TypeError("Field 'regression.delta_percent' must be number")
    if regression["status"] not in ("pass", "fail"):
        raise ValueError("Field 'regression.status' must be 'pass' or 'fail'")
    if not isinstance(regression["failed_metrics"], list) or any(
        not isinstance(item, str) for item in regression["failed_metrics"]
    ):
        raise TypeError("Field 'regression.failed_metrics' must be list[str]")

    if "tool_payload" in data and data["tool_payload"] is not None and not isinstance(data["tool_payload"], dict):
        raise TypeError("Field 'tool_payload' must be object or null")

    if "kv_cache" in data and data["kv_cache"] is not None:
        kv_cache = data["kv_cache"]
        if not isinstance(kv_cache, dict):
            raise TypeError("Field 'kv_cache' must be object or null")
        required = [
            "estimated_size_mb",
            "max_sequence_length",
            "batch_size",
            "num_layers",
            "num_heads",
            "head_size",
            "precision",
        ]
        _require_keys(kv_cache, required, "kv_cache.")
        for key in [
            "estimated_size_mb",
        ]:
            if not isinstance(kv_cache[key], (int, float)):
                raise TypeError(f"Field 'kv_cache.{key}' must be number")
        for key in [
            "max_sequence_length",
            "batch_size",
            "num_layers",
            "num_heads",
            "head_size",
        ]:
            if not isinstance(kv_cache[key], int):
                raise TypeError(f"Field 'kv_cache.{key}' must be int")
        if not isinstance(kv_cache["precision"], str):
            raise TypeError("Field 'kv_cache.precision' must be str")
        if "samples" in kv_cache and not isinstance(kv_cache["samples"], list):
            raise TypeError("Field 'kv_cache.samples' must be list when provided")


def validate_trackiq_result_obj(result: TrackiqResult) -> None:
    """Validate a TrackiqResult dataclass object."""
    validate_trackiq_result(result.to_dict())
