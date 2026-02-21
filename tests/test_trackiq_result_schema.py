"""Tests for canonical TrackIQ result schema and helpers."""

from datetime import datetime

import pytest

from trackiq_core.schema import (
    Metrics,
    PlatformInfo,
    RegressionInfo,
    TrackiqResult,
    WorkloadInfo,
)
from trackiq_core.serializer import load_trackiq_result, save_trackiq_result
from trackiq_core.validator import validate_trackiq_result


def _sample_result() -> TrackiqResult:
    """Build a fully populated sample TrackiqResult."""
    return TrackiqResult(
        tool_name="autoperfpy",
        tool_version="0.1.0",
        timestamp=datetime(2026, 2, 21, 12, 0, 0),
        platform=PlatformInfo(
            hardware_name="NVIDIA Orin",
            os="Linux 6.8",
            framework="pytorch",
            framework_version="2.7.0",
        ),
        workload=WorkloadInfo(
            name="resnet50",
            workload_type="inference",
            batch_size=4,
            steps=120,
        ),
        metrics=Metrics(
            throughput_samples_per_sec=98.5,
            latency_p50_ms=8.2,
            latency_p95_ms=11.6,
            latency_p99_ms=13.1,
            memory_utilization_percent=64.0,
            communication_overhead_percent=None,
            power_consumption_watts=None,
        ),
        regression=RegressionInfo(
            baseline_id="main",
            delta_percent=2.4,
            status="pass",
            failed_metrics=[],
        ),
    )


def test_trackiq_result_creation_all_fields() -> None:
    """TrackiqResult should be constructible with all required fields."""
    result = _sample_result()
    assert result.tool_name == "autoperfpy"
    assert result.platform.hardware_name == "NVIDIA Orin"
    assert result.workload.workload_type == "inference"
    assert result.metrics.latency_p99_ms == 13.1
    assert result.regression.status == "pass"


def test_trackiq_result_serialize_round_trip(tmp_path) -> None:
    """Serialization and deserialization should round-trip canonical data."""
    path = tmp_path / "trackiq_result.json"
    source = _sample_result()
    save_trackiq_result(source, path)
    loaded = load_trackiq_result(path)
    assert loaded.to_dict() == source.to_dict()


def test_trackiq_result_validation_passes_on_valid_payload() -> None:
    """Validator should accept a correct TrackiqResult payload."""
    payload = _sample_result().to_dict()
    validate_trackiq_result(payload)


def test_trackiq_result_validation_fails_on_missing_required_field() -> None:
    """Validator should raise clear error when required field is missing."""
    payload = _sample_result().to_dict()
    del payload["metrics"]
    with pytest.raises(ValueError, match="Missing required field: metrics"):
        validate_trackiq_result(payload)


def test_trackiq_result_null_handling_for_optional_fields(tmp_path) -> None:
    """Optional nullable fields should be preserved through save/load."""
    path = tmp_path / "trackiq_result_nulls.json"
    result = _sample_result()
    result.metrics.communication_overhead_percent = None
    result.metrics.power_consumption_watts = None
    result.regression.baseline_id = None
    save_trackiq_result(result, path)
    loaded = load_trackiq_result(path)
    assert loaded.metrics.communication_overhead_percent is None
    assert loaded.metrics.power_consumption_watts is None
    assert loaded.regression.baseline_id is None


def test_from_dict_defaults_schema_version_when_missing() -> None:
    """Loading payload without schema_version should default to 1.0.0."""
    payload = _sample_result().to_dict()
    del payload["schema_version"]
    loaded = TrackiqResult.from_dict(payload)
    assert loaded.schema_version == "1.0.0"


def test_new_power_fields_round_trip_through_dict_conversion() -> None:
    """New optional power/thermal fields should round-trip via to_dict/from_dict."""
    source = _sample_result()
    source.metrics.energy_per_step_joules = 12.5
    source.metrics.performance_per_watt = 3.2
    source.metrics.temperature_celsius = 71.4

    payload = source.to_dict()
    loaded = TrackiqResult.from_dict(payload)

    assert loaded.metrics.energy_per_step_joules == 12.5
    assert loaded.metrics.performance_per_watt == 3.2
    assert loaded.metrics.temperature_celsius == 71.4
