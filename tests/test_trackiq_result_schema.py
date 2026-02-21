"""Tests for canonical TrackIQ result schema and helpers."""

import json
from datetime import datetime

import pytest

from trackiq_core.schema import (
    KVCacheInfo,
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


def test_trackiq_result_validation_fails_on_invalid_llm_metric_type() -> None:
    """Validator should reject non-numeric LLM metric field values."""
    payload = _sample_result().to_dict()
    payload["metrics"]["ttft_ms"] = "fast"
    with pytest.raises(TypeError, match="metrics.ttft_ms"):
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


def test_new_llm_fields_round_trip_through_dict_conversion() -> None:
    """Optional LLM fields should round-trip via to_dict/from_dict."""
    source = _sample_result()
    source.metrics.ttft_ms = 812.4
    source.metrics.tokens_per_sec = 26.8
    source.metrics.decode_tpt_ms = 37.3

    loaded = TrackiqResult.from_dict(source.to_dict())
    assert loaded.metrics.ttft_ms == 812.4
    assert loaded.metrics.tokens_per_sec == 26.8
    assert loaded.metrics.decode_tpt_ms == 37.3


def test_kv_cache_round_trip_through_dict_conversion() -> None:
    """KV cache schema block should round-trip in canonical result payload."""
    source = _sample_result()
    source.kv_cache = KVCacheInfo(
        estimated_size_mb=256.0,
        max_sequence_length=2048,
        batch_size=1,
        num_layers=32,
        num_heads=32,
        head_size=128,
        precision="fp16",
        samples=[{"sequence_length": 1024, "kv_cache_mb": 128.0}],
    )
    loaded = TrackiqResult.from_dict(source.to_dict())
    assert loaded.kv_cache is not None
    assert loaded.kv_cache.max_sequence_length == 2048
    assert loaded.kv_cache.samples[0]["kv_cache_mb"] == 128.0


def test_kv_cache_from_tool_payload_backcompat() -> None:
    """from_dict should backfill kv_cache from tool_payload for legacy payloads."""
    payload = _sample_result().to_dict()
    payload["tool_payload"] = {
        "kv_cache": {
            "estimated_size_mb": 512.0,
            "max_sequence_length": 4096,
            "batch_size": 1,
            "num_layers": 40,
            "num_heads": 40,
            "head_size": 128,
            "precision": "fp16",
            "samples": [{"sequence_length": 4096, "kv_cache_mb": 512.0}],
        }
    }
    loaded = TrackiqResult.from_dict(payload)
    assert loaded.kv_cache is not None
    assert loaded.kv_cache.estimated_size_mb == 512.0


def test_llm_metrics_from_tool_payload_backcompat() -> None:
    """from_dict should backfill LLM metrics from tool_payload for legacy payloads."""
    payload = _sample_result().to_dict()
    payload["tool_payload"] = {
        "ttft_p50": 750.0,
        "throughput_tokens_per_sec": 31.5,
        "tpt_p50": 28.4,
    }
    loaded = TrackiqResult.from_dict(payload)
    assert loaded.metrics.ttft_ms == 750.0
    assert loaded.metrics.tokens_per_sec == 31.5
    assert loaded.metrics.decode_tpt_ms == 28.4


def test_llm_metrics_backcompat_falls_back_when_canonical_payload_values_are_null() -> None:
    """from_dict should use legacy keys when canonical tool_payload keys are present but null."""
    payload = _sample_result().to_dict()
    payload["tool_payload"] = {
        "ttft_ms": None,
        "ttft_p50": 750.0,
        "tokens_per_sec": None,
        "throughput_tokens_per_sec": 31.5,
        "decode_tpt_ms": None,
        "tpt_p50": 28.4,
    }
    loaded = TrackiqResult.from_dict(payload)
    assert loaded.metrics.ttft_ms == 750.0
    assert loaded.metrics.tokens_per_sec == 31.5
    assert loaded.metrics.decode_tpt_ms == 28.4


def test_save_trackiq_result_enforces_schema_contract(tmp_path) -> None:
    """save_trackiq_result should reject invalid dataclass payloads."""
    result = _sample_result()
    result.workload.workload_type = "invalid"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="workload.workload_type"):
        save_trackiq_result(result, tmp_path / "invalid.json")


def test_load_trackiq_result_enforces_schema_contract(tmp_path) -> None:
    """load_trackiq_result should reject invalid JSON payloads."""
    path = tmp_path / "broken_schema.json"
    payload = _sample_result().to_dict()
    del payload["metrics"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing required field: metrics"):
        load_trackiq_result(path)


def test_load_trackiq_result_backcompat_legacy_metrics_defaults_to_none(tmp_path) -> None:
    """Legacy payloads missing newer nullable metrics should load successfully."""
    path = tmp_path / "legacy_schema.json"
    payload = _sample_result().to_dict()
    payload["schema_version"] = "1.0.0"
    metrics = payload["metrics"]
    del metrics["communication_overhead_percent"]  # type: ignore[index]
    del metrics["power_consumption_watts"]  # type: ignore[index]
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_trackiq_result(path)
    assert loaded.metrics.communication_overhead_percent is None
    assert loaded.metrics.power_consumption_watts is None


@pytest.mark.parametrize(
    "missing_key",
    ["tool_name", "tool_version", "timestamp", "platform", "workload", "regression"],
)
def test_load_trackiq_result_missing_required_top_level_fields(tmp_path, missing_key: str) -> None:
    """load_trackiq_result should fail fast on missing top-level contract keys."""
    path = tmp_path / f"missing_{missing_key}.json"
    payload = _sample_result().to_dict()
    del payload[missing_key]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=f"Missing required field: {missing_key}"):
        load_trackiq_result(path)
