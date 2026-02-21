"""Compatibility tests for report JSON input normalization."""

from autoperfpy.cli import _normalize_report_input_data


def test_normalize_report_input_data_keeps_raw_payload() -> None:
    """Raw benchmark exports should pass through unchanged."""
    raw = {
        "collector_name": "synthetic",
        "summary": {"latency": {"p99_ms": 12.3}},
        "samples": [],
    }
    normalized = _normalize_report_input_data(raw)
    assert normalized == raw


def test_normalize_report_input_data_unwraps_trackiq_result() -> None:
    """TrackiqResult wrappers should be unwrapped to report payload shape."""
    wrapped = {
        "tool_name": "autoperfpy",
        "tool_payload": {
            "summary": {"latency": {"p99_ms": 45.6}},
            "samples": [],
        },
    }
    normalized = _normalize_report_input_data(wrapped)
    assert "summary" in normalized
    assert "samples" in normalized
    assert normalized.get("collector_name") == "autoperfpy"
