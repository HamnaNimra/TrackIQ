"""Tests for minicluster.ui.streamlit_app helpers."""

from __future__ import annotations

from minicluster.ui.streamlit_app import _build_demo_result, _build_result_summary_text


def test_build_demo_result_returns_training_trackiq_result() -> None:
    """Demo builder should return a MiniCluster training result with payload steps."""
    result = _build_demo_result()
    assert result.tool_name == "minicluster"
    assert result.workload.workload_type == "training"
    assert isinstance(result.tool_payload, dict)
    assert isinstance(result.tool_payload.get("steps"), list)


def test_build_result_summary_text_contains_p99_and_throughput() -> None:
    """Sidebar summary text should surface latency and throughput highlights."""
    result = _build_demo_result()
    summary = _build_result_summary_text(result)
    assert "P99" in summary
    assert "samples/s" in summary
