"""Tests for trackiq_core.ui.streamlit_app helpers."""

from __future__ import annotations

from trackiq_core.ui.streamlit_app import CoreDashboard, _build_demo_result, _discover_result_rows


def test_build_demo_result_returns_valid_trackiq_result() -> None:
    """Demo builder should produce a non-empty canonical result."""
    result = _build_demo_result()
    assert result.tool_name == "trackiq_core_demo"
    assert result.metrics.throughput_samples_per_sec > 0
    assert result.workload.workload_type == "inference"


def test_discover_result_rows_handles_browser_failures(monkeypatch) -> None:
    """Discovery helper should return empty list instead of raising."""

    class _FailingBrowser:
        def to_dict(self):  # noqa: D401
            raise RuntimeError("boom")

    import trackiq_core.ui.streamlit_app as app

    monkeypatch.setattr(app, "ResultBrowser", _FailingBrowser)
    rows = _discover_result_rows()
    assert rows == []


def test_core_dashboard_is_generic_and_accepts_any_tool() -> None:
    """Core dashboard should not restrict tool names."""
    result = _build_demo_result()
    dash = CoreDashboard(result=result)
    assert dash.expected_tool_names() is None
