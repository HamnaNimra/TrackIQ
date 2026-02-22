"""Tests for trackiq_core.ui.streamlit_app helpers."""

from __future__ import annotations

from trackiq_core.ui.streamlit_app import (
    CoreDashboard,
    _build_compare_rows,
    _build_demo_result,
    _discover_result_rows,
    _result_row_label,
)


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


def test_result_row_label_includes_filename() -> None:
    """Result row labels should include source filename for disambiguation."""
    row = {
        "tool_name": "autoperfpy",
        "workload_name": "inference",
        "timestamp": "2026-02-22T12:00:00Z",
        "path": "output/results/run_a.json",
    }
    label = _result_row_label(row, 0)
    assert "run_a.json" in label
    assert "autoperfpy" in label


def test_build_compare_rows_computes_delta_vs_baseline() -> None:
    """Compare rows should compute throughput delta percentage against baseline."""
    baseline = _build_demo_result()
    current = _build_demo_result()
    current.metrics.throughput_samples_per_sec = baseline.metrics.throughput_samples_per_sec * 1.1
    rows = _build_compare_rows([("baseline", baseline), ("current", current)], baseline_label="baseline")
    assert len(rows) == 2
    assert rows[0]["Delta vs Baseline (%)"] == 0.0
    assert float(rows[1]["Delta vs Baseline (%)"]) > 9.9
