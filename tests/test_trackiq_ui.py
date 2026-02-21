"""Tests for trackiq_core.ui package."""

from datetime import datetime
import sys
from pathlib import Path

import pytest

from trackiq_core.schema import (
    Metrics,
    PlatformInfo,
    RegressionInfo,
    TrackiqResult,
    WorkloadInfo,
)
from trackiq_core.ui import (
    DARK_THEME,
    LIGHT_THEME,
    ComparisonTable,
    DevicePanel,
    LossChart,
    MetricTable,
    PowerGauge,
    RegressionBadge,
    ResultBrowser,
    RunHistoryLoader,
    TrackiqDashboard,
    TrendChart,
    WorkerGrid,
    run_dashboard,
)
from trackiq_core.serializer import save_trackiq_result


def _result(
    throughput: float = 100.0,
    power: float | None = 120.0,
    energy: float | None = 2.0,
    perf_per_watt: float | None = 0.9,
    temp: float | None = 50.0,
    hardware: str = "HW-A",
    optional_none: bool = False,
) -> TrackiqResult:
    return TrackiqResult(
        tool_name="tool_a",
        tool_version="0.1.0",
        timestamp=datetime(2026, 2, 21, 12, 0, 0),
        platform=PlatformInfo(
            hardware_name=hardware,
            os="Linux",
            framework="pytorch",
            framework_version="2.7.0",
        ),
        workload=WorkloadInfo(
            name="demo",
            workload_type="inference",
            batch_size=1,
            steps=10,
        ),
        metrics=Metrics(
            throughput_samples_per_sec=throughput,
            latency_p50_ms=10.0,
            latency_p95_ms=11.0,
            latency_p99_ms=12.0,
            memory_utilization_percent=40.0,
            communication_overhead_percent=None if optional_none else 1.0,
            power_consumption_watts=power if not optional_none else None,
            energy_per_step_joules=energy if not optional_none else None,
            performance_per_watt=perf_per_watt if not optional_none else None,
            temperature_celsius=temp if not optional_none else None,
        ),
        regression=RegressionInfo(
            baseline_id=None,
            delta_percent=1.5,
            status="pass",
            failed_metrics=[],
        ),
        tool_payload=None,
    )


def test_themes_have_no_null_fields() -> None:
    """Prebuilt themes should define every field with non-null values."""
    for theme in (DARK_THEME, LIGHT_THEME):
        for key, value in theme.__dict__.items():
            assert value is not None, key
            if isinstance(value, str):
                assert value.strip() != "", key


def test_metric_table_single_mode_returns_all_metric_fields() -> None:
    """Single mode should expose all metric fields in to_dict output."""
    payload = MetricTable(result=_result(), mode="single").to_dict()
    metric_keys = set(payload["metrics"].keys())
    expected = set(_result().metrics.__dict__.keys())
    assert metric_keys == expected


def test_metric_table_comparison_winner_logic() -> None:
    """Comparison mode should use higher-is-better for throughput and lower for power."""
    a = _result(throughput=100.0, power=200.0)
    b = _result(throughput=120.0, power=150.0, hardware="HW-B")
    payload = MetricTable(result=[a, b], mode="comparison").to_dict()
    by_metric = {row["metric"]: row for row in payload["metrics"]}
    assert by_metric["throughput_samples_per_sec"]["winner"] == "B"
    assert by_metric["power_consumption_watts"]["winner"] == "B"


def test_loss_chart_to_dict_round_trip_fields() -> None:
    """LossChart should return core series arrays unchanged."""
    data = LossChart(steps=[0, 1], loss_values=[1.0, 0.9], baseline_values=[1.1, 1.0]).to_dict()
    assert data["steps"] == [0, 1]
    assert data["loss_values"] == [1.0, 0.9]
    assert data["baseline_values"] == [1.1, 1.0]


def test_regression_badge_to_dict_returns_status_and_delta() -> None:
    """Regression badge payload should include status and delta."""
    payload = RegressionBadge(_result().regression).to_dict()
    assert payload["status"] == "pass"
    assert payload["delta_percent"] == 1.5


def test_worker_grid_to_dict_returns_all_workers() -> None:
    """Worker grid payload should preserve all entries."""
    workers = [
        {"worker_id": "w0", "throughput": 10, "allreduce_time_ms": 1.1, "status": "healthy"},
        {"worker_id": "w1", "throughput": 8, "allreduce_time_ms": 2.3, "status": "slow"},
    ]
    payload = WorkerGrid(workers).to_dict()
    assert payload["workers"] == workers


def test_power_gauge_to_dict_placeholder_when_all_power_null() -> None:
    """Power gauge should emit placeholder when power profiling is absent."""
    result = _result(optional_none=True)
    payload = PowerGauge(metrics=result.metrics, tool_payload=result.tool_payload).to_dict()
    assert payload["placeholder"] == "Power profiling not available in this environment."


def test_comparison_table_to_dict_contains_platform_diff_and_metrics() -> None:
    """Comparison table should include platform differences and metric payload."""
    a = _result(hardware="HW-A")
    b = _result(hardware="HW-B", throughput=90.0)
    payload = ComparisonTable(a, b).to_dict()
    assert "hardware_name" in payload["platform_diff"]
    assert payload["metric_comparison"]["mode"] == "comparison"
    assert isinstance(payload["summary"], str) and payload["summary"]


def test_components_instantiate_with_optional_metrics_none() -> None:
    """Components should initialize when optional metrics are all None."""
    result = _result(optional_none=True)
    MetricTable(result=result, mode="single")
    MetricTable(result=[result, _result(optional_none=True)], mode="comparison")
    LossChart(steps=[0, 1, 2], loss_values=[1.0, 0.8, 0.7])
    RegressionBadge(result.regression)
    WorkerGrid(
        [{"worker_id": "w0", "throughput": 1, "allreduce_time_ms": 0.5, "status": "healthy"}]
    )
    PowerGauge(metrics=result.metrics, tool_payload=result.tool_payload)
    ComparisonTable(result, _result(optional_none=True, hardware="HW-B"))


def test_run_dashboard_raises_when_no_input_provided() -> None:
    """Launcher should raise ValueError when no result source is provided."""
    with pytest.raises(ValueError, match="Either result_path or result must be provided."):
        run_dashboard(dashboard_class=object)  # type: ignore[arg-type]


def test_components_import_without_streamlit_dependency() -> None:
    """UI component imports should not force import of streamlit."""
    preloaded = sys.modules.pop("streamlit", None)
    try:
        __import__("trackiq_core.ui.components.metric_table")
        __import__("trackiq_core.ui.components.loss_chart")
        __import__("trackiq_core.ui.components.regression_badge")
        __import__("trackiq_core.ui.components.worker_grid")
        __import__("trackiq_core.ui.components.power_gauge")
        __import__("trackiq_core.ui.components.comparison_table")
        __import__("trackiq_core.ui.components.device_panel")
        __import__("trackiq_core.ui.components.result_browser")
        __import__("trackiq_core.ui.components.trend_chart")
        assert "streamlit" not in sys.modules
    finally:
        if preloaded is not None:
            sys.modules["streamlit"] = preloaded


def test_device_panel_to_dict_has_device_on_any_machine() -> None:
    """DevicePanel should expose at least one device when detection is available."""
    from trackiq_core.hardware.devices import get_all_devices

    panel = DevicePanel(devices=get_all_devices())
    payload = panel.to_dict()
    assert isinstance(payload["devices"], list)
    assert len(payload["devices"]) >= 1


def test_device_panel_to_dict_handles_empty_devices() -> None:
    """DevicePanel should handle empty device lists gracefully."""
    panel = DevicePanel(devices=[])
    payload = panel.to_dict()
    assert payload["devices"] == []
    assert payload["selected_device_index"] is None


def test_result_browser_to_dict_empty_when_no_valid_files(tmp_path: Path) -> None:
    """ResultBrowser should return empty metadata list when no valid files exist."""
    browser = ResultBrowser(search_paths=[str(tmp_path)])
    assert browser.to_dict() == []


def test_result_browser_to_dict_returns_metadata_for_valid_result(tmp_path: Path) -> None:
    """ResultBrowser should return metadata for valid TrackiqResult files."""
    out = tmp_path / "run.json"
    save_trackiq_result(_result(), out)

    browser = ResultBrowser(search_paths=[str(tmp_path)])
    rows = browser.to_dict()
    assert len(rows) == 1
    assert rows[0]["tool_name"] == "tool_a"
    assert rows[0]["workload_name"] == "demo"
    assert rows[0]["path"] == str(out)


def test_run_history_loader_loads_sorted_results_and_skips_invalid(tmp_path: Path) -> None:
    """RunHistoryLoader should return valid results sorted by timestamp."""
    history_dir = tmp_path / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    older = _result(throughput=90.0, perf_per_watt=0.7)
    older.timestamp = datetime(2026, 2, 20, 10, 0, 0)
    newer = _result(throughput=120.0, perf_per_watt=1.0)
    newer.timestamp = datetime(2026, 2, 21, 10, 0, 0)
    save_trackiq_result(newer, history_dir / "newer.json")
    save_trackiq_result(older, history_dir / "older.json")
    (history_dir / "broken.json").write_text("{not-json", encoding="utf-8")

    loaded = RunHistoryLoader(str(history_dir)).load()
    assert len(loaded) == 2
    assert [item.metrics.throughput_samples_per_sec for item in loaded] == [90.0, 120.0]


def test_trend_chart_to_dict_default_metrics_filters_null_points() -> None:
    """TrendChart should include default metrics and omit null metric points."""
    run_a = _result(throughput=95.0, perf_per_watt=0.8)
    run_a.timestamp = datetime(2026, 2, 19, 8, 0, 0)
    run_a.metrics.latency_p99_ms = 16.0

    run_b = _result(throughput=110.0, perf_per_watt=1.1)
    run_b.timestamp = datetime(2026, 2, 20, 8, 0, 0)
    run_b.metrics.latency_p99_ms = 12.0

    run_c = _result(throughput=125.0, perf_per_watt=1.3)
    run_c.timestamp = datetime(2026, 2, 21, 8, 0, 0)
    run_c.metrics.latency_p99_ms = 10.0
    run_c.metrics.performance_per_watt = None

    payload = TrendChart(results=[run_c, run_a, run_b]).to_dict()

    assert payload["run_count"] == 3
    assert payload["metric_names"] == list(TrendChart.DEFAULT_METRICS)
    assert [point["value"] for point in payload["trends"]["throughput_samples_per_sec"]] == [
        95.0,
        110.0,
        125.0,
    ]
    assert len(payload["trends"]["performance_per_watt"]) == 2


def test_trend_chart_supports_custom_metric_names() -> None:
    """TrendChart should preserve custom metric ordering and unknown metrics."""
    payload = TrendChart(
        results=[_result()],
        metric_names=["latency_p95_ms", "missing_metric"],
    ).to_dict()
    assert payload["metric_names"] == ["latency_p95_ms", "missing_metric"]
    assert len(payload["trends"]["latency_p95_ms"]) == 1
    assert payload["trends"]["missing_metric"] == []


def test_dashboard_subclass_has_device_and_result_browser_methods() -> None:
    """TrackiqDashboard subclasses should expose device panel and result browser methods."""

    class _Dash(TrackiqDashboard):
        def render_body(self) -> None:
            return None

    dash = _Dash(result=_result(), theme=DARK_THEME, title="T")
    assert hasattr(dash, "render_device_panel")
    assert hasattr(dash, "render_result_browser")
    assert hasattr(dash, "render_trend_section")
    assert hasattr(dash, "render_download_section")


def test_power_gauge_placeholder_with_null_metrics_and_live_device_none() -> None:
    """PowerGauge to_dict should return placeholder when no benchmark/live power exists."""
    result = _result(optional_none=True)
    payload = PowerGauge(
        metrics=result.metrics, tool_payload=result.tool_payload, live_device=None
    ).to_dict()
    assert payload["placeholder"] == "Power profiling not available in this environment."


def test_apply_theme_css_contains_font_and_background(monkeypatch: pytest.MonkeyPatch) -> None:
    """apply_theme should inject CSS containing the configured font and background."""
    captured = {"css": ""}

    class _FakeStreamlit:
        @staticmethod
        def markdown(content, unsafe_allow_html=False):  # noqa: ANN001
            if unsafe_allow_html:
                captured["css"] = content

    monkeypatch.setitem(sys.modules, "streamlit", _FakeStreamlit)

    class _Dash(TrackiqDashboard):
        def render_body(self) -> None:
            return None

    dash = _Dash(result=_result(), theme=DARK_THEME, title="T")
    dash.apply_theme(DARK_THEME)
    assert DARK_THEME.font_family in captured["css"]
    assert DARK_THEME.background_color in captured["css"]


def test_render_download_section_emits_two_download_buttons(monkeypatch: pytest.MonkeyPatch) -> None:
    """render_download_section should render JSON and HTML download buttons."""
    calls = {"labels": []}

    class _Ctx:
        def __enter__(self):  # noqa: D401
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: D401, ANN001
            return False

    class _FakeStreamlit:
        @staticmethod
        def markdown(*args, **kwargs):  # noqa: ANN001
            return None

        @staticmethod
        def columns(n):  # noqa: ANN001
            return tuple(_Ctx() for _ in range(n))

        @staticmethod
        def download_button(label, **kwargs):  # noqa: ANN001
            calls["labels"].append(label)
            return True

    monkeypatch.setitem(sys.modules, "streamlit", _FakeStreamlit)

    class _Dash(TrackiqDashboard):
        def render_body(self) -> None:
            return None

    dash = _Dash(result=_result(), theme=DARK_THEME, title="T")
    dash.render_download_section()
    assert "Download JSON" in calls["labels"]
    assert "Download HTML Report" in calls["labels"]
