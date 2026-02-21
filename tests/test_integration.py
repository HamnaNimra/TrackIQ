"""Integration test: full benchmark -> JSON -> HTML -> Streamlit UI loads."""

import json
import sys
from pathlib import Path

import pytest

# Ensure project root on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoperfpy.auto_runner import run_single_benchmark
from autoperfpy.cli import _infer_trackiq_result
from autoperfpy.reports import HTMLReportGenerator, PerformanceVisualizer
from autoperfpy.runners import BenchmarkRunner
from trackiq_core.collectors import SyntheticCollector
from trackiq_core.hardware.devices import DEVICE_TYPE_CPU, DeviceProfile
from trackiq_core.inference import InferenceConfig
from trackiq_core.power_profiler import SimulatedPowerReader

try:
    from autoperfpy.reports import charts as shared_charts

    CHARTS_AVAILABLE = shared_charts.is_available()
except ImportError:
    CHARTS_AVAILABLE = False
    shared_charts = None


def test_full_benchmark_saves_json(tmp_path):
    """Run a short synthetic benchmark and save JSON results."""
    config = {"warmup_samples": 2, "seed": 42}
    collector = SyntheticCollector(config=config)
    runner = BenchmarkRunner(
        collector,
        duration_seconds=0.5,
        sample_interval_seconds=0.05,
        quiet=True,
    )
    export = runner.run()
    assert export is not None
    data = export.to_dict()
    json_path = tmp_path / "results.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    assert json_path.exists()
    with open(json_path) as f:
        loaded = json.load(f)
    assert "collector_name" in loaded
    assert "summary" in loaded or "samples" in loaded


def test_generate_html_report_from_benchmark_data(tmp_path):
    """Generate HTML report from benchmark-style data."""
    report = HTMLReportGenerator(
        title="Integration Test Report",
        author="AutoPerfPy",
        theme="light",
    )
    report.add_metadata("Source", "Integration test")
    report.add_summary_item("Samples", 10, "", "neutral")
    viz = PerformanceVisualizer()
    demo = {"A": {"P50": 25.0, "P95": 30.0, "P99": 35.0}}
    fig = viz.plot_latency_percentiles(demo)
    report.add_figure(fig, "Latency Percentiles", "Test")
    html_path = tmp_path / "report.html"
    out = report.generate_html(str(html_path))
    assert Path(out).exists()
    content = Path(out).read_text(encoding="utf-8")
    assert "Integration Test Report" in content
    assert "Latency Percentiles" in content


def test_streamlit_ui_module_loads_with_results(tmp_path):
    """Confirm Streamlit UI loads and accepts results JSON."""
    import autoperfpy.ui.streamlit_app as app_module

    assert hasattr(app_module, "load_json_data")
    assert hasattr(app_module, "load_csv_data")
    assert callable(app_module.load_json_data)
    # Write minimal results JSON and verify loader can read it
    results_file = tmp_path / "results.json"
    data = {
        "collector_name": "SyntheticCollector",
        "summary": {"latency": {"p99_ms": 50}},
        "samples": [],
    }
    with open(results_file, "w") as f:
        json.dump(data, f)
    loaded = app_module.load_json_data(str(results_file))
    assert loaded is not None
    assert loaded.get("collector_name") == "SyntheticCollector"
    assert "summary" in loaded


@pytest.mark.skipif(
    not CHARTS_AVAILABLE,
    reason="trackiq.reporting.charts (pandas/plotly) not available",
)
def test_html_report_with_chartjs_from_benchmark(tmp_path):
    """Run benchmark, build df/summary, add Chart.js charts via add_charts_to_html_report, generate HTML."""
    config = {"warmup_samples": 2, "seed": 42}
    collector = SyntheticCollector(config=config)
    runner = BenchmarkRunner(
        collector,
        duration_seconds=0.4,
        sample_interval_seconds=0.05,
        quiet=True,
    )
    export = runner.run()
    data = export.to_dict()

    samples = data.get("samples", [])
    summary = data.get("summary") or {}
    assert samples, "benchmark should produce samples"

    df = shared_charts.samples_to_dataframe(samples)
    if "latency_ms" in df.columns and "throughput_fps" not in df.columns:
        import numpy as np

        df["throughput_fps"] = 1000.0 / df["latency_ms"].replace(0, np.nan)

    report = HTMLReportGenerator(
        title="Benchmark Chart.js Report",
        author="Integration Test",
        theme="light",
    )
    report.add_metadata("Collector", data.get("collector_name", "Unknown"))
    report.add_summary_item("Samples", len(samples), "", "neutral")
    if summary.get("latency"):
        report.add_summary_item(
            "P99 Latency",
            f"{summary['latency'].get('p99_ms', 0):.2f}",
            "ms",
            "neutral",
        )

    shared_charts.add_charts_to_html_report(report, df, summary, chart_engine="chartjs")

    html_path = tmp_path / "benchmark_chartjs_report.html"
    out = report.generate_html(str(html_path))

    assert Path(out).exists()
    content = Path(out).read_text(encoding="utf-8")
    assert "Benchmark Chart.js Report" in content
    assert "chart.js" in content.lower() or "jsdelivr" in content.lower()
    assert "latency" in content.lower() or "summary-statistics" in content


def test_autoperfpy_power_profiler_integration_full_session(monkeypatch):
    """AutoPerfPy run should populate canonical TrackiqResult power metrics."""
    monkeypatch.setattr(
        "autoperfpy.auto_runner.detect_power_source",
        lambda: SimulatedPowerReader(tdp_watts=120.0),
    )
    device = DeviceProfile(
        device_id="cpu_0",
        device_type=DEVICE_TYPE_CPU,
        device_name="Test CPU",
        index=0,
    )
    config = InferenceConfig(
        precision="fp32",
        batch_size=2,
        accelerator="cpu_0",
        warmup_runs=1,
        iterations=10,
    )
    payload = run_single_benchmark(
        device=device,
        config=config,
        duration_seconds=0.25,
        sample_interval_seconds=0.05,
        quiet=True,
        enable_power=True,
    )
    result = _infer_trackiq_result(payload)
    assert result.metrics.power_consumption_watts is not None
    assert result.metrics.performance_per_watt is not None


def test_infer_trackiq_result_backfills_llm_metrics_from_payload() -> None:
    """LLM benchmark payloads should populate canonical LLM metric fields."""
    payload = {
        "run_label": "llm_latency",
        "ttft_p50": 820.0,
        "ttft_p95": 990.0,
        "ttft_p99": 1100.0,
        "tpt_p50": 35.0,
        "throughput_tokens_per_sec": 28.2,
    }
    result = _infer_trackiq_result(payload)
    assert result.metrics.ttft_ms == 820.0
    assert result.metrics.tokens_per_sec == 28.2
    assert result.metrics.decode_tpt_ms == 35.0
