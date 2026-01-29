"""Integration test: full benchmark -> JSON -> HTML -> Streamlit UI loads."""

import json
import sys
from pathlib import Path

import pytest

# Ensure project root on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trackiq.collectors import SyntheticCollector
from trackiq.runner import BenchmarkRunner
from trackiq.reporting import HTMLReportGenerator, PerformanceVisualizer


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
