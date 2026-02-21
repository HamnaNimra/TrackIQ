"""Contract tests for real tool output fixtures across the TrackIQ stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trackiq_compare.cli import run_compare
from trackiq_compare.comparator import MetricComparator
from trackiq_core.serializer import load_trackiq_result
from trackiq_core.validator import validate_trackiq_result, validate_trackiq_result_obj


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "tool_outputs"
AUTOPERFPY_FIXTURE = FIXTURE_DIR / "autoperfpy_real_output.json"
MINICLUSTER_FIXTURE = FIXTURE_DIR / "minicluster_real_output.json"


def _load_fixture_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_real_autoperfpy_output_fixture_contract() -> None:
    """Real autoperfpy output fixture should satisfy canonical TrackiqResult contract."""
    payload = _load_fixture_payload(AUTOPERFPY_FIXTURE)
    validate_trackiq_result(payload)
    result = load_trackiq_result(AUTOPERFPY_FIXTURE)
    validate_trackiq_result_obj(result)
    assert result.tool_name == "autoperfpy"
    assert result.workload.workload_type == "inference"


def test_real_minicluster_output_fixture_contract() -> None:
    """Real minicluster output fixture should satisfy canonical TrackiqResult contract."""
    payload = _load_fixture_payload(MINICLUSTER_FIXTURE)
    validate_trackiq_result(payload)
    result = load_trackiq_result(MINICLUSTER_FIXTURE)
    validate_trackiq_result_obj(result)
    assert result.tool_name == "minicluster"
    assert result.workload.workload_type == "training"


def test_real_fixtures_are_comparable_by_trackiq_compare() -> None:
    """trackiq-compare comparator should consume real tool fixtures without adaptation."""
    autoperfpy_result = load_trackiq_result(AUTOPERFPY_FIXTURE)
    minicluster_result = load_trackiq_result(MINICLUSTER_FIXTURE)
    comparison = MetricComparator("autoperfpy", "minicluster").compare(
        autoperfpy_result, minicluster_result
    )
    assert "throughput_samples_per_sec" in comparison.metrics
    assert "latency_p99_ms" in comparison.metrics


def test_trackiq_compare_run_with_real_fixtures_generates_html() -> None:
    """trackiq-compare run should produce a valid HTML report from real fixtures."""
    html_path = Path("output") / "fixture_compare_report_contract_test.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(
        result_a=str(AUTOPERFPY_FIXTURE),
        result_b=str(MINICLUSTER_FIXTURE),
        html=str(html_path),
        label_a="autoperfpy-real",
        label_b="minicluster-real",
        tolerance=0.5,
        regression_threshold=5.0,
    )
    code = run_compare(args)
    assert code == 0
    assert html_path.exists()
    html = html_path.read_text(encoding="utf-8")
    assert "TrackIQ Comparison Report" in html
    assert "autoperfpy-real" in html
    assert "minicluster-real" in html
    try:
        html_path.unlink()
    except OSError:
        pass
