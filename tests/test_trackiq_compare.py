"""Unit tests for trackiq comparison engine (run-to-run comparisons)."""

import pytest

from trackiq_core.utils.compare import (
    RegressionDetector,
    RegressionThreshold,
)


class TestRegressionDetectorCompare:
    """Run-to-run comparisons produce expected results."""

    def test_compare_metrics_latency_increase_is_regression(self):
        baseline = {"p99_latency_ms": 50.0}
        current = {"p99_latency_ms": 56.0}  # 12% increase
        th = RegressionThreshold(p99_percent=10.0)
        detector = RegressionDetector()
        comparisons = detector.compare_metrics(baseline, current, th)
        assert "p99_latency_ms" in comparisons
        assert comparisons["p99_latency_ms"].is_regression is True
        assert comparisons["p99_latency_ms"].percent_change == pytest.approx(
            12.0, rel=0.1
        )

    def test_compare_metrics_throughput_decrease_is_regression(self):
        baseline = {"throughput_fps": 100.0}
        current = {"throughput_fps": 90.0}  # 10% decrease
        th = RegressionThreshold(throughput_percent=5.0)
        detector = RegressionDetector()
        comparisons = detector.compare_metrics(baseline, current, th)
        assert "throughput_fps" in comparisons
        assert comparisons["throughput_fps"].is_regression is True

    def test_compare_metrics_no_regression_when_under_threshold(self):
        baseline = {"p99_latency_ms": 50.0}
        current = {"p99_latency_ms": 52.0}  # 4% increase
        th = RegressionThreshold(latency_percent=5.0)
        detector = RegressionDetector()
        comparisons = detector.compare_metrics(baseline, current, th)
        assert comparisons["p99_latency_ms"].is_regression is False

    def test_detect_regressions_returns_has_regressions_and_details(self, tmp_path):
        detector = RegressionDetector(baseline_dir=str(tmp_path))
        detector.save_baseline("ref", {"p99_ms": 50.0, "throughput": 100.0})
        result = detector.detect_regressions(
            "ref",
            {"p99_ms": 60.0, "throughput": 95.0},
            RegressionThreshold(p99_percent=10.0, throughput_percent=5.0),
        )
        assert "has_regressions" in result
        assert "regressions" in result
        assert "improvements" in result
        assert "comparisons" in result
        assert result["baseline"] == "ref"

    def test_list_baselines(self, tmp_path):
        detector = RegressionDetector(baseline_dir=str(tmp_path))
        detector.save_baseline("b1", {"a": 1})
        detector.save_baseline("b2", {"a": 2})
        names = detector.list_baselines()
        assert "b1" in names
        assert "b2" in names
        assert len(names) == 2
