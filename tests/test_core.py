"""Tests for AutoPerfPy core module."""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from autoperfpy.core import (
    DataLoader,
    LatencyStats,
    PerformanceComparator,
    RegressionDetector,
    RegressionThreshold,
)


class TestDataLoader:
    """Tests for DataLoader utility."""

    def test_load_csv_valid_file(self):
        """Test loading a valid CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("timestamp,workload,latency_ms\n")
            f.write("2024-01-01,inference,25.5\n")
            f.write("2024-01-01,inference,26.3\n")
            f.flush()
            
            df = DataLoader.load_csv(f.name)
            assert len(df) == 2
            assert list(df.columns) == ["timestamp", "workload", "latency_ms"]
            
            Path(f.name).unlink()

    def test_load_csv_file_not_found(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            DataLoader.load_csv("/nonexistent/path/file.csv")

    def test_validate_columns_success(self):
        """Test validation passes with all required columns."""
        df = pd.DataFrame({
            "timestamp": [1, 2],
            "workload": ["a", "b"],
            "latency_ms": [25, 26]
        })
        assert DataLoader.validate_columns(df, ["timestamp", "workload"]) is True

    def test_validate_columns_missing(self):
        """Test validation fails with missing columns."""
        df = pd.DataFrame({"timestamp": [1, 2]})
        with pytest.raises(ValueError, match="Missing required columns"):
            DataLoader.validate_columns(df, ["timestamp", "workload"])


class TestLatencyStats:
    """Tests for LatencyStats calculations."""

    def test_calculate_percentiles_basic(self):
        """Test basic percentile calculations."""
        latencies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        stats = LatencyStats.calculate_percentiles(latencies)
        
        assert "p50" in stats
        assert "p95" in stats
        assert "p99" in stats
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        
        assert stats["min"] == 10
        assert stats["max"] == 100
        assert stats["mean"] == 55
        assert stats["p50"] == 55

    def test_calculate_percentiles_empty(self):
        """Test percentiles with empty list."""
        stats = LatencyStats.calculate_percentiles([])
        assert stats == {}

    def test_calculate_percentiles_single_value(self):
        """Test percentiles with single value."""
        stats = LatencyStats.calculate_percentiles([42.5])
        assert stats["p50"] == 42.5
        assert stats["p99"] == 42.5
        assert stats["mean"] == 42.5

    def test_calculate_for_groups(self):
        """Test calculating statistics per group."""
        df = pd.DataFrame({
            "latency_ms": [10, 20, 30, 40, 50],
            "batch_size": [1, 1, 4, 4, 8]
        })
        
        stats = LatencyStats.calculate_for_groups(df, "latency_ms", "batch_size")
        
        assert 1 in stats
        assert 4 in stats
        assert 8 in stats
        assert stats[1]["mean"] == 15
        assert stats[8]["mean"] == 50


class TestPerformanceComparator:
    """Tests for PerformanceComparator."""

    def test_compare_latency_throughput(self):
        """Test latency vs throughput comparison."""
        batch_sizes = [1, 2, 4, 8]
        latencies = [10.0, 12.0, 11.0, 15.0]
        throughput = [100.0, 83.3, 90.9, 66.7]
        
        result = PerformanceComparator.compare_latency_throughput(
            batch_sizes, latencies, throughput
        )
        
        assert "optimal_batch_size" in result
        assert result["total_compared"] == 4


class TestRegressionDetector:
    """Tests for regression detection."""

    def test_save_and_load_baseline(self):
        """Test saving and loading baselines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            detector = RegressionDetector(baseline_dir=tmpdir)
            
            metrics = {
                "p99_latency": 50.0,
                "p95_latency": 40.0,
                "throughput_imgs_per_sec": 100.0
            }
            
            detector.save_baseline("main", metrics)
            loaded = detector.load_baseline("main")
            
            assert loaded == metrics

    def test_list_baselines(self):
        """Test listing available baselines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            detector = RegressionDetector(baseline_dir=tmpdir)
            
            detector.save_baseline("main", {"p99": 50.0})
            detector.save_baseline("v1.0", {"p99": 48.0})
            
            baselines = detector.list_baselines()
            assert "main" in baselines
            assert "v1.0" in baselines
            assert len(baselines) == 2

    def test_compare_metrics_latency_regression(self):
        """Test detection of latency regression."""
        baseline = {"p99_latency": 50.0, "mean_latency": 25.0}
        current = {"p99_latency": 57.0, "mean_latency": 26.0}  # 14% increase, 4% increase
        thresholds = RegressionThreshold(p99_percent=10.0, latency_percent=5.0)
        
        detector = RegressionDetector()
        comparisons = detector.compare_metrics(baseline, current, thresholds)
        
        assert comparisons["p99_latency"].is_regression is True  # 14% > 10%
        assert comparisons["mean_latency"].is_regression is False  # 4% < 5%

    def test_compare_metrics_throughput_regression(self):
        """Test detection of throughput regression."""
        baseline = {"throughput_imgs_per_sec": 1000.0}
        current = {"throughput_imgs_per_sec": 900.0}  # 10% decrease
        thresholds = RegressionThreshold(throughput_percent=5.0)
        
        detector = RegressionDetector()
        comparisons = detector.compare_metrics(baseline, current, thresholds)
        
        assert comparisons["throughput_imgs_per_sec"].is_regression is True

    def test_detect_regressions_full_workflow(self):
        """Test complete regression detection workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            detector = RegressionDetector(baseline_dir=tmpdir)
            
            baseline_metrics = {
                "p99_latency": 50.0,
                "p95_latency": 45.0,
                "throughput": 1000.0
            }
            detector.save_baseline("main", baseline_metrics)
            
            current_metrics = {
                "p99_latency": 56.0,  # 12% increase - regression
                "p95_latency": 46.0,  # 2% increase - no regression
                "throughput": 950.0   # 5% decrease - no regression (at threshold)
            }
            
            result = detector.detect_regressions(
                "main", 
                current_metrics,
                RegressionThreshold(p99_percent=10.0, throughput_percent=5.0)
            )
            
            assert result["has_regressions"] is True
            assert "p99_latency" in result["regressions"]
            assert "p95_latency" not in result["regressions"]

    def test_generate_report(self):
        """Test report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            detector = RegressionDetector(baseline_dir=tmpdir)
            
            baseline_metrics = {"p99_latency": 50.0}
            detector.save_baseline("main", baseline_metrics)
            
            current_metrics = {"p99_latency": 56.0}
            report = detector.generate_report(
                "main",
                current_metrics,
                RegressionThreshold(p99_percent=10.0)
            )
            
            assert "PERFORMANCE REGRESSION REPORT" in report
            assert "REGRESSIONS DETECTED" in report
            assert "p99_latency" in report
            assert "+12.00%" in report


class TestRegressionThreshold:
    """Tests for RegressionThreshold configuration."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = RegressionThreshold()
        assert thresholds.latency_percent == 5.0
        assert thresholds.throughput_percent == 5.0
        assert thresholds.p99_percent == 10.0

    def test_custom_thresholds(self):
        """Test custom threshold values."""
        thresholds = RegressionThreshold(
            latency_percent=2.0,
            throughput_percent=3.0,
            p99_percent=15.0
        )
        assert thresholds.latency_percent == 2.0
        assert thresholds.throughput_percent == 3.0
        assert thresholds.p99_percent == 15.0
