"""Tests for variability metrics analyzer."""

import pytest
import tempfile
from pathlib import Path

from autoperfpy.analyzers import VariabilityAnalyzer
from autoperfpy.core import LatencyStats


class TestVariabilityAnalyzer:
    """Tests for VariabilityAnalyzer class."""

    @pytest.fixture
    def temp_csv_consistent(self):
        """Create a temporary CSV file with consistent latencies (close before yield for Windows)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("workload,batch_size,latency_ms\n")
            for i in range(20):
                f.write(f"ResNet50,1,{10.0 + (i % 3) * 0.1}\n")
            f.flush()
            name = f.name
        try:
            yield name
        finally:
            Path(name).unlink(missing_ok=True)

    @pytest.fixture
    def temp_csv_variable(self):
        """Create a temporary CSV file with variable latencies (close before yield for Windows)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("workload,batch_size,latency_ms\n")
            latencies = [10.0, 25.0, 15.0, 50.0, 8.0, 35.0, 12.0, 45.0, 20.0, 30.0]
            for lat in latencies:
                f.write(f"YOLO,1,{lat}\n")
            f.flush()
            name = f.name
        try:
            yield name
        finally:
            Path(name).unlink(missing_ok=True)

    @pytest.fixture
    def temp_csv_with_groups(self):
        """Create a temporary CSV file with multiple workload groups (close before yield for Windows)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("workload,batch_size,latency_ms\n")
            for i in range(10):
                f.write(f"ResNet50,1,{10.0 + (i % 2) * 0.5}\n")
            variable_lats = [15.0, 25.0, 10.0, 40.0, 18.0]
            for lat in variable_lats:
                f.write(f"YOLO,1,{lat}\n")
            f.flush()
            name = f.name
        try:
            yield name
        finally:
            Path(name).unlink(missing_ok=True)

    @pytest.fixture
    def temp_csv_with_outliers(self):
        """Create a CSV file with clear outliers (close before yield for Windows)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("workload,latency_ms\n")
            for _ in range(50):
                f.write("test,10.0\n")
            f.write("test,100.0\n")
            f.write("test,150.0\n")
            f.flush()
            name = f.name
        try:
            yield name
        finally:
            Path(name).unlink(missing_ok=True)

    def test_analyze_consistent_data(self, temp_csv_consistent):
        """Test analysis of consistent data produces low CV."""
        analyzer = VariabilityAnalyzer()
        result = analyzer.analyze(temp_csv_consistent)

        assert result.name == "Variability Analysis"
        assert "all" in result.metrics

        metrics = result.metrics["all"]
        assert metrics["cv"] < 0.1  # Very consistent
        assert metrics["consistency_rating"] == "very_consistent"
        assert metrics["num_samples"] == 20

    def test_analyze_variable_data(self, temp_csv_variable):
        """Test analysis of variable data produces high CV."""
        analyzer = VariabilityAnalyzer()
        result = analyzer.analyze(temp_csv_variable)

        metrics = result.metrics["all"]
        assert metrics["cv"] > 0.3  # High variability
        assert metrics["consistency_rating"] == "high_variability"

    def test_analyze_with_grouping(self, temp_csv_with_groups):
        """Test analysis with group_by parameter."""
        analyzer = VariabilityAnalyzer()
        result = analyzer.analyze(temp_csv_with_groups, group_by="workload")

        assert "ResNet50" in result.metrics
        assert "YOLO" in result.metrics

        # ResNet50 should be more consistent
        resnet_cv = result.metrics["ResNet50"]["cv"]
        yolo_cv = result.metrics["YOLO"]["cv"]
        assert resnet_cv < yolo_cv

    def test_analyze_jitter_calculation(self, temp_csv_variable):
        """Test that jitter is calculated correctly."""
        analyzer = VariabilityAnalyzer()
        result = analyzer.analyze(temp_csv_variable)

        metrics = result.metrics["all"]
        assert "jitter" in metrics
        assert metrics["jitter"] > 0  # Variable data should have positive jitter

    def test_analyze_iqr_calculation(self, temp_csv_variable):
        """Test that IQR is calculated."""
        analyzer = VariabilityAnalyzer()
        result = analyzer.analyze(temp_csv_variable)

        metrics = result.metrics["all"]
        assert "iqr" in metrics
        assert metrics["iqr"] > 0

    def test_analyze_outlier_detection(self, temp_csv_with_outliers):
        """Test outlier detection with clear outliers."""
        analyzer = VariabilityAnalyzer()
        result = analyzer.analyze(temp_csv_with_outliers)

        metrics = result.metrics["all"]
        assert metrics["outlier_count"] >= 1
        assert metrics["outlier_percentage"] > 0

    def test_analyze_custom_sigma(self, temp_csv_with_outliers):
        """Test outlier detection with custom sigma threshold."""
        # Lower sigma should catch more outliers
        analyzer_strict = VariabilityAnalyzer(config={"sigma": 2.0})
        result_strict = analyzer_strict.analyze(temp_csv_with_outliers)

        # Higher sigma catches fewer
        analyzer_loose = VariabilityAnalyzer(config={"sigma": 4.0})
        result_loose = analyzer_loose.analyze(temp_csv_with_outliers)

        strict_outliers = result_strict.metrics["all"]["outlier_count"]
        loose_outliers = result_loose.metrics["all"]["outlier_count"]

        assert strict_outliers >= loose_outliers

    def test_analyze_from_list(self):
        """Test analysis from a list of latencies."""
        analyzer = VariabilityAnalyzer()
        latencies = [10.0, 11.0, 9.5, 10.5, 10.2]

        result = analyzer.analyze_from_list(latencies, name="test_workload")

        assert "test_workload" in result.metrics
        metrics = result.metrics["test_workload"]
        assert "jitter" in metrics
        assert "cv" in metrics
        assert "iqr" in metrics
        assert metrics["num_samples"] == 5

    def test_analyze_empty_list(self):
        """Test analysis of empty list returns zero metrics."""
        analyzer = VariabilityAnalyzer()
        result = analyzer.analyze_from_list([], name="empty")

        metrics = result.metrics["empty"]
        assert metrics["jitter"] == 0.0
        assert metrics["cv"] == 0.0
        assert metrics["outlier_count"] == 0
        assert metrics["consistency_rating"] == "insufficient_data"

    def test_summarize(self, temp_csv_with_groups):
        """Test summary generation across analyses."""
        analyzer = VariabilityAnalyzer()
        analyzer.analyze(temp_csv_with_groups, group_by="workload")

        summary = analyzer.summarize()

        assert summary["total_analyses"] == 1
        assert "most_consistent" in summary
        assert "most_variable" in summary
        assert "groups_analyzed" in summary
        assert len(summary["groups_analyzed"]) == 2

    def test_summarize_empty(self):
        """Test summary with no analyses."""
        analyzer = VariabilityAnalyzer()
        summary = analyzer.summarize()

        assert summary["total_analyses"] == 0

    def test_compare_variability(self):
        """Test variability comparison between configurations."""
        analyzer = VariabilityAnalyzer()

        baseline = {
            "cv": 0.3,
            "jitter": 5.0,
            "outlier_count": 10,
        }

        improved = {
            "cv": 0.1,
            "jitter": 2.0,
            "outlier_count": 2,
        }

        comparison = analyzer.compare_variability(baseline, improved)

        assert comparison["is_more_consistent"] is True
        assert comparison["is_less_jittery"] is True
        assert comparison["has_fewer_outliers"] is True
        assert comparison["overall_improvement"] is True
        assert comparison["cv_change_percent"] < 0  # Negative = improvement

    def test_compare_variability_regression(self):
        """Test comparison detects variability regression."""
        analyzer = VariabilityAnalyzer()

        baseline = {
            "cv": 0.1,
            "jitter": 2.0,
            "outlier_count": 2,
        }

        worse = {
            "cv": 0.4,
            "jitter": 8.0,
            "outlier_count": 15,
        }

        comparison = analyzer.compare_variability(baseline, worse)

        assert comparison["is_more_consistent"] is False
        assert comparison["is_less_jittery"] is False
        assert comparison["overall_improvement"] is False
        assert comparison["cv_change_percent"] > 0  # Positive = regression

    def test_metrics_include_percentiles(self, temp_csv_consistent):
        """Test that basic percentile stats are also included."""
        analyzer = VariabilityAnalyzer()
        result = analyzer.analyze(temp_csv_consistent)

        metrics = result.metrics["all"]
        assert "p50" in metrics
        assert "p95" in metrics
        assert "p99" in metrics
        assert "mean" in metrics
        assert "std" in metrics
        assert "min" in metrics
        assert "max" in metrics

    def test_result_stored_correctly(self, temp_csv_consistent):
        """Test that results are stored in analyzer."""
        analyzer = VariabilityAnalyzer()
        analyzer.analyze(temp_csv_consistent)

        assert len(analyzer.get_results()) == 1

        # Second analysis
        analyzer.analyze(temp_csv_consistent)
        assert len(analyzer.get_results()) == 2

    def test_analyze_missing_file(self):
        """Test error handling for missing file."""
        analyzer = VariabilityAnalyzer()

        with pytest.raises(FileNotFoundError):
            analyzer.analyze("/nonexistent/file.csv")

    def test_analyze_missing_column(self, temp_csv_consistent):
        """Test error handling for missing required column."""
        analyzer = VariabilityAnalyzer()

        with pytest.raises(ValueError, match="Missing required columns"):
            analyzer.analyze(temp_csv_consistent, latency_col="nonexistent_col")

    def test_consistency_rating_thresholds(self):
        """Test that consistency ratings are assigned correctly."""
        analyzer = VariabilityAnalyzer()

        # Very consistent (CV < 0.1)
        result = analyzer.analyze_from_list([10.0, 10.1, 10.0, 9.9, 10.0], "test1")
        assert result.metrics["test1"]["consistency_rating"] == "very_consistent"

        # Consistent (CV 0.1-0.2)
        result = analyzer.analyze_from_list([10.0, 11.0, 9.0, 10.5, 9.5], "test2")
        # CV should be around 0.08, so very_consistent

        # Moderately consistent (CV 0.2-0.3)
        result = analyzer.analyze_from_list([10.0, 13.0, 7.0, 12.0, 8.0], "test3")
        rating = result.metrics["test3"]["consistency_rating"]
        assert rating in ["consistent", "moderately_consistent", "high_variability"]

        # High variability (CV > 0.3)
        result = analyzer.analyze_from_list([5.0, 20.0, 8.0, 25.0, 10.0], "test4")
        assert result.metrics["test4"]["consistency_rating"] == "high_variability"


class TestLatencyStatsVariabilityMethods:
    """Additional tests for LatencyStats variability methods used by the analyzer."""

    def test_jitter_constant_values(self):
        """Test jitter is zero for constant values."""
        latencies = [10.0, 10.0, 10.0, 10.0]
        jitter = LatencyStats.calculate_jitter(latencies)
        assert jitter == 0.0

    def test_jitter_alternating_values(self):
        """Test jitter for alternating pattern."""
        latencies = [10.0, 20.0, 10.0, 20.0]
        jitter = LatencyStats.calculate_jitter(latencies)
        assert jitter == pytest.approx(10.0)

    def test_cv_zero_mean(self):
        """Test CV returns 0 when mean is zero."""
        latencies = [0.0, 0.0, 0.0]
        cv = LatencyStats.calculate_coefficient_of_variation(latencies)
        assert cv == 0.0

    def test_cv_empty_list(self):
        """Test CV returns 0 for empty list."""
        cv = LatencyStats.calculate_coefficient_of_variation([])
        assert cv == 0.0

    def test_iqr_small_dataset(self):
        """Test IQR on small dataset."""
        latencies = [1.0, 2.0, 3.0, 4.0, 5.0]
        iqr = LatencyStats.calculate_iqr(latencies)
        assert iqr > 0

    def test_count_outliers_no_variance(self):
        """Test outlier count when all values are identical."""
        latencies = [10.0, 10.0, 10.0, 10.0]
        outliers = LatencyStats.count_outliers(latencies)
        assert outliers == 0

    def test_count_outliers_single_value(self):
        """Test outlier count with single value."""
        outliers = LatencyStats.count_outliers([10.0])
        assert outliers == 0
