"""Tests for AutoPerfPy analyzers module."""

import pytest
import tempfile
from pathlib import Path

from autoperfpy.analyzers import PercentileLatencyAnalyzer, LogAnalyzer


class TestPercentileLatencyAnalyzer:
    """Tests for PercentileLatencyAnalyzer."""

    @pytest.fixture
    def sample_csv(self):
        """Create a sample CSV file for testing (close before yield for Windows)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("timestamp,workload,batch_size,latency_ms,power_w\n")
            f.write("2024-01-01 10:00:00,inference,1,25.5,15.2\n")
            f.write("2024-01-01 10:00:01,inference,1,26.3,15.3\n")
            f.write("2024-01-01 10:00:02,inference,1,24.8,15.1\n")
            f.write("2024-01-01 10:00:03,inference,4,10.5,18.2\n")
            f.write("2024-01-01 10:00:04,inference,4,11.2,18.5\n")
            f.write("2024-01-01 10:00:05,inference,4,10.8,18.3\n")
            f.flush()
            name = f.name
        try:
            yield name
        finally:
            Path(name).unlink(missing_ok=True)

    def test_analyze_valid_csv(self, sample_csv):
        """Test analyzing valid CSV file."""
        analyzer = PercentileLatencyAnalyzer()
        result = analyzer.analyze(sample_csv)

        assert result.name == "Percentile Latency Analysis"
        assert result.metrics is not None
        assert len(result.metrics) > 0

    def test_analyze_metrics_content(self, sample_csv):
        """Test that analysis produces expected metrics."""
        analyzer = PercentileLatencyAnalyzer()
        result = analyzer.analyze(sample_csv)

        # Check that both workload-batch combinations exist
        assert any("inference_batch1" in key for key in result.metrics.keys())
        assert any("inference_batch4" in key for key in result.metrics.keys())

        # Check that percentiles are calculated
        for key, metrics in result.metrics.items():
            assert "p50" in metrics
            assert "p95" in metrics
            assert "p99" in metrics
            assert "mean" in metrics
            assert "std" in metrics
            assert "num_samples" in metrics

    def test_analyze_stores_results(self, sample_csv):
        """Test that results are stored in analyzer."""
        analyzer = PercentileLatencyAnalyzer()
        analyzer.analyze(sample_csv)
        analyzer.analyze(sample_csv)

        assert len(analyzer.results) == 2

    def test_summarize(self, sample_csv):
        """Test summary generation."""
        analyzer = PercentileLatencyAnalyzer()
        analyzer.analyze(sample_csv)

        summary = analyzer.summarize()
        assert "total_analyses" in summary
        assert summary["total_analyses"] == 1
        assert "workloads_analyzed" in summary
        assert len(summary["workloads_analyzed"]) > 0

    def test_analyze_missing_file(self):
        """Test error handling for missing file."""
        analyzer = PercentileLatencyAnalyzer()
        with pytest.raises(FileNotFoundError):
            analyzer.analyze("/nonexistent/file.csv")

    def test_analyze_invalid_csv(self):
        """Test error handling for invalid CSV format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("invalid,csv,format\n")
            f.write("missing,required,columns\n")
            f.flush()
            name = f.name
        try:
            analyzer = PercentileLatencyAnalyzer()
            with pytest.raises(ValueError):
                analyzer.analyze(name)
        finally:
            Path(name).unlink(missing_ok=True)

    def test_analyze_missing_columns(self):
        """Test error handling for missing required columns."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("timestamp,workload\n")  # Missing batch_size, latency_ms, power_w
            f.write("2024-01-01,inference\n")
            f.flush()
            name = f.name
        try:
            analyzer = PercentileLatencyAnalyzer()
            with pytest.raises(ValueError, match="Missing required columns"):
                analyzer.analyze(name)
        finally:
            Path(name).unlink(missing_ok=True)


class TestLogAnalyzer:
    """Tests for LogAnalyzer."""

    @pytest.fixture
    def sample_log(self):
        """Create a sample log file for testing (close before yield for Windows)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("[2024-01-01 10:00:00] Frame 1: E2E: 25.5ms\n")
            f.write("[2024-01-01 10:00:01] Frame 2: E2E: 26.3ms\n")
            f.write("[2024-01-01 10:00:02] Frame 3: E2E: 75.5ms\n")  # spike
            f.write("[2024-01-01 10:00:03] Frame 4: E2E: 26.8ms\n")
            f.write("[2024-01-01 10:00:04] Frame 5: E2E: 120.0ms\n")  # spike
            f.flush()
            name = f.name
        try:
            yield name
        finally:
            Path(name).unlink(missing_ok=True)

    def test_analyze_valid_log(self, sample_log):
        """Test analyzing valid log file."""
        analyzer = LogAnalyzer()
        result = analyzer.analyze(sample_log, threshold_ms=50.0)

        assert result is not None

    def test_analyze_detects_spikes(self, sample_log):
        """Test that spike detection works."""
        analyzer = LogAnalyzer()
        result = analyzer.analyze(sample_log, threshold_ms=50.0)

        # Should detect spikes above 50ms
        metrics = result.metrics
        assert "spike_events" in metrics
        assert metrics["spike_events"] >= 2  # 75.5ms and 120.0ms

    def test_analyze_threshold_parameter(self, sample_log):
        """Test that threshold parameter affects results."""
        analyzer = LogAnalyzer()

        result_50 = analyzer.analyze(sample_log, threshold_ms=50.0)
        result_100 = analyzer.analyze(sample_log, threshold_ms=100.0)

        spikes_50 = result_50.metrics.get("spikes_detected", 0)
        spikes_100 = result_100.metrics.get("spikes_detected", 0)

        # Should detect fewer spikes with higher threshold
        assert spikes_100 <= spikes_50

    def test_analyze_missing_file(self):
        """Test error handling for missing log file."""
        analyzer = LogAnalyzer()
        with pytest.raises(FileNotFoundError):
            analyzer.analyze("/nonexistent/log.txt")
