"""Tests for efficiency metrics module."""

import pytest
import tempfile
import numpy as np
from pathlib import Path

from autoperfpy.core import (
    EfficiencyMetrics,
    EfficiencyCalculator,
    BatchEfficiencyAnalyzer,
    LatencyStats,
)
from autoperfpy.analyzers import EfficiencyAnalyzer


class TestEfficiencyCalculator:
    """Tests for EfficiencyCalculator class."""

    def test_calculate_perf_per_watt(self):
        """Test performance per watt calculation."""
        calculator = EfficiencyCalculator()

        # 100 images/sec at 200W = 0.5 images/W
        result = calculator.calculate_perf_per_watt(
            throughput=100.0,
            power_watts=200.0,
        )
        assert result == pytest.approx(0.5)

    def test_calculate_perf_per_watt_zero_power(self):
        """Test perf/watt with zero power returns zero."""
        calculator = EfficiencyCalculator()
        result = calculator.calculate_perf_per_watt(throughput=100.0, power_watts=0.0)
        assert result == 0.0

    def test_calculate_energy_per_inference(self):
        """Test energy per inference calculation."""
        calculator = EfficiencyCalculator()

        # 10ms latency at 100W = 1 Joule
        result = calculator.calculate_energy_per_inference(
            latency_ms=10.0,
            power_watts=100.0,
        )

        assert result["joules"] == pytest.approx(1.0)
        assert result["kwh"] == pytest.approx(1.0 / 3_600_000)

    def test_calculate_cost_per_inference(self):
        """Test cost calculation."""
        calculator = EfficiencyCalculator(electricity_rate=0.15)

        # 1 kWh at $0.15/kWh = $0.15
        result = calculator.calculate_cost_per_inference(energy_kwh=1.0)

        assert result["per_inference_usd"] == pytest.approx(0.15)
        assert result["per_1k_inferences_usd"] == pytest.approx(150.0)

    def test_calculate_power_efficiency(self):
        """Test power efficiency percentage calculation."""
        calculator = EfficiencyCalculator(gpu_tdp=400.0)

        # 300W actual vs 400W TDP = 75%
        result = calculator.calculate_power_efficiency(actual_power=300.0)
        assert result == pytest.approx(75.0)

    def test_calculate_efficiency_metrics_comprehensive(self):
        """Test comprehensive efficiency metrics calculation."""
        calculator = EfficiencyCalculator(
            electricity_rate=0.12,
            gpu_tdp=350.0,
        )

        metrics = calculator.calculate_efficiency_metrics(
            throughput=50.0,  # 50 images/sec
            latency_ms=20.0,  # 20ms per image
            power_samples=[250.0, 260.0, 255.0, 248.0],  # Power readings
            throughput_unit="images/sec",
            include_cost=True,
        )

        assert isinstance(metrics, EfficiencyMetrics)
        assert metrics.throughput == 50.0
        assert metrics.throughput_unit == "images/sec"
        assert metrics.avg_power_watts == pytest.approx(253.25)
        assert metrics.peak_power_watts == 260.0
        assert metrics.perf_per_watt == pytest.approx(50.0 / 253.25)
        assert metrics.energy_per_inference_joules == pytest.approx(253.25 * 0.020)
        assert metrics.cost_per_inference_usd is not None

    def test_compare_efficiency(self):
        """Test efficiency comparison between configurations."""
        calculator = EfficiencyCalculator()

        baseline = EfficiencyMetrics(
            perf_per_watt=0.5,
            energy_per_inference_joules=2.0,
            energy_per_inference_kwh=2.0 / 3_600_000,
            throughput=100.0,
            throughput_unit="images/sec",
            avg_power_watts=200.0,
            peak_power_watts=220.0,
            power_efficiency_percent=66.7,
        )

        current = EfficiencyMetrics(
            perf_per_watt=0.6,  # 20% improvement
            energy_per_inference_joules=1.67,
            energy_per_inference_kwh=1.67 / 3_600_000,
            throughput=120.0,
            throughput_unit="images/sec",
            avg_power_watts=200.0,
            peak_power_watts=215.0,
            power_efficiency_percent=66.7,
        )

        comparison = calculator.compare_efficiency(baseline, current)

        assert comparison["is_more_efficient"] is True
        assert comparison["perf_per_watt_change_percent"] == pytest.approx(20.0)
        assert comparison["throughput_change_percent"] == pytest.approx(20.0)

    def test_electricity_rates_presets(self):
        """Test that electricity rate presets are accessible."""
        calculator = EfficiencyCalculator()

        assert "us_average" in calculator.ELECTRICITY_RATES
        assert "cloud_estimate" in calculator.ELECTRICITY_RATES
        assert calculator.ELECTRICITY_RATES["us_average"] == 0.12

    def test_gpu_tdp_presets(self):
        """Test that GPU TDP presets are accessible."""
        calculator = EfficiencyCalculator()

        assert "nvidia_a100_40gb" in calculator.GPU_TDP
        assert "nvidia_h100_sxm" in calculator.GPU_TDP
        assert calculator.GPU_TDP["nvidia_a100_40gb"] == 400


class TestBatchEfficiencyAnalyzer:
    """Tests for BatchEfficiencyAnalyzer class."""

    def test_analyze_batch_efficiency(self):
        """Test batch efficiency analysis."""
        analyzer = BatchEfficiencyAnalyzer()

        batch_results = [
            {
                "batch_size": 1,
                "throughput": 50.0,
                "latency_ms": 20.0,
                "power_samples": [200.0, 205.0],
            },
            {
                "batch_size": 4,
                "throughput": 150.0,
                "latency_ms": 26.7,
                "power_samples": [280.0, 290.0],
            },
            {
                "batch_size": 8,
                "throughput": 200.0,
                "latency_ms": 40.0,
                "power_samples": [320.0, 330.0],
            },
        ]

        result = analyzer.analyze_batch_efficiency(batch_results)

        assert "batch_efficiencies" in result
        assert len(result["batch_efficiencies"]) == 3
        assert result["optimal_for_efficiency"] is not None
        assert result["optimal_for_throughput"] == 8  # Highest throughput
        assert result["optimal_for_energy"] == 1  # Lowest energy (smallest batch)

    def test_find_pareto_optimal(self):
        """Test Pareto frontier detection."""
        analyzer = BatchEfficiencyAnalyzer()

        # Create batch results where batch_size 4 is Pareto-optimal
        batch_results = [
            {
                "batch_size": 1,
                "throughput": 50.0,
                "latency_ms": 20.0,
                "power_samples": [200.0],
            },
            {
                "batch_size": 4,
                "throughput": 180.0,  # High throughput
                "latency_ms": 22.0,  # Low latency = low energy
                "power_samples": [250.0],
            },
            {
                "batch_size": 8,
                "throughput": 200.0,
                "latency_ms": 40.0,  # Higher latency = higher energy
                "power_samples": [350.0],
            },
        ]

        pareto = analyzer.find_pareto_optimal(batch_results)

        assert isinstance(pareto, list)
        assert len(pareto) > 0
        # Batch 4 should be Pareto-optimal (good throughput, lower energy than 8)
        assert 4 in pareto


class TestLatencyStatsExtended:
    """Tests for extended LatencyStats methods."""

    def test_calculate_jitter(self):
        """Test jitter calculation."""
        # Constant latencies = zero jitter
        latencies = [10.0, 10.0, 10.0, 10.0]
        jitter = LatencyStats.calculate_jitter(latencies)
        assert jitter == 0.0

        # Variable latencies
        latencies = [10.0, 12.0, 8.0, 14.0]
        jitter = LatencyStats.calculate_jitter(latencies)
        # Differences: |12-10|=2, |8-12|=4, |14-8|=6
        assert jitter == pytest.approx(4.0)

    def test_calculate_coefficient_of_variation(self):
        """Test CV calculation."""
        # Low variability
        latencies = [100.0, 101.0, 99.0, 100.0]
        cv = LatencyStats.calculate_coefficient_of_variation(latencies)
        assert cv < 0.1  # Very consistent

        # High variability
        latencies = [50.0, 100.0, 150.0, 200.0]
        cv = LatencyStats.calculate_coefficient_of_variation(latencies)
        assert cv > 0.3  # High variability

    def test_calculate_iqr(self):
        """Test IQR calculation."""
        latencies = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        iqr = LatencyStats.calculate_iqr(latencies)

        # Q1 = 3.25, Q3 = 7.75, IQR = 4.5
        assert iqr == pytest.approx(4.5, rel=0.1)

    def test_count_outliers(self):
        """Test outlier counting."""
        # Create a dataset with clear outliers
        # Use many normal values so the outlier is clearly beyond 3 sigma
        latencies = [10.0] * 20 + [100.0]  # 100 is a clear outlier
        outliers = LatencyStats.count_outliers(latencies, sigma=3.0)
        assert outliers >= 1

        # No outliers - all values are similar
        latencies = [10.0, 10.5, 9.5, 10.2, 9.8]
        outliers = LatencyStats.count_outliers(latencies, sigma=3.0)
        assert outliers == 0

        # Use many normal values to stabilize mean/std, then add extreme outlier
        # With 50 values of 10.0 plus one 500.0:
        # Mean ≈ 19.6, Std ≈ 68.5, so 2σ upper = 156.6
        # 500 is clearly beyond that
        latencies = [10.0] * 50 + [500.0]
        outliers = LatencyStats.count_outliers(latencies, sigma=2.0)
        assert outliers >= 1

    def test_calculate_extended_stats(self):
        """Test extended statistics calculation."""
        latencies = [10.0, 12.0, 11.0, 13.0, 10.5]
        power_samples = [200.0, 210.0, 205.0]

        stats = LatencyStats.calculate_extended_stats(
            latencies,
            power_samples=power_samples,
        )

        # Basic stats
        assert "p50" in stats
        assert "p95" in stats
        assert "p99" in stats
        assert "mean" in stats

        # Variability stats
        assert "jitter" in stats
        assert "cv" in stats
        assert "iqr" in stats
        assert "outlier_count" in stats

        # Efficiency stats (only with power data)
        assert "perf_per_watt" in stats
        assert "energy_per_inference_j" in stats
        assert "power_mean_w" in stats
        assert stats["power_mean_w"] == pytest.approx(205.0)

    def test_calculate_extended_stats_no_power(self):
        """Test extended stats without power data."""
        latencies = [10.0, 12.0, 11.0]

        stats = LatencyStats.calculate_extended_stats(latencies)

        # Should have variability stats
        assert "jitter" in stats
        assert "cv" in stats

        # Should NOT have efficiency stats
        assert "perf_per_watt" not in stats
        assert "energy_per_inference_j" not in stats


class TestEfficiencyAnalyzer:
    """Tests for EfficiencyAnalyzer class."""

    @pytest.fixture
    def temp_csv_with_power(self):
        """Create a temporary CSV file with power data (close before yield for Windows)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("workload,batch_size,latency_ms,power_w,throughput\n")
            f.write("ResNet50,1,20.0,200.0,50.0\n")
            f.write("ResNet50,1,21.0,205.0,47.6\n")
            f.write("ResNet50,1,19.5,198.0,51.3\n")
            f.write("ResNet50,4,25.0,280.0,160.0\n")
            f.write("ResNet50,4,26.0,285.0,153.8\n")
            f.write("ResNet50,4,24.5,275.0,163.3\n")
            f.write("YOLO,1,15.0,180.0,66.7\n")
            f.write("YOLO,1,16.0,185.0,62.5\n")
            f.flush()
            name = f.name
        try:
            yield name
        finally:
            Path(name).unlink(missing_ok=True)

    def test_analyze_efficiency(self, temp_csv_with_power):
        """Test efficiency analysis from CSV."""
        analyzer = EfficiencyAnalyzer()

        result = analyzer.analyze(
            temp_csv_with_power,
            throughput_col="throughput",
            latency_col="latency_ms",
            power_col="power_w",
            group_by="workload",
        )

        assert result.name == "Efficiency Analysis"
        assert "ResNet50" in result.metrics
        assert "YOLO" in result.metrics

        resnet_metrics = result.metrics["ResNet50"]
        assert "perf_per_watt" in resnet_metrics
        assert "energy_per_inference_joules" in resnet_metrics
        assert resnet_metrics["perf_per_watt"] > 0

    def test_analyze_batch_efficiency(self, temp_csv_with_power):
        """Test batch efficiency analysis."""
        analyzer = EfficiencyAnalyzer()

        result = analyzer.analyze_batch_efficiency(
            temp_csv_with_power,
            batch_col="batch_size",
            latency_col="latency_ms",
            power_col="power_w",
            throughput_col="throughput",
        )

        assert result.name == "Batch Efficiency Analysis"
        assert "batch_analysis" in result.metrics
        assert "pareto_optimal_batches" in result.metrics
        assert "recommendation" in result.metrics

    def test_compare_configurations(self):
        """Test configuration comparison."""
        analyzer = EfficiencyAnalyzer()

        baseline = {
            "throughput": 100.0,
            "latency_ms": 10.0,
            "power_w": 200.0,
        }

        current = {
            "throughput": 120.0,
            "latency_ms": 8.3,
            "power_w": 210.0,
        }

        comparison = analyzer.compare_configurations(baseline, current)

        assert "is_more_efficient" in comparison
        assert "efficiency_improvement_percent" in comparison
        assert comparison["throughput_change_percent"] == pytest.approx(20.0)

    def test_summarize(self, temp_csv_with_power):
        """Test analysis summary."""
        analyzer = EfficiencyAnalyzer()
        analyzer.analyze(temp_csv_with_power)

        summary = analyzer.summarize()

        assert summary["total_analyses"] == 1
        assert "best_perf_per_watt" in summary
        assert "configurations_analyzed" in summary


class TestEfficiencyMetricsDataclass:
    """Tests for EfficiencyMetrics dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = EfficiencyMetrics(
            perf_per_watt=0.5,
            energy_per_inference_joules=2.0,
            energy_per_inference_kwh=2.0 / 3_600_000,
            throughput=100.0,
            throughput_unit="images/sec",
            avg_power_watts=200.0,
            peak_power_watts=220.0,
            power_efficiency_percent=66.7,
            cost_per_inference_usd=0.0001,
            cost_per_1k_inferences_usd=0.1,
            electricity_rate_per_kwh=0.12,
        )

        d = metrics.to_dict()

        assert isinstance(d, dict)
        assert d["perf_per_watt"] == 0.5
        assert d["throughput"] == 100.0
        assert d["cost_per_inference_usd"] == 0.0001

    def test_optional_cost_fields(self):
        """Test that cost fields are optional."""
        metrics = EfficiencyMetrics(
            perf_per_watt=0.5,
            energy_per_inference_joules=2.0,
            energy_per_inference_kwh=2.0 / 3_600_000,
            throughput=100.0,
            throughput_unit="images/sec",
            avg_power_watts=200.0,
            peak_power_watts=220.0,
            power_efficiency_percent=66.7,
            # No cost fields provided
        )

        assert metrics.cost_per_inference_usd is None
        assert metrics.cost_per_1k_inferences_usd is None
