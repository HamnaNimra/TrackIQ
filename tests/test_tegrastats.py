"""Tests for tegrastats analyzer and parsing utilities."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from autoperfpy.analyzers.tegrastats import TegrastatsAnalyzer
from autoperfpy.core.tegrastats import (
    CPUCoreStats,
    GPUStats,
    MemoryStats,
    TegrastatsCalculator,
    TegrastatsParser,
    ThermalStats,
)

# Sample tegrastats output lines for testing
SAMPLE_TEGRASTATS_LINE = (
    "RAM 264/28409MB (lfb 7004x4MB) CPU [0%@1817,0%@1817,0%@1817,0%@1817,0%@1817,0%@1817,0%@1817,0%@1817] "
    "EMC_FREQ @2133 GR3D_FREQ 0%@1109 APE 245 AUX@30C CPU@31.5C Tdiode@30.75C AO@31C GPU@30.5C tj@41C"
)

SAMPLE_HIGH_LOAD_LINE = (
    "RAM 20000/28409MB (lfb 1000x4MB) CPU [85%@2200,90%@2200,78%@2200,92%@2200,88%@2200,95%@2200,82%@2200,89%@2200] "
    "EMC_FREQ @2133 GR3D_FREQ 95%@1377 APE 245 AUX@45C CPU@78.5C Tdiode@75.25C AO@72C GPU@82.5C tj@85C"
)

SAMPLE_THROTTLING_LINE = (
    "RAM 25000/28409MB (lfb 500x4MB) CPU [70%@1500,72%@1500,68%@1500,75%@1500,71%@1500,69%@1500,73%@1500,74%@1500] "
    "EMC_FREQ @1866 GR3D_FREQ 60%@900 APE 245 AUX@88C CPU@92C Tdiode@90C AO@87C GPU@95C tj@97C"
)


class TestTegrastatsParser:
    """Tests for TegrastatsParser class."""

    def test_parse_line_basic(self):
        """Test parsing a basic tegrastats line."""
        snapshot = TegrastatsParser.parse_line(SAMPLE_TEGRASTATS_LINE)

        assert snapshot is not None
        assert len(snapshot.cpu_cores) == 8
        assert snapshot.gpu.utilization_percent == 0.0
        assert snapshot.gpu.frequency_mhz == 1109

    def test_parse_cpu_cores(self):
        """Test CPU core parsing."""
        snapshot = TegrastatsParser.parse_line(SAMPLE_TEGRASTATS_LINE)

        assert len(snapshot.cpu_cores) == 8
        for core in snapshot.cpu_cores:
            assert core.utilization_percent == 0.0
            assert core.frequency_mhz == 1817

    def test_parse_cpu_high_load(self):
        """Test CPU parsing with high load."""
        snapshot = TegrastatsParser.parse_line(SAMPLE_HIGH_LOAD_LINE)

        assert len(snapshot.cpu_cores) == 8
        assert snapshot.cpu_cores[0].utilization_percent == 85.0
        assert snapshot.cpu_cores[5].utilization_percent == 95.0
        assert snapshot.cpu_avg_utilization == pytest.approx(87.375)

    def test_parse_gpu(self):
        """Test GPU (GR3D) parsing."""
        snapshot = TegrastatsParser.parse_line(SAMPLE_HIGH_LOAD_LINE)

        assert snapshot.gpu.utilization_percent == 95.0
        assert snapshot.gpu.frequency_mhz == 1377

    def test_parse_memory(self):
        """Test RAM parsing."""
        snapshot = TegrastatsParser.parse_line(SAMPLE_TEGRASTATS_LINE)

        assert snapshot.memory.used_mb == 264
        assert snapshot.memory.total_mb == 28409
        assert snapshot.memory.lfb_blocks == 7004
        assert snapshot.memory.lfb_size_mb == 4
        assert snapshot.memory.lfb_total_mb == 28016
        assert snapshot.memory.utilization_percent == pytest.approx(0.929, rel=0.01)

    def test_parse_memory_high_usage(self):
        """Test RAM parsing with high usage."""
        snapshot = TegrastatsParser.parse_line(SAMPLE_HIGH_LOAD_LINE)

        assert snapshot.memory.used_mb == 20000
        assert snapshot.memory.utilization_percent == pytest.approx(70.4, rel=0.1)

    def test_parse_emc_frequency(self):
        """Test EMC frequency parsing."""
        snapshot = TegrastatsParser.parse_line(SAMPLE_TEGRASTATS_LINE)

        assert snapshot.memory.emc_frequency_mhz == 2133

    def test_parse_thermal_zones(self):
        """Test thermal zone parsing."""
        snapshot = TegrastatsParser.parse_line(SAMPLE_TEGRASTATS_LINE)

        assert snapshot.thermal.cpu_temp_c == 31.5
        assert snapshot.thermal.gpu_temp_c == 30.5
        assert snapshot.thermal.aux_temp_c == 30.0
        assert snapshot.thermal.ao_temp_c == 31.0
        assert snapshot.thermal.tdiode_temp_c == 30.75
        assert snapshot.thermal.tj_temp_c == 41.0

    def test_parse_thermal_high_temps(self):
        """Test thermal parsing with high temperatures."""
        snapshot = TegrastatsParser.parse_line(SAMPLE_THROTTLING_LINE)

        assert snapshot.thermal.cpu_temp_c == 92.0
        assert snapshot.thermal.gpu_temp_c == 95.0
        assert snapshot.thermal.tj_temp_c == 97.0
        assert snapshot.thermal.max_temp_c == 97.0

    def test_parse_ape_frequency(self):
        """Test APE frequency parsing."""
        snapshot = TegrastatsParser.parse_line(SAMPLE_TEGRASTATS_LINE)

        assert snapshot.ape_frequency_mhz == 245

    def test_parse_file(self):
        """Test parsing from file (close before unlink for Windows)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(SAMPLE_TEGRASTATS_LINE + "\n")
            f.write(SAMPLE_HIGH_LOAD_LINE + "\n")
            f.write(SAMPLE_THROTTLING_LINE + "\n")
            f.flush()
            name = f.name
        try:
            snapshots = TegrastatsParser.parse_file(name)
            assert len(snapshots) == 3
        finally:
            Path(name).unlink(missing_ok=True)

    def test_parse_file_missing(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            TegrastatsParser.parse_file("/nonexistent/file.log")

    def test_parse_file_with_malformed_lines(self):
        """Test that malformed lines are skipped (close before unlink for Windows)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(SAMPLE_TEGRASTATS_LINE + "\n")
            f.write("This is not a valid tegrastats line\n")
            f.write(SAMPLE_HIGH_LOAD_LINE + "\n")
            f.write("Another invalid line without RAM\n")
            f.flush()
            name = f.name
        try:
            snapshots = TegrastatsParser.parse_file(name)
            assert len(snapshots) == 2
        finally:
            Path(name).unlink(missing_ok=True)


class TestTegrastatsSnapshot:
    """Tests for TegrastatsSnapshot dataclass."""

    def test_cpu_avg_utilization(self):
        """Test average CPU utilization calculation."""
        snapshot = TegrastatsParser.parse_line(SAMPLE_HIGH_LOAD_LINE)

        # 85+90+78+92+88+95+82+89 = 699 / 8 = 87.375
        assert snapshot.cpu_avg_utilization == pytest.approx(87.375)

    def test_cpu_max_utilization(self):
        """Test max CPU utilization."""
        snapshot = TegrastatsParser.parse_line(SAMPLE_HIGH_LOAD_LINE)

        assert snapshot.cpu_max_utilization == 95.0

    def test_num_cores(self):
        """Test core count."""
        snapshot = TegrastatsParser.parse_line(SAMPLE_TEGRASTATS_LINE)

        assert snapshot.num_cores == 8

    def test_to_dict(self):
        """Test conversion to dictionary."""
        snapshot = TegrastatsParser.parse_line(SAMPLE_TEGRASTATS_LINE)
        d = snapshot.to_dict()

        assert "cpu" in d
        assert "gpu" in d
        assert "memory" in d
        assert "thermal" in d
        assert d["cpu"]["num_cores"] == 8


class TestTegrastatsCalculator:
    """Tests for TegrastatsCalculator class."""

    @pytest.fixture
    def sample_snapshots(self):
        """Create sample snapshots for testing."""
        lines = [
            SAMPLE_TEGRASTATS_LINE,
            SAMPLE_HIGH_LOAD_LINE,
            SAMPLE_THROTTLING_LINE,
        ]
        snapshots = []
        base_time = datetime.now()
        for i, line in enumerate(lines):
            snapshot = TegrastatsParser.parse_line(line, timestamp=base_time + timedelta(seconds=i))
            snapshots.append(snapshot)
        return snapshots

    def test_calculate_aggregates(self, sample_snapshots):
        """Test aggregate calculation."""
        aggregates = TegrastatsCalculator.calculate_aggregates(sample_snapshots)

        assert aggregates.num_samples == 3
        assert aggregates.duration_seconds == pytest.approx(2.0)
        assert len(aggregates.cpu_per_core_avg) == 8

    def test_calculate_aggregates_empty(self):
        """Test aggregates with empty list."""
        aggregates = TegrastatsCalculator.calculate_aggregates([])

        assert aggregates.num_samples == 0
        assert aggregates.cpu_avg_utilization == 0.0

    def test_cpu_aggregates(self, sample_snapshots):
        """Test CPU aggregate statistics."""
        aggregates = TegrastatsCalculator.calculate_aggregates(sample_snapshots)

        # Should have reasonable values
        assert aggregates.cpu_avg_utilization > 0
        assert aggregates.cpu_max_utilization >= aggregates.cpu_avg_utilization
        assert aggregates.cpu_min_utilization <= aggregates.cpu_avg_utilization

    def test_gpu_aggregates(self, sample_snapshots):
        """Test GPU aggregate statistics."""
        aggregates = TegrastatsCalculator.calculate_aggregates(sample_snapshots)

        assert aggregates.gpu_max_utilization == 95.0  # From high load line
        assert aggregates.gpu_min_utilization == 0.0  # From idle line

    def test_memory_aggregates(self, sample_snapshots):
        """Test memory aggregate statistics."""
        aggregates = TegrastatsCalculator.calculate_aggregates(sample_snapshots)

        assert aggregates.ram_total_mb == 28409
        assert aggregates.ram_max_used_mb == 25000  # From throttling line

    def test_thermal_aggregates(self, sample_snapshots):
        """Test thermal aggregate statistics."""
        aggregates = TegrastatsCalculator.calculate_aggregates(sample_snapshots)

        assert aggregates.thermal_max_observed == 97.0  # tj from throttling
        assert "cpu" in aggregates.thermal_max_temps
        assert aggregates.thermal_max_temps["cpu"] == 92.0

    def test_detect_thermal_throttling(self, sample_snapshots):
        """Test thermal throttling detection."""
        result = TegrastatsCalculator.detect_thermal_throttling(sample_snapshots, throttle_temp_c=85.0)

        assert result["throttle_events"] >= 1  # Throttling line exceeds 85C
        assert result["max_temp_observed"] == 97.0

    def test_detect_thermal_throttling_none(self):
        """Test throttling detection with no throttling."""
        snapshot = TegrastatsParser.parse_line(SAMPLE_TEGRASTATS_LINE)
        result = TegrastatsCalculator.detect_thermal_throttling([snapshot], throttle_temp_c=85.0)

        assert result["throttle_events"] == 0
        assert result["throttle_percentage"] == 0.0

    def test_detect_memory_pressure(self, sample_snapshots):
        """Test memory pressure detection."""
        result = TegrastatsCalculator.detect_memory_pressure(sample_snapshots, pressure_threshold_percent=85.0)

        assert result["pressure_events"] >= 1  # Throttling line at ~88%

    def test_detect_memory_pressure_none(self):
        """Test memory pressure detection with no pressure."""
        snapshot = TegrastatsParser.parse_line(SAMPLE_TEGRASTATS_LINE)
        result = TegrastatsCalculator.detect_memory_pressure([snapshot], pressure_threshold_percent=90.0)

        assert result["pressure_events"] == 0


class TestTegrastatsAnalyzer:
    """Tests for TegrastatsAnalyzer class."""

    @pytest.fixture
    def temp_tegrastats_file(self):
        """Create a temporary tegrastats file (close before yield for Windows)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(SAMPLE_TEGRASTATS_LINE + "\n")
            f.write(SAMPLE_HIGH_LOAD_LINE + "\n")
            f.write(SAMPLE_THROTTLING_LINE + "\n")
            f.flush()
            name = f.name
        try:
            yield name
        finally:
            Path(name).unlink(missing_ok=True)

    def test_analyze_file(self, temp_tegrastats_file):
        """Test analyzing a tegrastats file."""
        analyzer = TegrastatsAnalyzer()
        result = analyzer.analyze(temp_tegrastats_file)

        assert result.name == "Tegrastats Analysis"
        assert result.metrics["num_samples"] == 3
        assert "cpu" in result.metrics
        assert "gpu" in result.metrics
        assert "memory" in result.metrics
        assert "thermal" in result.metrics

    def test_analyze_lines(self):
        """Test analyzing from lines."""
        analyzer = TegrastatsAnalyzer()
        lines = [SAMPLE_TEGRASTATS_LINE, SAMPLE_HIGH_LOAD_LINE]

        result = analyzer.analyze_lines(lines)

        assert result.metrics["num_samples"] == 2

    def test_analyze_cpu_metrics(self, temp_tegrastats_file):
        """Test CPU metrics in analysis."""
        analyzer = TegrastatsAnalyzer()
        result = analyzer.analyze(temp_tegrastats_file)

        cpu = result.metrics["cpu"]
        assert cpu["num_cores"] == 8
        assert "avg_utilization_percent" in cpu
        assert "max_utilization_percent" in cpu
        assert "per_core_avg_utilization" in cpu
        assert len(cpu["per_core_avg_utilization"]) == 8

    def test_analyze_gpu_metrics(self, temp_tegrastats_file):
        """Test GPU metrics in analysis."""
        analyzer = TegrastatsAnalyzer()
        result = analyzer.analyze(temp_tegrastats_file)

        gpu = result.metrics["gpu"]
        assert gpu["max_utilization_percent"] == 95.0
        assert "avg_frequency_mhz" in gpu

    def test_analyze_memory_metrics(self, temp_tegrastats_file):
        """Test memory metrics in analysis."""
        analyzer = TegrastatsAnalyzer()
        result = analyzer.analyze(temp_tegrastats_file)

        memory = result.metrics["memory"]
        assert memory["total_mb"] == 28409
        assert "avg_used_mb" in memory
        assert "emc_avg_frequency_mhz" in memory

    def test_analyze_thermal_metrics(self, temp_tegrastats_file):
        """Test thermal metrics in analysis."""
        analyzer = TegrastatsAnalyzer()
        result = analyzer.analyze(temp_tegrastats_file)

        thermal = result.metrics["thermal"]
        assert thermal["max_observed_c"] == 97.0
        assert "avg_temps_c" in thermal
        assert "max_temps_c" in thermal

    def test_analyze_throttling_detection(self, temp_tegrastats_file):
        """Test throttling detection in analysis."""
        analyzer = TegrastatsAnalyzer()
        result = analyzer.analyze(temp_tegrastats_file)

        throttle = result.metrics["thermal_throttling"]
        assert throttle["throttle_events"] >= 1
        assert "throttle_percentage" in throttle

    def test_analyze_memory_pressure_detection(self, temp_tegrastats_file):
        """Test memory pressure detection in analysis."""
        analyzer = TegrastatsAnalyzer()
        result = analyzer.analyze(temp_tegrastats_file)

        pressure = result.metrics["memory_pressure"]
        assert "pressure_events" in pressure
        assert "pressure_percentage" in pressure

    def test_analyze_health_assessment(self, temp_tegrastats_file):
        """Test health assessment in analysis."""
        analyzer = TegrastatsAnalyzer()
        result = analyzer.analyze(temp_tegrastats_file)

        health = result.metrics["health"]
        assert health["status"] in ["healthy", "warning", "critical"]
        assert "issues" in health
        assert "warnings" in health

    def test_analyze_empty_file(self):
        """Test analyzing empty file (close before unlink for Windows)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("")
            f.flush()
            name = f.name
        try:
            analyzer = TegrastatsAnalyzer()
            result = analyzer.analyze(name)
            assert result.metrics["num_samples"] == 0
            assert "error" in result.metrics
        finally:
            Path(name).unlink(missing_ok=True)

    def test_analyze_missing_file(self):
        """Test error handling for missing file."""
        analyzer = TegrastatsAnalyzer()

        with pytest.raises(FileNotFoundError):
            analyzer.analyze("/nonexistent/file.log")

    def test_custom_config(self, temp_tegrastats_file):
        """Test analyzer with custom config."""
        analyzer = TegrastatsAnalyzer(
            config={
                "throttle_temp_c": 90.0,
                "memory_pressure_percent": 95.0,
            }
        )
        result = analyzer.analyze(temp_tegrastats_file)

        # With higher thresholds, fewer events should be detected
        throttle = result.metrics["thermal_throttling"]
        assert throttle["throttle_threshold_c"] == 90.0

    def test_summarize(self, temp_tegrastats_file):
        """Test summary generation."""
        analyzer = TegrastatsAnalyzer()
        analyzer.analyze(temp_tegrastats_file)
        analyzer.analyze(temp_tegrastats_file)

        summary = analyzer.summarize()

        assert summary["total_analyses"] == 2
        assert summary["total_samples"] == 6
        assert "cpu_utilization_range" in summary
        assert "gpu_utilization_range" in summary
        assert "health_summary" in summary

    def test_summarize_empty(self):
        """Test summary with no analyses."""
        analyzer = TegrastatsAnalyzer()
        summary = analyzer.summarize()

        assert summary["total_analyses"] == 0

    def test_compare_runs(self):
        """Test comparing two runs."""
        analyzer = TegrastatsAnalyzer()

        baseline = {
            "cpu": {"avg_utilization_percent": 50.0},
            "gpu": {"avg_utilization_percent": 60.0},
            "memory": {"avg_utilization_percent": 40.0},
            "thermal": {"max_observed_c": 70.0},
        }

        improved = {
            "cpu": {"avg_utilization_percent": 45.0},
            "gpu": {"avg_utilization_percent": 55.0},
            "memory": {"avg_utilization_percent": 35.0},
            "thermal": {"max_observed_c": 65.0},
        }

        comparison = analyzer.compare_runs(baseline, improved)

        assert comparison["cpu_utilization_delta"] == -5.0
        assert comparison["cpu_more_efficient"] is True
        assert comparison["gpu_more_efficient"] is True
        assert comparison["runs_cooler"] is True

    def test_compare_runs_regression(self):
        """Test detecting regression between runs."""
        analyzer = TegrastatsAnalyzer()

        baseline = {
            "cpu": {"avg_utilization_percent": 50.0},
            "gpu": {"avg_utilization_percent": 60.0},
            "memory": {"avg_utilization_percent": 40.0},
            "thermal": {"max_observed_c": 70.0},
        }

        worse = {
            "cpu": {"avg_utilization_percent": 70.0},
            "gpu": {"avg_utilization_percent": 80.0},
            "memory": {"avg_utilization_percent": 60.0},
            "thermal": {"max_observed_c": 85.0},
        }

        comparison = analyzer.compare_runs(baseline, worse)

        assert comparison["cpu_utilization_delta"] == 20.0
        assert comparison["cpu_more_efficient"] is False
        assert comparison["runs_cooler"] is False

    def test_results_stored(self, temp_tegrastats_file):
        """Test that results are stored in analyzer."""
        analyzer = TegrastatsAnalyzer()
        analyzer.analyze(temp_tegrastats_file)

        assert len(analyzer.get_results()) == 1

        analyzer.analyze(temp_tegrastats_file)
        assert len(analyzer.get_results()) == 2


class TestDataclasses:
    """Tests for tegrastats dataclasses."""

    def test_cpu_core_stats_to_dict(self):
        """Test CPUCoreStats to_dict."""
        core = CPUCoreStats(core_id=0, utilization_percent=50.0, frequency_mhz=2000)
        d = core.to_dict()

        assert d["core_id"] == 0
        assert d["utilization_percent"] == 50.0
        assert d["frequency_mhz"] == 2000

    def test_gpu_stats_to_dict(self):
        """Test GPUStats to_dict."""
        gpu = GPUStats(utilization_percent=75.0, frequency_mhz=1500)
        d = gpu.to_dict()

        assert d["utilization_percent"] == 75.0
        assert d["frequency_mhz"] == 1500

    def test_memory_stats_properties(self):
        """Test MemoryStats computed properties."""
        mem = MemoryStats(used_mb=1000, total_mb=2000, lfb_blocks=100, lfb_size_mb=4)

        assert mem.utilization_percent == 50.0
        assert mem.available_mb == 1000
        assert mem.lfb_total_mb == 400

    def test_memory_stats_zero_total(self):
        """Test MemoryStats with zero total."""
        mem = MemoryStats(used_mb=0, total_mb=0, lfb_blocks=0, lfb_size_mb=0)

        assert mem.utilization_percent == 0.0

    def test_thermal_stats_max_temp(self):
        """Test ThermalStats max_temp property."""
        thermal = ThermalStats(
            cpu_temp_c=60.0,
            gpu_temp_c=70.0,
            tj_temp_c=75.0,
        )

        assert thermal.max_temp_c == 75.0

    def test_thermal_stats_empty(self):
        """Test ThermalStats with no temps."""
        thermal = ThermalStats()

        assert thermal.max_temp_c == 0.0
