"""Tests for DNN Pipeline analyzer and utilities."""

import pytest
import tempfile
from datetime import datetime
from pathlib import Path

from autoperfpy.analyzers.dnn_pipeline import DNNPipelineAnalyzer
from autoperfpy.core.dnn_pipeline import (
    LayerTiming,
    MemoryTransfer,
    InferenceRun,
    EngineOptimizationMetrics,
    DNNPipelineParser,
    DNNPipelineCalculator,
    DNNPipelineAggregateStats,
)


class TestLayerTiming:
    """Tests for LayerTiming dataclass."""

    def test_basic_creation(self):
        """Test creating a LayerTiming instance."""
        layer = LayerTiming(
            name="conv1",
            layer_type="Conv",
            execution_time_ms=1.5,
            device="GPU",
        )
        assert layer.name == "conv1"
        assert layer.layer_type == "Conv"
        assert layer.execution_time_ms == 1.5
        assert layer.device == "GPU"

    def test_execution_time_us_conversion(self):
        """Test microsecond conversion property."""
        layer = LayerTiming(
            name="conv1",
            layer_type="Conv",
            execution_time_ms=1.5,
            device="GPU",
        )
        assert layer.execution_time_us == 1500.0

    def test_optional_fields(self):
        """Test optional fields have correct defaults."""
        layer = LayerTiming(
            name="conv1",
            layer_type="Conv",
            execution_time_ms=1.5,
            device="GPU",
        )
        assert layer.input_dims is None
        assert layer.output_dims is None
        assert layer.workspace_size_bytes is None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        layer = LayerTiming(
            name="conv1",
            layer_type="Conv",
            execution_time_ms=1.5,
            device="GPU",
            input_dims=[1, 3, 224, 224],
            output_dims=[1, 64, 112, 112],
        )
        d = layer.to_dict()
        assert d["name"] == "conv1"
        assert d["layer_type"] == "Conv"
        assert d["execution_time_ms"] == 1.5
        assert d["execution_time_us"] == 1500.0
        assert d["device"] == "GPU"
        assert d["input_dims"] == [1, 3, 224, 224]
        assert d["output_dims"] == [1, 64, 112, 112]


class TestMemoryTransfer:
    """Tests for MemoryTransfer dataclass."""

    def test_basic_creation(self):
        """Test creating a MemoryTransfer instance."""
        transfer = MemoryTransfer(
            transfer_type="H2D",
            size_bytes=1024 * 1024,
            duration_ms=0.5,
        )
        assert transfer.transfer_type == "H2D"
        assert transfer.size_bytes == 1024 * 1024
        assert transfer.duration_ms == 0.5

    def test_bandwidth_calculation(self):
        """Test bandwidth calculation in GB/s."""
        # 1 GB in 1 second = 1 GB/s
        transfer = MemoryTransfer(
            transfer_type="H2D",
            size_bytes=1024**3,  # 1 GB
            duration_ms=1000,  # 1 second
        )
        assert abs(transfer.bandwidth_gbps - 1.0) < 0.001

    def test_bandwidth_zero_duration(self):
        """Test bandwidth with zero duration returns 0."""
        transfer = MemoryTransfer(
            transfer_type="H2D",
            size_bytes=1024,
            duration_ms=0,
        )
        assert transfer.bandwidth_gbps == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        transfer = MemoryTransfer(
            transfer_type="D2H",
            size_bytes=1024 * 1024,  # 1 MB
            duration_ms=0.5,
            stream_id=1,
        )
        d = transfer.to_dict()
        assert d["transfer_type"] == "D2H"
        assert d["size_bytes"] == 1024 * 1024
        assert d["size_mb"] == 1.0
        assert d["duration_ms"] == 0.5
        assert d["stream_id"] == 1
        assert "bandwidth_gbps" in d


class TestInferenceRun:
    """Tests for InferenceRun dataclass."""

    @pytest.fixture
    def sample_layers(self):
        """Create sample layers for testing."""
        return [
            LayerTiming("conv1", "Conv", 2.0, "GPU"),
            LayerTiming("conv2", "Conv", 1.5, "GPU"),
            LayerTiming("dla_conv1", "Conv", 3.0, "DLA0"),
            LayerTiming("dla_conv2", "Conv", 2.5, "DLA1"),
        ]

    @pytest.fixture
    def sample_transfers(self):
        """Create sample memory transfers for testing."""
        return [
            MemoryTransfer("H2D", 1024 * 1024, 0.5),
            MemoryTransfer("D2H", 512 * 1024, 0.3),
        ]

    def test_basic_creation(self, sample_layers):
        """Test creating an InferenceRun instance."""
        run = InferenceRun(
            timestamp=datetime.now(),
            batch_size=4,
            total_time_ms=10.0,
            layers=sample_layers,
        )
        assert run.batch_size == 4
        assert run.total_time_ms == 10.0
        assert len(run.layers) == 4

    def test_gpu_time_calculation(self, sample_layers):
        """Test GPU time calculation."""
        run = InferenceRun(
            timestamp=datetime.now(),
            batch_size=1,
            total_time_ms=10.0,
            layers=sample_layers,
        )
        # GPU layers: conv1 (2.0) + conv2 (1.5) = 3.5
        assert run.gpu_time_ms == 3.5

    def test_dla_time_calculation(self, sample_layers):
        """Test DLA time calculation."""
        run = InferenceRun(
            timestamp=datetime.now(),
            batch_size=1,
            total_time_ms=10.0,
            layers=sample_layers,
        )
        # DLA layers: dla_conv1 (3.0) + dla_conv2 (2.5) = 5.5
        assert run.dla_time_ms == 5.5

    def test_memory_transfer_times(self, sample_layers, sample_transfers):
        """Test memory transfer time calculations."""
        run = InferenceRun(
            timestamp=datetime.now(),
            batch_size=1,
            total_time_ms=10.0,
            layers=sample_layers,
            memory_transfers=sample_transfers,
        )
        assert run.h2d_time_ms == 0.5
        assert run.d2h_time_ms == 0.3
        assert run.memory_overhead_ms == 0.8

    def test_compute_time(self, sample_layers):
        """Test total compute time."""
        run = InferenceRun(
            timestamp=datetime.now(),
            batch_size=1,
            total_time_ms=10.0,
            layers=sample_layers,
        )
        # 3.5 (GPU) + 5.5 (DLA) = 9.0
        assert run.compute_time_ms == 9.0

    def test_device_percentages(self, sample_layers):
        """Test DLA/GPU percentage calculations."""
        run = InferenceRun(
            timestamp=datetime.now(),
            batch_size=1,
            total_time_ms=10.0,
            layers=sample_layers,
        )
        # GPU: 3.5/9.0 = 38.89%, DLA: 5.5/9.0 = 61.11%
        assert abs(run.gpu_percentage - 38.888) < 0.01
        assert abs(run.dla_percentage - 61.111) < 0.01

    def test_throughput_fps(self, sample_layers):
        """Test throughput calculation."""
        run = InferenceRun(
            timestamp=datetime.now(),
            batch_size=4,
            total_time_ms=10.0,
            layers=sample_layers,
        )
        # 4 * 1000 / 10 = 400 fps
        assert run.throughput_fps == 400.0

    def test_throughput_zero_time(self, sample_layers):
        """Test throughput with zero time."""
        run = InferenceRun(
            timestamp=datetime.now(),
            batch_size=4,
            total_time_ms=0,
            layers=sample_layers,
        )
        assert run.throughput_fps == 0.0

    def test_num_layers(self, sample_layers):
        """Test layer count."""
        run = InferenceRun(
            timestamp=datetime.now(),
            batch_size=1,
            total_time_ms=10.0,
            layers=sample_layers,
        )
        assert run.num_layers == 4

    def test_to_dict(self, sample_layers, sample_transfers):
        """Test conversion to dictionary."""
        run = InferenceRun(
            timestamp=datetime.now(),
            batch_size=4,
            total_time_ms=10.0,
            layers=sample_layers,
            memory_transfers=sample_transfers,
            engine_name="resnet50",
        )
        d = run.to_dict()
        assert d["batch_size"] == 4
        assert d["total_time_ms"] == 10.0
        assert d["engine_name"] == "resnet50"
        assert len(d["layers"]) == 4
        assert len(d["memory_transfers"]) == 2
        assert "timestamp" in d


class TestEngineOptimizationMetrics:
    """Tests for EngineOptimizationMetrics dataclass."""

    def test_basic_creation(self):
        """Test creating EngineOptimizationMetrics."""
        metrics = EngineOptimizationMetrics(
            engine_name="resnet50",
            build_time_seconds=120.5,
            input_shapes={"input": [1, 3, 224, 224]},
            output_shapes={"output": [1, 1000]},
            precision="FP16",
            dla_enabled=True,
            dla_cores_used=[0],
            gpu_fallback_layers=5,
            total_layers=50,
        )
        assert metrics.engine_name == "resnet50"
        assert metrics.precision == "FP16"
        assert metrics.dla_enabled

    def test_dla_coverage_percent(self):
        """Test DLA coverage calculation."""
        metrics = EngineOptimizationMetrics(
            engine_name="resnet50",
            build_time_seconds=120.5,
            input_shapes={},
            output_shapes={},
            precision="FP16",
            dla_enabled=True,
            gpu_fallback_layers=10,
            total_layers=50,
        )
        # 40/50 = 80%
        assert metrics.dla_coverage_percent == 80.0

    def test_dla_coverage_zero_layers(self):
        """Test DLA coverage with zero total layers."""
        metrics = EngineOptimizationMetrics(
            engine_name="test",
            build_time_seconds=0,
            input_shapes={},
            output_shapes={},
            precision="FP32",
            dla_enabled=False,
            total_layers=0,
        )
        assert metrics.dla_coverage_percent == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = EngineOptimizationMetrics(
            engine_name="resnet50",
            build_time_seconds=120.5,
            input_shapes={"input": [1, 3, 224, 224]},
            output_shapes={"output": [1, 1000]},
            precision="FP16",
            dla_enabled=True,
            dla_cores_used=[0, 1],
            gpu_fallback_layers=5,
            total_layers=50,
            workspace_size_mb=512.0,
            engine_size_mb=128.0,
        )
        d = metrics.to_dict()
        assert d["engine_name"] == "resnet50"
        assert d["dla_coverage_percent"] == 90.0
        assert d["workspace_size_mb"] == 512.0


class TestDNNPipelineParser:
    """Tests for DNNPipelineParser class."""

    def test_parse_profiler_output_basic(self):
        """Test parsing basic profiler output."""
        content = """
        Batch Size: 4
        conv1 Conv 2.5 ms GPU
        conv2 Conv 1.8 ms DLA0
        pool1 Pool 0.5 ms GPU
        Total: 5.0 ms
        """
        run = DNNPipelineParser.parse_profiler_output(content)

        assert run.batch_size == 4
        assert run.total_time_ms == 5.0
        assert len(run.layers) == 3

    def test_parse_profiler_output_with_memory(self):
        """Test parsing profiler output with memory transfers."""
        content = """
        Batch Size: 1
        conv1 Conv 2.0 ms GPU
        H2D 1048576 bytes 0.5 ms
        D2H 524288 bytes 0.3 ms
        Total: 3.0 ms
        """
        run = DNNPipelineParser.parse_profiler_output(content)

        assert len(run.memory_transfers) == 2
        assert run.h2d_time_ms == 0.5
        assert run.d2h_time_ms == 0.3

    def test_parse_profiler_output_no_total_time(self):
        """Test parsing when total time is computed from layers."""
        content = """
        conv1 Conv 2.0 ms GPU
        conv2 Conv 1.5 ms GPU
        """
        run = DNNPipelineParser.parse_profiler_output(content)

        # Should compute from layers
        assert run.total_time_ms == 3.5

    def test_parse_profiler_output_custom_timestamp(self):
        """Test parsing with custom timestamp."""
        content = "conv1 Conv 1.0 ms GPU"
        custom_time = datetime(2024, 1, 1, 12, 0, 0)
        run = DNNPipelineParser.parse_profiler_output(content, timestamp=custom_time)

        assert run.timestamp == custom_time

    @pytest.fixture
    def sample_csv_file(self):
        """Create a sample CSV file for testing (close before yield for Windows)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("layer_name,layer_type,time_ms,device\n")
            f.write("conv1,Conv,2.0,GPU\n")
            f.write("conv2,Conv,1.5,DLA0\n")
            f.write("pool1,Pool,0.5,GPU\n")
            f.flush()
            name = f.name
        try:
            yield name
        finally:
            Path(name).unlink(missing_ok=True)

    def test_parse_csv_layer_timing(self, sample_csv_file):
        """Test parsing CSV layer timing file."""
        run = DNNPipelineParser.parse_csv_layer_timing(sample_csv_file, batch_size=4)

        assert run.batch_size == 4
        assert len(run.layers) == 3
        assert run.total_time_ms == 4.0  # 2.0 + 1.5 + 0.5

    def test_parse_csv_missing_file(self):
        """Test error when CSV file is missing."""
        with pytest.raises(FileNotFoundError):
            DNNPipelineParser.parse_csv_layer_timing("/nonexistent/file.csv")


class TestDNNPipelineCalculator:
    """Tests for DNNPipelineCalculator class."""

    @pytest.fixture
    def sample_runs(self):
        """Create sample inference runs for testing."""
        runs = []
        for i in range(5):
            layers = [
                LayerTiming("conv1", "Conv", 2.0 + i * 0.1, "GPU"),
                LayerTiming("conv2", "Conv", 1.5, "DLA0"),
            ]
            transfers = [
                MemoryTransfer("H2D", 1024, 0.2),
                MemoryTransfer("D2H", 512, 0.1),
            ]
            run = InferenceRun(
                timestamp=datetime.now(),
                batch_size=4,
                total_time_ms=4.0 + i * 0.1,
                layers=layers,
                memory_transfers=transfers,
            )
            runs.append(run)
        return runs

    def test_calculate_aggregates(self, sample_runs):
        """Test calculating aggregate statistics."""
        stats = DNNPipelineCalculator.calculate_aggregates(sample_runs)

        assert stats.num_runs == 5
        assert stats.total_inferences == 20  # 5 runs * batch_size 4
        assert stats.batch_sizes_used == [4]
        assert stats.avg_total_time_ms > 0
        assert stats.avg_throughput_fps > 0

    def test_calculate_aggregates_empty(self):
        """Test aggregates with empty run list."""
        stats = DNNPipelineCalculator.calculate_aggregates([])

        assert stats.num_runs == 0
        assert stats.total_inferences == 0
        assert stats.slowest_layers == []

    def test_calculate_aggregates_slowest_layers(self, sample_runs):
        """Test slowest layers identification."""
        stats = DNNPipelineCalculator.calculate_aggregates(sample_runs, top_n_layers=2)

        assert len(stats.slowest_layers) == 2
        # conv1 should be slowest (2.0+ ms vs 1.5 ms)
        assert stats.slowest_layers[0]["name"] == "conv1"

    def test_analyze_batch_scaling(self):
        """Test batch scaling analysis."""
        runs = []
        for batch_size in [1, 2, 4, 8]:
            layers = [LayerTiming("conv1", "Conv", 2.0, "GPU")]
            run = InferenceRun(
                timestamp=datetime.now(),
                batch_size=batch_size,
                total_time_ms=2.0 + batch_size * 0.5,  # Latency increases with batch
                layers=layers,
            )
            runs.append(run)

        analysis = DNNPipelineCalculator.analyze_batch_scaling(runs)

        assert len(analysis["batch_metrics"]) == 4
        assert analysis["optimal_for_latency"] == 1  # Smallest batch has lowest latency
        assert "max_throughput_fps" in analysis

    def test_analyze_batch_scaling_empty(self):
        """Test batch scaling with empty runs."""
        analysis = DNNPipelineCalculator.analyze_batch_scaling([])
        assert "error" in analysis

    def test_compare_dla_vs_gpu(self, sample_runs):
        """Test DLA vs GPU comparison."""
        comparison = DNNPipelineCalculator.compare_dla_vs_gpu(sample_runs)

        assert comparison["gpu_layer_count"] == 1
        assert comparison["dla_layer_count"] == 1
        assert "gpu_percentage" in comparison
        assert "dla_percentage" in comparison
        assert "recommendation" in comparison

    def test_compare_dla_vs_gpu_empty(self):
        """Test DLA vs GPU comparison with empty runs."""
        comparison = DNNPipelineCalculator.compare_dla_vs_gpu([])
        assert "error" in comparison


class TestDNNPipelineAnalyzer:
    """Tests for DNNPipelineAnalyzer class."""

    def test_initialization_default(self):
        """Test default initialization."""
        analyzer = DNNPipelineAnalyzer()
        assert analyzer.top_n_layers == 5
        assert analyzer.memory_overhead_threshold == 20.0

    def test_initialization_custom_config(self):
        """Test initialization with custom config."""
        config = {"top_n_layers": 10, "memory_overhead_threshold": 15.0}
        analyzer = DNNPipelineAnalyzer(config=config)
        assert analyzer.top_n_layers == 10
        assert analyzer.memory_overhead_threshold == 15.0

    def test_analyze_profiler_output(self):
        """Test analyzing profiler output text."""
        analyzer = DNNPipelineAnalyzer()
        content = """
        Batch Size: 4
        conv1 Conv 2.5 ms GPU
        conv2 Conv 1.8 ms DLA0
        Total: 5.0 ms
        """
        result = analyzer.analyze_profiler_output(content)

        assert result.name == "DNN Pipeline Analysis"
        assert result.metrics["batch_size"] == 4
        assert result.metrics["num_layers"] == 2

    def test_analyze_runs_empty(self):
        """Test analyzing empty runs list."""
        analyzer = DNNPipelineAnalyzer()
        result = analyzer.analyze_runs([])

        assert result.metrics["num_runs"] == 0
        assert "error" in result.metrics

    def test_analyze_runs(self):
        """Test analyzing multiple inference runs."""
        analyzer = DNNPipelineAnalyzer()
        runs = []
        for _ in range(3):
            layers = [
                LayerTiming("conv1", "Conv", 2.0, "GPU"),
                LayerTiming("conv2", "Conv", 1.5, "DLA0"),
            ]
            run = InferenceRun(
                timestamp=datetime.now(),
                batch_size=4,
                total_time_ms=4.0,
                layers=layers,
            )
            runs.append(run)

        result = analyzer.analyze_runs(runs)

        assert result.metrics["num_runs"] == 3
        assert "timing" in result.metrics
        assert "throughput" in result.metrics
        assert "device_split" in result.metrics

    def test_analyze_from_data(self):
        """Test analyzing from raw data dictionaries."""
        analyzer = DNNPipelineAnalyzer()
        layer_timings = [
            {
                "name": "conv1",
                "layer_type": "Conv",
                "execution_time_ms": 2.0,
                "device": "GPU",
            },
            {
                "name": "conv2",
                "layer_type": "Conv",
                "execution_time_ms": 1.5,
                "device": "DLA0",
            },
        ]
        memory_transfers = [
            {"transfer_type": "H2D", "size_bytes": 1024, "duration_ms": 0.2},
        ]

        result = analyzer.analyze_from_data(
            layer_timings, memory_transfers=memory_transfers, batch_size=4
        )

        assert result.metrics["batch_size"] == 4
        assert result.metrics["num_layers"] == 2

    def test_analyze_generic_string(self):
        """Test generic analyze method with string input."""
        analyzer = DNNPipelineAnalyzer()
        content = "conv1 Conv 2.0 ms GPU"
        result = analyzer.analyze(content)

        assert result.name == "DNN Pipeline Analysis"

    def test_analyze_generic_inference_runs(self):
        """Test generic analyze method with InferenceRun list."""
        analyzer = DNNPipelineAnalyzer()
        layers = [LayerTiming("conv1", "Conv", 2.0, "GPU")]
        run = InferenceRun(
            timestamp=datetime.now(),
            batch_size=1,
            total_time_ms=2.0,
            layers=layers,
        )
        result = analyzer.analyze([run])

        assert result.metrics["num_runs"] == 1

    def test_analyze_generic_dict_list(self):
        """Test generic analyze method with dict list."""
        analyzer = DNNPipelineAnalyzer()
        data = [
            {"name": "conv1", "layer_type": "Conv", "execution_time_ms": 2.0},
        ]
        result = analyzer.analyze(data)

        assert result.name == "DNN Pipeline Analysis"

    def test_analyze_generic_invalid_type(self):
        """Test generic analyze method with invalid input type."""
        analyzer = DNNPipelineAnalyzer()
        with pytest.raises(ValueError, match="Unsupported data type"):
            analyzer.analyze(12345)

    @pytest.fixture
    def sample_csv_file(self):
        """Create a sample CSV file for testing (close before yield for Windows)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("layer_name,layer_type,time_ms,device\n")
            f.write("conv1,Conv,2.0,GPU\n")
            f.write("conv2,Conv,1.5,DLA0\n")
            f.flush()
            name = f.name
        try:
            yield name
        finally:
            Path(name).unlink(missing_ok=True)

    def test_analyze_layer_csv(self, sample_csv_file):
        """Test analyzing layer timing CSV file."""
        analyzer = DNNPipelineAnalyzer()
        result = analyzer.analyze_layer_csv(sample_csv_file, batch_size=4)

        assert result.metrics["batch_size"] == 4
        assert result.metrics["num_layers"] == 2

    def test_summarize_no_results(self):
        """Test summarize with no results."""
        analyzer = DNNPipelineAnalyzer()
        summary = analyzer.summarize()

        assert summary["total_analyses"] == 0

    def test_summarize_with_results(self):
        """Test summarize with multiple results."""
        analyzer = DNNPipelineAnalyzer()
        layers = [LayerTiming("conv1", "Conv", 2.0, "GPU")]

        for _ in range(3):
            run = InferenceRun(
                timestamp=datetime.now(),
                batch_size=4,
                total_time_ms=5.0,
                layers=layers,
            )
            analyzer.analyze_runs([run])

        summary = analyzer.summarize()

        assert summary["total_analyses"] == 3
        assert summary["total_runs"] == 3

    def test_compare_engines(self):
        """Test comparing two engine configurations."""
        analyzer = DNNPipelineAnalyzer()

        baseline = {
            "timing": {"total_time_ms": 10.0},
            "throughput_fps": 100.0,
            "memory_overhead": {"overhead_percentage": 15.0},
        }
        optimized = {
            "timing": {"total_time_ms": 8.0},
            "throughput_fps": 125.0,
            "memory_overhead": {"overhead_percentage": 10.0},
        }

        comparison = analyzer.compare_engines(baseline, optimized)

        assert comparison["is_faster"]
        assert comparison["is_higher_throughput"]
        assert comparison["latency_improvement_percent"] == 20.0
        assert comparison["throughput_improvement_percent"] == 25.0

    def test_compare_engines_with_nested_metrics(self):
        """Test comparing engines with nested metric structures."""
        analyzer = DNNPipelineAnalyzer()

        baseline = {
            "timing": {"avg_total_ms": 10.0},
            "throughput": {"avg_fps": 100.0},
            "memory_overhead": {"overhead_percentage": 15.0},
        }
        optimized = {
            "timing": {"avg_total_ms": 8.0},
            "throughput": {"avg_fps": 125.0},
            "memory_overhead": {"overhead_percentage": 10.0},
        }

        comparison = analyzer.compare_engines(baseline, optimized)

        assert comparison["is_faster"]
        assert comparison["baseline"]["latency_ms"] == 10.0
        assert comparison["optimized"]["latency_ms"] == 8.0

    def test_recommendations_high_memory_overhead(self):
        """Test recommendations for high memory overhead."""
        analyzer = DNNPipelineAnalyzer()
        layers = [LayerTiming("conv1", "Conv", 5.0, "GPU")]
        transfers = [
            MemoryTransfer("H2D", 1024, 3.0),  # High overhead
            MemoryTransfer("D2H", 512, 2.0),
        ]
        run = InferenceRun(
            timestamp=datetime.now(),
            batch_size=1,
            total_time_ms=10.0,
            layers=layers,
            memory_transfers=transfers,
        )
        result = analyzer.analyze_runs([run])

        recommendations = result.metrics.get("recommendations", [])
        assert any("memory transfer overhead" in r.lower() for r in recommendations)

    def test_recommendations_low_dla_utilization(self):
        """Test recommendations for low DLA utilization."""
        analyzer = DNNPipelineAnalyzer()
        layers = [
            LayerTiming("conv1", "Conv", 8.0, "GPU"),
            LayerTiming("conv2", "Conv", 2.0, "DLA0"),  # Low DLA usage
        ]
        run = InferenceRun(
            timestamp=datetime.now(),
            batch_size=1,
            total_time_ms=10.0,
            layers=layers,
        )
        result = analyzer.analyze_runs([run])

        recommendations = result.metrics.get("recommendations", [])
        assert any("dla utilization" in r.lower() for r in recommendations)

    def test_recommendations_dominant_layer(self):
        """Test recommendations for dominant layer."""
        analyzer = DNNPipelineAnalyzer()
        layers = [
            LayerTiming("slow_conv", "Conv", 8.0, "GPU"),  # Dominates
            LayerTiming("fast_conv", "Conv", 0.5, "GPU"),
        ]
        run = InferenceRun(
            timestamp=datetime.now(),
            batch_size=1,
            total_time_ms=10.0,
            layers=layers,
        )
        result = analyzer.analyze_runs([run])

        recommendations = result.metrics.get("recommendations", [])
        assert any("dominates execution" in r.lower() for r in recommendations)

    def test_stores_results(self):
        """Test that results are stored in analyzer."""
        analyzer = DNNPipelineAnalyzer()
        content = "conv1 Conv 2.0 ms GPU"

        analyzer.analyze_profiler_output(content)
        analyzer.analyze_profiler_output(content)

        assert len(analyzer.results) == 2
