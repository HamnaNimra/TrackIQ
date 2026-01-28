"""Tests for DNN pipeline analyzer and utilities."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from autoperfpy.analyzers.dnn_pipeline import DNNPipelineAnalyzer
from autoperfpy.core.dnn_pipeline import (
    DNNPipelineParser,
    DNNPipelineCalculator,
    InferenceRun,
    LayerTiming,
    MemoryTransfer,
    EngineOptimizationMetrics,
)


# Sample profiler output for testing
SAMPLE_PROFILER_OUTPUT = """
Batch Size: 4
Layer Timings:
conv1 Conv 0.5 ms GPU
bn1 BatchNorm 0.1 ms GPU
relu1 ReLU 0.05 ms GPU
conv2 Conv 1.2 ms DLA0
pool1 MaxPool 0.2 ms DLA0
fc1 FullyConnected 0.8 ms GPU
softmax Softmax 0.1 ms GPU

Memory Transfers:
H2D 4194304 bytes 0.3 ms
D2H 16384 bytes 0.05 ms

Total: 3.3 ms
"""

SAMPLE_DLA_HEAVY_OUTPUT = """
Batch Size: 8
Layer Timings:
conv1 Conv 0.3 ms DLA0
bn1 BatchNorm 0.1 ms DLA0
conv2 Conv 0.5 ms DLA0
conv3 Conv 0.6 ms DLA1
pool1 MaxPool 0.1 ms DLA1
fc1 FullyConnected 0.2 ms GPU

Memory Transfers:
H2D 8388608 bytes 0.5 ms
D2H 32768 bytes 0.1 ms

Total: 2.5 ms
"""


class TestLayerTiming:
    """Tests for LayerTiming dataclass."""

    def test_basic_properties(self):
        """Test basic layer timing properties."""
        layer = LayerTiming(
            name="conv1",
            layer_type="Conv",
            execution_time_ms=1.5,
            device="GPU",
        )

        assert layer.name == "conv1"
        assert layer.execution_time_ms == 1.5
        assert layer.execution_time_us == 1500.0
        assert layer.device == "GPU"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        layer = LayerTiming(
            name="conv1",
            layer_type="Conv",
            execution_time_ms=1.5,
            device="DLA0",
            input_dims=[1, 3, 224, 224],
        )

        d = layer.to_dict()
        assert d["name"] == "conv1"
        assert d["execution_time_ms"] == 1.5
        assert d["execution_time_us"] == 1500.0
        assert d["device"] == "DLA0"
        assert d["input_dims"] == [1, 3, 224, 224]


class TestMemoryTransfer:
    """Tests for MemoryTransfer dataclass."""

    def test_basic_properties(self):
        """Test basic memory transfer properties."""
        transfer = MemoryTransfer(
            transfer_type="H2D",
            size_bytes=4 * 1024 * 1024,  # 4 MB
            duration_ms=1.0,
        )

        assert transfer.transfer_type == "H2D"
        assert transfer.size_bytes == 4 * 1024 * 1024

    def test_bandwidth_calculation(self):
        """Test bandwidth calculation."""
        # 1 GB in 1 second = 1 GB/s
        transfer = MemoryTransfer(
            transfer_type="H2D",
            size_bytes=1024 ** 3,  # 1 GB
            duration_ms=1000.0,  # 1 second
        )

        assert transfer.bandwidth_gbps == pytest.approx(1.0)

    def test_bandwidth_zero_duration(self):
        """Test bandwidth with zero duration."""
        transfer = MemoryTransfer(
            transfer_type="D2H",
            size_bytes=1024,
            duration_ms=0.0,
        )

        assert transfer.bandwidth_gbps == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        transfer = MemoryTransfer(
            transfer_type="H2D",
            size_bytes=1024 * 1024,  # 1 MB
            duration_ms=0.5,
        )

        d = transfer.to_dict()
        assert d["transfer_type"] == "H2D"
        assert d["size_mb"] == 1.0
        assert "bandwidth_gbps" in d


class TestInferenceRun:
    """Tests for InferenceRun dataclass."""

    @pytest.fixture
    def sample_run(self):
        """Create a sample inference run."""
        layers = [
            LayerTiming("conv1", "Conv", 1.0, "GPU"),
            LayerTiming("conv2", "Conv", 0.5, "DLA0"),
            LayerTiming("fc1", "FC", 0.3, "GPU"),
        ]
        transfers = [
            MemoryTransfer("H2D", 1024 * 1024, 0.2),
            MemoryTransfer("D2H", 1024, 0.05),
        ]
        return InferenceRun(
            timestamp=datetime.now(),
            batch_size=4,
            total_time_ms=2.05,
            layers=layers,
            memory_transfers=transfers,
        )

    def test_gpu_time(self, sample_run):
        """Test GPU time calculation."""
        assert sample_run.gpu_time_ms == pytest.approx(1.3)  # 1.0 + 0.3

    def test_dla_time(self, sample_run):
        """Test DLA time calculation."""
        assert sample_run.dla_time_ms == pytest.approx(0.5)

    def test_memory_overhead(self, sample_run):
        """Test memory overhead calculation."""
        assert sample_run.h2d_time_ms == pytest.approx(0.2)
        assert sample_run.d2h_time_ms == pytest.approx(0.05)
        assert sample_run.memory_overhead_ms == pytest.approx(0.25)

    def test_compute_time(self, sample_run):
        """Test compute time calculation."""
        assert sample_run.compute_time_ms == pytest.approx(1.8)  # 1.3 + 0.5

    def test_dla_percentage(self, sample_run):
        """Test DLA percentage calculation."""
        # 0.5 / 1.8 * 100 = ~27.78%
        assert sample_run.dla_percentage == pytest.approx(27.78, rel=0.01)

    def test_throughput(self, sample_run):
        """Test throughput calculation."""
        # 4 samples / 2.05ms * 1000 = ~1951 fps
        assert sample_run.throughput_fps == pytest.approx(1951.2, rel=0.01)

    def test_to_dict(self, sample_run):
        """Test conversion to dictionary."""
        d = sample_run.to_dict()

        assert d["batch_size"] == 4
        assert d["total_time_ms"] == 2.05
        assert "gpu_time_ms" in d
        assert "dla_time_ms" in d
        assert "layers" in d
        assert len(d["layers"]) == 3


class TestDNNPipelineParser:
    """Tests for DNNPipelineParser class."""

    def test_parse_profiler_output(self):
        """Test parsing profiler output."""
        run = DNNPipelineParser.parse_profiler_output(SAMPLE_PROFILER_OUTPUT)

        assert run.batch_size == 4
        assert run.total_time_ms == pytest.approx(3.3)
        assert len(run.layers) == 7
        assert len(run.memory_transfers) == 2

    def test_parse_layer_timings(self):
        """Test parsing layer timings."""
        run = DNNPipelineParser.parse_profiler_output(SAMPLE_PROFILER_OUTPUT)

        # Check first layer
        assert run.layers[0].name == "conv1"
        assert run.layers[0].layer_type == "Conv"
        assert run.layers[0].execution_time_ms == 0.5
        assert run.layers[0].device == "GPU"

        # Check DLA layer
        dla_layers = [l for l in run.layers if l.device.startswith("DLA")]
        assert len(dla_layers) == 2

    def test_parse_memory_transfers(self):
        """Test parsing memory transfers."""
        run = DNNPipelineParser.parse_profiler_output(SAMPLE_PROFILER_OUTPUT)

        h2d = [t for t in run.memory_transfers if t.transfer_type == "H2D"]
        d2h = [t for t in run.memory_transfers if t.transfer_type == "D2H"]

        assert len(h2d) == 1
        assert len(d2h) == 1
        assert h2d[0].size_bytes == 4194304
        assert h2d[0].duration_ms == 0.3

    def test_parse_dla_heavy_output(self):
        """Test parsing DLA-heavy output."""
        run = DNNPipelineParser.parse_profiler_output(SAMPLE_DLA_HEAVY_OUTPUT)

        assert run.batch_size == 8
        assert run.dla_percentage > run.gpu_percentage

    def test_parse_csv_layer_timing(self):
        """Test parsing CSV layer timing file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("layer_name,layer_type,time_ms,device\n")
            f.write("conv1,Conv,1.0,GPU\n")
            f.write("conv2,Conv,0.5,DLA0\n")
            f.write("fc1,FC,0.3,GPU\n")
            f.flush()

            run = DNNPipelineParser.parse_csv_layer_timing(f.name, batch_size=2)
            Path(f.name).unlink()

        assert run.batch_size == 2
        assert len(run.layers) == 3
        assert run.total_time_ms == pytest.approx(1.8)

    def test_parse_csv_missing_file(self):
        """Test error handling for missing CSV file."""
        with pytest.raises(FileNotFoundError):
            DNNPipelineParser.parse_csv_layer_timing("/nonexistent/file.csv")


class TestDNNPipelineCalculator:
    """Tests for DNNPipelineCalculator class."""

    @pytest.fixture
    def sample_runs(self):
        """Create sample inference runs."""
        runs = []
        for batch_size in [1, 4, 8]:
            layers = [
                LayerTiming("conv1", "Conv", 1.0 * batch_size / 4, "GPU"),
                LayerTiming("conv2", "Conv", 0.5 * batch_size / 4, "DLA0"),
                LayerTiming("fc1", "FC", 0.3, "GPU"),
            ]
            transfers = [
                MemoryTransfer("H2D", 1024 * 1024 * batch_size, 0.1 * batch_size),
                MemoryTransfer("D2H", 1024 * batch_size, 0.02 * batch_size),
            ]
            total = sum(l.execution_time_ms for l in layers) + sum(t.duration_ms for t in transfers)
            runs.append(InferenceRun(
                timestamp=datetime.now(),
                batch_size=batch_size,
                total_time_ms=total,
                layers=layers,
                memory_transfers=transfers,
            ))
        return runs

    def test_calculate_aggregates(self, sample_runs):
        """Test aggregate calculation."""
        aggregates = DNNPipelineCalculator.calculate_aggregates(sample_runs)

        assert aggregates.num_runs == 3
        assert aggregates.total_inferences == 13  # 1 + 4 + 8
        assert set(aggregates.batch_sizes_used) == {1, 4, 8}

    def test_calculate_aggregates_empty(self):
        """Test aggregates with empty list."""
        aggregates = DNNPipelineCalculator.calculate_aggregates([])

        assert aggregates.num_runs == 0
        assert aggregates.avg_throughput_fps == 0.0

    def test_slowest_layers(self, sample_runs):
        """Test slowest layers identification."""
        aggregates = DNNPipelineCalculator.calculate_aggregates(sample_runs, top_n_layers=2)

        assert len(aggregates.slowest_layers) == 2
        # conv1 should be slowest
        assert aggregates.slowest_layers[0]["name"] == "conv1"

    def test_analyze_batch_scaling(self, sample_runs):
        """Test batch scaling analysis."""
        analysis = DNNPipelineCalculator.analyze_batch_scaling(sample_runs)

        assert "batch_metrics" in analysis
        assert len(analysis["batch_metrics"]) == 3
        assert "optimal_for_throughput" in analysis
        assert "optimal_for_latency" in analysis

    def test_analyze_batch_scaling_empty(self):
        """Test batch scaling with no runs."""
        analysis = DNNPipelineCalculator.analyze_batch_scaling([])

        assert "error" in analysis

    def test_compare_dla_vs_gpu(self, sample_runs):
        """Test DLA vs GPU comparison."""
        comparison = DNNPipelineCalculator.compare_dla_vs_gpu(sample_runs)

        assert "total_gpu_time_ms" in comparison
        assert "total_dla_time_ms" in comparison
        assert "gpu_layer_count" in comparison
        assert "dla_layer_count" in comparison
        assert comparison["gpu_layer_count"] == 2  # conv1, fc1
        assert comparison["dla_layer_count"] == 1  # conv2


class TestDNNPipelineAnalyzer:
    """Tests for DNNPipelineAnalyzer class."""

    def test_analyze_profiler_output(self):
        """Test analyzing profiler output."""
        analyzer = DNNPipelineAnalyzer()
        result = analyzer.analyze_profiler_output(SAMPLE_PROFILER_OUTPUT)

        assert result.name == "DNN Pipeline Analysis"
        assert result.metrics["batch_size"] == 4
        assert "timing" in result.metrics
        assert "device_split" in result.metrics
        assert "memory_overhead" in result.metrics

    def test_analyze_layer_csv(self):
        """Test analyzing CSV layer timing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("layer_name,layer_type,time_ms,device\n")
            f.write("conv1,Conv,1.0,GPU\n")
            f.write("conv2,Conv,0.5,DLA0\n")
            f.flush()

            analyzer = DNNPipelineAnalyzer()
            result = analyzer.analyze_layer_csv(f.name, batch_size=4)
            Path(f.name).unlink()

        assert result.metrics["batch_size"] == 4
        assert result.metrics["num_layers"] == 2

    def test_analyze_runs(self):
        """Test analyzing multiple runs."""
        runs = [
            InferenceRun(
                timestamp=datetime.now(),
                batch_size=4,
                total_time_ms=2.0,
                layers=[
                    LayerTiming("conv1", "Conv", 1.0, "GPU"),
                    LayerTiming("conv2", "Conv", 0.5, "DLA0"),
                ],
            ),
            InferenceRun(
                timestamp=datetime.now(),
                batch_size=8,
                total_time_ms=3.0,
                layers=[
                    LayerTiming("conv1", "Conv", 1.5, "GPU"),
                    LayerTiming("conv2", "Conv", 0.8, "DLA0"),
                ],
            ),
        ]

        analyzer = DNNPipelineAnalyzer()
        result = analyzer.analyze_runs(runs)

        assert result.metrics["num_runs"] == 2
        assert result.metrics["total_inferences"] == 12
        assert "batch_scaling" in result.metrics

    def test_analyze_runs_empty(self):
        """Test analyzing empty runs list."""
        analyzer = DNNPipelineAnalyzer()
        result = analyzer.analyze_runs([])

        assert result.metrics["num_runs"] == 0
        assert "error" in result.metrics

    def test_analyze_from_data(self):
        """Test analyzing from raw data dictionaries."""
        layer_timings = [
            {"name": "conv1", "layer_type": "Conv", "execution_time_ms": 1.0, "device": "GPU"},
            {"name": "conv2", "layer_type": "Conv", "execution_time_ms": 0.5, "device": "DLA0"},
        ]
        memory_transfers = [
            {"transfer_type": "H2D", "size_bytes": 1024 * 1024, "duration_ms": 0.2},
        ]

        analyzer = DNNPipelineAnalyzer()
        result = analyzer.analyze_from_data(
            layer_timings,
            memory_transfers=memory_transfers,
            batch_size=4,
        )

        assert result.metrics["batch_size"] == 4
        assert result.metrics["num_layers"] == 2

    def test_device_split_metrics(self):
        """Test device split metrics."""
        analyzer = DNNPipelineAnalyzer()
        result = analyzer.analyze_profiler_output(SAMPLE_DLA_HEAVY_OUTPUT)

        device_split = result.metrics["device_split"]
        assert device_split["dla_percentage"] > device_split["gpu_percentage"]

    def test_memory_overhead_metrics(self):
        """Test memory overhead metrics."""
        analyzer = DNNPipelineAnalyzer()
        result = analyzer.analyze_profiler_output(SAMPLE_PROFILER_OUTPUT)

        mem_overhead = result.metrics["memory_overhead"]
        assert mem_overhead["h2d_time_ms"] == pytest.approx(0.3)
        assert mem_overhead["d2h_time_ms"] == pytest.approx(0.05)
        assert mem_overhead["total_overhead_ms"] == pytest.approx(0.35)

    def test_slowest_layers_identification(self):
        """Test slowest layers identification."""
        analyzer = DNNPipelineAnalyzer(config={"top_n_layers": 3})
        result = analyzer.analyze_profiler_output(SAMPLE_PROFILER_OUTPUT)

        slowest = result.metrics["slowest_layers"]
        assert len(slowest) == 3
        # conv2 (1.2ms) should be slowest
        assert slowest[0]["name"] == "conv2"

    def test_recommendations_generated(self):
        """Test that recommendations are generated."""
        analyzer = DNNPipelineAnalyzer()
        result = analyzer.analyze_profiler_output(SAMPLE_PROFILER_OUTPUT)

        assert "recommendations" in result.metrics
        assert len(result.metrics["recommendations"]) > 0

    def test_summarize(self):
        """Test summary generation."""
        analyzer = DNNPipelineAnalyzer()
        analyzer.analyze_profiler_output(SAMPLE_PROFILER_OUTPUT)
        analyzer.analyze_profiler_output(SAMPLE_DLA_HEAVY_OUTPUT)

        summary = analyzer.summarize()

        assert summary["total_analyses"] == 2
        assert "avg_throughput_fps" in summary
        assert "avg_dla_percentage" in summary

    def test_summarize_empty(self):
        """Test summary with no analyses."""
        analyzer = DNNPipelineAnalyzer()
        summary = analyzer.summarize()

        assert summary["total_analyses"] == 0

    def test_compare_engines(self):
        """Test engine comparison."""
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

        assert comparison["is_faster"] is True
        assert comparison["is_higher_throughput"] is True
        assert comparison["latency_improvement_percent"] == pytest.approx(20.0)
        assert comparison["throughput_improvement_percent"] == pytest.approx(25.0)

    def test_compare_engines_regression(self):
        """Test detecting regression between engines."""
        analyzer = DNNPipelineAnalyzer()

        baseline = {
            "timing": {"total_time_ms": 8.0},
            "throughput_fps": 125.0,
            "memory_overhead": {"overhead_percentage": 10.0},
        }

        worse = {
            "timing": {"total_time_ms": 12.0},
            "throughput_fps": 83.0,
            "memory_overhead": {"overhead_percentage": 20.0},
        }

        comparison = analyzer.compare_engines(baseline, worse)

        assert comparison["is_faster"] is False
        assert comparison["is_higher_throughput"] is False
        assert comparison["latency_improvement_percent"] < 0

    def test_results_stored(self):
        """Test that results are stored in analyzer."""
        analyzer = DNNPipelineAnalyzer()
        analyzer.analyze_profiler_output(SAMPLE_PROFILER_OUTPUT)

        assert len(analyzer.get_results()) == 1

        analyzer.analyze_profiler_output(SAMPLE_DLA_HEAVY_OUTPUT)
        assert len(analyzer.get_results()) == 2


class TestEngineOptimizationMetrics:
    """Tests for EngineOptimizationMetrics dataclass."""

    def test_dla_coverage(self):
        """Test DLA coverage calculation."""
        metrics = EngineOptimizationMetrics(
            engine_name="resnet50",
            build_time_seconds=120.0,
            input_shapes={"input": [1, 3, 224, 224]},
            output_shapes={"output": [1, 1000]},
            precision="FP16",
            dla_enabled=True,
            dla_cores_used=[0],
            gpu_fallback_layers=5,
            total_layers=50,
        )

        assert metrics.dla_coverage_percent == pytest.approx(90.0)

    def test_dla_coverage_no_dla(self):
        """Test DLA coverage with no DLA."""
        metrics = EngineOptimizationMetrics(
            engine_name="resnet50",
            build_time_seconds=60.0,
            input_shapes={"input": [1, 3, 224, 224]},
            output_shapes={"output": [1, 1000]},
            precision="FP32",
            dla_enabled=False,
            gpu_fallback_layers=50,
            total_layers=50,
        )

        assert metrics.dla_coverage_percent == pytest.approx(0.0)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = EngineOptimizationMetrics(
            engine_name="yolo",
            build_time_seconds=90.0,
            input_shapes={"images": [1, 3, 640, 640]},
            output_shapes={"detections": [1, 100, 6]},
            precision="INT8",
            dla_enabled=True,
            dla_cores_used=[0, 1],
            gpu_fallback_layers=10,
            total_layers=100,
            workspace_size_mb=256.0,
            engine_size_mb=45.0,
        )

        d = metrics.to_dict()

        assert d["engine_name"] == "yolo"
        assert d["precision"] == "INT8"
        assert d["dla_coverage_percent"] == pytest.approx(90.0)
        assert d["workspace_size_mb"] == 256.0
