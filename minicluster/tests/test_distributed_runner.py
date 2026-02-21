"""Tests for distributed_runner module."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from minicluster.runner import (
    RunConfig,
    SimpleMLP,
    create_synthetic_dataset,
    load_metrics,
    run_distributed,
    save_metrics,
    train_single_process,
)
from minicluster.runner.distributed_runner import (
    HealthCheckpoint,
    WorkerSnapshot,
    determine_worker_status,
    write_health_checkpoint,
)
from trackiq_core.serializer import load_trackiq_result


class TestSimpleMLP:
    """Tests for SimpleMLP model."""

    def test_mlp_initialization(self):
        """Test MLP can be initialized with various configs."""
        mlp = SimpleMLP(input_size=10, hidden_size=128, output_size=1, num_layers=2)
        assert mlp is not None

    def test_mlp_forward_pass(self):
        """Test MLP forward pass produces correct output shape."""
        mlp = SimpleMLP(input_size=10, hidden_size=64, output_size=1, num_layers=3)
        x = torch.randn(32, 10)
        output = mlp(x)
        assert output.shape == (32, 1)

    def test_mlp_multiple_output_dims(self):
        """Test MLP with multiple output dimensions."""
        mlp = SimpleMLP(input_size=20, hidden_size=128, output_size=5, num_layers=2)
        x = torch.randn(16, 20)
        output = mlp(x)
        assert output.shape == (16, 5)


class TestSyntheticDataset:
    """Tests for synthetic dataset creation."""

    def test_dataset_creation(self):
        """Test synthetic dataset can be created."""
        dataset = create_synthetic_dataset(num_samples=100, input_size=10, output_size=1)
        assert len(dataset) == 100

    def test_dataset_determinism(self):
        """Test dataset creation is deterministic with same seed."""
        ds1 = create_synthetic_dataset(num_samples=50, seed=42)
        ds2 = create_synthetic_dataset(num_samples=50, seed=42)

        x1, y1 = ds1[0]
        x2, y2 = ds2[0]

        assert torch.allclose(x1, x2)
        assert torch.allclose(y1, y2)

    def test_dataset_shape(self):
        """Test dataset has correct shapes."""
        dataset = create_synthetic_dataset(num_samples=100, input_size=10, output_size=3, seed=42)
        x, y = dataset[0]
        assert x.shape == (10,)
        assert y.shape == (3,)


class TestRunConfig:
    """Tests for RunConfig (DistributedValidationConfig)."""

    def test_default_config(self):
        """Test RunConfig has sensible defaults."""
        config = RunConfig()
        assert config.num_steps == 100
        assert config.num_processes == 2  # trackiq_core default
        assert config.batch_size == 32

    def test_config_with_custom_values(self):
        """Test RunConfig with custom values."""
        config = RunConfig(num_steps=50, num_processes=4, batch_size=64)
        assert config.num_steps == 50
        assert config.num_processes == 4
        assert config.batch_size == 64


class TestSingleProcessTraining:
    """Tests for single-process training."""

    def test_single_process_run_completes(self):
        """Test single-process training run completes successfully."""
        config = RunConfig(num_steps=10, batch_size=32, num_processes=1, seed=42)
        metrics = train_single_process(config)

        assert metrics is not None
        assert len(metrics.steps) == 10
        assert metrics.num_workers == 1

    def test_single_process_loss_values(self):
        """Test single-process run returns valid loss values."""
        config = RunConfig(num_steps=20, seed=42)
        metrics = train_single_process(config)

        for i, step_metric in enumerate(metrics.steps):
            assert step_metric.step == i
            assert isinstance(step_metric.loss, float)
            assert step_metric.loss > 0
            assert isinstance(step_metric.throughput_samples_per_sec, float)
            assert step_metric.throughput_samples_per_sec > 0

    def test_single_process_metrics_structure(self):
        """Test single-process metrics have correct structure."""
        config = RunConfig(num_steps=15, seed=42)
        metrics = train_single_process(config)

        assert metrics.num_steps == 15
        assert metrics.total_time_sec > 0
        assert len(metrics.steps) == 15
        assert metrics.to_dict() is not None

    def test_single_process_loss_decreases(self):
        """Test loss generally decreases over training steps."""
        config = RunConfig(num_steps=50, seed=42)
        metrics = train_single_process(config)

        first_quarter_loss = metrics.steps[10].loss
        last_quarter_loss = metrics.steps[-1].loss

        # Loss should decrease overall (though may have noise)
        assert last_quarter_loss < first_quarter_loss

    def test_single_process_reproducibility(self):
        """Test single-process training is reproducible with same seed."""
        config1 = RunConfig(num_steps=20, seed=42)
        metrics1 = train_single_process(config1)

        config2 = RunConfig(num_steps=20, seed=42)
        metrics2 = train_single_process(config2)

        for m1, m2 in zip(metrics1.steps, metrics2.steps):
            assert abs(m1.loss - m2.loss) < 1e-5


class TestMetricsSerialization:
    """Tests for metrics serialization and deserialization."""

    def test_metrics_to_dict(self):
        """Test metrics can be converted to dictionary."""
        config = RunConfig(num_steps=10, seed=42)
        metrics = train_single_process(config)
        metrics_dict = metrics.to_dict()

        assert "config" in metrics_dict
        assert "steps" in metrics_dict
        assert "total_time_sec" in metrics_dict
        assert "average_loss" in metrics_dict

    def test_save_and_load_metrics(self):
        """Test metrics can be saved and loaded."""
        config = RunConfig(num_steps=10, seed=42)
        metrics = train_single_process(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "metrics.json")
            save_metrics(metrics, output_path)

            assert Path(output_path).exists()

            loaded_dict = load_metrics(output_path)
            assert loaded_dict is not None
            assert len(loaded_dict["steps"]) == 10

    def test_metrics_json_serializable(self):
        """Test metrics can be serialized to valid JSON."""
        config = RunConfig(num_steps=5, seed=42)
        metrics = train_single_process(config)
        metrics_dict = metrics.to_dict()

        # Should not raise
        json_str = json.dumps(metrics_dict)
        assert json_str is not None

        # Should be able to parse back
        parsed = json.loads(json_str)
        assert len(parsed["steps"]) == 5

    def test_runmetrics_allreduce_aggregates_none_with_fewer_than_two_steps(self):
        """All aggregate all-reduce fields should be None when fewer than 2 steps exist."""
        metrics = run_distributed(RunConfig(num_steps=1, num_processes=1, seed=42))
        payload = metrics.to_dict()

        assert payload["p50_allreduce_ms"] is None
        assert payload["p95_allreduce_ms"] is None
        assert payload["p99_allreduce_ms"] is None
        assert payload["max_allreduce_ms"] is None
        assert payload["allreduce_stdev_ms"] is None

    def test_runmetrics_allreduce_aggregates_present_with_two_or_more_steps(self):
        """All aggregate all-reduce fields should be populated with 2+ steps."""
        metrics = run_distributed(RunConfig(num_steps=3, num_processes=1, seed=42))
        payload = metrics.to_dict()

        assert isinstance(payload["p50_allreduce_ms"], float)
        assert isinstance(payload["p95_allreduce_ms"], float)
        assert isinstance(payload["p99_allreduce_ms"], float)
        assert isinstance(payload["max_allreduce_ms"], float)
        assert isinstance(payload["allreduce_stdev_ms"], float)
        assert payload["collective_backend"] == "gloo"
        assert payload["workload_type"] == "mlp"

    def test_runmetrics_scaling_efficiency_from_baseline_throughput(self):
        """Scaling efficiency should be computed when baseline throughput is provided."""
        config = RunConfig(num_steps=2, num_processes=2, baseline_throughput=50.0)
        metrics = run_distributed(RunConfig(num_steps=2, num_processes=1, seed=42))
        metrics.num_workers = 2
        metrics.config = config.__dict__.copy()
        metrics.steps = [
            metrics.steps[0],
            metrics.steps[1],
        ]
        metrics.steps[0].throughput_samples_per_sec = 100.0
        metrics.steps[1].throughput_samples_per_sec = 100.0

        payload = metrics.to_dict()
        assert payload["scaling_efficiency_pct"] == pytest.approx(100.0)


class TestMultiProcessTraining:
    """Tests for multi-process training via run_distributed function.

    Note: These tests use the run_distributed wrapper which internally
    handles process spawning. In testing environment, we verify it behaves
    correctly for num_workers=1 (single process mode).
    """

    def test_multiprocess_wrapper_singleworker(self):
        """Test run_distributed with num_processes=1 (single process fallback)."""
        config = RunConfig(num_steps=10, num_processes=1, seed=42)
        metrics = run_distributed(config)

        assert metrics is not None
        assert len(metrics.steps) == 10
        assert metrics.num_workers == 1

    def test_run_distributed_metrics_structure(self):
        """Test run_distributed returns properly structured metrics."""
        config = RunConfig(num_steps=15, num_processes=1, seed=42)
        metrics = run_distributed(config)

        assert hasattr(metrics, "steps")
        assert hasattr(metrics, "total_time_sec")
        assert hasattr(metrics, "config")
        assert metrics.to_dict() is not None

    def test_run_distributed_reproducibility(self):
        """Test run_distributed is reproducible with same seed."""
        config1 = RunConfig(num_steps=20, num_processes=1, seed=99)
        metrics1 = run_distributed(config1)

        config2 = RunConfig(num_steps=20, num_processes=1, seed=99)
        metrics2 = run_distributed(config2)

        assert len(metrics1.steps) == len(metrics2.steps)
        for m1, m2 in zip(metrics1.steps, metrics2.steps):
            assert abs(m1.loss - m2.loss) < 1e-5


def test_minicluster_power_profiler_integration_full_session(tmp_path):
    """MiniCluster run/save path should populate canonical TrackiqResult power metrics."""
    config = RunConfig(num_steps=6, num_processes=1, batch_size=16, seed=42, tdp_watts=150.0)
    metrics = run_distributed(config)

    output_path = tmp_path / "minicluster_result.json"
    save_metrics(metrics, str(output_path))
    result = load_trackiq_result(output_path)

    assert result.metrics.power_consumption_watts is not None
    assert result.metrics.performance_per_watt is not None


def test_minicluster_save_metrics_writes_scaling_efficiency_metric(tmp_path):
    """Canonical TrackiqResult should include scaling_efficiency_pct when available."""
    metrics = run_distributed(RunConfig(num_steps=2, num_processes=1, seed=42))
    metrics.num_workers = 2
    metrics.config["baseline_throughput"] = 50.0
    metrics.steps[0].throughput_samples_per_sec = 100.0
    metrics.steps[1].throughput_samples_per_sec = 100.0

    output_path = tmp_path / "minicluster_scaling_result.json"
    save_metrics(metrics, str(output_path))
    result = load_trackiq_result(output_path)

    assert result.metrics.scaling_efficiency_pct == pytest.approx(100.0)


def test_determine_worker_status_healthy() -> None:
    """Status should be healthy when throughput is above threshold."""
    worker = WorkerSnapshot(
        worker_id=0,
        step=1,
        loss=0.2,
        throughput_samples_per_sec=80.0,
        allreduce_time_ms=1.0,
        compute_time_ms=1.0,
        status="healthy",
        timestamp="2026-02-21T00:00:00",
    )
    assert determine_worker_status(worker, baseline_throughput=100.0) == "healthy"


def test_determine_worker_status_slow() -> None:
    """Status should be slow when throughput is below 70% baseline."""
    worker = WorkerSnapshot(
        worker_id=0,
        step=1,
        loss=0.2,
        throughput_samples_per_sec=69.0,
        allreduce_time_ms=1.0,
        compute_time_ms=1.0,
        status="healthy",
        timestamp="2026-02-21T00:00:00",
    )
    assert determine_worker_status(worker, baseline_throughput=100.0) == "slow"


def test_determine_worker_status_failed() -> None:
    """Status should be failed when throughput is zero."""
    worker = WorkerSnapshot(
        worker_id=0,
        step=1,
        loss=0.2,
        throughput_samples_per_sec=0.0,
        allreduce_time_ms=1.0,
        compute_time_ms=1.0,
        status="healthy",
        timestamp="2026-02-21T00:00:00",
    )
    assert determine_worker_status(worker, baseline_throughput=100.0) == "failed"


def test_write_health_checkpoint_writes_valid_json(tmp_path) -> None:
    """Checkpoint writer should emit valid JSON payload."""
    out = tmp_path / "health.json"
    checkpoint = HealthCheckpoint(
        run_id="run-1",
        total_steps=10,
        completed_steps=1,
        workers=[
            WorkerSnapshot(
                worker_id=0,
                step=1,
                loss=0.2,
                throughput_samples_per_sec=100.0,
                allreduce_time_ms=1.0,
                compute_time_ms=1.0,
                status="healthy",
                timestamp="2026-02-21T00:00:00",
            )
        ],
        timestamp="2026-02-21T00:00:01",
        is_complete=False,
    )
    write_health_checkpoint(checkpoint, str(out))
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["run_id"] == "run-1"
    assert payload["workers"][0]["worker_id"] == 0


def test_write_health_checkpoint_atomic_output_not_partial(tmp_path) -> None:
    """Repeated checkpoint writes should leave readable, non-partial output."""
    out = tmp_path / "health.json"
    for i in range(5):
        checkpoint = HealthCheckpoint(
            run_id="run-atomic",
            total_steps=10,
            completed_steps=i + 1,
            workers=[],
            timestamp=f"2026-02-21T00:00:0{i}",
            is_complete=False,
        )
        write_health_checkpoint(checkpoint, str(out))
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["completed_steps"] == i + 1
    # Ensure temp files were replaced out of final path.
    assert out.exists()


def test_run_with_health_checkpoint_writes_complete_final_checkpoint(tmp_path) -> None:
    """Run with checkpoint path should write final is_complete=true payload."""
    checkpoint_path = tmp_path / "health.json"
    config = RunConfig(num_steps=6, num_processes=1, seed=42)
    metrics = run_distributed(config, health_checkpoint_path=str(checkpoint_path))
    assert metrics is not None
    assert checkpoint_path.exists()
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert payload["is_complete"] is True
    assert payload["completed_steps"] == 6


def test_run_without_health_checkpoint_path_writes_no_checkpoint_file(tmp_path) -> None:
    """Run without checkpoint path should not emit health checkpoint output."""
    checkpoint_path = tmp_path / "health.json"
    config = RunConfig(num_steps=6, num_processes=1, seed=42)
    metrics_without = run_distributed(config, health_checkpoint_path=None)
    metrics_default = run_distributed(config)

    assert metrics_without is not None
    assert metrics_default is not None
    assert len(metrics_without.steps) == len(metrics_default.steps)
    assert not checkpoint_path.exists()
