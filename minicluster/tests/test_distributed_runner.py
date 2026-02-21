"""Tests for distributed_runner module."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from minicluster.runner import (
    RunConfig,
    RunMetrics,
    SimpleMLP,
    create_synthetic_dataset,
    train_single_process,
    run_distributed,
    save_metrics,
    load_metrics,
)


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
        dataset = create_synthetic_dataset(
            num_samples=100, input_size=10, output_size=3, seed=42
        )
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
