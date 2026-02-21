"""Tests for distributed validator module."""

import json
import pytest
import torch
from unittest.mock import patch, MagicMock

from trackiq_core.distributed_validator import (
    DistributedValidator,
    DistributedValidationConfig,
    SimpleMLP,
    create_synthetic_dataset,
    train_single_process,
    train_multi_process,
)


class TestDistributedValidationConfig:
    """Tests for DistributedValidationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DistributedValidationConfig()
        assert config.num_steps == 100
        assert config.learning_rate == 0.01
        assert config.batch_size == 32
        assert config.hidden_size == 128
        assert config.num_layers == 2
        assert config.input_size == 10
        assert config.output_size == 1
        assert config.loss_tolerance == 0.01
        assert config.num_processes == 2
        assert config.regression_threshold == 5.0


class TestSimpleMLP:
    """Tests for SimpleMLP model."""

    def test_model_creation(self):
        """Test model can be created and forward pass works."""
        model = SimpleMLP(input_size=10, hidden_size=64, output_size=1, num_layers=2)
        x = torch.randn(5, 10)
        output = model(x)
        assert output.shape == (5, 1)

    def test_different_layer_counts(self):
        """Test model with different number of layers."""
        model = SimpleMLP(input_size=5, hidden_size=32, output_size=3, num_layers=3)
        x = torch.randn(2, 5)
        output = model(x)
        assert output.shape == (2, 3)


class TestSyntheticDataset:
    """Tests for synthetic dataset creation."""

    def test_dataset_creation(self):
        """Test synthetic dataset creation."""
        dataset = create_synthetic_dataset(num_samples=100, input_size=5, output_size=2)
        assert len(dataset) == 100
        x, y = dataset[0]
        assert x.shape == (5,)
        assert y.shape == (2,)


class TestTrainingFunctions:
    """Tests for training functions."""

    def test_single_process_training(self):
        """Test single process training returns losses."""
        config = DistributedValidationConfig(num_steps=10)
        losses = train_single_process(config)
        assert len(losses) == 10
        assert all(isinstance(loss, float) for loss in losses)
        assert all(loss >= 0 for loss in losses)

    @patch('multiprocessing.get_context')
    def test_multi_process_training(self, mock_get_context):
        """Test multi-process training."""
        # Mock the multiprocessing context and queue
        mock_ctx = MagicMock()
        mock_queue = MagicMock()
        mock_ctx.Queue.return_value = mock_queue
        mock_ctx.Process.return_value = MagicMock()
        mock_get_context.return_value = mock_ctx

        # Mock the queue to return losses
        mock_queue.get.return_value = [0.5] * 10

        config = DistributedValidationConfig(num_steps=10, num_processes=2)
        losses = train_multi_process(config)

        assert len(losses) == 10
        mock_ctx.Process.assert_called()


class TestDistributedValidator:
    """Tests for DistributedValidator class."""

    def test_validator_creation(self):
        """Test validator can be created."""
        validator = DistributedValidator()
        assert validator.baseline_dir.name == "baselines"
        assert validator.regression_detector is not None

    @patch('trackiq_core.distributed_validator.train_single_process')
    @patch('trackiq_core.distributed_validator.train_multi_process')
    def test_run_validation(self, mock_multi_train, mock_single_train):
        """Test validation run."""
        mock_single_train.return_value = [1.0, 0.9, 0.8, 0.7, 0.6]
        mock_multi_train.return_value = [1.0, 0.9, 0.8, 0.7, 0.6]

        validator = DistributedValidator()
        config = DistributedValidationConfig(num_steps=5, loss_tolerance=0.1)
        results = validator.run_validation(config)

        assert "config" in results
        assert "comparisons" in results
        assert "summary" in results
        assert results["summary"]["overall_pass"] is True
        assert results["summary"]["passed_steps"] == 5

    @patch('trackiq_core.distributed_validator.train_single_process')
    @patch('trackiq_core.distributed_validator.train_multi_process')
    def test_run_validation_with_differences(self, mock_multi_train, mock_single_train):
        """Test validation with loss differences exceeding tolerance."""
        mock_single_train.return_value = [1.0, 0.9, 0.8, 0.7, 0.6]
        mock_multi_train.return_value = [1.5, 1.4, 1.3, 1.2, 1.1]  # Large differences

        validator = DistributedValidator()
        config = DistributedValidationConfig(num_steps=5, loss_tolerance=0.01)
        results = validator.run_validation(config)

        assert results["summary"]["overall_pass"] is False
        assert results["summary"]["passed_steps"] == 0

    def test_save_baseline(self):
        """Test saving baseline."""
        validator = DistributedValidator()
        results = {
            "comparisons": [
                {"step": 0, "relative_delta": 0.01},
                {"step": 1, "relative_delta": 0.02},
            ]
        }

        with patch.object(validator.regression_detector, 'save_baseline') as mock_save:
            validator.save_baseline("test_baseline", results)
            mock_save.assert_called_once()

    def test_detect_regression(self):
        """Test regression detection."""
        validator = DistributedValidator()
        results = {
            "comparisons": [
                {"step": 0, "relative_delta": 0.01},
                {"step": 1, "relative_delta": 0.02},
            ],
            "config": {"regression_threshold": 5.0}
        }

        mock_result = {"has_regressions": False, "regressions": {}}
        with patch.object(validator.regression_detector, 'detect_regressions', return_value=mock_result):
            regression = validator.detect_regression("test_baseline", results)
            assert regression == mock_result

    def test_generate_report_json(self):
        """Test JSON report generation."""
        validator = DistributedValidator()
        results = {"test": "data"}

        report = validator.generate_report(results, output_format="json")
        parsed = json.loads(report)
        assert parsed == results

    def test_generate_report_text(self):
        """Test text report generation."""
        validator = DistributedValidator()
        results = {
            "config": {"num_steps": 10, "num_processes": 2, "loss_tolerance": 0.01},
            "summary": {
                "total_steps": 10,
                "passed_steps": 8,
                "failed_steps": 2,
                "pass_rate": 0.8,
                "overall_pass": False
            }
        }

        report = validator.generate_report(results, output_format="text")
        assert "DISTRIBUTED TRAINING VALIDATION REPORT" in report
        assert "Total Steps: 10" in report
        assert "Overall: FAIL" in report


class TestCLIntegration:
    """Tests for CLI integration."""

    @patch('autoperfpy.cli.DistributedValidator')
    def test_cli_run_benchmark_distributed(self, mock_validator_class):
        """Test CLI integration for distributed benchmark."""
        from autoperfpy.cli import run_benchmark_distributed

        # Mock args
        args = MagicMock()
        args.steps = 50
        args.processes = 3
        args.tolerance = 0.05
        args.baseline = None
        args.save_baseline = None
        args.output = None

        # Mock validator
        mock_validator = MagicMock()
        mock_results = {
            "summary": {
                "total_steps": 50,
                "passed_steps": 45,
                "failed_steps": 5,
                "pass_rate": 0.9,
                "overall_pass": True
            }
        }
        mock_validator.run_validation.return_value = mock_results
        mock_validator_class.return_value = mock_validator

        config = MagicMock()
        result = run_benchmark_distributed(args, config)

        assert result == mock_results
        mock_validator.run_validation.assert_called_once()
