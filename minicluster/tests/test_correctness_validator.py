"""Tests for correctness_validator module."""

import json
import tempfile
from pathlib import Path

import pytest

from minicluster.runner import RunConfig, train_single_process
from minicluster.validators import CorrectnessValidator


class TestCorrectnessValidator:
    """Tests for correctness validation."""

    def test_validator_initialization(self):
        """Test validator can be initialized."""
        validator = CorrectnessValidator(tolerance=0.01)
        assert validator.tolerance == 0.01

    def test_compare_identical_runs(self):
        """Test validator passes when runs are identical."""
        config = RunConfig(num_steps=20, seed=42)
        metrics = train_single_process(config)
        metrics_dict = metrics.to_dict()

        validator = CorrectnessValidator(tolerance=0.01)
        report = validator.compare_runs(metrics_dict, metrics_dict)

        assert report.overall_passed is True
        assert report.num_steps_failed == 0
        assert report.num_steps_passed == 20

    def test_compare_within_tolerance(self):
        """Test validator passes when runs differ within tolerance."""
        config = RunConfig(num_steps=15, seed=42)
        metrics1 = train_single_process(config)
        metrics1_dict = metrics1.to_dict()

        # Create slightly perturbed metrics within 0.5% tolerance
        metrics2_dict = json.loads(json.dumps(metrics1_dict))
        for step in metrics2_dict["steps"]:
            step["loss"] *= 1.002  # 0.2% increase

        validator = CorrectnessValidator(tolerance=0.005)  # 0.5% tolerance
        report = validator.compare_runs(metrics1_dict, metrics2_dict)

        assert report.overall_passed is True
        assert report.num_steps_passed == 15

    def test_compare_exceeds_tolerance(self):
        """Test validator fails when runs diverge beyond tolerance."""
        config = RunConfig(num_steps=10, seed=42)
        metrics1 = train_single_process(config)
        metrics1_dict = metrics1.to_dict()

        # Create significantly perturbed metrics beyond tolerance
        metrics2_dict = json.loads(json.dumps(metrics1_dict))
        for step in metrics2_dict["steps"]:
            step["loss"] *= 1.15  # 15% increase

        validator = CorrectnessValidator(tolerance=0.01)  # 1% tolerance
        report = validator.compare_runs(metrics1_dict, metrics2_dict)

        assert report.overall_passed is False
        assert report.num_steps_failed == 10

    def test_step_comparison_details(self):
        """Test step-by-step comparison details."""
        config = RunConfig(num_steps=5, seed=42)
        metrics1 = train_single_process(config)
        metrics1_dict = metrics1.to_dict()

        metrics2_dict = json.loads(json.dumps(metrics1_dict))
        metrics2_dict["steps"][0]["loss"] *= 1.05  # 5% vary on first step

        validator = CorrectnessValidator(tolerance=0.01)
        report = validator.compare_runs(metrics1_dict, metrics2_dict)

        assert len(report.step_comparisons) == 5
        assert not report.step_comparisons[0].passed  # First step should fail
        assert report.step_comparisons[1].passed  # Others should pass

    def test_comparison_mismatched_step_count(self):
        """Test validator raises error when step counts don't match."""
        config = RunConfig(num_steps=20, seed=42)
        metrics1 = train_single_process(config)
        metrics1_dict = metrics1.to_dict()

        metrics2_dict = json.loads(json.dumps(metrics1_dict))
        metrics2_dict["steps"] = metrics2_dict["steps"][:10]  # Truncate to 10 steps

        validator = CorrectnessValidator()
        with pytest.raises(ValueError):
            validator.compare_runs(metrics1_dict, metrics2_dict)

    def test_report_summary_generation(self):
        """Test report generates appropriate summary."""
        config = RunConfig(num_steps=10, seed=42)
        metrics_dict = train_single_process(config).to_dict()

        validator = CorrectnessValidator(tolerance=0.01)
        report = validator.compare_runs(metrics_dict, metrics_dict)

        assert "✓ PASSED" in report.summary
        assert "10" in report.summary

    def test_validator_print_report(self, capsys):
        """Test validator can print human-readable report."""
        config = RunConfig(num_steps=5, seed=42)
        metrics_dict = train_single_process(config).to_dict()

        validator = CorrectnessValidator()
        report = validator.compare_runs(metrics_dict, metrics_dict)
        validator.print_report(report, verbose=False)

        captured = capsys.readouterr()
        assert "CORRECTNESS VALIDATION REPORT" in captured.out
        assert "✓" in captured.out or "PASSED" in captured.out

    def test_validator_save_report(self):
        """Test validator can save report to JSON file."""
        config = RunConfig(num_steps=5, seed=42)
        metrics_dict = train_single_process(config).to_dict()

        validator = CorrectnessValidator()
        report = validator.compare_runs(metrics_dict, metrics_dict)

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = str(Path(tmpdir) / "report.json")
            validator.save_report(report, report_path)

            assert Path(report_path).exists()
            with open(report_path) as f:
                saved_report = json.load(f)
            assert saved_report["overall_passed"] is True

    def test_validator_validate_file_pair(self):
        """Test validator can validate metrics from files."""
        config = RunConfig(num_steps=10, seed=42)
        metrics = train_single_process(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            single_path = str(Path(tmpdir) / "single.json")
            multi_path = str(Path(tmpdir) / "multi.json")
            report_path = str(Path(tmpdir) / "report.json")

            from minicluster.runner import save_metrics

            save_metrics(metrics, single_path)
            save_metrics(metrics, multi_path)

            validator = CorrectnessValidator()
            report = validator.validate_file_pair(single_path, multi_path, report_path)

            assert report.overall_passed is True
            assert Path(report_path).exists()

    def test_validator_file_not_found(self):
        """Test validator raises error when metrics file doesn't exist."""
        validator = CorrectnessValidator()

        with pytest.raises(FileNotFoundError):
            validator.validate_file_pair("nonexistent1.json", "nonexistent2.json")

    def test_tolerance_strictness(self):
        """Test that stricter tolerance rejects more runs."""
        config = RunConfig(num_steps=10, seed=42)
        metrics1 = train_single_process(config)
        metrics1_dict = metrics1.to_dict()

        metrics2_dict = json.loads(json.dumps(metrics1_dict))
        for step in metrics2_dict["steps"]:
            step["loss"] *= 1.03  # 3% increase

        # Loose tolerance should pass
        validator_loose = CorrectnessValidator(tolerance=0.05)
        report_loose = validator_loose.compare_runs(metrics1_dict, metrics2_dict)
        assert report_loose.overall_passed is True

        # Strict tolerance should fail
        validator_strict = CorrectnessValidator(tolerance=0.01)
        report_strict = validator_strict.compare_runs(metrics1_dict, metrics2_dict)
        assert report_strict.overall_passed is False
