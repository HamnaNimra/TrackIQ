"""Tests for fault_injector module."""

import json
import tempfile
from pathlib import Path

import pytest

from minicluster.runner import RunConfig, train_single_process
from minicluster.validators import FaultInjector, FaultType


class TestFaultInjector:
    """Tests for fault injection framework."""

    def test_injector_initialization(self):
        """Test fault injector can be initialized."""
        config = RunConfig(num_steps=10)
        injector = FaultInjector(config)
        assert injector.base_config == config

    def test_fault_injection_report_structure(self):
        """Test fault injection report has correct structure."""
        config = RunConfig(num_steps=10)
        injector = FaultInjector(config)

        # Run a simplified fault injection test
        report = injector.run_fault_injection_tests()

        assert report.num_faults >= 0
        assert report.num_detected >= 0
        assert report.num_missed >= 0
        assert len(report.results) > 0

    def test_slow_worker_detection(self):
        """Test fault injector can test slow worker detection."""
        config = RunConfig(num_steps=10)
        injector = FaultInjector(config)

        report = injector.run_fault_injection_tests()

        # Check that slow worker test was performed
        slow_worker_results = [r for r in report.results if r.fault_type == FaultType.SLOW_WORKER]
        assert len(slow_worker_results) > 0

    def test_gradient_anomaly_detection(self):
        """Test fault injector can test gradient anomaly detection."""
        config = RunConfig(num_steps=10)
        injector = FaultInjector(config)

        report = injector.run_fault_injection_tests()

        # Check that gradient anomaly test was performed
        anom_results = [r for r in report.results if r.fault_type == FaultType.GRADIENT_SYNC_ANOMALY]
        assert len(anom_results) > 0

    def test_timeout_detection(self):
        """Test fault injector can test timeout detection."""
        config = RunConfig(num_steps=10)
        injector = FaultInjector(config)

        report = injector.run_fault_injection_tests()

        # Check that timeout test was performed
        timeout_results = [r for r in report.results if r.fault_type == FaultType.WORKER_TIMEOUT]
        assert len(timeout_results) > 0

    def test_report_summary(self):
        """Test fault injection report generates summary."""
        config = RunConfig(num_steps=10)
        injector = FaultInjector(config)

        report = injector.run_fault_injection_tests()

        assert report.summary is not None
        assert "fault" in report.summary.lower()

    def test_report_save(self):
        """Test fault injection report can be saved to JSON."""
        config = RunConfig(num_steps=10)
        injector = FaultInjector(config)
        report = injector.run_fault_injection_tests()

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = str(Path(tmpdir) / "fault_report.json")
            injector.save_report(report, report_path)

            assert Path(report_path).exists()
            with open(report_path) as f:
                saved = json.load(f)
            assert "num_faults" in saved
            assert "results" in saved

    def test_report_to_dict(self):
        """Test fault injection report can be converted to dict."""
        config = RunConfig(num_steps=10)
        injector = FaultInjector(config)
        report = injector.run_fault_injection_tests()

        report_dict = report.to_dict()
        assert isinstance(report_dict, dict)
        assert "num_faults" in report_dict
        assert "num_detected" in report_dict
        assert "num_missed" in report_dict

    def test_injector_print_report(self, capsys):
        """Test fault injector can print human-readable report."""
        config = RunConfig(num_steps=5)
        injector = FaultInjector(config)
        report = injector.run_fault_injection_tests()

        injector.print_report(report)

        captured = capsys.readouterr()
        assert "FAULT INJECTION TEST REPORT" in captured.out

    def test_detection_results_have_fault_types(self):
        """Test that all detection results have valid fault types."""
        config = RunConfig(num_steps=10)
        injector = FaultInjector(config)
        report = injector.run_fault_injection_tests()

        valid_types = {FaultType.SLOW_WORKER, FaultType.GRADIENT_SYNC_ANOMALY, FaultType.WORKER_TIMEOUT}

        for result in report.results:
            assert result.fault_type in valid_types
            assert hasattr(result, "was_detected")
            assert isinstance(result.was_detected, bool)
            assert hasattr(result, "reason")

    def test_detection_results_consistency(self):
        """Test that detection results are internally consistent."""
        config = RunConfig(num_steps=5)
        injector = FaultInjector(config)
        report = injector.run_fault_injection_tests()

        detected_count = sum(1 for r in report.results if r.was_detected)
        missed_count = sum(1 for r in report.results if not r.was_detected)

        assert detected_count == report.num_detected
        assert missed_count == report.num_missed
        assert report.num_detected + report.num_missed == report.num_faults
