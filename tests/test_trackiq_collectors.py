"""Unit tests for trackiq collectors (GPU/CPU/memory metrics)."""

import time

import pytest

from trackiq.collectors import (
    CollectorBase,
    CollectorSample,
    CollectorExport,
    SyntheticCollector,
)
from trackiq.platform import get_memory_metrics, get_performance_metrics


class TestSyntheticCollector:
    """Verify synthetic collector returns correct metric shape and ranges."""

    def test_sample_returns_expected_keys(self):
        """SyntheticCollector.sample() returns latency, CPU, GPU, memory, power."""
        collector = SyntheticCollector(config={"seed": 42, "warmup_samples": 2})
        collector.start()
        try:
            m = collector.sample(time.time())
            assert m is not None
            assert "latency_ms" in m
            assert "cpu_percent" in m
            assert "gpu_percent" in m
            assert "memory_used_mb" in m
            assert "memory_total_mb" in m
            assert "memory_percent" in m
            assert "power_w" in m
            assert "is_warmup" in m
        finally:
            collector.stop()

    def test_sample_values_in_reasonable_range(self):
        """GPU/CPU/memory metrics are in valid ranges."""
        collector = SyntheticCollector(config={"seed": 42, "warmup_samples": 0})
        collector.start()
        try:
            m = collector.sample(time.time())
            assert 0 <= m["cpu_percent"] <= 100
            assert 0 <= m["gpu_percent"] <= 100
            assert m["memory_used_mb"] >= 0
            assert m["memory_total_mb"] > 0
            assert 0 <= m["memory_percent"] <= 100
            assert m["power_w"] >= 0
            assert m["latency_ms"] > 0
        finally:
            collector.stop()

    def test_export_has_summary_and_samples(self):
        """Export contains collector_name, samples, summary."""
        collector = SyntheticCollector(config={"seed": 42, "warmup_samples": 1})
        collector.start()
        collector.sample(time.time())
        collector.sample(time.time())
        collector.stop()
        export = collector.export()
        assert isinstance(export, CollectorExport)
        assert export.collector_name == "SyntheticCollector"
        assert len(export.samples) == 2
        assert (
            "latency" in export.summary or "latency_ms" in str(export.summary).lower()
        )


class TestPlatformGpuMetrics:
    """Verify platform GPU metrics (when nvidia-smi available)."""

    def test_get_memory_metrics_returns_dict_or_none(self):
        """get_memory_metrics returns dict with expected keys or None."""
        result = get_memory_metrics()
        if result is None:
            pytest.skip("nvidia-smi not available")
        assert "gpu_memory_used_mb" in result or "gpu_memory_total_mb" in result

    def test_get_performance_metrics_returns_dict_or_none(self):
        """get_performance_metrics returns dict or None."""
        result = get_performance_metrics()
        if result is None:
            pytest.skip("nvidia-smi not available")
        assert isinstance(result, dict)
