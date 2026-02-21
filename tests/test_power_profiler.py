"""Tests for trackiq_core.power_profiler."""

from dataclasses import replace

import pytest

from trackiq_core.power_profiler import (
    PowerProfiler,
    PowerReading,
    PowerSourceUnavailableError,
    RocmSmiReader,
    SimulatedPowerReader,
    detect_power_source,
)
from trackiq_core.schema import Metrics


class _DummyReader:
    """Deterministic test reader."""

    def __init__(self, values):
        self._values = list(values)
        self._idx = 0

    @classmethod
    def is_available(cls) -> bool:
        return True

    def read(self) -> PowerReading:
        value = self._values[min(self._idx, len(self._values) - 1)]
        self._idx += 1
        return PowerReading(power_watts=float(value), temperature_celsius=55.0)


def _base_metrics() -> Metrics:
    return Metrics(
        throughput_samples_per_sec=100.0,
        latency_p50_ms=10.0,
        latency_p95_ms=12.0,
        latency_p99_ms=14.0,
        memory_utilization_percent=60.0,
        communication_overhead_percent=None,
        power_consumption_watts=None,
    )


def test_simulated_power_reader_is_available_true() -> None:
    """Simulated reader is always available."""
    assert SimulatedPowerReader.is_available() is True


def test_simulated_power_reader_within_tdp_bounds(monkeypatch) -> None:
    """Simulated power should stay within idle and TDP bounds."""
    monkeypatch.setattr("trackiq_core.power_profiler.psutil.cpu_percent", lambda interval=None: 80.0)
    reader = SimulatedPowerReader(tdp_watts=120.0, idle_floor_watts=30.0)
    sample = reader.read()
    assert 30.0 <= sample.power_watts <= 120.0


def test_detect_power_source_falls_back_to_simulated(monkeypatch) -> None:
    """detect_power_source should return simulated when others unavailable."""
    monkeypatch.setattr("trackiq_core.power_profiler.RocmSmiReader.is_available", classmethod(lambda cls: False))
    monkeypatch.setattr("trackiq_core.power_profiler.TegrastatsReader.is_available", classmethod(lambda cls: False))
    source = detect_power_source()
    assert isinstance(source, SimulatedPowerReader)


def test_end_session_energy_computation(monkeypatch) -> None:
    """Energy should equal mean_power * elapsed_time."""
    reader = _DummyReader([100.0, 110.0, 90.0, 100.0])
    profiler = PowerProfiler(reader)
    times = iter([100.0, 101.0, 102.0, 103.0])
    monkeypatch.setattr("trackiq_core.power_profiler.time.time", lambda: next(times))
    profiler.start_session()
    profiler.record_step(0, 50.0)
    profiler.record_step(1, 50.0)
    summary = profiler.end_session()
    assert summary.mean_power_watts == pytest.approx(100.0)
    assert summary.total_energy_joules == pytest.approx(300.0)


def test_to_metrics_update_returns_new_object_no_mutation(monkeypatch) -> None:
    """to_metrics_update must not mutate input Metrics object."""
    reader = _DummyReader([100.0, 100.0, 100.0])
    profiler = PowerProfiler(reader)
    times = iter([1.0, 2.0, 3.0])
    monkeypatch.setattr("trackiq_core.power_profiler.time.time", lambda: next(times))
    metrics = _base_metrics()
    original = replace(metrics)
    profiler.start_session()
    profiler.record_step(0, 120.0)
    profiler.end_session()
    updated = profiler.to_metrics_update(metrics)
    assert updated is not metrics
    assert metrics == original
    assert updated.power_consumption_watts is not None


def test_to_tool_payload_contains_step_readings(monkeypatch) -> None:
    """Tool payload should include per-step readings."""
    reader = _DummyReader([90.0, 95.0, 92.0])
    profiler = PowerProfiler(reader)
    times = iter([10.0, 11.0, 12.0])
    monkeypatch.setattr("trackiq_core.power_profiler.time.time", lambda: next(times))
    profiler.start_session()
    profiler.record_step(0, 80.0)
    profiler.end_session()
    payload = profiler.to_tool_payload()
    assert isinstance(payload, dict)
    assert "power_profile" in payload
    assert len(payload["power_profile"]["step_readings"]) >= 2


def test_full_session_lifecycle_updates_metrics(monkeypatch) -> None:
    """Full session should populate non-null power fields."""
    reader = _DummyReader([120.0, 130.0, 110.0, 125.0, 128.0])
    profiler = PowerProfiler(reader)
    times = iter([100.0, 101.0, 102.0, 103.0, 104.0])
    monkeypatch.setattr("trackiq_core.power_profiler.time.time", lambda: next(times))
    profiler.start_session()
    profiler.record_step(0, 100.0)
    profiler.record_step(1, 120.0)
    profiler.record_step(2, 110.0)
    profiler.end_session()
    updated = profiler.to_metrics_update(_base_metrics())
    assert updated.power_consumption_watts is not None
    assert updated.energy_per_step_joules is not None
    assert updated.performance_per_watt is not None


def test_rocm_unavailable_raises_clear_error(monkeypatch) -> None:
    """Rocm reader should raise clear error when rocm-smi is unavailable."""
    monkeypatch.setattr("trackiq_core.power_profiler.shutil.which", lambda _: None)
    with pytest.raises(PowerSourceUnavailableError, match="rocm-smi"):
        RocmSmiReader()

