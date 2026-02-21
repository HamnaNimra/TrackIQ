"""Unified power profiling for TrackIQ tools.

This module provides runtime-selectable power readers and a session-based
profiler that computes derived efficiency metrics for `trackiq_core.schema.Metrics`.

Integration Notes
-----------------
```python
# In AutoPerfPy inference runner:
profiler = PowerProfiler(detect_power_source())
profiler.start_session()
for step, batch in enumerate(dataloader):
    throughput = run_inference(batch)
    profiler.record_step(step, throughput)
profiler.end_session()
result.metrics = profiler.to_metrics_update(result.metrics)
result.tool_payload = profiler.to_tool_payload()

# In MiniCluster distributed runner:
profiler = PowerProfiler(SimulatedPowerReader(tdp_watts=150))
profiler.start_session()
for step in range(num_steps):
    throughput = train_step()
    profiler.record_step(step, throughput)
profiler.end_session()
result.metrics = profiler.to_metrics_update(result.metrics)
result.tool_payload = profiler.to_tool_payload()
```
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, replace
from statistics import mean
from typing import Any, Dict, List, Optional, Protocol

try:
    import psutil
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal envs
    psutil = None  # type: ignore[assignment]

from trackiq_core.schema import Metrics


LOGGER = logging.getLogger(__name__)


class PowerSourceUnavailableError(RuntimeError):
    """Raised when a requested power source tool/hardware is not available."""


@dataclass
class PowerReading:
    """Single sampled power/thermal reading."""

    power_watts: float
    temperature_celsius: Optional[float]


class PowerReader(Protocol):
    """Protocol for pluggable power source readers."""

    @classmethod
    def is_available(cls) -> bool:
        """Return True if this reader can run in current environment."""

    def read(self) -> PowerReading:
        """Read a single instantaneous power/thermal sample."""


class TegrastatsReader:
    """Read Jetson board power and thermal data from `tegrastats` output."""

    @classmethod
    def is_available(cls) -> bool:
        return shutil.which("tegrastats") is not None

    def __init__(self) -> None:
        if not self.is_available():
            raise PowerSourceUnavailableError(
                "TegrastatsReader unavailable: 'tegrastats' command not found."
            )

    def read(self) -> PowerReading:
        if not self.is_available():
            raise PowerSourceUnavailableError(
                "TegrastatsReader unavailable: 'tegrastats' command not found."
            )

        proc = subprocess.Popen(
            ["tegrastats", "--interval", "1000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            line = ""
            if proc.stdout is not None:
                line = proc.stdout.readline().strip()
            if not line:
                raise PowerSourceUnavailableError(
                    "TegrastatsReader could not read tegrastats output."
                )
            return self._parse_tegrastats_line(line)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except Exception:
                proc.kill()

    @staticmethod
    def _parse_tegrastats_line(line: str) -> PowerReading:
        # Common format fragments include "VDD_IN 12345mW/..." and "GPU@57C".
        power_match = re.search(r"VDD_IN\s+(\d+(?:\.\d+)?)mW", line)
        temp_match = re.search(r"GPU@(\d+(?:\.\d+)?)C", line)
        if not power_match:
            raise PowerSourceUnavailableError(
                f"TegrastatsReader could not parse board power from: {line}"
            )
        power_watts = float(power_match.group(1)) / 1000.0
        temp_c = float(temp_match.group(1)) if temp_match else None
        return PowerReading(power_watts=power_watts, temperature_celsius=temp_c)


class RocmSmiReader:
    """Read AMD GPU power and thermal data from `rocm-smi` output."""

    @classmethod
    def is_available(cls) -> bool:
        return shutil.which("rocm-smi") is not None

    def __init__(self) -> None:
        if not self.is_available():
            raise PowerSourceUnavailableError(
                "RocmSmiReader unavailable: 'rocm-smi' command not found."
            )

    def read(self) -> PowerReading:
        if not self.is_available():
            raise PowerSourceUnavailableError(
                "RocmSmiReader unavailable: 'rocm-smi' command not found."
            )
        try:
            result = subprocess.run(
                ["rocm-smi", "--showpower", "--showtemp", "--json"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except Exception as exc:
            raise PowerSourceUnavailableError(
                f"RocmSmiReader failed to execute rocm-smi: {exc}"
            ) from exc

        output = (result.stdout or "").strip()
        if not output:
            raise PowerSourceUnavailableError("RocmSmiReader received empty output.")
        return self._parse_rocm_smi_output(output)

    @staticmethod
    def _parse_rocm_smi_output(output: str) -> PowerReading:
        # Prefer JSON parsing first.
        try:
            payload = json.loads(output)
            if isinstance(payload, dict):
                device = next(iter(payload.values()))
                if isinstance(device, dict):
                    power = None
                    temp = None
                    for key, value in device.items():
                        low = key.lower()
                        if power is None and "power" in low:
                            power = _extract_float(value)
                        if temp is None and "temp" in low:
                            temp = _extract_float(value)
                    if power is not None:
                        return PowerReading(power_watts=power, temperature_celsius=temp)
        except Exception:
            pass

        # Fallback regex for plain text output.
        power_match = re.search(r"(\d+(?:\.\d+)?)\s*W", output)
        temp_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:C|c)", output)
        if not power_match:
            raise PowerSourceUnavailableError(
                "RocmSmiReader could not parse power draw from rocm-smi output."
            )
        power_watts = float(power_match.group(1))
        temp_c = float(temp_match.group(1)) if temp_match else None
        return PowerReading(power_watts=power_watts, temperature_celsius=temp_c)


class SimulatedPowerReader:
    """Simulate power readings from CPU utilization and configured TDP."""

    def __init__(self, tdp_watts: float = 150.0, idle_floor_watts: float = 20.0):
        self.tdp_watts = float(tdp_watts)
        self.idle_floor_watts = float(idle_floor_watts)

    @classmethod
    def is_available(cls) -> bool:
        return True

    def read(self) -> PowerReading:
        if psutil is None:
            raise PowerSourceUnavailableError(
                "SimulatedPowerReader unavailable: 'psutil' is not installed. "
                "Install with `pip install psutil`."
            )
        utilization = float(psutil.cpu_percent(interval=None))
        utilization = max(0.0, min(100.0, utilization))
        span = max(0.0, self.tdp_watts - self.idle_floor_watts)
        power = self.idle_floor_watts + (utilization / 100.0) * span
        power = max(self.idle_floor_watts, min(self.tdp_watts, power))
        temperature = 35.0 + (utilization / 100.0) * 45.0
        return PowerReading(power_watts=power, temperature_celsius=temperature)


def detect_power_source() -> PowerReader:
    """Detect and instantiate the best available power source reader.

    Priority:
    1. ROCm SMI
    2. Tegrastats
    3. Simulated fallback
    """

    if RocmSmiReader.is_available():
        return RocmSmiReader()
    if TegrastatsReader.is_available():
        return TegrastatsReader()
    return SimulatedPowerReader()


@dataclass
class SessionSummary:
    """Computed power session summary."""

    mean_power_watts: Optional[float]
    peak_power_watts: Optional[float]
    total_energy_joules: Optional[float]
    performance_per_watt: Optional[float]
    mean_temperature_celsius: Optional[float]
    elapsed_seconds: Optional[float]


class PowerProfiler:
    """Session-based power profiler with derived efficiency metrics."""

    def __init__(self, reader: Optional[PowerReader] = None):
        try:
            self.reader = reader or detect_power_source()
            if not self.reader.is_available():
                raise PowerSourceUnavailableError(
                    f"{self.reader.__class__.__name__} is unavailable."
                )
        except PowerSourceUnavailableError as exc:
            LOGGER.warning(
                "Power source unavailable (%s). Falling back to SimulatedPowerReader.",
                exc,
            )
            self.reader = SimulatedPowerReader()

        self._started = False
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._readings: List[Dict[str, Any]] = []
        self._summary: Optional[SessionSummary] = None

    def start_session(self) -> None:
        """Begin power monitoring session and record initial reading."""
        self._started = True
        self._start_time = time.time()
        initial = self.reader.read()
        self._readings.append(
            {
                "step": -1,
                "timestamp": self._start_time,
                "throughput": None,
                "power_watts": initial.power_watts,
                "temperature_celsius": initial.temperature_celsius,
            }
        )

    def record_step(self, step: int, throughput: float) -> None:
        """Record one step's throughput and current power reading."""
        if not self._started:
            raise RuntimeError("Power profiling session not started.")
        sample = self.reader.read()
        self._readings.append(
            {
                "step": int(step),
                "timestamp": time.time(),
                "throughput": float(throughput),
                "power_watts": sample.power_watts,
                "temperature_celsius": sample.temperature_celsius,
            }
        )

    def end_session(self) -> SessionSummary:
        """End session and compute summary statistics."""
        if not self._started or self._start_time is None:
            raise RuntimeError("Power profiling session not started.")
        self._end_time = time.time()
        elapsed = max(0.0, self._end_time - self._start_time)

        power_values = [float(r["power_watts"]) for r in self._readings]
        step_rows = [r for r in self._readings if r["step"] >= 0]
        throughput_values = [
            float(r["throughput"])
            for r in step_rows
            if r.get("throughput") is not None
        ]
        temp_values = [
            float(r["temperature_celsius"])
            for r in self._readings
            if r.get("temperature_celsius") is not None
        ]

        mean_power = mean(power_values) if power_values else None
        peak_power = max(power_values) if power_values else None
        total_energy = (mean_power * elapsed) if mean_power is not None else None
        mean_throughput = mean(throughput_values) if throughput_values else None
        perf_per_watt = (
            (mean_throughput / mean_power)
            if mean_throughput is not None and mean_power is not None and mean_power > 0
            else None
        )
        mean_temp = mean(temp_values) if temp_values else None

        self._summary = SessionSummary(
            mean_power_watts=mean_power,
            peak_power_watts=peak_power,
            total_energy_joules=total_energy,
            performance_per_watt=perf_per_watt,
            mean_temperature_celsius=mean_temp,
            elapsed_seconds=elapsed,
        )
        return self._summary

    def to_metrics_update(self, metrics: Metrics) -> Metrics:
        """Return a new Metrics object with power fields populated."""
        if self._summary is None:
            raise RuntimeError("Power profiling session summary not available.")
        step_count = max(1, len([r for r in self._readings if r["step"] >= 0]))
        energy_per_step = (
            (self._summary.total_energy_joules / step_count)
            if self._summary.total_energy_joules is not None
            else None
        )
        return replace(
            metrics,
            power_consumption_watts=self._summary.mean_power_watts,
            energy_per_step_joules=energy_per_step,
            performance_per_watt=self._summary.performance_per_watt,
            temperature_celsius=self._summary.mean_temperature_celsius,
        )

    def to_tool_payload(self) -> Dict[str, Any]:
        """Return per-step power/thermal data and summary details."""
        if self._summary is None:
            raise RuntimeError("Power profiling session summary not available.")
        return {
            "power_profile": {
                "reader": self.reader.__class__.__name__,
                "start_time": self._start_time,
                "end_time": self._end_time,
                "step_readings": self._readings,
                "summary": {
                    "mean_power_watts": self._summary.mean_power_watts,
                    "peak_power_watts": self._summary.peak_power_watts,
                    "total_energy_joules": self._summary.total_energy_joules,
                    "performance_per_watt": self._summary.performance_per_watt,
                    "mean_temperature_celsius": self._summary.mean_temperature_celsius,
                    "elapsed_seconds": self._summary.elapsed_seconds,
                },
            }
        }


def _extract_float(value: Any) -> Optional[float]:
    """Extract first float from mixed textual value."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None
