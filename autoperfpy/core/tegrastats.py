"""Tegrastats parsing utilities for NVIDIA DriveOS/Jetson platforms.

This module provides utilities for parsing tegrastats output from NVIDIA
DriveOS (Orin/Thor) and Jetson platforms. Tegrastats reports real-time
system metrics including CPU, GPU, memory, and thermal information.

Example tegrastats output format:
RAM 264/28409MB (lfb 7004x4MB) CPU [0%@1817,0%@1817,0%@1817,0%@1817,0%@1817,0%@1817,0%@1817,0%@1817]
EMC_FREQ @2133 GR3D_FREQ 0%@1109 APE 245 AUX@30C CPU@31.5C Tdiode@30.75C AO@31C GPU@30.5C tj@41C

# Classes:
- TegrastatsSnapshot: Represents a single tegrastats reading.
- TegrastatsParser: Parses tegrastats output lines into TegrastatsSnapshot objects.
- TegrastatsCalculator: Computes aggregate statistics from multiple snapshots.
- TegrastatsAggregateStats: Holds aggregated statistics from multiple snapshots.
- CPUCoreStats: Statistics for a single CPU core.
- GPUStats: GPU (GR3D) statistics.
- MemoryStats: RAM and memory controller statistics.
- ThermalStats: Thermal zone temperatures.

Users can utilize these classes to monitor system performance, analyze resource usage,
and detect potential thermal or memory pressure issues.

Authors:
    Hamna Nimra
"""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class CPUCoreStats:
    """Statistics for a single CPU core."""

    core_id: int
    utilization_percent: float
    frequency_mhz: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "core_id": self.core_id,
            "utilization_percent": self.utilization_percent,
            "frequency_mhz": self.frequency_mhz,
        }


@dataclass
class GPUStats:
    """GPU (GR3D) statistics."""

    utilization_percent: float
    frequency_mhz: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "utilization_percent": self.utilization_percent,
            "frequency_mhz": self.frequency_mhz,
        }


@dataclass
class MemoryStats:
    """RAM and memory controller statistics."""

    used_mb: int
    total_mb: int
    lfb_blocks: int  # Largest free block count
    lfb_size_mb: int  # Size of each lfb block
    emc_frequency_mhz: int | None = None

    @property
    def utilization_percent(self) -> float:
        if self.total_mb == 0:
            return 0.0
        return (self.used_mb / self.total_mb) * 100

    @property
    def available_mb(self) -> int:
        return self.total_mb - self.used_mb

    @property
    def lfb_total_mb(self) -> int:
        return self.lfb_blocks * self.lfb_size_mb

    def to_dict(self) -> dict[str, Any]:
        return {
            "used_mb": self.used_mb,
            "total_mb": self.total_mb,
            "available_mb": self.available_mb,
            "utilization_percent": self.utilization_percent,
            "lfb_blocks": self.lfb_blocks,
            "lfb_size_mb": self.lfb_size_mb,
            "lfb_total_mb": self.lfb_total_mb,
            "emc_frequency_mhz": self.emc_frequency_mhz,
        }


@dataclass
class ThermalStats:
    """Thermal zone temperatures."""

    cpu_temp_c: float | None = None
    gpu_temp_c: float | None = None
    aux_temp_c: float | None = None
    ao_temp_c: float | None = None  # Always-On domain
    tdiode_temp_c: float | None = None
    tj_temp_c: float | None = None  # Junction temperature (max)
    pll_temp_c: float | None = None
    board_temp_c: float | None = None

    @property
    def max_temp_c(self) -> float:
        """Return the maximum temperature across all zones."""
        temps = [
            t
            for t in [
                self.cpu_temp_c,
                self.gpu_temp_c,
                self.aux_temp_c,
                self.ao_temp_c,
                self.tdiode_temp_c,
                self.tj_temp_c,
                self.pll_temp_c,
                self.board_temp_c,
            ]
            if t is not None
        ]
        return max(temps) if temps else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_temp_c": self.cpu_temp_c,
            "gpu_temp_c": self.gpu_temp_c,
            "aux_temp_c": self.aux_temp_c,
            "ao_temp_c": self.ao_temp_c,
            "tdiode_temp_c": self.tdiode_temp_c,
            "tj_temp_c": self.tj_temp_c,
            "pll_temp_c": self.pll_temp_c,
            "board_temp_c": self.board_temp_c,
            "max_temp_c": self.max_temp_c,
        }


@dataclass
class TegrastatsSnapshot:
    """A single tegrastats reading."""

    timestamp: datetime
    cpu_cores: list[CPUCoreStats]
    gpu: GPUStats
    memory: MemoryStats
    thermal: ThermalStats
    ape_frequency_mhz: int | None = None  # Audio Processing Engine
    raw_line: str = ""

    @property
    def cpu_avg_utilization(self) -> float:
        """Average CPU utilization across all cores."""
        if not self.cpu_cores:
            return 0.0
        return sum(c.utilization_percent for c in self.cpu_cores) / len(self.cpu_cores)

    @property
    def cpu_max_utilization(self) -> float:
        """Maximum CPU utilization across cores."""
        if not self.cpu_cores:
            return 0.0
        return max(c.utilization_percent for c in self.cpu_cores)

    @property
    def num_cores(self) -> int:
        """Number of CPU cores."""
        return len(self.cpu_cores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu": {
                "cores": [c.to_dict() for c in self.cpu_cores],
                "num_cores": self.num_cores,
                "avg_utilization_percent": self.cpu_avg_utilization,
                "max_utilization_percent": self.cpu_max_utilization,
            },
            "gpu": self.gpu.to_dict(),
            "memory": self.memory.to_dict(),
            "thermal": self.thermal.to_dict(),
            "ape_frequency_mhz": self.ape_frequency_mhz,
        }


class TegrastatsParser:
    """Parser for tegrastats output lines.

    Handles various tegrastats output formats from DriveOS and Jetson platforms.
    """

    # Regex patterns for parsing tegrastats output
    RAM_PATTERN = re.compile(r"RAM\s+(\d+)/(\d+)MB\s+\(lfb\s+(\d+)x(\d+)MB\)")
    CPU_PATTERN = re.compile(r"CPU\s+\[([^\]]+)\]")
    CPU_CORE_PATTERN = re.compile(r"(\d+)%@(\d+)")
    GPU_PATTERN = re.compile(r"GR3D_FREQ\s+(\d+)%@(\d+)")
    EMC_PATTERN = re.compile(r"EMC_FREQ\s+(?:(\d+)%)?@(\d+)")
    APE_PATTERN = re.compile(r"APE\s+(\d+)")

    # Thermal patterns - handle both formats: "CPU@31.5C" and "cpu@31.5C"
    THERMAL_PATTERNS = {
        "cpu": re.compile(r"(?:^|\s)CPU@([\d.]+)C", re.IGNORECASE),
        "gpu": re.compile(r"(?:^|\s)GPU@([\d.]+)C", re.IGNORECASE),
        "aux": re.compile(r"AUX@([\d.]+)C", re.IGNORECASE),
        "ao": re.compile(r"AO@([\d.]+)C", re.IGNORECASE),
        "tdiode": re.compile(r"Tdiode@([\d.]+)C", re.IGNORECASE),
        "tj": re.compile(r"tj@([\d.]+)C", re.IGNORECASE),
        "pll": re.compile(r"PLL@([\d.]+)C", re.IGNORECASE),
        "board": re.compile(r"(?:board|BOARD)@([\d.]+)C", re.IGNORECASE),
    }

    @classmethod
    def parse_line(cls, line: str, timestamp: datetime | None = None) -> TegrastatsSnapshot:
        """Parse a single tegrastats output line.

        Args:
            line: Raw tegrastats output line
            timestamp: Optional timestamp (defaults to now)

        Returns:
            TegrastatsSnapshot with parsed metrics
        """
        if timestamp is None:
            timestamp = datetime.now()

        cpu_cores = cls._parse_cpu(line)
        gpu = cls._parse_gpu(line)
        memory = cls._parse_memory(line)
        thermal = cls._parse_thermal(line)
        ape = cls._parse_ape(line)

        # Add EMC frequency to memory stats
        emc_match = cls.EMC_PATTERN.search(line)
        if emc_match:
            memory.emc_frequency_mhz = int(emc_match.group(2))

        return TegrastatsSnapshot(
            timestamp=timestamp,
            cpu_cores=cpu_cores,
            gpu=gpu,
            memory=memory,
            thermal=thermal,
            ape_frequency_mhz=ape,
            raw_line=line.strip(),
        )

    @classmethod
    def _parse_cpu(cls, line: str) -> list[CPUCoreStats]:
        """Parse CPU core statistics."""
        cores = []
        cpu_match = cls.CPU_PATTERN.search(line)
        if cpu_match:
            cpu_str = cpu_match.group(1)
            for i, core_match in enumerate(cls.CPU_CORE_PATTERN.finditer(cpu_str)):
                cores.append(
                    CPUCoreStats(
                        core_id=i,
                        utilization_percent=float(core_match.group(1)),
                        frequency_mhz=int(core_match.group(2)),
                    )
                )
        return cores

    @classmethod
    def _parse_gpu(cls, line: str) -> GPUStats:
        """Parse GPU (GR3D) statistics."""
        gpu_match = cls.GPU_PATTERN.search(line)
        if gpu_match:
            return GPUStats(
                utilization_percent=float(gpu_match.group(1)),
                frequency_mhz=int(gpu_match.group(2)),
            )
        return GPUStats(utilization_percent=0.0, frequency_mhz=0)

    @classmethod
    def _parse_memory(cls, line: str) -> MemoryStats:
        """Parse RAM and memory statistics."""
        ram_match = cls.RAM_PATTERN.search(line)
        if ram_match:
            return MemoryStats(
                used_mb=int(ram_match.group(1)),
                total_mb=int(ram_match.group(2)),
                lfb_blocks=int(ram_match.group(3)),
                lfb_size_mb=int(ram_match.group(4)),
            )
        return MemoryStats(used_mb=0, total_mb=0, lfb_blocks=0, lfb_size_mb=0)

    @classmethod
    def _parse_thermal(cls, line: str) -> ThermalStats:
        """Parse thermal zone temperatures."""
        thermal = ThermalStats()

        for zone, pattern in cls.THERMAL_PATTERNS.items():
            match = pattern.search(line)
            if match:
                temp = float(match.group(1))
                if zone == "cpu":
                    thermal.cpu_temp_c = temp
                elif zone == "gpu":
                    thermal.gpu_temp_c = temp
                elif zone == "aux":
                    thermal.aux_temp_c = temp
                elif zone == "ao":
                    thermal.ao_temp_c = temp
                elif zone == "tdiode":
                    thermal.tdiode_temp_c = temp
                elif zone == "tj":
                    thermal.tj_temp_c = temp
                elif zone == "pll":
                    thermal.pll_temp_c = temp
                elif zone == "board":
                    thermal.board_temp_c = temp

        return thermal

    @classmethod
    def _parse_ape(cls, line: str) -> int | None:
        """Parse APE (Audio Processing Engine) frequency."""
        ape_match = cls.APE_PATTERN.search(line)
        if ape_match:
            return int(ape_match.group(1))
        return None

    @classmethod
    def parse_file(cls, filepath: str) -> list[TegrastatsSnapshot]:
        """Parse a file containing tegrastats output.

        Args:
            filepath: Path to file with tegrastats lines

        Returns:
            List of TegrastatsSnapshot objects
        """
        snapshots = []
        try:
            with open(filepath) as f:
                for line in f:
                    line = line.strip()
                    # Validate line matches RAM pattern (actual tegrastats format)
                    if line and cls.RAM_PATTERN.search(line):
                        try:
                            snapshot = cls.parse_line(line)
                            snapshots.append(snapshot)
                        except Exception:
                            continue  # Skip malformed lines
        except FileNotFoundError:
            raise FileNotFoundError(f"Tegrastats file not found: {filepath}")

        return snapshots


@dataclass
class TegrastatsAggregateStats:
    """Aggregated statistics from multiple tegrastats snapshots."""

    num_samples: int
    duration_seconds: float

    # CPU aggregate stats
    cpu_avg_utilization: float
    cpu_max_utilization: float
    cpu_min_utilization: float
    cpu_per_core_avg: list[float]

    # GPU aggregate stats
    gpu_avg_utilization: float
    gpu_max_utilization: float
    gpu_min_utilization: float
    gpu_avg_frequency_mhz: float

    # Memory aggregate stats
    ram_avg_used_mb: float
    ram_max_used_mb: int
    ram_avg_utilization: float
    ram_total_mb: int
    emc_avg_frequency_mhz: float | None

    # Thermal aggregate stats
    thermal_avg_temps: dict[str, float]
    thermal_max_temps: dict[str, float]
    thermal_max_observed: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "duration_seconds": self.duration_seconds,
            "cpu": {
                "avg_utilization": self.cpu_avg_utilization,
                "max_utilization": self.cpu_max_utilization,
                "min_utilization": self.cpu_min_utilization,
                "per_core_avg": self.cpu_per_core_avg,
            },
            "gpu": {
                "avg_utilization": self.gpu_avg_utilization,
                "max_utilization": self.gpu_max_utilization,
                "min_utilization": self.gpu_min_utilization,
                "avg_frequency_mhz": self.gpu_avg_frequency_mhz,
            },
            "memory": {
                "avg_used_mb": self.ram_avg_used_mb,
                "max_used_mb": self.ram_max_used_mb,
                "avg_utilization": self.ram_avg_utilization,
                "total_mb": self.ram_total_mb,
                "emc_avg_frequency_mhz": self.emc_avg_frequency_mhz,
            },
            "thermal": {
                "avg_temps": self.thermal_avg_temps,
                "max_temps": self.thermal_max_temps,
                "max_observed": self.thermal_max_observed,
            },
        }


class TegrastatsCalculator:
    """Calculate aggregate statistics from tegrastats snapshots."""

    @staticmethod
    def calculate_aggregates(snapshots: list[TegrastatsSnapshot]) -> TegrastatsAggregateStats:
        """Calculate aggregate statistics from a list of snapshots.

        Args:
            snapshots: List of TegrastatsSnapshot objects

        Returns:
            TegrastatsAggregateStats with computed aggregates
        """
        if not snapshots:
            return TegrastatsAggregateStats(
                num_samples=0,
                duration_seconds=0.0,
                cpu_avg_utilization=0.0,
                cpu_max_utilization=0.0,
                cpu_min_utilization=0.0,
                cpu_per_core_avg=[],
                gpu_avg_utilization=0.0,
                gpu_max_utilization=0.0,
                gpu_min_utilization=0.0,
                gpu_avg_frequency_mhz=0.0,
                ram_avg_used_mb=0.0,
                ram_max_used_mb=0,
                ram_avg_utilization=0.0,
                ram_total_mb=0,
                emc_avg_frequency_mhz=None,
                thermal_avg_temps={},
                thermal_max_temps={},
                thermal_max_observed=0.0,
            )

        n = len(snapshots)

        # Calculate duration
        if n > 1:
            duration = (snapshots[-1].timestamp - snapshots[0].timestamp).total_seconds()
        else:
            duration = 0.0

        # CPU stats
        cpu_utils = [s.cpu_avg_utilization for s in snapshots]
        cpu_avg = sum(cpu_utils) / n
        cpu_max = max(cpu_utils)
        cpu_min = min(cpu_utils)

        # Per-core averages
        num_cores = snapshots[0].num_cores if snapshots else 0
        per_core_avg = []
        for core_id in range(num_cores):
            core_utils = [s.cpu_cores[core_id].utilization_percent for s in snapshots if core_id < len(s.cpu_cores)]
            if core_utils:
                per_core_avg.append(sum(core_utils) / len(core_utils))

        # GPU stats
        gpu_utils = [s.gpu.utilization_percent for s in snapshots]
        gpu_freqs = [s.gpu.frequency_mhz for s in snapshots]

        # Memory stats
        ram_used = [s.memory.used_mb for s in snapshots]
        ram_utils = [s.memory.utilization_percent for s in snapshots]
        ram_total = snapshots[0].memory.total_mb if snapshots else 0

        emc_freqs = [s.memory.emc_frequency_mhz for s in snapshots if s.memory.emc_frequency_mhz]
        emc_avg = sum(emc_freqs) / len(emc_freqs) if emc_freqs else None

        # Thermal stats
        thermal_zones = ["cpu", "gpu", "aux", "ao", "tdiode", "tj", "pll", "board"]
        thermal_avg = {}
        thermal_max = {}

        for zone in thermal_zones:
            temps = []
            for s in snapshots:
                temp = getattr(s.thermal, f"{zone}_temp_c", None)
                if temp is not None:
                    temps.append(temp)
            if temps:
                thermal_avg[zone] = sum(temps) / len(temps)
                thermal_max[zone] = max(temps)

        max_observed = max((s.thermal.max_temp_c for s in snapshots), default=0.0)

        return TegrastatsAggregateStats(
            num_samples=n,
            duration_seconds=duration,
            cpu_avg_utilization=cpu_avg,
            cpu_max_utilization=cpu_max,
            cpu_min_utilization=cpu_min,
            cpu_per_core_avg=per_core_avg,
            gpu_avg_utilization=sum(gpu_utils) / n,
            gpu_max_utilization=max(gpu_utils),
            gpu_min_utilization=min(gpu_utils),
            gpu_avg_frequency_mhz=sum(gpu_freqs) / n,
            ram_avg_used_mb=sum(ram_used) / n,
            ram_max_used_mb=max(ram_used),
            ram_avg_utilization=sum(ram_utils) / n,
            ram_total_mb=ram_total,
            emc_avg_frequency_mhz=emc_avg,
            thermal_avg_temps=thermal_avg,
            thermal_max_temps=thermal_max,
            thermal_max_observed=max_observed,
        )

    @staticmethod
    def detect_thermal_throttling(
        snapshots: list[TegrastatsSnapshot],
        throttle_temp_c: float = 85.0,
    ) -> dict[str, Any]:
        """Detect potential thermal throttling events.

        Args:
            snapshots: List of TegrastatsSnapshot objects
            throttle_temp_c: Temperature threshold for throttling detection

        Returns:
            Dictionary with throttling analysis
        """
        if not snapshots:
            return {
                "throttle_events": 0,
                "throttle_percentage": 0.0,
                "max_temp_observed": 0.0,
                "avg_temp_during_throttle": 0.0,
            }

        throttle_events = 0
        throttle_temps = []

        for s in snapshots:
            if s.thermal.max_temp_c >= throttle_temp_c:
                throttle_events += 1
                throttle_temps.append(s.thermal.max_temp_c)

        return {
            "throttle_events": throttle_events,
            "throttle_percentage": (throttle_events / len(snapshots)) * 100,
            "max_temp_observed": max(s.thermal.max_temp_c for s in snapshots),
            "avg_temp_during_throttle": (sum(throttle_temps) / len(throttle_temps) if throttle_temps else 0.0),
            "throttle_threshold_c": throttle_temp_c,
        }

    @staticmethod
    def detect_memory_pressure(
        snapshots: list[TegrastatsSnapshot],
        pressure_threshold_percent: float = 90.0,
    ) -> dict[str, Any]:
        """Detect memory pressure events.

        Args:
            snapshots: List of TegrastatsSnapshot objects
            pressure_threshold_percent: Memory utilization threshold

        Returns:
            Dictionary with memory pressure analysis
        """
        if not snapshots:
            return {
                "pressure_events": 0,
                "pressure_percentage": 0.0,
                "max_utilization": 0.0,
                "avg_utilization": 0.0,
            }

        pressure_events = sum(1 for s in snapshots if s.memory.utilization_percent >= pressure_threshold_percent)

        return {
            "pressure_events": pressure_events,
            "pressure_percentage": (pressure_events / len(snapshots)) * 100,
            "max_utilization": max(s.memory.utilization_percent for s in snapshots),
            "avg_utilization": sum(s.memory.utilization_percent for s in snapshots) / len(snapshots),
            "pressure_threshold_percent": pressure_threshold_percent,
        }


__all__ = [
    "CPUCoreStats",
    "GPUStats",
    "MemoryStats",
    "ThermalStats",
    "TegrastatsSnapshot",
    "TegrastatsParser",
    "TegrastatsAggregateStats",
    "TegrastatsCalculator",
]
