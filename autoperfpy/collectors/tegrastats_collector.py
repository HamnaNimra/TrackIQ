"""Tegrastats collector for NVIDIA Jetson/DriveOS platforms.

This module provides a collector that gathers system metrics from NVIDIA
Jetson and DriveOS platforms using the tegrastats utility. It supports:
- Live collection by spawning tegrastats process
- File-based collection from pre-recorded tegrastats output
- Real-time parsing of tegrastats output format

Tegrastats provides metrics including:
- Per-core CPU utilization and frequency
- GPU (GR3D) utilization and frequency
- RAM usage and EMC frequency
- Thermal zone temperatures

Example usage:
    from autoperfpy.collectors import TegrastatsCollector

    # Live collection (requires tegrastats available on system)
    collector = TegrastatsCollector(mode="live", interval_ms=1000)
    collector.start()

    for _ in range(60):
        metrics = collector.sample(time.time())
        print(f"GPU: {metrics['gpu_percent']:.1f}%, Temp: {metrics['temperature_c']:.1f}C")
        time.sleep(1)

    collector.stop()
    export = collector.export()

    # File-based collection
    collector = TegrastatsCollector(mode="file", filepath="tegrastats.log")
    collector.start()
    # ... process all lines from file

Authors:
    AutoPerfPy Team
"""

import subprocess
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

from trackiq_core.hardware.env import command_available
from trackiq_core.collectors import CollectorBase, CollectorExport

# Import tegrastats parser from core module
try:
    from ..core.tegrastats import TegrastatsParser, TegrastatsSnapshot
except ImportError:
    TegrastatsParser = None
    TegrastatsSnapshot = None


class TegrastatsCollector(CollectorBase):
    """Collector for NVIDIA Jetson/DriveOS tegrastats metrics.

    This collector provides system metrics from NVIDIA embedded platforms
    by parsing tegrastats output. It supports both live collection (spawning
    the tegrastats process) and file-based collection (reading from a log file).

    Capability Flags:
        supports_power: False - tegrastats doesn't report power directly
        supports_utilization: True - CPU and GPU utilization
        supports_temperature: True - Multiple thermal zones
        supports_memory: True - RAM usage and EMC frequency

    Attributes:
        mode: Collection mode ("live" or "file")
        filepath: Path to tegrastats log file (file mode only)
        interval_ms: Sampling interval in milliseconds (live mode only)
    """

    # Capability flags
    supports_power = False  # tegrastats doesn't report power directly
    supports_utilization = True
    supports_temperature = True
    supports_memory = True

    def __init__(
        self,
        mode: str = "live",
        filepath: Optional[str] = None,
        interval_ms: int = 1000,
        config: Optional[Dict[str, Any]] = None,
        name: str = "TegrastatsCollector",
    ):
        """Initialize the Tegrastats collector.

        Args:
            mode: Collection mode - "live" to spawn tegrastats process,
                  "file" to read from a log file
            filepath: Path to tegrastats log file (required for file mode)
            interval_ms: Sampling interval in milliseconds for live mode
            config: Optional configuration dictionary with:
                - warmup_samples: Number of warmup samples to mark (default: 0)
                - tegrastats_path: Path to tegrastats binary (default: "tegrastats")
                - buffer_size: Max lines to buffer in live mode (default: 100)
            name: Name for this collector instance

        Raises:
            ValueError: If mode is "file" but no filepath provided
        """
        super().__init__(name, config)

        if mode not in ("live", "file"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'live' or 'file'")

        if mode == "file" and not filepath:
            raise ValueError("filepath is required for file mode")

        self.mode = mode
        self.filepath = filepath
        self.interval_ms = interval_ms

        # Configuration
        self._cfg = {
            "warmup_samples": 0,
            "tegrastats_path": "tegrastats",
            "buffer_size": 100,
            **(config or {}),
        }

        self._sample_index = 0
        self._process = None
        self._reader_thread = None
        self._line_buffer: deque = deque(maxlen=self._cfg["buffer_size"])
        self._file_lines: List[str] = []
        self._file_index = 0
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start tegrastats collection.

        For live mode: Spawns the tegrastats process and starts reading output.
        For file mode: Loads the log file into memory.

        Raises:
            RuntimeError: If tegrastats process fails to start (live mode)
            FileNotFoundError: If log file not found (file mode)
        """
        if TegrastatsParser is None:
            raise ImportError(
                "TegrastatsParser not available. "
                "Ensure autoperfpy.core.tegrastats module is accessible."
            )

        self._is_running = True
        self._start_time = time.time()
        self._sample_index = 0
        self._samples.clear()
        self._stop_event.clear()

        if self.mode == "live":
            self._start_live_collection()
        else:
            self._start_file_collection()

    def _start_live_collection(self) -> None:
        """Start live tegrastats collection."""
        tegrastats_path = self._cfg["tegrastats_path"]
        interval_arg = f"--interval {self.interval_ms}"

        try:
            self._process = subprocess.Popen(
                f"{tegrastats_path} {interval_arg}",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"tegrastats not found at '{tegrastats_path}'. "
                "Ensure you're running on a Jetson/DriveOS platform."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to start tegrastats: {e}")

        # Start reader thread
        self._reader_thread = threading.Thread(target=self._read_tegrastats_output)
        self._reader_thread.daemon = True
        self._reader_thread.start()

    def _start_file_collection(self) -> None:
        """Start file-based tegrastats collection."""
        try:
            with open(self.filepath, "r") as f:
                self._file_lines = [
                    line.strip() for line in f if line.strip() and "RAM" in line
                ]
        except FileNotFoundError:
            raise FileNotFoundError(f"Tegrastats file not found: {self.filepath}")

        self._file_index = 0

        if not self._file_lines:
            raise ValueError(f"No valid tegrastats data found in {self.filepath}")

    def _read_tegrastats_output(self) -> None:
        """Reader thread for live tegrastats output."""
        if self._process is None:
            return

        try:
            for line in self._process.stdout:
                if self._stop_event.is_set():
                    break
                line = line.strip()
                if line and "RAM" in line:  # Valid tegrastats line
                    self._line_buffer.append((time.time(), line))
        except Exception:
            pass  # Process terminated or pipe closed

    def sample(self, timestamp: float) -> Optional[Dict[str, Any]]:
        """Collect tegrastats metrics at the given timestamp.

        For live mode: Returns the most recent tegrastats reading.
        For file mode: Returns the next line from the file.

        Args:
            timestamp: Unix timestamp for this sample

        Returns:
            Dictionary containing system metrics:
            - cpu_percent: Average CPU utilization (0-100)
            - cpu_per_core: List of per-core utilization
            - gpu_percent: GPU (GR3D) utilization (0-100)
            - gpu_frequency_mhz: GPU frequency
            - memory_used_mb: RAM used in MB
            - memory_total_mb: Total RAM in MB
            - memory_percent: RAM utilization percentage
            - emc_frequency_mhz: Memory controller frequency
            - temperature_c: Maximum temperature across zones
            - thermal_zones: Dictionary of all thermal readings
            - is_warmup: Boolean indicating warmup period

        Raises:
            RuntimeError: If collector is not running
        """
        if not self._is_running:
            raise RuntimeError("Collector not started. Call start() first.")

        if self.mode == "live":
            return self._sample_live(timestamp)
        else:
            return self._sample_file(timestamp)

    def _sample_live(self, timestamp: float) -> Optional[Dict[str, Any]]:
        """Get sample from live tegrastats process."""
        if not self._line_buffer:
            # No data available yet
            return None

        # Get most recent line
        line_timestamp, line = self._line_buffer[-1]

        try:
            snapshot = TegrastatsParser.parse_line(line)
            metrics = self._snapshot_to_metrics(snapshot)
            metrics["is_warmup"] = self._sample_index < self._cfg.get(
                "warmup_samples", 0
            )

            # Store sample
            metadata = {
                "sample_index": self._sample_index,
                "line_timestamp": line_timestamp,
                "mode": "live",
            }
            self._store_sample(timestamp, metrics, metadata)
            self._sample_index += 1

            return metrics
        except Exception as e:
            return {"error": str(e)}

    def _sample_file(self, timestamp: float) -> Optional[Dict[str, Any]]:
        """Get next sample from file."""
        if self._file_index >= len(self._file_lines):
            # End of file
            return None

        line = self._file_lines[self._file_index]
        self._file_index += 1

        try:
            snapshot = TegrastatsParser.parse_line(line)
            metrics = self._snapshot_to_metrics(snapshot)
            metrics["is_warmup"] = self._sample_index < self._cfg.get(
                "warmup_samples", 0
            )

            # Store sample
            metadata = {
                "sample_index": self._sample_index,
                "file_line": self._file_index,
                "mode": "file",
            }
            self._store_sample(timestamp, metrics, metadata)
            self._sample_index += 1

            return metrics
        except Exception as e:
            return {"error": str(e)}

    def _snapshot_to_metrics(self, snapshot: TegrastatsSnapshot) -> Dict[str, Any]:
        """Convert TegrastatsSnapshot to metrics dictionary.

        Args:
            snapshot: Parsed tegrastats snapshot

        Returns:
            Dictionary with flattened metrics
        """
        metrics = {
            # CPU metrics
            "cpu_percent": snapshot.cpu_avg_utilization,
            "cpu_max_percent": snapshot.cpu_max_utilization,
            "cpu_per_core": [
                {
                    "core_id": c.core_id,
                    "utilization": c.utilization_percent,
                    "frequency_mhz": c.frequency_mhz,
                }
                for c in snapshot.cpu_cores
            ],
            "cpu_num_cores": snapshot.num_cores,
            # GPU metrics
            "gpu_percent": snapshot.gpu.utilization_percent,
            "gpu_frequency_mhz": snapshot.gpu.frequency_mhz,
            # Memory metrics
            "memory_used_mb": snapshot.memory.used_mb,
            "memory_total_mb": snapshot.memory.total_mb,
            "memory_percent": snapshot.memory.utilization_percent,
            "memory_available_mb": snapshot.memory.available_mb,
            "emc_frequency_mhz": snapshot.memory.emc_frequency_mhz,
            # Thermal metrics
            "temperature_c": snapshot.thermal.max_temp_c,
            "thermal_zones": {
                "cpu": snapshot.thermal.cpu_temp_c,
                "gpu": snapshot.thermal.gpu_temp_c,
                "aux": snapshot.thermal.aux_temp_c,
                "ao": snapshot.thermal.ao_temp_c,
                "tdiode": snapshot.thermal.tdiode_temp_c,
                "tj": snapshot.thermal.tj_temp_c,
            },
            # APE frequency
            "ape_frequency_mhz": snapshot.ape_frequency_mhz,
        }

        return metrics

    def stop(self) -> None:
        """Stop tegrastats collection.

        For live mode: Terminates the tegrastats process.
        Safe to call multiple times.
        """
        if self._is_running:
            self._is_running = False
            self._end_time = time.time()
            self._stop_event.set()

            if self._process is not None:
                try:
                    self._process.terminate()
                    self._process.wait(timeout=2)
                except Exception:
                    try:
                        self._process.kill()
                    except Exception:
                        pass
                self._process = None

            if self._reader_thread is not None:
                self._reader_thread.join(timeout=1)
                self._reader_thread = None

    def export(self) -> CollectorExport:
        """Export all collected tegrastats metrics.

        Returns:
            CollectorExport containing all samples and summary statistics
        """
        summary = self._calculate_summary()

        return CollectorExport(
            collector_name=self.name,
            start_time=self._start_time,
            end_time=self._end_time,
            samples=self._samples,
            summary=summary,
            config={
                **self._cfg,
                "mode": self.mode,
                "filepath": self.filepath,
                "interval_ms": self.interval_ms,
            },
        )

    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics for collected samples.

        Returns:
            Dictionary with summary statistics
        """
        if not self._samples:
            return {}

        warmup_count = self._cfg.get("warmup_samples", 0)
        steady_samples = (
            self._samples[warmup_count:]
            if len(self._samples) > warmup_count
            else self._samples
        )

        def safe_mean(key: str) -> Optional[float]:
            values = [
                s.metrics.get(key)
                for s in steady_samples
                if s.metrics.get(key) is not None
            ]
            return sum(values) / len(values) if values else None

        def safe_max(key: str) -> Optional[float]:
            values = [
                s.metrics.get(key)
                for s in steady_samples
                if s.metrics.get(key) is not None
            ]
            return max(values) if values else None

        def safe_min(key: str) -> Optional[float]:
            values = [
                s.metrics.get(key)
                for s in steady_samples
                if s.metrics.get(key) is not None
            ]
            return min(values) if values else None

        return {
            "sample_count": len(self._samples),
            "warmup_samples": min(warmup_count, len(self._samples)),
            "duration_seconds": (
                (self._end_time - self._start_time) if self._end_time else None
            ),
            "mode": self.mode,
            "cpu": {
                "mean_percent": safe_mean("cpu_percent"),
                "max_percent": safe_max("cpu_percent"),
                "min_percent": safe_min("cpu_percent"),
            },
            "gpu": {
                "mean_percent": safe_mean("gpu_percent"),
                "max_percent": safe_max("gpu_percent"),
                "min_percent": safe_min("gpu_percent"),
                "mean_frequency_mhz": safe_mean("gpu_frequency_mhz"),
            },
            "memory": {
                "mean_mb": safe_mean("memory_used_mb"),
                "max_mb": safe_max("memory_used_mb"),
                "min_mb": safe_min("memory_used_mb"),
                "total_mb": (
                    steady_samples[0].metrics.get("memory_total_mb")
                    if steady_samples
                    else None
                ),
                "mean_percent": safe_mean("memory_percent"),
            },
            "temperature": {
                "mean_c": safe_mean("temperature_c"),
                "max_c": safe_max("temperature_c"),
            },
        }

    def get_remaining_samples(self) -> int:
        """Get number of remaining samples (file mode only).

        Returns:
            Number of unread lines in file mode, -1 for live mode
        """
        if self.mode == "file":
            return len(self._file_lines) - self._file_index
        return -1

    @staticmethod
    def is_available() -> bool:
        """Check if tegrastats is available on the system.

        Returns:
            True if tegrastats command is available
        """
        return command_available("tegrastats", timeout=5)


__all__ = ["TegrastatsCollector"]
