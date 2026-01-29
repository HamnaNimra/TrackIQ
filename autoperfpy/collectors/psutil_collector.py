"""Psutil collector for cross-platform system metrics.

This module provides a collector that uses psutil to gather system metrics
across all platforms (Linux, Windows, macOS). It captures:
- CPU utilization (overall and per-core)
- Memory usage (physical and swap)
- Disk I/O statistics
- Network I/O statistics
- Process-specific metrics (optional)

Requires: psutil (pip install psutil)

Example usage:
    from autoperfpy.collectors import PsutilCollector

    collector = PsutilCollector()
    collector.start()

    for _ in range(100):
        metrics = collector.sample(time.time())
        print(f"CPU: {metrics['cpu_percent']:.1f}%, Memory: {metrics['memory_percent']:.1f}%")
        time.sleep(0.1)

    collector.stop()
    export = collector.export()

Authors:
    AutoPerfPy Team
"""

import time
from typing import Any, Dict, List, Optional

from .base import CollectorBase, CollectorExport, CollectorSample


class PsutilCollector(CollectorBase):
    """Cross-platform system metrics collector using psutil.

    This collector provides comprehensive system metrics via the psutil
    library, which works across Linux, Windows, and macOS. It's useful
    for collecting CPU and memory metrics when GPU-specific tools are
    not available or not needed.

    Capability Flags:
        supports_power: False - psutil doesn't report power
        supports_utilization: True - CPU utilization metrics
        supports_temperature: True (Linux only) - CPU temperature if available
        supports_memory: True - RAM and swap memory metrics

    Attributes:
        include_per_cpu: Include per-CPU core metrics
        include_disk_io: Include disk I/O metrics
        include_network_io: Include network I/O metrics
        process_pid: Optional PID to track specific process
    """

    # Capability flags
    supports_power = False
    supports_utilization = True
    supports_temperature = True  # Platform dependent
    supports_memory = True

    def __init__(
        self,
        include_per_cpu: bool = True,
        include_disk_io: bool = False,
        include_network_io: bool = False,
        process_pid: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
        name: str = "PsutilCollector",
    ):
        """Initialize the psutil collector.

        Args:
            include_per_cpu: Include per-CPU core utilization (default: True)
            include_disk_io: Include disk I/O statistics (default: False)
            include_network_io: Include network I/O statistics (default: False)
            process_pid: Optional PID of process to monitor (for process-specific metrics)
            config: Optional configuration dictionary with:
                - warmup_samples: Number of warmup samples to mark (default: 0)
                - cpu_interval: CPU sampling interval in seconds (default: None for instant)
            name: Name for this collector instance

        Raises:
            ImportError: If psutil is not installed
        """
        super().__init__(name, config)

        self.include_per_cpu = include_per_cpu
        self.include_disk_io = include_disk_io
        self.include_network_io = include_network_io
        self.process_pid = process_pid

        # Configuration
        self._cfg = {
            "warmup_samples": 0,
            "cpu_interval": None,  # None for non-blocking
            **(config or {}),
        }

        self._sample_index = 0
        self._process = None
        self._prev_disk_io = None
        self._prev_net_io = None
        self._prev_sample_time = None

    def start(self) -> None:
        """Start system monitoring.

        Initializes psutil and optionally attaches to a process.

        Raises:
            ImportError: If psutil is not installed
            RuntimeError: If process PID is invalid
        """
        try:
            import psutil
            self._psutil = psutil
        except ImportError:
            raise ImportError(
                "psutil is required for PsutilCollector. "
                "Install with: pip install psutil"
            )

        # Attach to process if PID specified
        if self.process_pid is not None:
            try:
                self._process = psutil.Process(self.process_pid)
            except psutil.NoSuchProcess:
                raise RuntimeError(f"Process with PID {self.process_pid} not found")

        # Initialize baseline readings for I/O deltas
        if self.include_disk_io:
            self._prev_disk_io = psutil.disk_io_counters()

        if self.include_network_io:
            self._prev_net_io = psutil.net_io_counters()

        self._is_running = True
        self._start_time = time.time()
        self._prev_sample_time = self._start_time
        self._sample_index = 0
        self._samples.clear()

    def sample(self, timestamp: float) -> Optional[Dict[str, Any]]:
        """Collect system metrics at the given timestamp.

        Args:
            timestamp: Unix timestamp for this sample

        Returns:
            Dictionary containing system metrics:
            - cpu_percent: Overall CPU utilization (0-100)
            - cpu_per_core: List of per-core utilization (if enabled)
            - cpu_count: Number of CPU cores
            - memory_used_mb: Physical memory used in MB
            - memory_total_mb: Total physical memory in MB
            - memory_percent: Memory utilization percentage
            - memory_available_mb: Available memory in MB
            - swap_used_mb: Swap memory used in MB
            - swap_total_mb: Total swap memory in MB
            - swap_percent: Swap utilization percentage
            - temperature_c: CPU temperature if available
            - disk_read_mbps: Disk read speed (if enabled)
            - disk_write_mbps: Disk write speed (if enabled)
            - net_sent_mbps: Network upload speed (if enabled)
            - net_recv_mbps: Network download speed (if enabled)
            - process_*: Process-specific metrics (if PID specified)
            - is_warmup: Boolean indicating warmup period

        Raises:
            RuntimeError: If collector is not running
        """
        if not self._is_running:
            raise RuntimeError("Collector not started. Call start() first.")

        psutil = self._psutil
        metrics = {}

        # CPU metrics
        cpu_interval = self._cfg.get("cpu_interval")
        metrics["cpu_percent"] = psutil.cpu_percent(interval=cpu_interval)
        metrics["cpu_count"] = psutil.cpu_count()
        metrics["cpu_count_physical"] = psutil.cpu_count(logical=False)

        if self.include_per_cpu:
            metrics["cpu_per_core"] = psutil.cpu_percent(interval=None, percpu=True)

        # CPU frequency
        try:
            freq = psutil.cpu_freq()
            if freq:
                metrics["cpu_freq_current_mhz"] = freq.current
                metrics["cpu_freq_min_mhz"] = freq.min
                metrics["cpu_freq_max_mhz"] = freq.max
        except Exception:
            pass

        # Memory metrics
        mem = psutil.virtual_memory()
        metrics["memory_used_mb"] = mem.used / (1024 * 1024)
        metrics["memory_total_mb"] = mem.total / (1024 * 1024)
        metrics["memory_available_mb"] = mem.available / (1024 * 1024)
        metrics["memory_percent"] = mem.percent

        # Swap metrics
        swap = psutil.swap_memory()
        metrics["swap_used_mb"] = swap.used / (1024 * 1024)
        metrics["swap_total_mb"] = swap.total / (1024 * 1024)
        metrics["swap_percent"] = swap.percent

        # Temperature (Linux only typically)
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Try common temperature sensor names
                for name in ["coretemp", "cpu_thermal", "k10temp", "zenpower"]:
                    if name in temps:
                        # Get highest temperature
                        max_temp = max(t.current for t in temps[name] if t.current is not None)
                        metrics["temperature_c"] = max_temp
                        break
                else:
                    # Use first available sensor
                    for name, entries in temps.items():
                        valid_temps = [t.current for t in entries if t.current is not None]
                        if valid_temps:
                            metrics["temperature_c"] = max(valid_temps)
                            break
        except (AttributeError, Exception):
            # sensors_temperatures not available on this platform
            pass

        # Calculate time delta for I/O rates
        time_delta = timestamp - self._prev_sample_time if self._prev_sample_time else 1.0
        time_delta = max(time_delta, 0.001)  # Avoid division by zero

        # Disk I/O metrics
        if self.include_disk_io:
            try:
                disk_io = psutil.disk_io_counters()
                if disk_io and self._prev_disk_io:
                    read_bytes = disk_io.read_bytes - self._prev_disk_io.read_bytes
                    write_bytes = disk_io.write_bytes - self._prev_disk_io.write_bytes

                    metrics["disk_read_mbps"] = (read_bytes / time_delta) / (1024 * 1024)
                    metrics["disk_write_mbps"] = (write_bytes / time_delta) / (1024 * 1024)
                    metrics["disk_read_count"] = disk_io.read_count - self._prev_disk_io.read_count
                    metrics["disk_write_count"] = disk_io.write_count - self._prev_disk_io.write_count

                self._prev_disk_io = disk_io
            except Exception:
                pass

        # Network I/O metrics
        if self.include_network_io:
            try:
                net_io = psutil.net_io_counters()
                if net_io and self._prev_net_io:
                    sent_bytes = net_io.bytes_sent - self._prev_net_io.bytes_sent
                    recv_bytes = net_io.bytes_recv - self._prev_net_io.bytes_recv

                    metrics["net_sent_mbps"] = (sent_bytes / time_delta) / (1024 * 1024)
                    metrics["net_recv_mbps"] = (recv_bytes / time_delta) / (1024 * 1024)
                    metrics["net_packets_sent"] = net_io.packets_sent - self._prev_net_io.packets_sent
                    metrics["net_packets_recv"] = net_io.packets_recv - self._prev_net_io.packets_recv

                self._prev_net_io = net_io
            except Exception:
                pass

        # Process-specific metrics
        if self._process is not None:
            try:
                if self._process.is_running():
                    with self._process.oneshot():
                        metrics["process_cpu_percent"] = self._process.cpu_percent()
                        metrics["process_memory_mb"] = self._process.memory_info().rss / (1024 * 1024)
                        metrics["process_memory_percent"] = self._process.memory_percent()
                        metrics["process_threads"] = self._process.num_threads()

                        try:
                            metrics["process_fds"] = self._process.num_fds()
                        except (AttributeError, Exception):
                            pass  # Not available on Windows
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Load average (Unix only)
        try:
            load = psutil.getloadavg()
            metrics["load_avg_1m"] = load[0]
            metrics["load_avg_5m"] = load[1]
            metrics["load_avg_15m"] = load[2]
        except (AttributeError, Exception):
            pass

        # Warmup marker
        metrics["is_warmup"] = self._sample_index < self._cfg.get("warmup_samples", 0)

        # Update timing
        self._prev_sample_time = timestamp

        # Store sample
        metadata = {
            "sample_index": self._sample_index,
            "platform": self._get_platform_info(),
        }
        self._store_sample(timestamp, metrics, metadata)
        self._sample_index += 1

        return metrics

    def _get_platform_info(self) -> Dict[str, Any]:
        """Get platform information."""
        import platform
        return {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        }

    def stop(self) -> None:
        """Stop system monitoring.

        Safe to call multiple times.
        """
        if self._is_running:
            self._is_running = False
            self._end_time = time.time()
            self._process = None

    def export(self) -> CollectorExport:
        """Export all collected system metrics.

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
                "include_per_cpu": self.include_per_cpu,
                "include_disk_io": self.include_disk_io,
                "include_network_io": self.include_network_io,
                "process_pid": self.process_pid,
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
        steady_samples = self._samples[warmup_count:] if len(self._samples) > warmup_count else self._samples

        def safe_stats(key: str) -> Dict[str, Optional[float]]:
            """Calculate stats for a metric, handling missing values."""
            values = [s.metrics.get(key) for s in steady_samples if s.metrics.get(key) is not None]
            if not values:
                return {}
            return {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }

        summary = {
            "sample_count": len(self._samples),
            "warmup_samples": min(warmup_count, len(self._samples)),
            "duration_seconds": (self._end_time - self._start_time) if self._end_time else None,
            "cpu": {
                "mean_percent": safe_stats("cpu_percent").get("mean"),
                "max_percent": safe_stats("cpu_percent").get("max"),
                "min_percent": safe_stats("cpu_percent").get("min"),
            },
            "memory": {
                "mean_mb": safe_stats("memory_used_mb").get("mean"),
                "max_mb": safe_stats("memory_used_mb").get("max"),
                "min_mb": safe_stats("memory_used_mb").get("min"),
                "total_mb": steady_samples[0].metrics.get("memory_total_mb") if steady_samples else None,
                "mean_percent": safe_stats("memory_percent").get("mean"),
            },
            "swap": {
                "mean_mb": safe_stats("swap_used_mb").get("mean"),
                "max_mb": safe_stats("swap_used_mb").get("max"),
                "mean_percent": safe_stats("swap_percent").get("mean"),
            },
        }

        # Temperature stats if available
        temp_stats = safe_stats("temperature_c")
        if temp_stats:
            summary["temperature"] = {
                "mean_c": temp_stats.get("mean"),
                "max_c": temp_stats.get("max"),
            }

        # Load average stats if available
        load_stats = safe_stats("load_avg_1m")
        if load_stats:
            summary["load_avg"] = {
                "mean_1m": load_stats.get("mean"),
                "max_1m": load_stats.get("max"),
            }

        # Process stats if available
        process_cpu = safe_stats("process_cpu_percent")
        if process_cpu:
            summary["process"] = {
                "mean_cpu_percent": process_cpu.get("mean"),
                "max_cpu_percent": process_cpu.get("max"),
                "mean_memory_mb": safe_stats("process_memory_mb").get("mean"),
                "max_memory_mb": safe_stats("process_memory_mb").get("max"),
            }

        return summary

    def get_system_info(self) -> Dict[str, Any]:
        """Get detailed system information.

        Returns:
            Dictionary with system specifications
        """
        if not self._is_running:
            try:
                import psutil
                self._psutil = psutil
            except ImportError:
                return {"error": "psutil not installed"}

        psutil = self._psutil
        info = {}

        # CPU info
        info["cpu"] = {
            "cores_logical": psutil.cpu_count(),
            "cores_physical": psutil.cpu_count(logical=False),
        }

        try:
            freq = psutil.cpu_freq()
            if freq:
                info["cpu"]["freq_max_mhz"] = freq.max
                info["cpu"]["freq_min_mhz"] = freq.min
        except Exception:
            pass

        # Memory info
        mem = psutil.virtual_memory()
        info["memory"] = {
            "total_mb": mem.total / (1024 * 1024),
            "total_gb": mem.total / (1024 * 1024 * 1024),
        }

        swap = psutil.swap_memory()
        info["swap"] = {
            "total_mb": swap.total / (1024 * 1024),
        }

        # Disk info
        try:
            disk = psutil.disk_usage("/")
            info["disk"] = {
                "total_gb": disk.total / (1024 * 1024 * 1024),
                "used_gb": disk.used / (1024 * 1024 * 1024),
                "free_gb": disk.free / (1024 * 1024 * 1024),
            }
        except Exception:
            pass

        # Platform info
        import platform
        info["platform"] = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        }

        # Boot time
        try:
            info["boot_time"] = psutil.boot_time()
        except Exception:
            pass

        return info

    @staticmethod
    def is_available() -> bool:
        """Check if psutil is available.

        Returns:
            True if psutil is installed
        """
        try:
            import psutil
            return True
        except ImportError:
            return False


__all__ = ["PsutilCollector"]
