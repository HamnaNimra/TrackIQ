"""GPU and system monitoring for TrackIQ."""

import time
import threading
from typing import Dict, Any, List, Optional
from trackiq_core.utils.base import BaseMonitor
from trackiq_core.hardware import get_memory_metrics


class GPUMemoryMonitor(BaseMonitor):
    """Monitor GPU memory usage during inference."""

    def __init__(self, config=None):
        """Initialize monitor.

        Args:
            config: Optional configuration object
        """
        super().__init__("GPUMemoryMonitor")
        self.config = config
        self.is_running = False
        self.thread = None
        self._metrics_lock = threading.Lock()

    def start(self) -> None:
        """Start GPU memory monitoring."""
        self.is_running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop GPU memory monitoring."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5)

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        interval = 1
        if self.config:
            if hasattr(self.config, "monitoring"):
                interval = getattr(self.config.monitoring, "interval_seconds", 1)
            elif isinstance(self.config, dict):
                interval = self.config.get("interval_seconds", 1)

        while self.is_running:
            try:
                metric = self._get_gpu_metrics()
                if metric:
                    with self._metrics_lock:
                        self.metrics.append(metric)
            except Exception:
                pass  # Continue monitoring even if metrics unavailable

            time.sleep(interval)

    def _get_gpu_metrics(self) -> Optional[Dict[str, Any]]:
        """Get current GPU metrics using shared utilities.

        Returns:
            Dictionary with GPU metrics or None
        """
        metrics = get_memory_metrics()
        if metrics:
            metrics["timestamp"] = time.time()
        return metrics

    def get_metrics(self) -> List[Dict[str, Any]]:
        """Get collected metrics (thread-safe copy).

        Returns:
            Copy of metric snapshots to avoid race conditions
        """
        with self._metrics_lock:
            return self.metrics.copy()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of collected metrics (thread-safe).

        Returns:
            Summary dictionary
        """
        with self._metrics_lock:
            if not self.metrics:
                return {}

            memory_used = [m.get("gpu_memory_used_mb", 0) for m in self.metrics]
            utilization = [m.get("gpu_utilization_percent", 0) for m in self.metrics]

            return {
                "avg_memory_mb": sum(memory_used) / len(memory_used),
                "max_memory_mb": max(memory_used),
                "avg_utilization_percent": sum(utilization) / len(utilization),
                "max_utilization_percent": max(utilization),
                "samples_collected": len(self.metrics),
            }


class LLMKVCacheMonitor(BaseMonitor):
    """Monitor KV cache growth during LLM inference."""

    def __init__(self, config=None):
        """Initialize monitor.

        Args:
            config: Optional configuration object
        """
        super().__init__("LLMKVCacheMonitor")
        self.config = config

    def start(self) -> None:
        """Start KV cache monitoring."""
        pass

    def stop(self) -> None:
        """Stop KV cache monitoring."""
        pass

    def estimate_kv_cache_size(
        self, sequence_length: int, model_config: Dict[str, int]
    ) -> float:
        """Estimate KV cache size in MB.

        Args:
            sequence_length: Current sequence length
            model_config: Dict with num_layers, num_heads, head_size,
                         batch_size, precision

        Returns:
            Estimated KV cache size in MB
        """
        # KV cache = 2 * layers * batch * seq_len * heads * head_size * bytes_per_value
        precision = str(model_config.get("precision", "fp32")).lower()
        bytes_per_value_map = {
            "fp32": 4.0,
            "fp16": 2.0,
            "bf16": 2.0,
            "int8": 1.0,
            "int4": 0.5,
            "mixed": 2.0,
        }
        bytes_per_value = bytes_per_value_map.get(precision, 4.0)

        kv_cache_bytes = (
            2  # K and V
            * model_config.get("num_layers", 32)
            * model_config.get("batch_size", 1)
            * sequence_length
            * model_config.get("num_heads", 32)
            * model_config.get("head_size", 128)
            * bytes_per_value
        )

        return kv_cache_bytes / (1024 * 1024)  # Convert to MB

    def get_metrics(self) -> List[Dict[str, Any]]:
        """Get collected metrics.

        Returns:
            List of metric snapshots
        """
        return self.metrics


__all__ = ["GPUMemoryMonitor", "LLMKVCacheMonitor"]
