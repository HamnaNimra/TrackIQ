"""Benchmarking modules for AutoPerfPy."""

import numpy as np
from typing import Dict, Any, List
from ..core import BaseBenchmark


class BatchingTradeoffBenchmark(BaseBenchmark):
    """Benchmark batch size impact on latency and throughput."""

    def __init__(self, config=None):
        """Initialize benchmark.

        Args:
            config: Optional configuration object
        """
        super().__init__("BatchingTradeoffBenchmark")
        self.config = config

    def run(
        self,
        batch_sizes: List[int] = None,
        num_images: int = 1000,
        base_overhead: float = 0.01,
        time_per_image: float = 0.005,
    ) -> Dict[str, Any]:
        """Run batching trade-off benchmark.

        Args:
            batch_sizes: List of batch sizes to test
            num_images: Total number of images to process
            base_overhead: Fixed kernel overhead in seconds
            time_per_image: Time to process one image in seconds

        Returns:
            Dictionary with benchmark results
        """
        if batch_sizes is None:
            batch_sizes = self.config.benchmark.batch_sizes if self.config else [1, 4, 8, 16, 32]

        results = {
            "batch_size": [],
            "latency_ms": [],
            "throughput_img_per_sec": [],
        }

        for batch_size in batch_sizes:
            # Simulate inference with batching
            batch_latency = base_overhead + (time_per_image * batch_size)
            latency_per_image = batch_latency / batch_size
            throughput = 1.0 / latency_per_image

            results["batch_size"].append(batch_size)
            results["latency_ms"].append(latency_per_image * 1000)
            results["throughput_img_per_sec"].append(throughput)

        self.results = results
        return results

    def get_optimal_batch_size(self, optimize_for: str = "latency") -> int:
        """Get optimal batch size.

        Args:
            optimize_for: "latency" or "throughput"

        Returns:
            Optimal batch size
        """
        if not self.results or "batch_size" not in self.results:
            return None

        if optimize_for == "latency":
            idx = np.argmin(self.results["latency_ms"])
        else:  # throughput
            idx = np.argmax(self.results["throughput_img_per_sec"])

        return self.results["batch_size"][idx]


class LLMLatencyBenchmark(BaseBenchmark):
    """Benchmark LLM inference latency metrics."""

    def __init__(self, config=None):
        """Initialize benchmark.

        Args:
            config: Optional configuration object
        """
        super().__init__("LLMLatencyBenchmark")
        self.config = config

    def run(self, prompt_tokens: int = 512, output_tokens: int = 256, num_runs: int = 10) -> Dict[str, Any]:
        """Run LLM latency benchmark.

        Args:
            prompt_tokens: Number of input tokens
            output_tokens: Number of output tokens to generate
            num_runs: Number of benchmark runs

        Returns:
            Dictionary with latency metrics
        """
        ttft_times = []
        tpt_times = []

        for _ in range(num_runs):
            # Simulate prefill phase (TTFT)
            ttft = np.random.normal(800, 100)  # ~800ms ± 100ms
            ttft_times.append(max(100, ttft))  # Min 100ms

            # Simulate decode phase (Time-per-token)
            tpt = np.random.normal(50, 10)  # ~50ms ± 10ms
            tpt_times.append(max(10, tpt))  # Min 10ms

        self.results = {
            "ttft_p50": np.percentile(ttft_times, 50),
            "ttft_p95": np.percentile(ttft_times, 95),
            "ttft_p99": np.percentile(ttft_times, 99),
            "tpt_p50": np.percentile(tpt_times, 50),
            "tpt_p95": np.percentile(tpt_times, 95),
            "tpt_p99": np.percentile(tpt_times, 99),
            "throughput_tokens_per_sec": 1000.0 / np.mean(tpt_times),
            "total_time_ms": np.mean(ttft_times) + (output_tokens * np.mean(tpt_times)),
        }

        return self.results


__all__ = ["BatchingTradeoffBenchmark", "LLMLatencyBenchmark"]
