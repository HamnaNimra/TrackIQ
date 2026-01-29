"""Benchmark run orchestration."""

import time

from trackiq.collectors import CollectorBase, CollectorExport


class BenchmarkRunner:
    """Run a collector for a fixed duration and return exported results."""

    def __init__(
        self,
        collector: CollectorBase,
        duration_seconds: float,
        sample_interval_seconds: float = 0.1,
        quiet: bool = False,
    ):
        self.collector = collector
        self.duration_seconds = duration_seconds
        self.sample_interval_seconds = sample_interval_seconds
        self.quiet = quiet

    def run(self) -> CollectorExport:
        """Run collection for duration_seconds, sampling at sample_interval_seconds."""
        self.collector.start()
        start = time.time()
        sample_count = 0

        try:
            while time.time() - start < self.duration_seconds:
                ts = time.time()
                metrics = self.collector.sample(ts)
                if not self.quiet and metrics:
                    warmup = metrics.get("is_warmup", False)
                    latency = metrics.get("latency_ms", 0)
                    gpu = metrics.get("gpu_percent", 0)
                    power = metrics.get("power_w", 0)
                    marker = " [WARMUP]" if warmup else ""
                    print(
                        f"[{sample_count:4d}] Latency: {latency:6.2f}ms | "
                        f"GPU: {gpu:5.1f}% | Power: {power:5.1f}W{marker}"
                    )
                sample_count += 1
                time.sleep(self.sample_interval_seconds)
        except KeyboardInterrupt:
            if not self.quiet:
                print("\nCollection interrupted by user")

        self.collector.stop()
        return self.collector.export()
