"""Default configuration for AutoPerfPy."""

from dataclasses import dataclass, field


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking tasks."""

    batch_sizes: list = field(default_factory=lambda: [1, 4, 8, 16, 32])
    num_images: int = 1000
    timeout_seconds: int = 3600
    iterations: int = 3


@dataclass
class LLMConfig:
    """Configuration for LLM inference."""

    prompt_length: int = 512
    output_tokens: int = 256
    batch_size: int = 8
    max_sequence_length: int = 2048
    model_size: str = "7b"  # 7b, 13b, 70b


@dataclass
class MonitoringConfig:
    """Configuration for monitoring tasks."""

    duration_seconds: int = 300
    interval_seconds: int = 1
    enable_gpu_monitoring: bool = True
    log_file: str = "monitoring.log"
    oem_threshold_percent: int = 85


@dataclass
class AnalysisConfig:
    """Configuration for analysis tasks."""

    latency_threshold_ms: float = 50.0
    percentiles: list = field(default_factory=lambda: [50, 95, 99])
    group_by: str = "workload"  # workload, batch_size, timestamp


@dataclass
class ProcessMonitorConfig:
    """Configuration for process monitoring."""

    timeout_seconds: int = 1800
    enable_zombie_detection: bool = True
    sigterm_wait_seconds: int = 10
    log_file: str = "process_monitor.log"


DEFAULT_CONFIG = {
    "benchmark": BenchmarkConfig(),
    "llm": LLMConfig(),
    "monitoring": MonitoringConfig(),
    "analysis": AnalysisConfig(),
    "process": ProcessMonitorConfig(),
}
