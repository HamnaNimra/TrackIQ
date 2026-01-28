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


@dataclass
class DNNPipelineConfig:
    """Configuration for DNN pipeline analysis."""

    top_n_layers: int = 5
    memory_overhead_threshold: float = 20.0
    dla_utilization_target: float = 30.0
    enable_layer_profiling: bool = True
    batch_sizes: list = field(default_factory=lambda: [1, 2, 4, 8])


@dataclass
class TegrastatsConfig:
    """Configuration for Tegrastats analysis."""

    throttle_temp_threshold: float = 85.0
    memory_pressure_threshold: float = 90.0
    sample_interval_ms: int = 1000
    enable_thermal_monitoring: bool = True
    enable_emc_monitoring: bool = True


@dataclass
class EfficiencyConfig:
    """Configuration for efficiency analysis."""

    power_column: str = "power_w"
    latency_column: str = "latency_ms"
    throughput_column: str = "throughput_fps"
    group_by: str = "workload"
    include_pareto_analysis: bool = True


@dataclass
class VariabilityConfig:
    """Configuration for variability analysis."""

    latency_column: str = "latency_ms"
    cv_threshold_low: float = 5.0  # Very consistent
    cv_threshold_moderate: float = 10.0  # Consistent
    cv_threshold_high: float = 20.0  # Moderately consistent
    outlier_method: str = "iqr"  # iqr or zscore
    zscore_threshold: float = 3.0


DEFAULT_CONFIG = {
    "benchmark": BenchmarkConfig(),
    "llm": LLMConfig(),
    "monitoring": MonitoringConfig(),
    "analysis": AnalysisConfig(),
    "process": ProcessMonitorConfig(),
    "dnn_pipeline": DNNPipelineConfig(),
    "tegrastats": TegrastatsConfig(),
    "efficiency": EfficiencyConfig(),
    "variability": VariabilityConfig(),
}
