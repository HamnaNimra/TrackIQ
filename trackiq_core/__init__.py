"""Shared abstractions for TrackIQ and AutoPerfPy.

Provides:
- dict_utils (safe_get)
- stats (percentile, stats_from_values)
- env (command_available, nvidia_smi_available)
- config_io (load/save YAML/JSON)
- configs (Config, ConfigManager)
- inference (InferenceConfig, enumerate_inference_configs)
- runners (BenchmarkRunner, run_single_benchmark, run_auto_benchmarks)
- monitoring (GPUMemoryMonitor, LLMKVCacheMonitor)
- benchmarks (BatchingTradeoffBenchmark, LLMLatencyBenchmark)
"""

from trackiq_core.utils.dict_utils import safe_get
from trackiq_core.utils.stats import percentile, stats_from_values
from trackiq_core.hardware.env import command_available, nvidia_smi_available
from trackiq_core.configs.config_io import (
    load_yaml_file,
    load_json_file,
    save_yaml_file,
    save_json_file,
    ensure_parent_dir,
)
from trackiq_core.configs.config import Config, ConfigManager

# Inference configuration
from trackiq_core.inference import (
    InferenceConfig,
    enumerate_inference_configs,
    PRECISION_FP32,
    PRECISION_FP16,
    PRECISION_INT8,
    PRECISIONS,
    DEFAULT_BATCH_SIZES,
    DEFAULT_WARMUP_RUNS,
    DEFAULT_ITERATIONS,
)

# Runners
from trackiq_core.runners import (
    BenchmarkRunner,
    run_single_benchmark,
    run_auto_benchmarks,
)

# Monitoring
from trackiq_core.monitoring import GPUMemoryMonitor, LLMKVCacheMonitor

# Benchmarks
from trackiq_core.benchmarks import BatchingTradeoffBenchmark, LLMLatencyBenchmark

# Distributed validation
from trackiq_core.distributed_validator import DistributedValidator, DistributedValidationConfig
from trackiq_core.schema import (
    TrackiqResult,
    PlatformInfo,
    WorkloadInfo,
    Metrics,
    RegressionInfo,
)
from trackiq_core.serializer import save_trackiq_result, load_trackiq_result
from trackiq_core.validator import validate_trackiq_result, validate_trackiq_result_obj

__all__ = [
    # Utils
    "safe_get",
    "percentile",
    "stats_from_values",
    "command_available",
    "nvidia_smi_available",
    # Config I/O
    "load_yaml_file",
    "load_json_file",
    "save_yaml_file",
    "save_json_file",
    "ensure_parent_dir",
    # Config
    "Config",
    "ConfigManager",
    # Inference
    "InferenceConfig",
    "enumerate_inference_configs",
    "PRECISION_FP32",
    "PRECISION_FP16",
    "PRECISION_INT8",
    "PRECISIONS",
    "DEFAULT_BATCH_SIZES",
    "DEFAULT_WARMUP_RUNS",
    "DEFAULT_ITERATIONS",
    # Runners
    "BenchmarkRunner",
    "run_single_benchmark",
    "run_auto_benchmarks",
    # Monitoring
    "GPUMemoryMonitor",
    "LLMKVCacheMonitor",
    # Benchmarks
    "BatchingTradeoffBenchmark",
    "LLMLatencyBenchmark",
    # Distributed validation
    "DistributedValidator",
    "DistributedValidationConfig",
    # Canonical result schema
    "TrackiqResult",
    "PlatformInfo",
    "WorkloadInfo",
    "Metrics",
    "RegressionInfo",
    "save_trackiq_result",
    "load_trackiq_result",
    "validate_trackiq_result",
    "validate_trackiq_result_obj",
]
