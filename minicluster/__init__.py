"""MiniCluster - Local distributed training validation tool.

MiniCluster simulates a distributed AI training cluster locally using PyTorch
distributed training with the Gloo backend (CPU-only). It validates distributed
training workloads using correctness first, performance second, fault tolerance
third approach - the same validation methodology used by production cluster
engineers.
"""

__version__ = "0.1.0"
__author__ = "Hamna Nimra"
__description__ = "Local distributed training validation tool for AI workloads"

from minicluster.runner import (
    RunConfig,
    RunMetrics,
    StepMetrics,
    SimpleMLP,
    run_distributed,
    train_single_process,
    train_distributed,
    save_metrics,
    load_metrics,
)
from minicluster.validators import (
    CorrectnessValidator,
    CorrectnessReport,
    FaultInjector,
    FaultInjectionReport,
    FaultType,
)

__all__ = [
    "RunConfig",
    "RunMetrics",
    "StepMetrics",
    "SimpleMLP",
    "run_distributed",
    "train_single_process",
    "train_distributed",
    "save_metrics",
    "load_metrics",
    "CorrectnessValidator",
    "CorrectnessReport",
    "FaultInjector",
    "FaultInjectionReport",
    "FaultType",
]
