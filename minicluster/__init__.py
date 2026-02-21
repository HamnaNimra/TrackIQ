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
    EmbeddingWorkload,
    RunConfig,
    RunMetrics,
    SimpleMLP,
    StepMetrics,
    TransformerWorkload,
    load_metrics,
    run_distributed,
    save_metrics,
    train_distributed,
    train_single_process,
)
from minicluster.validators import (
    CorrectnessReport,
    CorrectnessValidator,
    FaultInjectionReport,
    FaultInjector,
    FaultType,
)

__all__ = [
    "RunConfig",
    "RunMetrics",
    "StepMetrics",
    "SimpleMLP",
    "TransformerWorkload",
    "EmbeddingWorkload",
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
