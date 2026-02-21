"""Distributed training runner module for minicluster.

Wraps trackiq_core's distributed training and provides minicluster-specific
metrics formatting and serialization.
"""

from minicluster.runner.distributed_runner import (
    RunConfig,
    RunMetrics,
    StepMetrics,
    SimpleMLP,
    create_synthetic_dataset,
    train_single_process,
    train_distributed,
    run_distributed,
    save_metrics,
    load_metrics,
)

__all__ = [
    "RunConfig",
    "RunMetrics",
    "StepMetrics",
    "SimpleMLP",
    "create_synthetic_dataset",
    "train_single_process",
    "train_distributed",
    "run_distributed",
    "save_metrics",
    "load_metrics",
]
