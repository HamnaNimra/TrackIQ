"""Distributed training runner wrapper for minicluster.

This module wraps trackiq_core's distributed training functionality and adds
convenience methods for metrics serialization and metrics formatting compatible
with minicluster's validation framework.
"""

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# Import from trackiq_core - the single source of truth for distributed training
from trackiq_core.distributed_validator import (
    SimpleMLP,
    train_single_process as _train_single_process_impl,
    train_distributed as _train_distributed_impl,
)
from trackiq_core.distributed_validator import DistributedValidationConfig

from minicluster.deps import load_json_file, save_json_file, ensure_parent_dir


@dataclass
class StepMetrics:
    """Metrics for a single training step (minicluster format)."""

    step: int
    loss: float
    throughput_samples_per_sec: float
    allreduce_time_ms: float = 0.0
    compute_time_ms: float = 0.0


@dataclass
class RunMetrics:
    """Aggregated metrics for a training run (minicluster format)."""

    config: Dict[str, Any]
    num_workers: int
    num_steps: int
    steps: List[StepMetrics] = field(default_factory=list)
    total_time_sec: float = 0.0
    total_allreduce_time_ms: float = 0.0
    total_compute_time_ms: float = 0.0
    start_timestamp: str = ""
    end_timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "config": self.config,
            "num_workers": self.num_workers,
            "num_steps": self.num_steps,
            "steps": [asdict(s) for s in self.steps],
            "total_time_sec": self.total_time_sec,
            "total_allreduce_time_ms": self.total_allreduce_time_ms,
            "total_compute_time_ms": self.total_compute_time_ms,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "average_loss": sum(s.loss for s in self.steps) / len(self.steps) if self.steps else 0.0,
            "final_loss": self.steps[-1].loss if self.steps else 0.0,
            "average_throughput_samples_per_sec": (
                sum(s.throughput_samples_per_sec for s in self.steps) / len(self.steps)
                if self.steps
                else 0.0
            ),
        }


# Alias for compatibility with trackiq_core's config
RunConfig = DistributedValidationConfig

# Re-export SimpleMLP for convenience
__all__ = [
    "SimpleMLP",
    "RunConfig",
    "RunMetrics",
    "StepMetrics",
    "train_single_process",
    "train_distributed",
    "run_distributed",
    "save_metrics",
    "load_metrics",
]


def train_single_process(config: RunConfig) -> RunMetrics:
    """Train in single-process mode using trackiq_core implementation.

    Args:
        config: RunConfig (DistributedValidationConfig) with training parameters

    Returns:
        RunMetrics with per-step metrics
    """
    import torch

    torch.manual_seed(config.seed if hasattr(config, 'seed') else 42)

    # Call trackiq_core implementation
    metrics_dict = _train_single_process_impl(config)

    # Convert to minicluster RunMetrics format
    return _convert_to_run_metrics(metrics_dict, num_workers=1, config=config)


def train_distributed(rank: int, world_size: int, config: RunConfig) -> Optional[RunMetrics]:
    """Train in distributed mode using trackiq_core implementation.

    Args:
        rank: Process rank
        world_size: Total number of processes
        config: RunConfig with training parameters

    Returns:
        RunMetrics on rank 0, None on other ranks
    """
    import torch

    torch.manual_seed((config.seed if hasattr(config, 'seed') else 42) + rank)

    # Call trackiq_core implementation
    metrics_dict = _train_distributed_impl(rank, world_size, config)

    if rank == 0 and metrics_dict is not None:
        return _convert_to_run_metrics(metrics_dict, num_workers=world_size, config=config)

    return None


def run_distributed(config: RunConfig) -> RunMetrics:
    """Spawn distributed training processes and return metrics.

    Uses multiprocessing to spawn worker processes with torch.distributed,
    wrapping trackiq_core's train_distributed function.

    Args:
        config: RunConfig with training parameters including num_workers

    Returns:
        RunMetrics from rank 0 process
    """
    import multiprocessing as mp

    if config.num_processes == 1:
        # Single process mode
        return train_single_process(config)

    # Multi-process mode
    metrics_list = []

    def run_worker(rank: int, world_size: int, config: RunConfig):
        """Wrapper to run train_distributed in subprocess."""
        result = train_distributed(rank, world_size, config)
        if rank == 0 and result is not None:
            metrics_list.append(result)

    world_size = config.num_processes
    # Set start method before spawning
    ctx = mp.get_context("spawn")
    processes = []

    for rank in range(world_size):
        p = ctx.Process(target=run_worker, args=(rank, world_size, config))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    if metrics_list:
        return metrics_list[0]

    # Fallback if metrics not captured (shouldn't happen)
    return train_single_process(config)


def _convert_to_run_metrics(
    metrics_dict: Dict[str, Any], num_workers: int, config: RunConfig
) -> RunMetrics:
    """Convert trackiq_core metrics to minicluster RunMetrics format.

    Args:
        metrics_dict: Metrics dictionary from trackiq_core
        num_workers: Number of workers used
        config: Training configuration

    Returns:
        RunMetrics in minicluster format
    """
    steps = []
    if "losses" in metrics_dict:
        for i, loss in enumerate(metrics_dict["losses"]):
            steps.append(
                StepMetrics(
                    step=i,
                    loss=loss,
                    throughput_samples_per_sec=0.0,  # Not tracked in trackiq_core
                    allreduce_time_ms=0.0,
                    compute_time_ms=0.0,
                )
            )

    return RunMetrics(
        config=asdict(config) if hasattr(config, '__dataclass_fields__') else config,
        num_workers=num_workers,
        num_steps=config.num_steps,
        steps=steps,
        total_time_sec=metrics_dict.get("total_time_sec", 0.0),
        total_allreduce_time_ms=0.0,
        total_compute_time_ms=0.0,
        start_timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        end_timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def save_metrics(metrics: RunMetrics, output_path: str) -> None:
    """Save run metrics to JSON file.

    Args:
        metrics: RunMetrics to save
        output_path: Path to output JSON file
    """
    ensure_parent_dir(output_path)
    save_json_file(output_path, metrics.to_dict())


def load_metrics(output_path: str) -> Dict[str, Any]:
    """Load run metrics from JSON file.

    Args:
        output_path: Path to JSON metrics file

    Returns:
        Dictionary with metrics
    """
    return load_json_file(output_path)
