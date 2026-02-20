"""Distributed training runner for minicluster.

This module provides a training harness using torchrun and torch.distributed
that trains a small MLP on synthetic data with configurable steps, workers,
and batch size. Supports both single-process and multi-process modes with
Gloo backend.
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from minicluster.deps import AnalysisResult, load_json_file, save_json_file, ensure_parent_dir


@dataclass
class RunConfig:
    """Configuration for distributed training run."""

    num_steps: int = 100
    learning_rate: float = 0.01
    batch_size: int = 32
    hidden_size: int = 128
    num_layers: int = 2
    input_size: int = 10
    output_size: int = 1
    num_workers: int = 1
    seed: int = 42
    output_dir: str = "./minicluster_results"


@dataclass
class StepMetrics:
    """Metrics for a single training step."""

    step: int
    loss: float
    throughput_samples_per_sec: float
    allreduce_time_ms: float = 0.0
    compute_time_ms: float = 0.0


@dataclass
class RunMetrics:
    """Aggregated metrics for a training run."""

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


class SimpleMLP(nn.Module):
    """Simple MLP model for training."""

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, num_layers: int
    ):
        """Initialize MLP with configurable architecture.

        Args:
            input_size: Input feature dimension
            hidden_size: Hidden layer dimension
            output_size: Output dimension
            num_layers: Total number of layers (≥2)
        """
        super().__init__()
        assert num_layers >= 2, "num_layers must be at least 2"

        layers = []
        in_size = input_size

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_size, hidden_size),
                nn.ReLU(),
            ])
            in_size = hidden_size

        # Output layer
        layers.append(nn.Linear(in_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        return self.net(x)


def create_synthetic_dataset(
    num_samples: int = 1000, input_size: int = 10, output_size: int = 1, seed: int = 42
) -> TensorDataset:
    """Create deterministic synthetic dataset.

    Args:
        num_samples: Number of samples to generate
        input_size: Feature dimension
        output_size: Target dimension
        seed: Random seed for reproducibility

    Returns:
        TensorDataset with inputs and targets
    """
    torch.manual_seed(seed)

    X = torch.randn(num_samples, input_size)
    # Simple linear relationship with noise
    W = torch.randn(input_size, output_size)
    y = X @ W + 0.1 * torch.randn(num_samples, output_size)

    return TensorDataset(X, y)


def train_single_process(config: RunConfig) -> RunMetrics:
    """Train in single-process mode.

    Args:
        config: RunConfig with training parameters

    Returns:
        RunMetrics with per-step metrics
    """
    torch.manual_seed(config.seed)

    model = SimpleMLP(
        config.input_size, config.hidden_size, config.output_size, config.num_layers
    )
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    dataset = create_synthetic_dataset(
        input_size=config.input_size, output_size=config.output_size, seed=config.seed
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    metrics = RunMetrics(
        config=asdict(config),
        num_workers=1,
        num_steps=config.num_steps,
        start_timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    step = 0
    start_time = time.time()

    while step < config.num_steps:
        for X_batch, y_batch in dataloader:
            if step >= config.num_steps:
                break

            step_start = time.time()

            # Forward pass
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)

            # Backward pass
            loss.backward()
            compute_time = time.time() - step_start

            # Optimizer step (simulates all-reduce for consistency)
            optimizer.step()

            elapsed = time.time() - step_start
            samples_per_sec = len(X_batch) / elapsed if elapsed > 0 else 0

            metrics.steps.append(
                StepMetrics(
                    step=step,
                    loss=loss.item(),
                    throughput_samples_per_sec=samples_per_sec,
                    allreduce_time_ms=0.0,
                    compute_time_ms=compute_time * 1000,
                )
            )
            metrics.total_compute_time_ms += compute_time * 1000

            step += 1

    metrics.total_time_sec = time.time() - start_time
    metrics.end_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    return metrics


def train_distributed(rank: int, world_size: int, config: RunConfig) -> Optional[RunMetrics]:
    """Train in distributed mode using torch.distributed.

    Args:
        rank: Process rank
        world_size: Total number of processes
        config: RunConfig with training parameters

    Returns:
        RunMetrics on rank 0, None on other ranks
    """
    torch.manual_seed(config.seed + rank)

    # Initialize distributed process group
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    try:
        model = SimpleMLP(
            config.input_size, config.hidden_size, config.output_size, config.num_layers
        )

        # Wrap model for distributed data parallel
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(model)

        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.MSELoss()

        dataset = create_synthetic_dataset(
            input_size=config.input_size, output_size=config.output_size, seed=config.seed
        )

        # Create sampler for distributed training
        from torch.utils.data import DistributedSampler

        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=config.seed
        )
        dataloader = DataLoader(
            dataset, batch_size=config.batch_size, sampler=sampler, shuffle=False
        )

        metrics = None
        if rank == 0:
            metrics = RunMetrics(
                config=asdict(config),
                num_workers=world_size,
                num_steps=config.num_steps,
                start_timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )

        step = 0
        start_time = time.time()

        while step < config.num_steps:
            for X_batch, y_batch in dataloader:
                if step >= config.num_steps:
                    break

                step_start = time.time()

                # Forward pass
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)

                # Backward pass
                loss.backward()
                compute_time = time.time() - step_start

                # All-reduce (implicit in DDP)
                allreduce_start = time.time()
                optimizer.step()
                allreduce_time = time.time() - allreduce_start

                elapsed = time.time() - step_start
                samples_per_sec = len(X_batch) / elapsed if elapsed > 0 else 0

                if rank == 0:
                    # Gather loss from all ranks for monitoring
                    loss_tensor = loss.detach().clone()
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)

                    assert metrics is not None
                    metrics.steps.append(
                        StepMetrics(
                            step=step,
                            loss=loss_tensor.item(),
                            throughput_samples_per_sec=samples_per_sec,
                            allreduce_time_ms=allreduce_time * 1000,
                            compute_time_ms=compute_time * 1000,
                        )
                    )
                    metrics.total_allreduce_time_ms += allreduce_time * 1000
                    metrics.total_compute_time_ms += compute_time * 1000

                step += 1

        if rank == 0:
            assert metrics is not None
            metrics.total_time_sec = time.time() - start_time
            metrics.end_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

        return metrics if rank == 0 else None

    finally:
        dist.destroy_process_group()


def run_distributed(config: RunConfig) -> RunMetrics:
    """Spawn distributed training processes and return metrics.

    Uses multiprocessing to spawn worker processes with torch.distributed.

    Args:
        config: RunConfig with training parameters including num_workers

    Returns:
        RunMetrics from rank 0 process
    """
    import multiprocessing as mp

    if config.num_workers == 1:
        # Single process mode
        return train_single_process(config)

    # Multi-process mode
    metrics_list = []

    def run_worker(rank: int, world_size: int, config: RunConfig):
        """Wrapper to run train_distributed in subprocess."""
        result = train_distributed(rank, world_size, config)
        if rank == 0 and result is not None:
            metrics_list.append(result)

    world_size = config.num_workers
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
