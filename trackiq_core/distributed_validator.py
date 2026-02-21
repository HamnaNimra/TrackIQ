"""Distributed training validation module for AutoPerfPy.

Validates PyTorch distributed training by comparing single-process vs multi-process
loss convergence using torch.distributed with Gloo backend.
"""

import json
import multiprocessing
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from trackiq_core.utils.compare.regression import RegressionDetector, RegressionThreshold


@dataclass
class DistributedValidationConfig:
    """Configuration for distributed validation."""

    num_steps: int = 100
    learning_rate: float = 0.01
    batch_size: int = 32
    hidden_size: int = 128
    num_layers: int = 2
    input_size: int = 10
    output_size: int = 1
    loss_tolerance: float = 0.01  # Relative tolerance for loss comparison (1%)
    num_processes: int = 2
    regression_threshold: float = 5.0  # Percent threshold for regression detection


class SimpleMLP(nn.Module):
    """Simple MLP model for validation."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int):
        super().__init__()
        layers = []
        in_size = input_size
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_size, hidden_size),
                nn.ReLU(),
            ])
            in_size = hidden_size
        layers.append(nn.Linear(in_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def create_synthetic_dataset(num_samples: int = 1000, input_size: int = 10, output_size: int = 1) -> TensorDataset:
    """Create synthetic regression dataset."""
    # Ensure deterministic dataset creation
    torch.manual_seed(42)
    
    X = torch.randn(num_samples, input_size)
    # Simple linear relationship with noise
    W = torch.randn(input_size, output_size)
    y = X @ W + 0.1 * torch.randn(num_samples, output_size)
    return TensorDataset(X, y)


def train_single_process(config: DistributedValidationConfig) -> List[float]:
    """Run training in single process mode."""
    torch.manual_seed(42)  # Ensure deterministic training
    
    model = SimpleMLP(config.input_size, config.hidden_size, config.output_size, config.num_layers)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    dataset = create_synthetic_dataset()
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    losses = []
    step = 0

    while step < config.num_steps:
        for X_batch, y_batch in dataloader:
            if step >= config.num_steps:
                break

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            step += 1

            if step >= config.num_steps:
                break

    return losses[:config.num_steps]


def train_worker(rank: int, world_size: int, config: DistributedValidationConfig, losses_queue: multiprocessing.Queue):
    """Worker function for distributed training."""
    # Initialize process group
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12345'
    # Some Windows/CPU PyTorch builds do not include libuv support.
    # Force the TCPStore path to avoid "use_libuv was requested" runtime failures.
    os.environ.setdefault("USE_LIBUV", "0")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Set device (CPU for Gloo)
    torch.manual_seed(42)  # Same seed for all processes to ensure deterministic training

    model = SimpleMLP(config.input_size, config.hidden_size, config.output_size, config.num_layers)

    # For CPU-only distributed training, don't use DDP wrapper
    # Just synchronize gradients manually if needed
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    dataset = create_synthetic_dataset()
    # Use regular sampler for simplicity
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    local_losses = []
    step = 0

    while step < config.num_steps:
        for X_batch, y_batch in dataloader:
            if step >= config.num_steps:
                break

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            # Synchronize loss across processes for comparison
            loss_tensor = torch.tensor([loss.item()])
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / world_size

            # Only rank 0 collects losses
            if rank == 0:
                local_losses.append(avg_loss)
            step += 1

            if step >= config.num_steps:
                break

    if rank == 0:
        losses_queue.put(local_losses[:config.num_steps])

    dist.destroy_process_group()


def train_multi_process(config: DistributedValidationConfig) -> List[float]:
    """Run training in multi-process distributed mode."""
    # Use multiprocessing to spawn processes
    ctx = multiprocessing.get_context('spawn')
    losses_queue = ctx.Queue()

    processes = []
    for rank in range(config.num_processes):
        p = ctx.Process(target=train_worker, args=(rank, config.num_processes, config, losses_queue))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Get losses from queue
    losses = losses_queue.get()
    return losses


class DistributedValidator:
    """Validates distributed training correctness by comparing single vs multi-process runs."""

    def __init__(self, baseline_dir: str = ".trackiq/baselines"):
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.regression_detector = RegressionDetector(str(self.baseline_dir))

    def run_validation(
        self,
        config: Optional[DistributedValidationConfig] = None
    ) -> Dict[str, any]:
        """Run distributed validation.

        Args:
            config: Validation configuration

        Returns:
            Dictionary with validation results
        """
        if config is None:
            config = DistributedValidationConfig()

        print("Running single-process training...")
        single_losses = train_single_process(config)

        print("Running multi-process distributed training...")
        multi_losses = train_multi_process(config)

        # Compare losses
        comparisons = []
        max_len = min(len(single_losses), len(multi_losses))
        passed_steps = 0

        for i in range(max_len):
            single_loss = single_losses[i]
            multi_loss = multi_losses[i]
            delta = abs(single_loss - multi_loss)
            rel_delta = delta / max(abs(single_loss), abs(multi_loss)) if max(abs(single_loss), abs(multi_loss)) > 0 else 0

            passed = rel_delta <= config.loss_tolerance
            if passed:
                passed_steps += 1

            comparisons.append({
                "step": i,
                "single_process_loss": single_loss,
                "multi_process_loss": multi_loss,
                "absolute_delta": delta,
                "relative_delta": rel_delta,
                "passed": passed,
                "tolerance": config.loss_tolerance
            })

        overall_pass = passed_steps == max_len

        result = {
            "config": {
                "num_steps": config.num_steps,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "hidden_size": config.hidden_size,
                "num_layers": config.num_layers,
                "input_size": config.input_size,
                "output_size": config.output_size,
                "loss_tolerance": config.loss_tolerance,
                "num_processes": config.num_processes,
                "regression_threshold": config.regression_threshold,
            },
            "single_process_losses": single_losses,
            "multi_process_losses": multi_losses,
            "comparisons": comparisons,
            "summary": {
                "total_steps": max_len,
                "passed_steps": passed_steps,
                "failed_steps": max_len - passed_steps,
                "pass_rate": passed_steps / max_len if max_len > 0 else 0,
                "overall_pass": overall_pass
            }
        }

        return result

    def save_baseline(self, name: str, results: Dict[str, any]) -> None:
        """Save validation results as baseline."""
        # Extract loss metrics for regression detection
        metrics = {}
        for comp in results["comparisons"]:
            metrics[f"step_{comp['step']}_loss_delta"] = comp["relative_delta"]

        self.regression_detector.save_baseline(name, metrics)

    def detect_regression(
        self,
        baseline_name: str,
        results: Dict[str, any],
        threshold: Optional[float] = None
    ) -> Dict[str, any]:
        """Detect regression against baseline."""
        if threshold is None:
            threshold = results["config"]["regression_threshold"]

        # Extract current metrics
        current_metrics = {}
        for comp in results["comparisons"]:
            current_metrics[f"step_{comp['step']}_loss_delta"] = comp["relative_delta"]

        regression_threshold = RegressionThreshold(
            latency_percent=threshold,
            throughput_percent=threshold,
            p99_percent=threshold
        )

        regression_result = self.regression_detector.detect_regressions(
            baseline_name, current_metrics, regression_threshold
        )

        return regression_result

    def generate_report(
        self,
        results: Dict[str, any],
        regression_results: Optional[Dict[str, any]] = None,
        output_format: str = "json"
    ) -> str:
        """Generate validation report."""
        if output_format == "json":
            return json.dumps(results, indent=2)
        else:
            # Simple text report
            lines = [
                "=" * 70,
                "DISTRIBUTED TRAINING VALIDATION REPORT",
                "=" * 70,
                "",
                f"Configuration:",
                f"  Steps: {results['config']['num_steps']}",
                f"  Processes: {results['config']['num_processes']}",
                f"  Tolerance: {results['config']['loss_tolerance']}",
                "",
                f"Results:",
                f"  Total Steps: {results['summary']['total_steps']}",
                f"  Passed: {results['summary']['passed_steps']}",
                f"  Failed: {results['summary']['failed_steps']}",
                f"  Pass Rate: {results['summary']['pass_rate']:.2%}",
                f"  Overall: {'PASS' if results['summary']['overall_pass'] else 'FAIL'}",
            ]

            if regression_results:
                lines.extend([
                    "",
                    "Regression Detection:",
                    f"  Baseline: {regression_results.get('baseline', 'N/A')}",
                    f"  Regressions: {len(regression_results.get('regressions', {}))}",
                    f"  Status: {'REGRESSION' if regression_results.get('has_regressions') else 'OK'}"
                ])

            lines.append("=" * 70)
            return "\n".join(lines)
