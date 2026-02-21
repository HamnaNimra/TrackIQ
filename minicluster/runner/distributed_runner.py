"""Distributed training runner wrapper for minicluster.

This module wraps trackiq_core's distributed training functionality and adds
minicluster-compatible config and metrics serialization.
"""

import time
import platform as _platform
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

# Import from trackiq_core - the single source of truth for distributed training
from trackiq_core.distributed_validator import (
    DistributedValidationConfig,
    SimpleMLP,
    create_synthetic_dataset as _create_synthetic_dataset_impl,
    train_multi_process as _train_multi_process_impl,
    train_single_process as _train_single_process_impl,
)

from minicluster.deps import load_json_file, save_json_file, ensure_parent_dir
from trackiq_core.schema import (
    Metrics,
    PlatformInfo,
    RegressionInfo,
    TrackiqResult,
    WorkloadInfo,
)
from trackiq_core.serializer import save_trackiq_result, load_trackiq_result


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


@dataclass
class RunConfig:
    """Minicluster run configuration.

    This mirrors trackiq_core.DistributedValidationConfig and adds a local
    `seed` for deterministic wrappers/tests.
    """

    num_steps: int = 100
    learning_rate: float = 0.01
    batch_size: int = 32
    hidden_size: int = 128
    num_layers: int = 2
    input_size: int = 10
    output_size: int = 1
    loss_tolerance: float = 0.01
    num_processes: int = 2
    regression_threshold: float = 5.0
    seed: int = 42

    @property
    def num_workers(self) -> int:
        """Backward-compatible alias used in docs/tests."""
        return self.num_processes

    def to_core(self) -> DistributedValidationConfig:
        """Convert to trackiq_core config."""
        return DistributedValidationConfig(
            num_steps=self.num_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            input_size=self.input_size,
            output_size=self.output_size,
            loss_tolerance=self.loss_tolerance,
            num_processes=self.num_processes,
            regression_threshold=self.regression_threshold,
        )

# Re-export SimpleMLP for convenience
__all__ = [
    "SimpleMLP",
    "RunConfig",
    "RunMetrics",
    "StepMetrics",
    "train_single_process",
    "train_distributed",
    "run_distributed",
    "create_synthetic_dataset",
    "save_metrics",
    "load_metrics",
]


def create_synthetic_dataset(
    num_samples: int = 1000,
    input_size: int = 10,
    output_size: int = 1,
    seed: int = 42,
):
    """Create deterministic synthetic dataset with optional seed."""
    import torch

    torch.manual_seed(seed)
    return _create_synthetic_dataset_impl(
        num_samples=num_samples,
        input_size=input_size,
        output_size=output_size,
    )


def train_single_process(config: RunConfig) -> RunMetrics:
    """Train in single-process mode using trackiq_core implementation.

    Args:
        config: RunConfig (DistributedValidationConfig) with training parameters

    Returns:
        RunMetrics with per-step metrics
    """
    import torch

    torch.manual_seed(config.seed)

    # Call trackiq_core implementation
    start = time.time()
    losses = _train_single_process_impl(config.to_core())
    elapsed = time.time() - start
    metrics_dict: Dict[str, Any] = {"losses": losses, "total_time_sec": elapsed}

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
    if rank != 0:
        return None
    if world_size <= 1:
        return train_single_process(config)

    # Compatibility wrapper: build equivalent multi-process run and return rank-0 view.
    core_cfg = config.to_core()
    core_cfg.num_processes = world_size
    start = time.time()
    losses = _train_multi_process_impl(core_cfg)
    elapsed = time.time() - start
    metrics_dict = {"losses": losses, "total_time_sec": elapsed}
    return _convert_to_run_metrics(metrics_dict, num_workers=world_size, config=config)


def run_distributed(config: RunConfig) -> RunMetrics:
    """Spawn distributed training processes and return metrics.

    Uses multiprocessing to spawn worker processes with torch.distributed,
    wrapping trackiq_core's train_distributed function.

    Args:
        config: RunConfig with training parameters including num_workers

    Returns:
        RunMetrics from rank 0 process
    """
    if config.num_processes == 1:
        # Single process mode
        return train_single_process(config)

    core_cfg = config.to_core()
    start = time.time()
    losses = _train_multi_process_impl(core_cfg)
    elapsed = time.time() - start
    return _convert_to_run_metrics(
        {"losses": losses, "total_time_sec": elapsed},
        num_workers=config.num_processes,
        config=config,
    )


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
    if isinstance(metrics_dict, list):
        metrics_dict = {"losses": metrics_dict}

    steps = []
    total_time = float(metrics_dict.get("total_time_sec", 0.0) or 0.0)
    losses = metrics_dict.get("losses", [])
    avg_step_time = (total_time / len(losses)) if losses and total_time > 0 else 0.0
    throughput = (config.batch_size / avg_step_time) if avg_step_time > 0 else 0.0

    if "losses" in metrics_dict:
        for i, loss in enumerate(metrics_dict["losses"]):
            steps.append(
                StepMetrics(
                    step=i,
                    loss=loss,
                    throughput_samples_per_sec=throughput,
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
    metrics_dict = metrics.to_dict()
    comm_overhead = None
    denom = metrics.total_allreduce_time_ms + metrics.total_compute_time_ms
    if denom > 0:
        comm_overhead = (metrics.total_allreduce_time_ms / denom) * 100.0

    result = TrackiqResult(
        tool_name="minicluster",
        tool_version="0.1.0",
        timestamp=datetime.utcnow(),
        platform=PlatformInfo(
            hardware_name="CPU",
            os=f"{_platform.system()} {_platform.release()}",
            framework="pytorch",
            framework_version=_safe_torch_version(),
        ),
        workload=WorkloadInfo(
            name="distributed_training_validation",
            workload_type="training",
            batch_size=int(metrics.config.get("batch_size", 1)),
            steps=metrics.num_steps,
        ),
        metrics=Metrics(
            throughput_samples_per_sec=float(
                metrics_dict.get("average_throughput_samples_per_sec", 0.0)
            ),
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            latency_p99_ms=0.0,
            memory_utilization_percent=0.0,
            communication_overhead_percent=comm_overhead,
            power_consumption_watts=None,
        ),
        regression=RegressionInfo(
            baseline_id=None,
            delta_percent=0.0,
            status="pass",
            failed_metrics=[],
        ),
        tool_payload=metrics_dict,
    )
    save_trackiq_result(result, output_path)


def load_metrics(output_path: str) -> Dict[str, Any]:
    """Load run metrics from JSON file.

    Args:
        output_path: Path to JSON metrics file

    Returns:
        Dictionary with metrics
    """
    data = load_json_file(output_path)
    if isinstance(data, dict) and "tool_name" in data and "workload" in data and "metrics" in data:
        result = load_trackiq_result(output_path)
        if isinstance(result.tool_payload, dict):
            return result.tool_payload
        return result.to_dict()
    return data


def _safe_torch_version() -> str:
    """Best-effort torch version lookup."""
    try:
        import torch  # local import to avoid hard dependency at module import time

        return str(torch.__version__)
    except Exception:
        return "unknown"
