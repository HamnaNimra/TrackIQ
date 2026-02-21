"""Distributed training runner wrapper for minicluster.

This module wraps trackiq_core's distributed training functionality and adds
minicluster-compatible config and metrics serialization.
"""

import json
import os
import platform as _platform
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from minicluster.deps import ensure_parent_dir, load_json_file

# Import from trackiq_core - the single source of truth for distributed training
from trackiq_core.distributed_validator import (
    DistributedValidationConfig,
    SimpleMLP,
)
from trackiq_core.distributed_validator import create_synthetic_dataset as _create_synthetic_dataset_impl
from trackiq_core.distributed_validator import train_multi_process as _train_multi_process_impl
from trackiq_core.distributed_validator import train_single_process as _train_single_process_impl
from trackiq_core.power_profiler import PowerProfiler, SimulatedPowerReader
from trackiq_core.schema import (
    Metrics,
    PlatformInfo,
    RegressionInfo,
    TrackiqResult,
    WorkloadInfo,
)
from trackiq_core.serializer import load_trackiq_result, save_trackiq_result


@dataclass
class StepMetrics:
    """Metrics for a single training step (minicluster format)."""

    step: int
    loss: float
    throughput_samples_per_sec: float
    allreduce_time_ms: float = 0.0
    compute_time_ms: float = 0.0


@dataclass
class WorkerSnapshot:
    """Point-in-time snapshot of a single worker's state."""

    worker_id: int
    step: int
    loss: float
    throughput_samples_per_sec: float
    allreduce_time_ms: float
    compute_time_ms: float
    status: Literal["healthy", "slow", "failed"]
    timestamp: str  # ISO format


@dataclass
class HealthCheckpoint:
    """Incremental health data written during a training run."""

    run_id: str
    total_steps: int
    completed_steps: int
    workers: list[WorkerSnapshot]
    timestamp: str  # ISO format
    is_complete: bool = False


@dataclass
class RunMetrics:
    """Aggregated metrics for a training run (minicluster format)."""

    config: dict[str, Any]
    num_workers: int
    num_steps: int
    steps: list[StepMetrics] = field(default_factory=list)
    total_time_sec: float = 0.0
    total_allreduce_time_ms: float = 0.0
    total_compute_time_ms: float = 0.0
    start_timestamp: str = ""
    end_timestamp: str = ""
    power_metrics: dict[str, Any] | None = None
    power_tool_payload: dict[str, Any] | None = None
    worker_snapshots: list[dict[str, Any]] = field(default_factory=list)
    health_checkpoint_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
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
            "power_metrics": self.power_metrics,
            "power_tool_payload": self.power_tool_payload,
            "worker_snapshots": self.worker_snapshots,
            "health_checkpoint_path": self.health_checkpoint_path,
            "average_loss": sum(s.loss for s in self.steps) / len(self.steps) if self.steps else 0.0,
            "final_loss": self.steps[-1].loss if self.steps else 0.0,
            "average_throughput_samples_per_sec": (
                sum(s.throughput_samples_per_sec for s in self.steps) / len(self.steps) if self.steps else 0.0
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
    tdp_watts: float = 150.0

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


def train_single_process(config: RunConfig, health_checkpoint_path: str | None = None) -> RunMetrics:
    """Train in single-process mode using trackiq_core implementation.

    Args:
        config: RunConfig (DistributedValidationConfig) with training parameters

    Returns:
        RunMetrics with per-step metrics
    """
    import torch

    torch.manual_seed(config.seed)

    profiler = PowerProfiler(SimulatedPowerReader(tdp_watts=config.tdp_watts))
    profiler.start_session()
    # Call trackiq_core implementation
    start = time.time()
    losses = _train_single_process_impl(config.to_core())
    elapsed = time.time() - start
    throughput = (config.batch_size * len(losses) / elapsed) if elapsed > 0 and losses else 0.0
    for idx in range(len(losses)):
        profiler.record_step(idx, throughput)
    profiler.end_session()
    base_metrics = Metrics(
        throughput_samples_per_sec=throughput,
        latency_p50_ms=0.0,
        latency_p95_ms=0.0,
        latency_p99_ms=0.0,
        memory_utilization_percent=0.0,
        communication_overhead_percent=None,
        power_consumption_watts=None,
    )
    updated_metrics = profiler.to_metrics_update(base_metrics)
    metrics_dict: dict[str, Any] = {
        "losses": losses,
        "total_time_sec": elapsed,
        "power_metrics": asdict(updated_metrics),
        "power_tool_payload": profiler.to_tool_payload(),
    }

    # Convert to minicluster RunMetrics format
    return _convert_to_run_metrics(
        metrics_dict,
        num_workers=1,
        config=config,
        health_checkpoint_path=health_checkpoint_path,
    )


def train_distributed(
    rank: int,
    world_size: int,
    config: RunConfig,
    health_checkpoint_path: str | None = None,
) -> RunMetrics | None:
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
        return train_single_process(config, health_checkpoint_path=health_checkpoint_path)

    # Compatibility wrapper: build equivalent multi-process run and return rank-0 view.
    core_cfg = config.to_core()
    core_cfg.num_processes = world_size
    profiler = PowerProfiler(SimulatedPowerReader(tdp_watts=config.tdp_watts))
    profiler.start_session()
    start = time.time()
    losses = _train_multi_process_impl(core_cfg)
    elapsed = time.time() - start
    throughput = (config.batch_size * len(losses) / elapsed) if elapsed > 0 and losses else 0.0
    for idx in range(len(losses)):
        profiler.record_step(idx, throughput)
    profiler.end_session()
    base_metrics = Metrics(
        throughput_samples_per_sec=throughput,
        latency_p50_ms=0.0,
        latency_p95_ms=0.0,
        latency_p99_ms=0.0,
        memory_utilization_percent=0.0,
        communication_overhead_percent=None,
        power_consumption_watts=None,
    )
    updated_metrics = profiler.to_metrics_update(base_metrics)
    metrics_dict = {
        "losses": losses,
        "total_time_sec": elapsed,
        "power_metrics": asdict(updated_metrics),
        "power_tool_payload": profiler.to_tool_payload(),
    }
    return _convert_to_run_metrics(
        metrics_dict,
        num_workers=world_size,
        config=config,
        health_checkpoint_path=health_checkpoint_path,
    )


def run_distributed(config: RunConfig, health_checkpoint_path: str | None = None) -> RunMetrics:
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
        return train_single_process(config, health_checkpoint_path=health_checkpoint_path)

    core_cfg = config.to_core()
    profiler = PowerProfiler(SimulatedPowerReader(tdp_watts=config.tdp_watts))
    profiler.start_session()
    start = time.time()
    losses = _train_multi_process_impl(core_cfg)
    elapsed = time.time() - start
    throughput = (config.batch_size * len(losses) / elapsed) if elapsed > 0 and losses else 0.0
    for idx in range(len(losses)):
        profiler.record_step(idx, throughput)
    profiler.end_session()
    base_metrics = Metrics(
        throughput_samples_per_sec=throughput,
        latency_p50_ms=0.0,
        latency_p95_ms=0.0,
        latency_p99_ms=0.0,
        memory_utilization_percent=0.0,
        communication_overhead_percent=None,
        power_consumption_watts=None,
    )
    updated_metrics = profiler.to_metrics_update(base_metrics)
    return _convert_to_run_metrics(
        {
            "losses": losses,
            "total_time_sec": elapsed,
            "power_metrics": asdict(updated_metrics),
            "power_tool_payload": profiler.to_tool_payload(),
        },
        num_workers=config.num_processes,
        config=config,
        health_checkpoint_path=health_checkpoint_path,
    )


def write_health_checkpoint(checkpoint: HealthCheckpoint, output_path: str) -> None:
    """Write health checkpoint atomically so readers never see partial JSON."""
    ensure_parent_dir(output_path)
    parent = os.path.dirname(os.path.abspath(output_path)) or "."
    payload = {
        "run_id": checkpoint.run_id,
        "total_steps": checkpoint.total_steps,
        "completed_steps": checkpoint.completed_steps,
        "workers": [asdict(w) for w in checkpoint.workers],
        "timestamp": checkpoint.timestamp,
        "is_complete": checkpoint.is_complete,
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".tmp",
        prefix="health_",
        dir=parent,
        delete=False,
    ) as handle:
        temp_path = handle.name
        json.dump(payload, handle, indent=2)
    os.replace(temp_path, output_path)


def determine_worker_status(
    worker: WorkerSnapshot, baseline_throughput: float, slow_threshold: float = 0.7
) -> Literal["healthy", "slow", "failed"]:
    """Classify worker status from current throughput relative to baseline."""
    if worker.throughput_samples_per_sec == 0:
        return "failed"
    if baseline_throughput > 0 and worker.throughput_samples_per_sec < (baseline_throughput * slow_threshold):
        return "slow"
    return "healthy"


def _convert_to_run_metrics(
    metrics_dict: dict[str, Any],
    num_workers: int,
    config: RunConfig,
    health_checkpoint_path: str | None = None,
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
        run_id = str(uuid.uuid4())
        all_worker_snapshots: list[WorkerSnapshot] = []
        baseline_throughput = throughput
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
            worker_snapshots: list[WorkerSnapshot] = []
            now_iso = datetime.now(UTC).isoformat()
            for worker_id in range(num_workers):
                snap = WorkerSnapshot(
                    worker_id=worker_id,
                    step=i,
                    loss=float(loss),
                    throughput_samples_per_sec=float(throughput),
                    allreduce_time_ms=0.0,
                    compute_time_ms=0.0,
                    status="healthy",
                    timestamp=now_iso,
                )
                snap.status = determine_worker_status(snap, baseline_throughput)
                worker_snapshots.append(snap)
                all_worker_snapshots.append(snap)

            if health_checkpoint_path:
                write_health_checkpoint(
                    HealthCheckpoint(
                        run_id=run_id,
                        total_steps=config.num_steps,
                        completed_steps=i + 1,
                        workers=worker_snapshots,
                        timestamp=now_iso,
                        is_complete=False,
                    ),
                    health_checkpoint_path,
                )

        if health_checkpoint_path and all_worker_snapshots:
            latest_step = all_worker_snapshots[-1].step + 1
            write_health_checkpoint(
                HealthCheckpoint(
                    run_id=run_id,
                    total_steps=config.num_steps,
                    completed_steps=latest_step,
                    workers=[s for s in all_worker_snapshots if s.step == all_worker_snapshots[-1].step],
                    timestamp=datetime.now(UTC).isoformat(),
                    is_complete=True,
                ),
                health_checkpoint_path,
            )
        metrics_dict["worker_snapshots"] = [asdict(s) for s in all_worker_snapshots]
        metrics_dict["health_checkpoint_path"] = health_checkpoint_path

    return RunMetrics(
        config=asdict(config) if hasattr(config, "__dataclass_fields__") else config,
        num_workers=num_workers,
        num_steps=config.num_steps,
        steps=steps,
        total_time_sec=metrics_dict.get("total_time_sec", 0.0),
        total_allreduce_time_ms=0.0,
        total_compute_time_ms=0.0,
        start_timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        end_timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        power_metrics=metrics_dict.get("power_metrics"),
        power_tool_payload=metrics_dict.get("power_tool_payload"),
        worker_snapshots=metrics_dict.get("worker_snapshots", []),
        health_checkpoint_path=metrics_dict.get("health_checkpoint_path"),
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
        timestamp=datetime.now(UTC),
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
            throughput_samples_per_sec=float(metrics_dict.get("average_throughput_samples_per_sec", 0.0)),
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            latency_p99_ms=0.0,
            memory_utilization_percent=0.0,
            communication_overhead_percent=comm_overhead,
            power_consumption_watts=(
                float(metrics.power_metrics.get("power_consumption_watts"))
                if metrics.power_metrics and metrics.power_metrics.get("power_consumption_watts") is not None
                else None
            ),
            energy_per_step_joules=(
                float(metrics.power_metrics.get("energy_per_step_joules"))
                if metrics.power_metrics and metrics.power_metrics.get("energy_per_step_joules") is not None
                else None
            ),
            performance_per_watt=(
                float(metrics.power_metrics.get("performance_per_watt"))
                if metrics.power_metrics and metrics.power_metrics.get("performance_per_watt") is not None
                else None
            ),
            temperature_celsius=(
                float(metrics.power_metrics.get("temperature_celsius"))
                if metrics.power_metrics and metrics.power_metrics.get("temperature_celsius") is not None
                else None
            ),
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


def load_metrics(output_path: str) -> dict[str, Any]:
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
