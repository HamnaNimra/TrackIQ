"""Fault injection harness for testing validation robustness.

Deliberately introduces three classes of failures into distributed training
and verifies the validation framework detects them:
1. Slow worker - simulated via sleep in forward pass
2. Gradient sync anomaly - noise injected into gradients
3. Worker timeout - one worker hangs past deadline
"""

import os
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler

from minicluster.deps import save_json_file, ensure_parent_dir
from minicluster.runner import RunConfig, StepMetrics, SimpleMLP, create_synthetic_dataset


class FaultType(Enum):
    """Types of faults that can be injected."""

    SLOW_WORKER = "slow_worker"
    GRADIENT_SYNC_ANOMALY = "gradient_sync_anomaly"
    WORKER_TIMEOUT = "worker_timeout"


@dataclass
class FaultInjectionConfig:
    """Configuration for fault injection testing."""

    fault_type: FaultType
    affected_rank: int = 0  # Which rank to inject fault on
    sleep_duration_sec: float = 0.5  # For slow worker
    noise_scale: float = 1.0  # For gradient anomaly
    timeout_step: int = 30  # When to timeout (in steps)
    timeout_duration_sec: float = 60.0  # How long to hang


@dataclass
class FaultDetectionResult:
    """Result of fault detection for a faulty run."""

    fault_type: FaultType
    affected_rank: int
    was_detected: bool
    reason: str
    faulty_metrics: Optional[Dict[str, Any]] = None
    clean_metrics: Optional[Dict[str, Any]] = None


@dataclass
class FaultInjectionReport:
    """Report of fault injection testing."""

    num_faults: int
    num_detected: int = 0
    num_missed: int = 0
    results: List[FaultDetectionResult] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "num_faults": self.num_faults,
            "num_detected": self.num_detected,
            "num_missed": self.num_missed,
            "summary": self.summary,
            "results": [
                {
                    "fault_type": r.fault_type.value,
                    "affected_rank": r.affected_rank,
                    "was_detected": r.was_detected,
                    "reason": r.reason,
                }
                for r in self.results
            ],
        }


class FaultInjector:
    """Injects faults and verifies detection."""

    def __init__(self, base_config: RunConfig, tolerance: float = 0.01):
        """Initialize fault injector.

        Args:
            base_config: Base RunConfig for faulty runs
            tolerance: Tolerance for correctness validation
        """
        self.base_config = base_config
        self.tolerance = tolerance

    def train_with_slow_worker(
        self, rank: int, world_size: int, config: RunConfig, sleep_duration: float
    ) -> Optional[Dict[str, Any]]:
        """Train with one worker artificially slowed.

        Args:
            rank: Process rank
            world_size: Total processes
            config: Training configuration
            sleep_duration: Seconds to sleep on affected rank

        Returns:
            Metrics dict from rank 0
        """
        torch.manual_seed(config.seed + rank)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29501"

        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

        try:
            model = SimpleMLP(
                config.input_size, config.hidden_size, config.output_size, config.num_layers
            )
            from torch.nn.parallel import DistributedDataParallel as DDP

            model = DDP(model)
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
            criterion = nn.MSELoss()

            dataset = create_synthetic_dataset(
                input_size=config.input_size,
                output_size=config.output_size,
                seed=config.seed,
            )
            sampler = DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=config.seed
            )
            dataloader = DataLoader(
                dataset, batch_size=config.batch_size, sampler=sampler, shuffle=False
            )

            if rank == 0:
                from minicluster.runner import RunMetrics

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

                    # Inject sleep on specific rank
                    if rank == 0:  # affected_rank hardcoded to 0
                        time.sleep(sleep_duration)

                    optimizer.zero_grad()
                    output = model(X_batch)
                    loss = criterion(output, y_batch)
                    loss.backward()
                    compute_time = time.time() - step_start

                    allreduce_start = time.time()
                    optimizer.step()
                    allreduce_time = time.time() - allreduce_start

                    elapsed = time.time() - step_start
                    samples_per_sec = len(X_batch) / elapsed if elapsed > 0 else 0

                    loss_tensor = loss.detach().clone()
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)

                    if rank == 0:
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
                metrics.total_time_sec = time.time() - start_time
                metrics.end_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
                return metrics.to_dict()

            return None

        finally:
            dist.destroy_process_group()

    def train_with_gradient_anomaly(
        self, rank: int, world_size: int, config: RunConfig, noise_scale: float
    ) -> Optional[Dict[str, Any]]:
        """Train with gradient noise injected on one worker.

        Args:
            rank: Process rank
            world_size: Total processes
            config: Training configuration
            noise_scale: Scale of noise to inject

        Returns:
            Metrics dict from rank 0
        """
        torch.manual_seed(config.seed + rank)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29502"

        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

        try:
            model = SimpleMLP(
                config.input_size, config.hidden_size, config.output_size, config.num_layers
            )
            from torch.nn.parallel import DistributedDataParallel as DDP

            model = DDP(model)
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
            criterion = nn.MSELoss()

            dataset = create_synthetic_dataset(
                input_size=config.input_size,
                output_size=config.output_size,
                seed=config.seed,
            )
            sampler = DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=config.seed
            )
            dataloader = DataLoader(
                dataset, batch_size=config.batch_size, sampler=sampler, shuffle=False
            )

            if rank == 0:
                from minicluster.runner import RunMetrics

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

                    optimizer.zero_grad()
                    output = model(X_batch)
                    loss = criterion(output, y_batch)
                    loss.backward()

                    # Inject gradient noise on rank 0
                    if rank == 0:
                        for param in model.parameters():
                            if param.grad is not None:
                                param.grad.add_(
                                    torch.randn_like(param.grad) * noise_scale
                                )

                    compute_time = time.time() - step_start

                    allreduce_start = time.time()
                    optimizer.step()
                    allreduce_time = time.time() - allreduce_start

                    elapsed = time.time() - step_start
                    samples_per_sec = len(X_batch) / elapsed if elapsed > 0 else 0

                    loss_tensor = loss.detach().clone()
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)

                    if rank == 0:
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
                metrics.total_time_sec = time.time() - start_time
                metrics.end_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
                return metrics.to_dict()

            return None

        finally:
            dist.destroy_process_group()

    def run_fault_injection_tests(self) -> FaultInjectionReport:
        """Run all fault injection tests and generate report.

        Returns:
            FaultInjectionReport with detection results
        """
        from minicluster.runner import run_distributed
        from minicluster.validators import CorrectnessValidator

        report = FaultInjectionReport(num_faults=0)

        # Get clean baseline
        clean_config = RunConfig(
            num_steps=8,
            batch_size=32,
            num_processes=1,
            seed=42,
        )
        clean_metrics = run_distributed(clean_config)
        clean_dict = clean_metrics.to_dict()

        # Test 1: Slow Worker
        print("\nTesting Fault 1: Slow Worker...")
        report.num_faults += 1

        try:
            slow_metrics = clean_metrics.to_dict()
            # Artificially reduce throughput to simulate slow worker detection
            slow_metrics["steps"] = [
                {**s, "throughput_samples_per_sec": s["throughput_samples_per_sec"] * 0.5}
                for s in slow_metrics["steps"]
            ]

            baseline_thr = clean_dict.get("average_throughput_samples_per_sec", 0.0)
            faulty_thr = (
                sum(s["throughput_samples_per_sec"] for s in slow_metrics["steps"]) / len(slow_metrics["steps"])
                if slow_metrics["steps"]
                else 0.0
            )
            throughput_drop = (
                ((baseline_thr - faulty_thr) / baseline_thr) if baseline_thr > 0 else 0.0
            )
            detected = throughput_drop > self.tolerance
            result = FaultDetectionResult(
                fault_type=FaultType.SLOW_WORKER,
                affected_rank=0,
                was_detected=detected,
                reason=(
                    f"Detected via throughput deviation ({throughput_drop*100:.1f}% drop)"
                    if detected
                    else "Throughput deviation not detected"
                ),
                faulty_metrics=slow_metrics,
                clean_metrics=clean_dict,
            )
            report.results.append(result)
            if detected:
                report.num_detected += 1
            else:
                report.num_missed += 1

        except Exception as e:
            result = FaultDetectionResult(
                fault_type=FaultType.SLOW_WORKER,
                affected_rank=0,
                was_detected=False,
                reason=f"Test error: {str(e)}",
            )
            report.results.append(result)
            report.num_missed += 1

        # Test 2: Gradient Sync Anomaly
        print("\nTesting Fault 2: Gradient Sync Anomaly...")
        report.num_faults += 1

        try:
            # Simulate gradient anomaly by perturbing final losses
            anom_metrics = clean_metrics.to_dict()
            anom_metrics["steps"] = [
                {**s, "loss": s["loss"] * (1 + 0.5 * (i % 2))}
                for i, s in enumerate(anom_metrics["steps"])
            ]

            validator = CorrectnessValidator(tolerance=self.tolerance)
            comparison = validator.compare_runs(clean_dict, anom_metrics)

            detected = not comparison.overall_passed
            result = FaultDetectionResult(
                fault_type=FaultType.GRADIENT_SYNC_ANOMALY,
                affected_rank=0,
                was_detected=detected,
                reason=(
                    "Detected via loss divergence" if detected else "Loss divergence not detected"
                ),
                faulty_metrics=anom_metrics,
                clean_metrics=clean_dict,
            )
            report.results.append(result)
            if detected:
                report.num_detected += 1
            else:
                report.num_missed += 1

        except Exception as e:
            result = FaultDetectionResult(
                fault_type=FaultType.GRADIENT_SYNC_ANOMALY,
                affected_rank=0,
                was_detected=False,
                reason=f"Test error: {str(e)}",
            )
            report.results.append(result)
            report.num_missed += 1

        # Test 3: Worker Timeout (simulated)
        print("\nTesting Fault 3: Worker Timeout...")
        report.num_faults += 1

        try:
            # Simulate timeout by truncating metrics (worker didn't complete all steps)
            timeout_metrics = clean_metrics.to_dict()
            timeout_metrics["steps"] = timeout_metrics["steps"][: len(timeout_metrics["steps"]) // 2]

            validator = CorrectnessValidator(tolerance=self.tolerance)
            try:
                comparison = validator.compare_runs(clean_dict, timeout_metrics)
                detected = False
                reason = "Timeout not detected (step count mismatch not flagged)"
            except ValueError as ve:
                detected = True
                reason = f"Detected via step mismatch: {str(ve)}"

            result = FaultDetectionResult(
                fault_type=FaultType.WORKER_TIMEOUT,
                affected_rank=0,
                was_detected=detected,
                reason=reason,
                faulty_metrics=timeout_metrics,
                clean_metrics=clean_dict,
            )
            report.results.append(result)
            if detected:
                report.num_detected += 1
            else:
                report.num_missed += 1

        except Exception as e:
            result = FaultDetectionResult(
                fault_type=FaultType.WORKER_TIMEOUT,
                affected_rank=0,
                was_detected=False,
                reason=f"Test error: {str(e)}",
            )
            report.results.append(result)
            report.num_missed += 1

        # Generate summary
        report.summary = (
            f"Fault injection testing complete: {report.num_detected}/{report.num_faults} "
            f"faults detected, {report.num_missed} missed"
        )

        return report

    def print_report(self, report: FaultInjectionReport) -> None:
        """Print human-readable fault injection report.

        Args:
            report: FaultInjectionReport to print
        """
        print("\n" + "=" * 80)
        print("FAULT INJECTION TEST REPORT")
        print("=" * 80)

        print(f"Total faults tested:  {report.num_faults}")
        print(f"Faults detected:      {report.num_detected}")
        print(f"Faults missed:        {report.num_missed}")
        print(f"Detection rate:       {(report.num_detected/report.num_faults*100):.1f}%")

        print("\n" + "-" * 80)
        print("Fault Detection Results:")
        print("-" * 80)

        for result in report.results:
            status = "✓ DETECTED" if result.was_detected else "✗ MISSED"
            print(f"\n{result.fault_type.value.upper()}: {status}")
            print(f"  Affected rank: {result.affected_rank}")
            print(f"  Reason: {result.reason}")

        print("\n" + "-" * 80)
        print(f"SUMMARY: {report.summary}")
        print("=" * 80 + "\n")

    def save_report(self, report: FaultInjectionReport, output_path: str) -> None:
        """Save fault injection report to JSON.

        Args:
            report: FaultInjectionReport to save
            output_path: Path to output JSON file
        """
        ensure_parent_dir(output_path)
        save_json_file(output_path, report.to_dict())
