"""All-reduce-only benchmark for isolating communication fabric performance."""

from __future__ import annotations

import json
import multiprocessing
import os
import socket
import time
from typing import Any, Literal

from minicluster.deps import ensure_parent_dir


def _percentile(values: list[float], p: float) -> float:
    """Compute percentile with linear interpolation."""
    if not values:
        raise ValueError("Cannot compute percentile on empty input.")
    if p <= 0:
        return float(min(values))
    if p >= 100:
        return float(max(values))
    ordered = sorted(float(v) for v in values)
    rank = (len(ordered) - 1) * (p / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def compute_allreduce_bandwidth_gbps(size_bytes: int, time_seconds: float, workers: int) -> float:
    """Compute all-reduce bus bandwidth in GB/s."""
    if workers <= 1 or size_bytes <= 0 or time_seconds <= 0:
        return 0.0
    return (2.0 * (workers - 1) / workers * float(size_bytes)) / time_seconds / 1_000_000_000.0


def summarize_bandwidth_gbps(values: list[float], iterations: int) -> dict[str, Any]:
    """Summarize per-iteration bandwidth samples."""
    if not values:
        raise ValueError("No bandwidth values to summarize.")
    return {
        "mean_bandwidth_gbps": float(sum(values) / len(values)),
        "p50_bandwidth_gbps": float(_percentile(values, 50.0)),
        "p99_bandwidth_gbps": float(_percentile(values, 99.0)),
        "min_bandwidth_gbps": float(min(values)),
        "iterations": int(iterations),
    }


def _pick_free_port() -> int:
    """Pick a best-effort free localhost port for process group setup."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _bench_worker(
    rank: int,
    world_size: int,
    backend: Literal["gloo", "nccl"],
    tensor_numel: int,
    size_bytes: int,
    iterations: int,
    port: int,
    queue: multiprocessing.Queue,
) -> None:
    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ.setdefault("USE_LIBUV", "0")

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    try:
        tensor = torch.ones(tensor_numel, dtype=torch.float32)
        bandwidths: list[float] = []
        for _ in range(iterations):
            dist.barrier()
            start = time.perf_counter()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            dist.barrier()
            elapsed = time.perf_counter() - start
            bandwidths.append(compute_allreduce_bandwidth_gbps(size_bytes=size_bytes, time_seconds=elapsed, workers=world_size))
        if rank == 0:
            queue.put(bandwidths)
    finally:
        dist.destroy_process_group()


def run_collective_benchmark(
    *,
    workers: int = 2,
    size_mb: float = 256.0,
    iterations: int = 50,
    backend: Literal["gloo", "nccl"] = "gloo",
) -> dict[str, Any]:
    """Run all-reduce-only benchmark and return summarized stats.

    This isolates fabric performance from compute. Compare against hardware spec
    (MI300X Infinity Fabric: ~33 GB/s P2P, H100 NVLink: ~900 GB/s). A gap between
    achieved and spec indicates interconnect misconfiguration.
    """
    if workers < 2:
        raise ValueError("workers must be >= 2")
    if size_mb <= 0:
        raise ValueError("size_mb must be > 0")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    size_bytes = int(size_mb * 1024 * 1024)
    tensor_numel = max(1, size_bytes // 4)  # float32
    port = _pick_free_port()

    ctx = multiprocessing.get_context("spawn")
    queue: multiprocessing.Queue = ctx.Queue()
    procs: list[multiprocessing.Process] = []
    for rank in range(workers):
        proc = ctx.Process(
            target=_bench_worker,
            args=(
                rank,
                workers,
                backend,
                tensor_numel,
                size_bytes,
                iterations,
                port,
                queue,
            ),
        )
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()
    if any(proc.exitcode not in (0, None) for proc in procs):
        exit_codes = [proc.exitcode for proc in procs]
        raise RuntimeError(f"Collective benchmark failed (worker exit codes: {exit_codes})")

    if queue.empty():
        raise RuntimeError("Collective benchmark did not return rank-0 measurements.")
    bandwidths = queue.get()
    if not isinstance(bandwidths, list) or not bandwidths:
        raise RuntimeError("Collective benchmark returned invalid bandwidth payload.")

    summary = summarize_bandwidth_gbps([float(v) for v in bandwidths], iterations=iterations)
    summary["backend"] = backend
    summary["workers"] = workers
    summary["size_mb"] = float(size_mb)
    return summary


def save_collective_benchmark(result: dict[str, Any], output_path: str) -> None:
    """Persist collective benchmark JSON payload."""
    ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

