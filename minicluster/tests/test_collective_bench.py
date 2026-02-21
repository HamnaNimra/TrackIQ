"""Tests for communication-only collective benchmark helpers."""

from __future__ import annotations

import pytest

from minicluster.benchmarks.collective_bench import (
    compute_allreduce_bandwidth_gbps,
    run_collective_benchmark,
    summarize_bandwidth_gbps,
)


def test_compute_allreduce_bandwidth_formula() -> None:
    """Bus-bandwidth formula should match expected all-reduce expression."""
    size_bytes = 1_000_000_000  # 1 GB
    time_seconds = 1.0
    workers = 2
    expected = 1.0  # (2*(2-1)/2 * 1GB) / 1s
    observed = compute_allreduce_bandwidth_gbps(size_bytes=size_bytes, time_seconds=time_seconds, workers=workers)
    assert observed == pytest.approx(expected)


def test_summarize_bandwidth_outputs_required_fields() -> None:
    """Summary payload should include required aggregate fields."""
    summary = summarize_bandwidth_gbps([10.0, 12.0, 14.0, 11.0], iterations=4)
    assert set(summary.keys()) == {
        "mean_bandwidth_gbps",
        "p50_bandwidth_gbps",
        "p99_bandwidth_gbps",
        "min_bandwidth_gbps",
        "iterations",
    }
    assert summary["iterations"] == 4
    assert summary["min_bandwidth_gbps"] == 10.0


def test_collective_benchmark_rejects_invalid_inputs() -> None:
    """Runner should fail fast for invalid user inputs."""
    with pytest.raises(ValueError, match="workers must be >= 2"):
        run_collective_benchmark(workers=1, size_mb=1.0, iterations=2)
    with pytest.raises(ValueError, match="size_mb must be > 0"):
        run_collective_benchmark(workers=2, size_mb=0.0, iterations=2)
    with pytest.raises(ValueError, match="iterations must be >= 1"):
        run_collective_benchmark(workers=2, size_mb=1.0, iterations=0)
