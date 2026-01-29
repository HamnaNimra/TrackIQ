"""Pytest configuration and fixtures."""

import pytest
import tempfile
import pandas as pd
from pathlib import Path


def _temp_csv_file():
    """Create a temporary CSV file for testing (close before yield for Windows)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("timestamp,workload,batch_size,latency_ms,power_w\n")
        f.write("2024-01-01 10:00:00,inference,1,25.5,15.2\n")
        f.write("2024-01-01 10:00:01,inference,1,26.3,15.3\n")
        f.write("2024-01-01 10:00:02,inference,1,24.8,15.1\n")
        f.write("2024-01-01 10:00:03,inference,4,10.5,18.2\n")
        f.write("2024-01-01 10:00:04,inference,4,11.2,18.5\n")
        f.write("2024-01-01 10:00:05,inference,4,10.8,18.3\n")
        f.flush()
        name = f.name
    try:
        yield name
    finally:
        Path(name).unlink(missing_ok=True)


@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file for testing."""
    yield from _temp_csv_file()


def _temp_log_file():
    """Create a temporary log file for testing (close before yield for Windows)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        f.write("[2024-01-01 10:00:00] Frame 1: E2E: 25.5ms\n")
        f.write("[2024-01-01 10:00:01] Frame 2: E2E: 26.3ms\n")
        f.write("[2024-01-01 10:00:02] Frame 3: E2E: 75.5ms\n")
        f.write("[2024-01-01 10:00:03] Frame 4: E2E: 26.8ms\n")
        f.write("[2024-01-01 10:00:04] Frame 5: E2E: 120.0ms\n")
        f.flush()
        name = f.name
    try:
        yield name
    finally:
        Path(name).unlink(missing_ok=True)


@pytest.fixture
def temp_log_file():
    """Create a temporary log file for testing."""
    yield from _temp_log_file()


@pytest.fixture
def sample_metrics():
    """Provide sample performance metrics."""
    return {
        "p99_latency": 50.0,
        "p95_latency": 45.0,
        "p50_latency": 30.0,
        "mean_latency": 32.5,
        "throughput_imgs_per_sec": 1000.0,
    }


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
