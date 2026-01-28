"""Shared GPU utilities for AutoPerfPy monitoring.

This module provides common functionality for GPU monitoring,
used by both the package (autoperfpy.monitoring) and legacy scripts (monitoring/).
"""

import subprocess
from typing import Dict, List, Optional, Any

# Default timeout for nvidia-smi commands
DEFAULT_NVIDIA_SMI_TIMEOUT = 5


def query_nvidia_smi(
    query_fields: List[str],
    timeout: int = DEFAULT_NVIDIA_SMI_TIMEOUT,
) -> Optional[str]:
    """Execute nvidia-smi query and return raw output.

    Args:
        query_fields: List of nvidia-smi query fields (e.g., ["memory.used", "utilization.gpu"])
        timeout: Command timeout in seconds

    Returns:
        Raw stdout string if successful, None otherwise

    Example:
        >>> output = query_nvidia_smi(["memory.used", "memory.total"])
        >>> print(output)  # "1234, 8192"
    """
    try:
        query_string = ",".join(query_fields)
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query_string}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode == 0:
            return result.stdout.strip()
        return None

    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def parse_gpu_metrics(
    output: str,
    field_names: List[str],
    separator: str = ",",
) -> Optional[Dict[str, float]]:
    """Parse nvidia-smi CSV output into a dictionary.

    Args:
        output: Raw CSV output from nvidia-smi
        field_names: Names to assign to each parsed value
        separator: CSV separator (default: comma)

    Returns:
        Dictionary mapping field names to float values, or None on parse error

    Example:
        >>> output = "1234, 8192, 75"
        >>> fields = ["memory_used", "memory_total", "utilization"]
        >>> result = parse_gpu_metrics(output, fields)
        >>> print(result)  # {"memory_used": 1234.0, "memory_total": 8192.0, "utilization": 75.0}
    """
    try:
        values = [v.strip() for v in output.split(separator)]

        if len(values) != len(field_names):
            return None

        return {name: float(value) for name, value in zip(field_names, values)}

    except (ValueError, IndexError):
        return None


def get_memory_metrics(timeout: int = DEFAULT_NVIDIA_SMI_TIMEOUT) -> Optional[Dict[str, Any]]:
    """Get GPU memory metrics.

    Returns:
        Dictionary with memory metrics or None if unavailable:
        - gpu_memory_used_mb: Memory used in MB
        - gpu_memory_total_mb: Total memory in MB
        - gpu_utilization_percent: GPU utilization percentage
        - gpu_memory_percent: Memory usage percentage
    """
    output = query_nvidia_smi(
        ["memory.used", "memory.total", "utilization.gpu"],
        timeout=timeout,
    )

    if output is None:
        return None

    parsed = parse_gpu_metrics(
        output,
        ["gpu_memory_used_mb", "gpu_memory_total_mb", "gpu_utilization_percent"],
    )

    if parsed is None:
        return None

    # Calculate memory percentage
    if parsed["gpu_memory_total_mb"] > 0:
        parsed["gpu_memory_percent"] = (
            parsed["gpu_memory_used_mb"] / parsed["gpu_memory_total_mb"] * 100
        )
    else:
        parsed["gpu_memory_percent"] = 0.0

    return parsed


def get_performance_metrics(timeout: int = DEFAULT_NVIDIA_SMI_TIMEOUT) -> Optional[Dict[str, float]]:
    """Get GPU performance metrics (utilization, temperature, power).

    Returns:
        Dictionary with performance metrics or None if unavailable:
        - utilization: GPU utilization percentage
        - temperature: GPU temperature in Celsius
        - power: Power draw in Watts
    """
    output = query_nvidia_smi(
        ["utilization.gpu", "temperature.gpu", "power.draw"],
        timeout=timeout,
    )

    if output is None:
        return None

    return parse_gpu_metrics(
        output,
        ["utilization", "temperature", "power"],
    )


__all__ = [
    "query_nvidia_smi",
    "parse_gpu_metrics",
    "get_memory_metrics",
    "get_performance_metrics",
    "DEFAULT_NVIDIA_SMI_TIMEOUT",
]
