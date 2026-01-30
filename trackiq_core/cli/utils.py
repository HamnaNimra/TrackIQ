"""Generic CLI utility functions for trackiq_core."""

import json
import os
import sys
import tempfile
from typing import Optional, Tuple, Any

from trackiq_core.utils.errors import HardwareNotFoundError
from trackiq_core.hardware import DeviceProfile


def output_path(args, filename: str) -> str:
    """Return path for an output file inside the output directory (create dir if needed)."""
    out_dir = getattr(args, "output_dir", None) or "output"
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, os.path.basename(filename))


def write_result_to_csv(result: dict, path: str) -> bool:
    """Write run result samples to a CSV file. Returns True if written."""
    samples = result.get("samples", [])
    if not samples:
        return False
    batch_size = result.get("inference_config", {}).get("batch_size", 1)
    rows = []
    for s in samples:
        ts = s.get("timestamp", 0)
        m = s.get("metrics", s) if isinstance(s, dict) else {}
        lat = m.get("latency_ms", 0)
        pwr = m.get("power_w", 0)
        throughput = (1000 / lat) if lat else 0
        rows.append((ts, "default", batch_size, lat, pwr, throughput))
    with open(path, "w", encoding="utf-8") as f:
        f.write("timestamp,workload,batch_size,latency_ms,power_w,throughput\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    return True


def run_default_benchmark(
    device_resolver_fn: Any,
    benchmark_runner_fn: Any,
    device_id: Optional[str] = None,
    duration_seconds: int = 10,
) -> Tuple[dict, Optional[str], Optional[str]]:
    """Run a short benchmark and return (data_dict, temp_csv_path, temp_json_path).

    Args:
        device_resolver_fn: Function to resolve device ID to DeviceProfile
        benchmark_runner_fn: Function to run benchmark (device, config, ...) -> result
        device_id: Device ID to use (default: cpu_0)
        duration_seconds: Benchmark duration

    Returns:
        Tuple of (result_dict, csv_path, json_path)
    """
    from trackiq_core.inference import InferenceConfig, PRECISION_FP32, DEFAULT_WARMUP_RUNS

    device = device_resolver_fn(device_id or "cpu_0")
    if not device:
        raise HardwareNotFoundError(
            "No devices detected. Use --device or install GPU/CPU support."
        )
    config = InferenceConfig(
        precision=PRECISION_FP32,
        batch_size=1,
        accelerator=device.device_id,
        streams=1,
        warmup_runs=DEFAULT_WARMUP_RUNS,
        iterations=min(100, max(20, duration_seconds * 10)),
    )
    result = benchmark_runner_fn(
        device,
        config,
        duration_seconds=float(duration_seconds),
        sample_interval_seconds=0.2,
        quiet=True,
    )
    batch_size = result.get("inference_config", {}).get("batch_size", 1)
    rows = []
    for s in result.get("samples", []):
        ts = s.get("timestamp", 0)
        m = s.get("metrics", s) if isinstance(s, dict) else {}
        lat = m.get("latency_ms", 0)
        pwr = m.get("power_w", 0)
        throughput = (1000 / lat) if lat else 0
        rows.append((ts, "default", batch_size, lat, pwr, throughput))
    path_csv = None
    path_json = None
    if rows:
        fd_csv, path_csv = tempfile.mkstemp(suffix=".csv", prefix="trackiq_")
        try:
            with os.fdopen(fd_csv, "w", encoding="utf-8") as f:
                f.write("timestamp,workload,batch_size,latency_ms,power_w,throughput\n")
                for r in rows:
                    f.write(",".join(str(x) for x in r) + "\n")
        except Exception:
            if path_csv and os.path.exists(path_csv):
                try:
                    os.unlink(path_csv)
                except OSError:
                    pass
            path_csv = None
    fd_json, path_json = tempfile.mkstemp(suffix=".json", prefix="trackiq_")
    try:
        with os.fdopen(fd_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    except Exception:
        if path_json and os.path.exists(path_json):
            try:
                os.unlink(path_json)
            except OSError:
                pass
        path_json = None
    return (result, path_csv, path_json)
