"""CSV export helpers shared by AutoPerfPy CLI/report paths."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _power_profile_lookup(payload: dict[str, Any]) -> dict[int, float]:
    """Build step->power lookup from optional power_profile.step_readings."""
    power_by_step: dict[int, float] = {}
    power_profile = payload.get("power_profile")
    if not isinstance(power_profile, dict):
        return power_by_step
    step_readings = power_profile.get("step_readings")
    if not isinstance(step_readings, list):
        return power_by_step
    for reading in step_readings:
        if not isinstance(reading, dict):
            continue
        step = reading.get("step")
        watts = reading.get("power_watts")
        if isinstance(step, int) and step >= 0 and isinstance(watts, (int, float)):
            power_by_step[step] = float(watts)
    return power_by_step


def _throughput_from_latency(latency_ms: Any) -> float:
    if isinstance(latency_ms, (int, float)) and float(latency_ms) > 0:
        return 1000.0 / float(latency_ms)
    return 0.0


def _iter_run_rows(
    run: dict[str, Any],
    *,
    include_run_label: bool,
    default_run_label: str,
) -> list[dict[str, Any]]:
    """Convert one run payload into normalized CSV row dictionaries."""
    samples = run.get("samples")
    if not isinstance(samples, list) or not samples:
        return []

    batch_size = 1
    inference_cfg = run.get("inference_config")
    if isinstance(inference_cfg, dict):
        raw_batch = inference_cfg.get("batch_size")
        if isinstance(raw_batch, (int, float)):
            batch_size = int(raw_batch)

    run_label = str(run.get("run_label") or run.get("collector_name") or default_run_label)
    power_by_step = _power_profile_lookup(run)
    rows: list[dict[str, Any]] = []
    for row_idx, sample in enumerate(samples):
        if not isinstance(sample, dict):
            continue
        timestamp = sample.get("timestamp", 0)
        metrics = sample.get("metrics", sample)
        if not isinstance(metrics, dict):
            continue

        latency_ms = metrics.get("latency_ms", 0)
        power_w = metrics.get("power_w")
        if not isinstance(power_w, (int, float)) or float(power_w) == 0:
            metadata = sample.get("metadata")
            sample_index = None
            if isinstance(metadata, dict):
                raw_idx = metadata.get("sample_index")
                if isinstance(raw_idx, int):
                    sample_index = raw_idx
            if sample_index is None:
                sample_index = row_idx
            power_w = power_by_step.get(sample_index, 0)

        row: dict[str, Any] = {
            "timestamp": timestamp,
            "workload": "default",
            "batch_size": batch_size,
            "latency_ms": latency_ms,
            "power_w": power_w,
            "throughput": _throughput_from_latency(latency_ms),
        }
        if include_run_label:
            row["run_label"] = run_label
        rows.append(row)
    return rows


def write_single_run_csv(result: dict[str, Any], path: str) -> bool:
    """Write single-run CSV payload in stable legacy column order."""
    rows = _iter_run_rows(result, include_run_label=False, default_run_label="default")
    if not rows:
        return False
    headers = ["timestamp", "workload", "batch_size", "latency_ms", "power_w", "throughput"]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return True


def write_multi_run_csv(runs: list[dict[str, Any]], path: str) -> bool:
    """Write consolidated multi-run CSV payload in stable legacy column order."""
    all_rows: list[dict[str, Any]] = []
    for idx, run in enumerate(runs):
        if not isinstance(run, dict):
            continue
        all_rows.extend(_iter_run_rows(run, include_run_label=True, default_run_label=f"run_{idx + 1}"))
    if not all_rows:
        return False
    headers = ["timestamp", "run_label", "workload", "batch_size", "latency_ms", "power_w", "throughput"]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(all_rows)
    return True

