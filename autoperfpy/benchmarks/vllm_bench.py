"""Inference benchmark helpers for mock and vLLM backends."""

from __future__ import annotations

import importlib.util
import json
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal


def _percentile(values: list[float], p: float) -> float:
    """Compute percentile with linear interpolation."""
    if not values:
        return 0.0
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


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _mock_samples(num_prompts: int, seed: int = 42) -> tuple[list[float], list[float]]:
    rng = random.Random(seed)
    ttft_samples = [_clamp(rng.gauss(125.0, 30.0), 50.0, 200.0) for _ in range(max(1, num_prompts))]
    tpot_samples = [_clamp(rng.gauss(45.0, 12.0), 20.0, 80.0) for _ in range(max(1, num_prompts))]
    return ttft_samples, tpot_samples


def _parse_float_pattern(text: str, patterns: list[str]) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if not match:
            continue
        try:
            return float(match.group(1))
        except (TypeError, ValueError):
            continue
    return None


def _parse_vllm_stdout(stdout: str) -> dict[str, float]:
    """Best-effort parse of vLLM benchmark stdout for required metrics."""
    throughput = _parse_float_pattern(
        stdout,
        [
            r"throughput[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*(?:tokens?/s|tok/s)",
            r"tokens_per_sec[^0-9]*([0-9]+(?:\.[0-9]+)?)",
        ],
    )
    mean_ttft = _parse_float_pattern(stdout, [r"(?:mean|avg)[^\n]*ttft[^0-9]*([0-9]+(?:\.[0-9]+)?)"])
    p99_ttft = _parse_float_pattern(stdout, [r"p99[^\n]*ttft[^0-9]*([0-9]+(?:\.[0-9]+)?)"])
    mean_tpot = _parse_float_pattern(
        stdout, [r"(?:mean|avg)[^\n]*(?:tpot|time per output token)[^0-9]*([0-9]+(?:\.[0-9]+)?)"]
    )
    p99_tpot = _parse_float_pattern(stdout, [r"p99[^\n]*(?:tpot|time per output token)[^0-9]*([0-9]+(?:\.[0-9]+)?)"])
    return {
        "throughput_tokens_per_sec": throughput or 0.0,
        "mean_ttft_ms": mean_ttft or 0.0,
        "p99_ttft_ms": p99_ttft or 0.0,
        "mean_tpot_ms": mean_tpot or 0.0,
        "p99_tpot_ms": p99_tpot or 0.0,
    }


def run_inference_benchmark(
    *,
    model: str,
    backend: Literal["vllm", "mock"] = "mock",
    num_prompts: int = 100,
    input_len: int = 128,
    output_len: int = 128,
) -> dict[str, Any]:
    """Run inference benchmark and return normalized JSON payload.

    TTFT = Time to First Token (measures prefill latency). TPOT = Time Per Output
    Token (measures decode throughput). These are the two primary SLOs for
    production LLM inference clusters.
    """
    if num_prompts < 1:
        raise ValueError("num_prompts must be >= 1")
    if input_len < 1:
        raise ValueError("input_len must be >= 1")
    if output_len < 1:
        raise ValueError("output_len must be >= 1")

    if backend == "mock":
        ttft_samples, tpot_samples = _mock_samples(num_prompts=num_prompts, seed=42)
        total_decode_seconds = sum((output_len * sample) / 1000.0 for sample in tpot_samples)
        throughput_tokens_per_sec = (num_prompts * output_len) / max(total_decode_seconds, 1e-9)
        return {
            "backend": backend,
            "model": model,
            "num_prompts": int(num_prompts),
            "mean_ttft_ms": float(sum(ttft_samples) / len(ttft_samples)),
            "p99_ttft_ms": float(_percentile(ttft_samples, 99.0)),
            "mean_tpot_ms": float(sum(tpot_samples) / len(tpot_samples)),
            "p99_tpot_ms": float(_percentile(tpot_samples, 99.0)),
            "throughput_tokens_per_sec": float(throughput_tokens_per_sec),
        }

    if importlib.util.find_spec("vllm") is None:
        raise RuntimeError("vLLM backend requested but package 'vllm' is not installed.")

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.benchmark_throughput",
        "--model",
        str(model),
        "--num-prompts",
        str(int(num_prompts)),
        "--input-len",
        str(int(input_len)),
        "--output-len",
        str(int(output_len)),
    ]
    process = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if process.returncode != 0:
        stderr = process.stderr.strip() or process.stdout.strip()
        raise RuntimeError(f"vLLM benchmark failed (exit={process.returncode}): {stderr}")

    parsed = _parse_vllm_stdout(process.stdout)
    return {
        "backend": backend,
        "model": model,
        "num_prompts": int(num_prompts),
        "mean_ttft_ms": float(parsed["mean_ttft_ms"]),
        "p99_ttft_ms": float(parsed["p99_ttft_ms"]),
        "mean_tpot_ms": float(parsed["mean_tpot_ms"]),
        "p99_tpot_ms": float(parsed["p99_tpot_ms"]),
        "throughput_tokens_per_sec": float(parsed["throughput_tokens_per_sec"]),
    }


def save_inference_benchmark(result: dict[str, Any], output_path: str) -> str:
    """Persist benchmark result JSON to disk."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return str(out)
