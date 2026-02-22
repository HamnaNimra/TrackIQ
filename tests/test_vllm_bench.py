"""Tests for AutoPerfPy inference benchmark helpers."""

from __future__ import annotations

import json

import pytest

import autoperfpy.cli as autoperf_cli
from autoperfpy.benchmarks.vllm_bench import run_inference_benchmark, save_inference_benchmark


def test_run_inference_benchmark_mock_backend_is_deterministic() -> None:
    """Mock backend should be deterministic and produce required fields."""
    first = run_inference_benchmark(
        model="mock-model",
        backend="mock",
        num_prompts=32,
        input_len=128,
        output_len=64,
    )
    second = run_inference_benchmark(
        model="mock-model",
        backend="mock",
        num_prompts=32,
        input_len=128,
        output_len=64,
    )

    assert first == second
    assert set(first.keys()) == {
        "backend",
        "model",
        "num_prompts",
        "mean_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "p99_tpot_ms",
        "throughput_tokens_per_sec",
    }
    assert first["backend"] == "mock"
    assert first["num_prompts"] == 32
    assert first["mean_ttft_ms"] > 0
    assert first["p99_ttft_ms"] >= first["mean_ttft_ms"]
    assert first["mean_tpot_ms"] > 0
    assert first["p99_tpot_ms"] >= first["mean_tpot_ms"]
    assert first["throughput_tokens_per_sec"] > 0


def test_run_inference_benchmark_rejects_invalid_inputs() -> None:
    """Benchmark helper should reject invalid numeric arguments."""
    with pytest.raises(ValueError, match="num_prompts must be >= 1"):
        run_inference_benchmark(model="m", backend="mock", num_prompts=0, input_len=1, output_len=1)
    with pytest.raises(ValueError, match="input_len must be >= 1"):
        run_inference_benchmark(model="m", backend="mock", num_prompts=1, input_len=0, output_len=1)
    with pytest.raises(ValueError, match="output_len must be >= 1"):
        run_inference_benchmark(model="m", backend="mock", num_prompts=1, input_len=1, output_len=0)


def test_save_inference_benchmark_writes_json(tmp_path) -> None:
    """Save helper should write JSON file to disk."""
    payload = run_inference_benchmark(
        model="mock-model",
        backend="mock",
        num_prompts=8,
        input_len=32,
        output_len=16,
    )
    output = tmp_path / "bench_inference.json"
    saved_path = save_inference_benchmark(payload, str(output))

    assert saved_path == str(output)
    assert output.exists()
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == payload


def test_cli_parser_accepts_bench_inference_and_analyze_latency_json() -> None:
    """CLI parser should support new bench-inference and analyze latency JSON flags."""
    parser = autoperf_cli.setup_parser()

    args = parser.parse_args(
        [
            "bench-inference",
            "--model",
            "mock-model",
            "--backend",
            "mock",
            "--num-prompts",
            "10",
            "--input-len",
            "64",
            "--output-len",
            "64",
            "--output",
            "bench.json",
        ]
    )
    assert args.command == "bench-inference"
    assert args.model == "mock-model"
    assert args.backend == "mock"
    assert args.output == "bench.json"

    analyze_args = parser.parse_args(["analyze", "latency", "--json", "run.json"])
    assert analyze_args.command == "analyze"
    assert analyze_args.analyze_type == "latency"
    assert analyze_args.json == "run.json"
