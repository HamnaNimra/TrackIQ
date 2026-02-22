"""Benchmark command handlers for AutoPerfPy CLI."""

from __future__ import annotations

import sys
from typing import Any

from autoperfpy.benchmarks import (
    BatchingTradeoffBenchmark,
    LLMLatencyBenchmark,
    run_inference_benchmark,
    save_inference_benchmark,
)


def run_benchmark_batching(args: Any, config: Any) -> dict[str, Any]:
    """Run batching trade-off benchmark."""
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    benchmark = BatchingTradeoffBenchmark(config)
    results = benchmark.run(batch_sizes=batch_sizes, num_images=args.images)

    print("\nBatching Trade-off Analysis")
    print("=" * 60)
    print(f"{'Batch':<10} {'Latency (ms)':<15} {'Throughput (img/s)':<20}")
    print("-" * 60)
    for i, batch in enumerate(results["batch_size"]):
        latency = results["latency_ms"][i]
        throughput = results["throughput_img_per_sec"][i]
        print(f"{batch:<10} {latency:<15.2f} {throughput:<20.2f}")

    return results


def run_benchmark_llm(args: Any, config: Any) -> dict[str, Any]:
    """Run LLM latency benchmark."""
    benchmark = LLMLatencyBenchmark(config)
    results = benchmark.run(
        prompt_tokens=args.prompt_length,
        output_tokens=args.output_tokens,
        num_runs=args.runs,
    )

    print("\nLLM Latency Benchmark")
    print("=" * 60)
    print("TTFT (Time-to-First-Token):")
    print(f"  P50: {results['ttft_p50']:.1f}ms")
    print(f"  P95: {results['ttft_p95']:.1f}ms")
    print(f"  P99: {results['ttft_p99']:.1f}ms")
    print("\nTime-per-Token (Decode):")
    print(f"  P50: {results['tpt_p50']:.1f}ms")
    print(f"  P95: {results['tpt_p95']:.1f}ms")
    print(f"  P99: {results['tpt_p99']:.1f}ms")
    print(f"\nThroughput: {results['throughput_tokens_per_sec']:.1f} tokens/sec")

    return results


def run_bench_inference(args: Any, _config: Any, *, output_path: Any) -> dict[str, Any] | None:
    """Run inference benchmark for mock/vLLM backend and write JSON output."""
    try:
        result = run_inference_benchmark(
            model=str(args.model),
            backend=str(args.backend),
            num_prompts=int(args.num_prompts),
            input_len=int(args.input_len),
            output_len=int(args.output_len),
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"[ERROR] {exc}", file=sys.stderr)
        return None

    output_file = output_path(args, args.output)
    save_inference_benchmark(result, output_file)
    print(f"[OK] Inference benchmark saved: {output_file}")
    print(
        "[OK] "
        f"TTFT(mean/p99)={result['mean_ttft_ms']:.2f}/{result['p99_ttft_ms']:.2f} ms, "
        f"TPOT(mean/p99)={result['mean_tpot_ms']:.2f}/{result['p99_tpot_ms']:.2f} ms, "
        f"throughput={result['throughput_tokens_per_sec']:.2f} tok/s"
    )
    return result
