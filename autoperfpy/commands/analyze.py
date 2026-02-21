"""Analyze command handlers for AutoPerfPy CLI."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import Any

from autoperfpy.analyzers import (
    DNNPipelineAnalyzer,
    EfficiencyAnalyzer,
    LogAnalyzer,
    PercentileLatencyAnalyzer,
    TegrastatsAnalyzer,
    VariabilityAnalyzer,
)
from trackiq_core.utils.errors import DependencyError, HardwareNotFoundError


def _resolve_csv_input(
    args: Any,
    run_default_benchmark: Callable[[str | None, int], tuple[dict, str | None, str | None]],
) -> tuple[str | None, list[str]]:
    """Return csv path for analyzers and list of generated temp files to clean up."""
    csv_path = getattr(args, "csv", None)
    cleanup_paths: list[str] = []
    if csv_path:
        return csv_path, cleanup_paths

    print("No --csv provided; running a quick benchmark to generate data...")
    _, csv_path, json_path = run_default_benchmark(
        device_id=getattr(args, "device", None),
        duration_seconds=getattr(args, "duration", 10),
    )
    if json_path:
        cleanup_paths.append(json_path)
    if csv_path:
        cleanup_paths.append(csv_path)
    return csv_path, cleanup_paths


def _cleanup_paths(paths: list[str]) -> None:
    """Best-effort temp file cleanup."""
    for path in paths:
        try:
            if path and os.path.exists(path):
                os.unlink(path)
        except OSError:
            pass


def run_analyze_latency(
    args: Any,
    config: Any,
    *,
    run_default_benchmark: Callable[[str | None, int], tuple[dict, str | None, str | None]],
) -> Any:
    """Run latency analysis."""
    csv_path = None
    cleanup_paths: list[str] = []
    try:
        csv_path, cleanup_paths = _resolve_csv_input(args, run_default_benchmark)
    except (HardwareNotFoundError, DependencyError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return None

    if not csv_path:
        print("[ERROR] Could not generate benchmark CSV", file=sys.stderr)
        return None

    try:
        analyzer = PercentileLatencyAnalyzer(config)
        result = analyzer.analyze(csv_path)

        print("\nPercentile Latency Analysis")
        print("=" * 60)
        for key, metrics in result.metrics.items():
            print(f"\n{key}:")
            print(f"  P99: {metrics.get('p99', 0):.2f}ms")
            print(f"  P95: {metrics.get('p95', 0):.2f}ms")
            print(f"  P50: {metrics.get('p50', 0):.2f}ms")
            print(f"  Mean: {metrics.get('mean', 0):.2f}ms +/- {metrics.get('std', 0):.2f}ms")
        return result
    finally:
        _cleanup_paths(cleanup_paths)


def run_analyze_logs(args: Any, config: Any) -> Any:
    """Run log analysis."""
    analyzer = LogAnalyzer(config)
    result = analyzer.analyze(args.log, args.threshold)

    print("\nLog Analysis")
    print("=" * 60)
    print(f"Threshold: {result.metrics['threshold_ms']}ms")
    print(f"Total events: {result.metrics['total_events']}")
    print(f"Spike events: {result.metrics['spike_events']}")
    print(f"Spike percentage: {result.metrics['spike_percentage']:.2f}%")
    return result


def run_analyze_dnn_pipeline(args: Any, config: Any) -> Any:
    """Run DNN pipeline analysis."""
    analyzer = DNNPipelineAnalyzer(config={"top_n_layers": args.top_layers})

    if args.csv:
        result = analyzer.analyze_layer_csv(args.csv, batch_size=args.batch_size)
    elif args.profiler:
        with open(args.profiler, encoding="utf-8") as handle:
            content = handle.read()
        result = analyzer.analyze_profiler_output(content)
    else:
        print("[ERROR] Either --csv or --profiler must be specified", file=sys.stderr)
        return None

    print("\nDNN Pipeline Analysis")
    print("=" * 60)
    metrics = result.metrics
    print(f"\nSource: {metrics.get('source', 'unknown')}")
    print(f"Batch Size: {metrics.get('batch_size', 1)}")
    print(f"Number of Layers: {metrics.get('num_layers', 0)}")

    timing = metrics.get("timing", {})
    print("\nTiming:")
    print(f"  Total Time: {timing.get('total_time_ms', timing.get('avg_total_ms', 0)):.2f}ms")
    print(f"  GPU Time: {timing.get('gpu_time_ms', 0):.2f}ms")
    print(f"  DLA Time: {timing.get('dla_time_ms', 0):.2f}ms")

    device_split = metrics.get("device_split", {})
    print("\nDevice Split:")
    print(f"  GPU: {device_split.get('gpu_percentage', 0):.1f}%")
    print(f"  DLA: {device_split.get('dla_percentage', 0):.1f}%")

    throughput = metrics.get("throughput_fps", metrics.get("throughput", {}).get("avg_fps", 0))
    print(f"\nThroughput: {throughput:.1f} FPS")

    slowest_layers = metrics.get("slowest_layers", [])
    if slowest_layers:
        print("\nSlowest Layers:")
        for layer in slowest_layers[:5]:
            name = layer.get("name", "unknown")
            time_ms = layer.get("time_ms", layer.get("avg_time_ms", 0))
            device = layer.get("device", "GPU")
            print(f"  {name}: {time_ms:.2f}ms ({device})")

    recommendations = metrics.get("recommendations", [])
    if recommendations:
        print("\nRecommendations:")
        for rec in recommendations:
            print(f"  - {rec}")

    return result


def run_analyze_tegrastats(args: Any, _config: Any) -> Any:
    """Run tegrastats analysis."""
    analyzer = TegrastatsAnalyzer(config={"throttle_temp_c": getattr(args, "throttle_threshold", 85.0)})
    result = analyzer.analyze(args.log)
    metrics = result.metrics

    print("\nTegrastats Analysis")
    print("=" * 60)
    print(f"\nSamples Analyzed: {metrics.get('num_samples', 0)}")

    cpu = metrics.get("cpu", {})
    print("\nCPU:")
    print(f"  Average Utilization: {cpu.get('avg_utilization', 0):.1f}%")
    print(f"  Max Utilization: {cpu.get('max_utilization', 0):.1f}%")

    gpu = metrics.get("gpu", {})
    print("\nGPU:")
    print(f"  Average Utilization: {gpu.get('avg_utilization', 0):.1f}%")
    print(f"  Max Utilization: {gpu.get('max_utilization', 0):.1f}%")
    print(f"  Average Frequency: {gpu.get('avg_frequency_mhz', 0):.0f} MHz")

    memory = metrics.get("memory", {})
    print("\nMemory:")
    print(f"  Average Used: {memory.get('avg_used_mb', 0):.0f} MB")
    print(f"  Max Used: {memory.get('max_used_mb', 0):.0f} MB")

    thermal = metrics.get("thermal", {})
    print("\nThermal:")
    print(f"  Average Temperature: {thermal.get('avg_temperature', 0):.1f}C")
    print(f"  Max Temperature: {thermal.get('max_temperature', 0):.1f}C")
    print(f"  Throttling Events: {thermal.get('throttle_events', 0)}")

    health = metrics.get("health", {})
    status = health.get("status", "unknown")
    print(f"\nHealth Status: {str(status).upper()}")
    warnings = health.get("warnings", [])
    if warnings:
        print("  Warnings:")
        for warning in warnings:
            print(f"    - {warning}")
    return result


def run_analyze_efficiency(
    args: Any,
    config: Any,
    *,
    run_default_benchmark: Callable[[str | None, int], tuple[dict, str | None, str | None]],
) -> Any:
    """Run efficiency analysis."""
    csv_path = None
    cleanup_paths: list[str] = []
    try:
        csv_path, cleanup_paths = _resolve_csv_input(args, run_default_benchmark)
    except (HardwareNotFoundError, DependencyError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return None

    if not csv_path:
        print("[ERROR] Could not generate benchmark CSV", file=sys.stderr)
        return None

    try:
        analyzer = EfficiencyAnalyzer(config)
        result = analyzer.analyze(csv_path)

        print("\nEfficiency Analysis")
        print("=" * 60)
        metrics = result.metrics
        for workload, data in metrics.items():
            if not isinstance(data, dict):
                continue
            print(f"\n{workload}:")
            print(f"  Performance/Watt: {data.get('perf_per_watt', 0):.2f} infer/s/W")
            print(f"  Energy/Inference: {data.get('energy_per_inference_j', 0):.4f} J")
            print(f"  Throughput: {data.get('throughput_fps', 0):.1f} FPS")
            print(f"  Average Power: {data.get('avg_power_w', 0):.1f} W")
        return result
    finally:
        _cleanup_paths(cleanup_paths)


def run_analyze_variability(
    args: Any,
    config: Any,
    *,
    run_default_benchmark: Callable[[str | None, int], tuple[dict, str | None, str | None]],
) -> Any:
    """Run variability analysis."""
    csv_path = None
    cleanup_paths: list[str] = []
    try:
        csv_path, cleanup_paths = _resolve_csv_input(args, run_default_benchmark)
    except (HardwareNotFoundError, DependencyError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return None

    if not csv_path:
        print("[ERROR] Could not generate benchmark CSV", file=sys.stderr)
        return None

    try:
        analyzer = VariabilityAnalyzer(config)
        result = analyzer.analyze(csv_path, latency_column=args.column)
        metrics = result.metrics

        print("\nVariability Analysis")
        print("=" * 60)
        print(f"\nCoefficient of Variation: {metrics.get('cv_percent', 0):.2f}%")
        print(f"Jitter (Std Dev): {metrics.get('jitter_ms', 0):.2f}ms")
        print(f"IQR: {metrics.get('iqr_ms', 0):.2f}ms")
        print(f"Outliers: {metrics.get('outlier_count', 0)}")
        print(f"Consistency Rating: {metrics.get('consistency_rating', 'unknown')}")
        print("\nPercentiles:")
        print(f"  P50: {metrics.get('p50_ms', 0):.2f}ms")
        print(f"  P95: {metrics.get('p95_ms', 0):.2f}ms")
        print(f"  P99: {metrics.get('p99_ms', 0):.2f}ms")
        return result
    finally:
        _cleanup_paths(cleanup_paths)

