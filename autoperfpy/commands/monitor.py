"""Monitoring command handlers for AutoPerfPy CLI."""

from __future__ import annotations

import time
from typing import Any

from autoperfpy.monitoring import GPUMemoryMonitor, LLMKVCacheMonitor


def run_monitor_gpu(args: Any, config: Any) -> dict[str, Any]:
    """Run GPU monitoring."""
    monitor = GPUMemoryMonitor(config)
    print(f"\nMonitoring GPU for {args.duration} seconds...")
    monitor.start()

    try:
        remaining = args.duration
        while remaining > 0:
            metrics = monitor.get_metrics()
            if metrics:
                latest = metrics[-1]
                print(
                    f"GPU Memory: {latest['gpu_memory_used_mb']:.0f}MB / "
                    f"{latest['gpu_memory_total_mb']:.0f}MB "
                    f"({latest['gpu_memory_percent']:.1f}%), "
                    f"Utilization: {latest['gpu_utilization_percent']:.1f}%"
                )
            remaining -= args.interval
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    finally:
        monitor.stop()

    summary = monitor.get_summary()
    if summary:
        print("\nSummary:")
        print(f"  Avg Memory: {summary['avg_memory_mb']:.0f}MB")
        print(f"  Max Memory: {summary['max_memory_mb']:.0f}MB")
        print(f"  Avg Utilization: {summary['avg_utilization_percent']:.1f}%")

    return summary


def run_monitor_kv_cache(args: Any, config: Any) -> dict[str, Any]:
    """Run KV cache estimation monitor."""
    monitor = LLMKVCacheMonitor(config)
    model_config = {
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "head_size": args.head_size,
        "batch_size": args.batch_size,
        "precision": args.precision,
    }
    max_length = int(args.max_length)
    step = max(1, max_length // 10)
    samples = []
    for seq_len in range(step, max_length + 1, step):
        size_mb = monitor.estimate_kv_cache_size(seq_len, model_config)
        samples.append(
            {
                "sequence_length": seq_len,
                "kv_cache_mb": round(float(size_mb), 4),
                "timestamp": time.time(),
            }
        )

    final_size = samples[-1]["kv_cache_mb"] if samples else 0.0
    print("\nKV Cache Monitor")
    print("=" * 60)
    print(f"Precision: {args.precision}")
    print(f"Model: layers={args.num_layers}, heads={args.num_heads}, head_size={args.head_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max sequence length: {max_length}")
    print(f"Estimated KV cache @ max length: {final_size:.2f} MB")

    return {
        "kv_cache": {
            "estimated_size_mb": float(final_size),
            "max_sequence_length": max_length,
            "batch_size": int(args.batch_size),
            "num_layers": int(args.num_layers),
            "num_heads": int(args.num_heads),
            "head_size": int(args.head_size),
            "precision": str(args.precision),
            "samples": samples,
        },
        "summary": {
            "sample_count": len(samples),
            "latency": {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0},
            "throughput": {"mean_fps": 0.0},
            "power": {"mean_w": None},
            "memory": {"mean_percent": 0.0},
        },
        "run_label": "kv_cache_monitor",
    }
