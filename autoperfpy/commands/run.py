"""Run/profile/device command handlers for AutoPerfPy CLI."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from typing import Any

from autoperfpy.auto_runner import run_auto_benchmarks, run_single_benchmark
from autoperfpy.collectors import SyntheticCollector
from autoperfpy.device_config import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_ITERATIONS,
    DEFAULT_WARMUP_RUNS,
    PRECISION_FP32,
    PRECISIONS,
    InferenceConfig,
    get_devices_and_configs_auto,
    resolve_precision_for_device,
)
from autoperfpy.profiles import (
    CollectorType,
    ProfileValidationError,
    get_profile,
    get_profile_info,
    validate_profile_collector,
    validate_profile_precision,
)
from trackiq_core.hardware import DeviceProfile, get_memory_metrics
from trackiq_core.power_profiler import PowerProfiler, detect_power_source
from trackiq_core.utils.errors import DependencyError, HardwareNotFoundError


def run_profiles(args: Any) -> int:
    """Handle profiles command."""
    if args.info:
        try:
            profile = get_profile(args.info)
        except ValueError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1

        print(f"\nProfile: {profile.name}")
        print("=" * 60)
        print(f"Description: {profile.description}")
        print("\nLatency Requirements:")
        print(f"  Threshold (P99): {profile.latency_threshold_ms}ms")
        print(f"  Target: {profile.latency_target_ms}ms")
        print(f"  Percentiles: {profile.latency_percentiles}")
        print("\nThroughput Requirements:")
        print(f"  Minimum: {profile.throughput_min_fps} FPS")
        print(f"  Target: {profile.throughput_target_fps} FPS")
        print("\nConstraints:")
        print(f"  Power Budget: {profile.power_budget_w}W" if profile.power_budget_w else "  Power Budget: None")
        print(f"  Thermal Limit: {profile.thermal_limit_c}C")
        print(f"  Memory Limit: {profile.memory_limit_mb}MB" if profile.memory_limit_mb else "  Memory Limit: None")
        print("\nBenchmark Settings:")
        print(f"  Batch Sizes: {profile.batch_sizes}")
        print(f"  Warmup Iterations: {profile.warmup_iterations}")
        print(f"  Test Iterations: {profile.test_iterations}")
        print(f"  Runs: {profile.num_runs}")
        print("\nMonitoring Settings:")
        print(f"  Sample Interval: {profile.sample_interval_ms}ms")
        print(f"  Duration: {profile.duration_seconds}s")
        print("\nSupported Collectors:")
        for collector in profile.supported_collectors:
            print(f"  - {collector.value}")
        print("\nSupported Precisions:")
        for precision in profile.supported_precisions:
            print(f"  - {precision}")
        print(f"\nTags: {', '.join(profile.tags)}")
        return 0

    print("\nAvailable Performance Profiles")
    print("=" * 60)
    info = get_profile_info()
    for name, details in info.items():
        print(f"\n{name}")
        print(f"  {details['description']}")
        print(
            f"  Latency threshold: {details['latency_threshold_ms']}ms | "
            f"Throughput target: {details['throughput_target_fps']} FPS"
        )
        power = f"{details['power_budget_w']}W" if details["power_budget_w"] else "None"
        print(f"  Power budget: {power} | Tags: {', '.join(details['tags'][:3])}")
    print("\nUse 'autoperfpy profiles --info <name>' for detailed information.")
    return 0


def run_devices_list(args: Any) -> int:
    """List all detected devices (uses trackiq_core.cli.commands.devices)."""
    from trackiq_core.cli.commands.devices import run_devices_list as trackiq_run_devices_list

    result = trackiq_run_devices_list(args)
    if result == 0:
        print("\nUsage: autoperfpy run --auto or --device <id>")
    return result


def run_auto_benchmarks_cli(
    args: Any,
    *,
    parse_precision_list: Callable[[str], tuple[list[str], list[str]]],
    output_path: Callable[[Any, str], str],
    save_trackiq_wrapped_json: Callable[[str, Any, str, str], None],
    write_result_to_csv: Callable[[dict[str, Any], str], bool],
    run_auto_benchmarks_fn: Callable[..., list[dict[str, Any]]] = run_auto_benchmarks,
) -> int:
    """Run automatic benchmarks on all detected devices and configs."""
    device_ids_filter = None
    if getattr(args, "devices", None):
        device_ids_filter = [s.strip() for s in args.devices.split(",") if s.strip()]
    raw_precisions = getattr(args, "precisions", ",".join(PRECISIONS))
    precisions, invalid_precisions = parse_precision_list(raw_precisions)
    if invalid_precisions:
        print(
            f"[WARN] Ignoring unsupported precision(s): {', '.join(invalid_precisions)}. "
            f"Supported values: {', '.join(PRECISIONS)}",
            file=sys.stderr,
        )
    if not precisions:
        precisions = [PRECISION_FP32]
        print(
            f"[WARN] No valid precision requested; defaulting to {PRECISION_FP32}.",
            file=sys.stderr,
        )
    batch_sizes: list[int] = []
    for size in getattr(args, "batch_sizes", ",".join(map(str, DEFAULT_BATCH_SIZES))).split(","):
        try:
            batch_sizes.append(int(size.strip()))
        except ValueError:
            continue
    if not batch_sizes:
        batch_sizes = list(DEFAULT_BATCH_SIZES)
    pairs = get_devices_and_configs_auto(
        device_ids_filter=device_ids_filter,
        precisions=precisions,
        batch_sizes=batch_sizes,
        max_configs_per_device=getattr(args, "max_configs_per_device", 6),
    )
    if not pairs:
        print("No (device, config) pairs to run.", file=sys.stderr)
        return 1
    duration = float(getattr(args, "duration", None) or 10)
    if not args.quiet:
        print("Auto mode: running benchmarks on all detected devices and configs")
        print("=" * 60)
        device_ids = list(dict.fromkeys(pair[0].device_id for pair in pairs))
        print(f"Devices: {device_ids}")
        print(f"Runs: {len(pairs)} (duration {duration}s each)")
        print("=" * 60)
    results = run_auto_benchmarks_fn(
        pairs,
        duration_seconds=duration,
        sample_interval_seconds=0.2,
        quiet=args.quiet,
        enable_power=not getattr(args, "no_power", False),
        progress_callback=(
            None
            if args.quiet
            else lambda i, t, device, cfg: print(f"[{i}/{t}] {device.device_id} {cfg.precision} bs{cfg.batch_size}...")
        ),
    )
    if args.export:
        export_path = output_path(args, args.export)
        save_trackiq_wrapped_json(export_path, results, "auto_run_batch", "inference")
        print(f"\n[OK] Exported {len(results)} runs to {export_path}")
    if getattr(args, "export_csv", None):
        base = args.export_csv.rstrip(".csv") if args.export_csv.endswith(".csv") else args.export_csv
        for idx, result in enumerate(results):
            if "error" in result:
                continue
            label = result.get("run_label", str(idx))
            safe_label = label.replace(" ", "_").replace(",", "_")
            path = output_path(args, f"{base}_{safe_label}.csv")
            if write_result_to_csv(result, path) and not args.quiet:
                print(f"[OK] CSV: {path}")
    for result in results:
        if "error" in result:
            print(f"[FAIL] {result.get('run_label', '?')}: {result['error']}", file=sys.stderr)
        elif not args.quiet:
            summary = result.get("summary", {})
            latency = summary.get("latency", {}).get("p99_ms", "N/A")
            throughput = summary.get("throughput", {}).get("mean_fps", "N/A")
            print(f"[OK] {result.get('run_label', '?')} P99={latency}ms Throughput={throughput} FPS")
    return 0


def run_manual_single(
    args: Any,
    *,
    resolve_device_fn: Callable[[str], DeviceProfile | None],
    output_path: Callable[[Any, str], str],
    save_trackiq_wrapped_json: Callable[[str, Any, str, str], None],
    write_result_to_csv: Callable[[dict[str, Any], str], bool],
    run_single_benchmark_fn: Callable[..., dict[str, Any]] = run_single_benchmark,
) -> Any:
    """Run a single benchmark with manually selected device and config."""
    device_id = getattr(args, "device", None) or "cpu_0"
    device = resolve_device_fn(device_id)
    if device is None:
        print("No device found. Use --device nvidia_0, cpu_0, or 0.", file=sys.stderr)
        return None
    requested_precision = str(getattr(args, "precision", None) or PRECISION_FP32).lower()
    effective_precision = resolve_precision_for_device(device, requested_precision)
    if effective_precision != requested_precision:
        print(
            f"[WARN] Precision '{requested_precision}' is not supported on "
            f"{device.device_id}; falling back to '{effective_precision}'.",
            file=sys.stderr,
        )
    config = InferenceConfig(
        precision=effective_precision,
        batch_size=getattr(args, "batch_size", None) or 1,
        accelerator=device.device_id,
        streams=1,
        warmup_runs=getattr(args, "warmup", None) or DEFAULT_WARMUP_RUNS,
        iterations=getattr(args, "iterations", None) or DEFAULT_ITERATIONS,
    )
    duration = float(getattr(args, "duration", None) or 10)
    if not args.quiet:
        print("Manual mode: single run")
        print("=" * 60)
        print(f"Device: {device.device_name} ({device.device_id})")
        print(f"Precision: {config.precision}  Batch: {config.batch_size}")
        print("=" * 60)
    result = run_single_benchmark_fn(
        device,
        config,
        duration_seconds=duration,
        sample_interval_seconds=0.2,
        quiet=args.quiet,
        enable_power=not getattr(args, "no_power", False),
    )
    if args.export:
        export_path = output_path(args, args.export)
        save_trackiq_wrapped_json(export_path, result, "manual_run", "inference")
        print(f"\n[OK] Exported to {export_path}")
    if getattr(args, "export_csv", None):
        csv_path = output_path(args, args.export_csv)
        if write_result_to_csv(result, csv_path):
            print(f"\n[OK] CSV exported to: {csv_path}")
        else:
            print("No samples to export as CSV", file=sys.stderr)
    return result


def run_with_profile(
    args: Any,
    _config: Any,
    *,
    resolve_device_fn: Callable[[str], DeviceProfile | None],
    output_path: Callable[[Any, str], str],
    save_trackiq_wrapped_json: Callable[[str, Any, str, str], None],
    write_result_to_csv: Callable[[dict[str, Any], str], bool],
) -> Any:
    """Run performance test with a profile."""
    profile_name = args.profile
    try:
        profile = get_profile(profile_name)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return None

    collector_map = {
        "synthetic": CollectorType.SYNTHETIC,
        "nvml": CollectorType.NVML,
        "tegrastats": CollectorType.TEGRASTATS,
        "psutil": CollectorType.PSUTIL,
    }
    collector_type = collector_map.get(args.collector, CollectorType.SYNTHETIC)

    try:
        validate_profile_collector(profile, collector_type)
    except ProfileValidationError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return None

    requested_precision = str(getattr(args, "precision", PRECISION_FP32) or PRECISION_FP32).lower()
    try:
        validate_profile_precision(profile, requested_precision)
    except ProfileValidationError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return None

    if args.validate_only:
        print(f"Profile '{profile_name}' validated successfully with collector '{args.collector}'")
        return {
            "status": "validated",
            "profile": profile_name,
            "collector": args.collector,
        }

    duration = args.duration if args.duration else profile.duration_seconds
    iterations = args.iterations if args.iterations else profile.test_iterations
    warmup = args.warmup if args.warmup else profile.warmup_iterations

    if not args.quiet:
        print(f"\nRunning with profile: {profile_name}")
        print("=" * 60)
        print(f"Collector: {args.collector}")
        print(f"Duration: {duration}s")
        print(f"Iterations: {iterations}")
        print(f"Warmup: {warmup}")
        print(f"Latency Threshold: {profile.latency_threshold_ms}ms")
        print("=" * 60)

    collector = None
    if collector_type == CollectorType.NVML:
        try:
            from autoperfpy.collectors import NVMLCollector
        except ImportError as exc:
            print(f"[ERROR] NVML collector requires nvidia-ml-py. {exc}", file=sys.stderr)
            raise DependencyError(
                "NVML collector requires nvidia-ml-py. Install with: pip install nvidia-ml-py"
            ) from exc

        if get_memory_metrics() is None:
            raise HardwareNotFoundError(
                "No NVIDIA GPU or nvidia-smi not available. Use --collector synthetic for simulation."
            )
        device_index = 0
        if getattr(args, "device", None) is not None:
            try:
                device_index = int(args.device)
            except ValueError:
                pass
        collector = NVMLCollector(device_index=device_index, config=profile.get_synthetic_config() or {})
    elif collector_type == CollectorType.PSUTIL:
        try:
            from autoperfpy.collectors import PsutilCollector
        except ImportError as exc:
            print(f"[ERROR] Psutil collector requires psutil. {exc}", file=sys.stderr)
            raise DependencyError("Psutil collector requires psutil. Install with: pip install psutil") from exc
        collector = PsutilCollector(config=profile.get_synthetic_config() or {})
    elif collector_type == CollectorType.TEGRASTATS:
        try:
            from autoperfpy.collectors import TegrastatsCollector
        except ImportError as exc:
            print(f"[ERROR] Tegrastats collector not available. {exc}", file=sys.stderr)
            raise DependencyError(
                "Tegrastats collector requires Jetson/tegrastats. Use --collector synthetic on non-Jetson."
            ) from exc
        collector = TegrastatsCollector(config=profile.get_synthetic_config() or {})

    if collector_type == CollectorType.SYNTHETIC:
        collector_config = profile.get_synthetic_config()
        collector_config["warmup_samples"] = warmup
        if args.batch_size:
            collector_config["batch_sizes"] = [args.batch_size]
        collector = SyntheticCollector(config=collector_config)

    device_id = getattr(args, "device", None)
    effective_precision = requested_precision
    if device_id is not None:
        resolved = resolve_device_fn(device_id)
        if resolved is not None:
            resolved_precision = resolve_precision_for_device(resolved, requested_precision)
            if resolved_precision != requested_precision:
                print(
                    f"[WARN] Precision '{requested_precision}' is not supported on "
                    f"{resolved.device_id}. Falling back to '{resolved_precision}'.",
                    file=sys.stderr,
                )
            effective_precision = resolved_precision
    if not args.quiet and (device_id is not None or effective_precision != PRECISION_FP32):
        print(f"Device: {device_id or 'default'} | Precision: {effective_precision}")

    profiler = None if getattr(args, "no_power", False) else PowerProfiler(detect_power_source())
    if profiler is not None:
        profiler.start_session()
    collector.start()
    sample_count = 0
    sample_interval = profile.sample_interval_ms / 1000.0

    start_time = time.time()
    try:
        while time.time() - start_time < duration and sample_count < iterations:
            timestamp = time.time()
            metrics = collector.sample(timestamp)
            if profiler is not None and metrics:
                profiler.record_step(sample_count, float(metrics.get("throughput_fps", 0.0) or 0.0))

            if not args.quiet and metrics:
                warmup_marker = "[WARMUP]" if metrics.get("is_warmup") else ""
                latency = metrics.get("latency_ms", 0)
                gpu = metrics.get("gpu_percent", 0)
                power = metrics.get("power_w", 0)
                print(
                    f"[{sample_count:4d}] "
                    f"Latency: {latency:6.2f}ms | "
                    f"GPU: {gpu:5.1f}% | "
                    f"Power: {power:5.1f}W "
                    f"{warmup_marker}"
                )

            sample_count += 1
            time.sleep(sample_interval)

    except KeyboardInterrupt:
        print("\nCollection interrupted by user")

    collector.stop()
    if profiler is not None:
        profiler.end_session()

    export = collector.export()
    summary = export.summary
    if profiler is not None:
        profile_payload = profiler.to_tool_payload().get("power_profile", {})
        profile_summary = profile_payload.get("summary", {})
        summary.setdefault("power", {})
        summary["power"]["mean_w"] = profile_summary.get("mean_power_watts")
        summary["power"]["peak_w"] = profile_summary.get("peak_power_watts")

    latency_p99 = summary.get("latency", {}).get("p99_ms", 0)
    throughput = summary.get("throughput", {}).get("mean_fps", 0)
    power_avg = summary.get("power", {}).get("mean_w", 0)

    latency_pass = latency_p99 <= profile.latency_threshold_ms
    throughput_pass = throughput >= profile.throughput_min_fps
    power_pass = profile.power_budget_w is None or power_avg <= profile.power_budget_w

    print(f"\n{'=' * 60}")
    print("Results Summary")
    print("=" * 60)
    print(f"Samples Collected: {summary.get('sample_count', 0)}")

    print("\nLatency (excluding warmup):")
    latency_status = "PASS" if latency_pass else "FAIL"
    print(f"  P99: {latency_p99:.2f}ms (threshold: {profile.latency_threshold_ms}ms) [{latency_status}]")
    print(f"  P95: {summary.get('latency', {}).get('p95_ms', 0):.2f}ms")
    print(f"  P50: {summary.get('latency', {}).get('p50_ms', 0):.2f}ms")
    print(f"  Mean: {summary.get('latency', {}).get('mean_ms', 0):.2f}ms")

    print("\nThroughput:")
    throughput_status = "PASS" if throughput_pass else "FAIL"
    print(f"  Mean: {throughput:.1f} FPS (min: {profile.throughput_min_fps} FPS) [{throughput_status}]")

    print("\nPower:")
    if profile.power_budget_w:
        power_status = "PASS" if power_pass else "FAIL"
        print(f"  Mean: {power_avg:.1f}W (budget: {profile.power_budget_w}W) [{power_status}]")
    else:
        print(f"  Mean: {power_avg:.1f}W (no budget constraint)")

    print("\nResource Utilization:")
    print(f"  GPU: {summary.get('gpu', {}).get('mean_percent', 0):.1f}% avg")
    print(f"  CPU: {summary.get('cpu', {}).get('mean_percent', 0):.1f}% avg")
    print(f"  Memory: {summary.get('memory', {}).get('mean_mb', 0):.0f}MB avg")

    overall_pass = latency_pass and throughput_pass and power_pass
    overall_status = "PASS" if overall_pass else "FAIL"
    print(f"\nOverall Status: [{overall_status}]")

    if args.export:
        export_path = output_path(args, args.export)
        export_data = export.to_dict()
        if profiler is not None:
            export_data["power_profile"] = profiler.to_tool_payload().get("power_profile")
        export_data.setdefault("inference_config", {})
        export_data["inference_config"]["precision"] = effective_precision
        export_data["profile"] = profile.name
        export_data["validation"] = {
            "latency_pass": latency_pass,
            "throughput_pass": throughput_pass,
            "power_pass": power_pass,
            "overall_pass": overall_pass,
        }
        save_trackiq_wrapped_json(
            export_path,
            export_data,
            workload_name=f"profile_{profile.name}",
            workload_type="inference",
        )
        print(f"\nResults exported to: {export_path}")

    if getattr(args, "export_csv", None):
        csv_path = output_path(args, args.export_csv)
        export_data = export.to_dict()
        export_data.setdefault("inference_config", {})["batch_size"] = args.batch_size or (
            profile.batch_sizes[0] if profile.batch_sizes else 1
        )
        if write_result_to_csv(export_data, csv_path):
            print(f"CSV exported to: {csv_path}")
        else:
            print("No samples to export as CSV", file=sys.stderr)

    return export

