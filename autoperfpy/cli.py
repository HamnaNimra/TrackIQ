"""Command-line interface for AutoPerfPy.
This CLI uses generic utilities from trackiq_core.cli and adds automotive-specific
commands for profiles, tegrastats analysis, and DNN pipeline analysis.
"""

import argparse
import json
import os
import platform as _platform
import sys
import tempfile
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Any

from autoperfpy.auto_runner import run_auto_benchmarks, run_single_benchmark
from autoperfpy.config import ConfigManager
from autoperfpy.device_config import (
    DEFAULT_WARMUP_RUNS,
    PRECISION_BF16,
    PRECISION_FP16,
    PRECISION_FP32,
    PRECISION_INT4,
    PRECISION_INT8,
    PRECISIONS,
    InferenceConfig,
    resolve_device,
)
from autoperfpy.profiles import ProfileValidationError, list_profiles
from trackiq_core.distributed_validator import DistributedValidationConfig, DistributedValidator
from trackiq_core.hardware import DeviceProfile
from trackiq_core.reporting import PDF_BACKEND_AUTO, PDF_BACKENDS, PdfBackendError
from trackiq_core.schema import (
    KVCacheInfo,
)
from trackiq_core.schema import Metrics as TrackiqMetrics
from trackiq_core.schema import (
    PlatformInfo,
    RegressionInfo,
    TrackiqResult,
    WorkloadInfo,
)
from trackiq_core.serializer import save_trackiq_result
from trackiq_core.utils.errors import DependencyError, HardwareNotFoundError

try:
    AUTOPERFPY_CLI_VERSION = package_version("autoperfpy")
except PackageNotFoundError:
    AUTOPERFPY_CLI_VERSION = "1.0"

DEFAULT_PROFILE_NAMES = [
    "automotive_safety",
    "edge_max_perf",
    "edge_low_power",
    "ci_smoke",
]


def _available_profile_names() -> list[str]:
    """Return profile names for help text, with safe fallback."""
    try:
        names = [str(name).strip() for name in list_profiles()]
        names = [name for name in names if name]
        if names:
            return sorted(dict.fromkeys(names))
    except Exception:
        pass
    return list(DEFAULT_PROFILE_NAMES)


def setup_parser() -> argparse.ArgumentParser:
    """Setup command-line argument parser.

    Returns:
        ArgumentParser configured for AutoPerfPy
    """
    parser = argparse.ArgumentParser(
        prog="autoperfpy",
        description="AutoPerfPy - Performance Analysis Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with a profile (export JSON and/or CSV)
  autoperfpy run --profile automotive_safety --batch-size 4 --export results.json --export-csv results.csv
  autoperfpy run --profile ci_smoke --duration 10
  autoperfpy run --manual --device nvidia_0 --export-csv run.csv

  # List available profiles
  autoperfpy profiles --list
  autoperfpy profiles --info automotive_safety

  # List detected devices (for run --auto or --device)
  autoperfpy devices --list

  # Latency analysis
  autoperfpy analyze latency --csv data.csv
  autoperfpy analyze logs --log performance.log --threshold 50

  # DNN pipeline analysis
  autoperfpy analyze dnn-pipeline --csv layer_times.csv --batch-size 4
  autoperfpy analyze dnn-pipeline --profiler profiler_output.txt

  # Tegrastats analysis
  autoperfpy analyze tegrastats --log tegrastats.log

  # Efficiency / variability (omit --csv to auto-run a quick benchmark)
  autoperfpy analyze efficiency
  autoperfpy analyze efficiency --csv benchmark_data.csv --device cpu_0
  autoperfpy analyze variability --duration 15

  # Benchmarking
  autoperfpy benchmark batching --batch-sizes 1,4,8,16
  autoperfpy benchmark llm --prompt-length 512
  autoperfpy bench-inference --model meta-llama/Llama-3-8B --backend mock --output llm_bench.json

  # Monitoring
  autoperfpy monitor gpu --duration 300

  # Report generation (omit --csv/--json to auto-run a quick benchmark)
  autoperfpy report html
  autoperfpy report html --csv data.csv --output report.html --title "My Report"
  autoperfpy report pdf --device nvidia_0 --duration 15
  autoperfpy report pdf --json results.json --output report.pdf

Options:
  --output-dir DIR      Put report/export files in DIR (default: output). Used by report and run.

Environment Variables:
  AUTOPERFPY_PROFILE    Default profile name
  AUTOPERFPY_CONFIG     Default config file path
  AUTOPERFPY_COLLECTOR  Default collector type (synthetic, nvml, tegrastats, psutil)
        """,
    )

    parser.add_argument("--config", help="Path to configuration file (YAML/JSON)")
    parser.add_argument(
        "--version",
        action="version",
        version=f"autoperfpy {AUTOPERFPY_CLI_VERSION}",
        help="Show CLI version and exit",
    )
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument(
        "--output-dir",
        default="output",
        metavar="DIR",
        help="Directory for report and export files (default: output). Created if missing.",
    )
    profile_names = _available_profile_names()
    profile_list_help = ", ".join(profile_names)

    parser.add_argument(
        "--profile",
        "-p",
        help=f"Performance profile to use ({profile_list_help})",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Profiles command
    profiles_parser = subparsers.add_parser("profiles", help="List and inspect performance profiles")
    profiles_parser.add_argument("--list", "-l", action="store_true", help="List all available profiles")
    profiles_parser.add_argument("--info", "-i", metavar="NAME", help="Show detailed info for a profile")

    # Devices command (list detected hardware for run --auto / --device)
    devices_parser = subparsers.add_parser(
        "devices",
        help="List detected devices (GPUs, CPU, Jetson/DRIVE). Use before run --auto.",
    )
    devices_parser.add_argument("--list", "-l", action="store_true", help="List all detected devices (default)")

    # Run command (profile-based, auto, or manual)
    run_parser = subparsers.add_parser("run", help="Run performance test (auto-detect devices, profile, or manual)")
    run_parser.add_argument(
        "--profile",
        "-p",
        default=None,
        help=(f"Profile to use ({profile_list_help}). " "If omitted, auto mode runs on all detected devices."),
    )
    run_parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatic mode: detect all devices, enumerate configs, run benchmarks (default if no --profile)",
    )
    run_parser.add_argument(
        "--manual",
        action="store_true",
        help="Manual mode: use --device, --collector, --precision, etc. for a single run",
    )
    run_parser.add_argument(
        "--collector",
        "-c",
        default=os.environ.get("AUTOPERFPY_COLLECTOR", "synthetic"),
        choices=["synthetic", "nvml", "tegrastats", "psutil"],
        help="Collector type (manual mode; default: synthetic)",
    )
    run_parser.add_argument(
        "--duration",
        "-d",
        type=int,
        help="Duration in seconds (default: 10 for auto, profile value for profile)",
    )
    run_parser.add_argument("--batch-size", "-b", type=int, help="Batch size (manual or override)")
    run_parser.add_argument("--iterations", "-n", type=int, help="Override number of iterations")
    run_parser.add_argument("--warmup", "-w", type=int, help="Override warmup iterations")
    run_parser.add_argument("--export", "-e", metavar="FILE", help="Export results to JSON file")
    run_parser.add_argument(
        "--export-csv",
        metavar="FILE",
        help="Export run samples to CSV (timestamp, workload, batch_size, latency_ms, power_w, throughput). Use with analyze/report.",
    )
    run_parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output (summary only)")
    run_parser.add_argument("--validate-only", action="store_true", help="Validate profile and exit")
    run_parser.add_argument(
        "--device",
        "-D",
        help="Device ID (e.g. nvidia_0, cpu_0) or GPU index (e.g. 0). Manual mode.",
    )
    run_parser.add_argument(
        "--devices",
        metavar="IDS",
        help="Comma-separated device IDs for auto mode (e.g. nvidia_0,cpu_0). Default: all detected.",
    )
    run_parser.add_argument(
        "--precision",
        "-P",
        choices=PRECISIONS,
        default=PRECISION_FP32,
        help=("Inference precision " f"(default: {PRECISION_FP32}; supported: {', '.join(PRECISIONS)})"),
    )
    run_parser.add_argument(
        "--precisions",
        default=f"{PRECISION_FP32},{PRECISION_FP16}",
        help=("Auto mode: comma-separated precisions " f"(default: {PRECISION_FP32},{PRECISION_FP16})"),
    )
    run_parser.add_argument(
        "--batch-sizes",
        default="1,4",
        help="Auto mode: comma-separated batch sizes (default: 1,4)",
    )
    run_parser.add_argument(
        "--max-configs-per-device",
        type=int,
        default=6,
        help="Auto mode: max (precision x batch) configs per device (default: 6)",
    )
    run_parser.add_argument(
        "--no-power",
        action="store_true",
        help="Disable power profiling for run commands",
    )

    # Compare command (uses trackiq comparison module)
    compare_parser = subparsers.add_parser("compare", help="Compare run results against a baseline (trackiq)")
    compare_parser.add_argument("--baseline", "-b", required=True, help="Baseline name or path to baseline JSON")
    compare_parser.add_argument("--current", "-c", required=True, help="Path to current run JSON (metrics)")
    compare_parser.add_argument(
        "--baseline-dir",
        default=".trackiq/baselines",
        help="Directory for baseline files",
    )
    compare_parser.add_argument("--latency-pct", type=float, default=5.0, help="Latency regression threshold %%")
    compare_parser.add_argument(
        "--throughput-pct",
        type=float,
        default=5.0,
        help="Throughput regression threshold %%",
    )
    compare_parser.add_argument("--p99-pct", type=float, default=10.0, help="P99 regression threshold %%")
    compare_parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save --current as new baseline named --baseline",
    )

    # Analyze commands
    analyze_parser = subparsers.add_parser("analyze", help="Analyze performance data")
    analyze_subparsers = analyze_parser.add_subparsers(dest="analyze_type")

    # Analyze latency
    latency_parser = analyze_subparsers.add_parser("latency", help="Analyze percentile latencies")
    latency_parser.add_argument("--csv", help="CSV file with benchmark data (default: run a quick benchmark)")
    latency_parser.add_argument("--device", "-D", help="Device to use when no --csv (e.g. nvidia_0, cpu_0)")
    latency_parser.add_argument(
        "--duration",
        "-d",
        type=int,
        default=10,
        help="Benchmark duration (s) when no --csv",
    )

    # Analyze logs
    log_parser = analyze_subparsers.add_parser("logs", help="Analyze performance logs")
    log_parser.add_argument("--log", required=True, help="Log file to analyze")
    log_parser.add_argument("--threshold", type=float, default=50.0, help="Latency threshold (ms)")

    # Analyze DNN pipeline
    dnn_parser = analyze_subparsers.add_parser("dnn-pipeline", help="Analyze DNN inference pipeline")
    dnn_parser.add_argument("--csv", help="CSV file with layer timings")
    dnn_parser.add_argument("--profiler", help="Profiler output text file")
    dnn_parser.add_argument("--batch-size", type=int, default=1, help="Batch size used")
    dnn_parser.add_argument("--top-layers", type=int, default=5, help="Number of slowest layers to report")

    # Analyze tegrastats
    tegra_parser = analyze_subparsers.add_parser("tegrastats", help="Analyze Tegrastats output")
    tegra_parser.add_argument("--log", required=True, help="Tegrastats log file")
    tegra_parser.add_argument(
        "--throttle-threshold",
        type=float,
        default=85.0,
        help="Thermal throttling threshold (Â°C)",
    )

    # Analyze efficiency
    eff_parser = analyze_subparsers.add_parser("efficiency", help="Analyze power efficiency")
    eff_parser.add_argument("--csv", help="CSV file with benchmark data (default: run a quick benchmark)")
    eff_parser.add_argument("--device", "-D", help="Device to use when no --csv (e.g. nvidia_0, cpu_0)")
    eff_parser.add_argument(
        "--duration",
        "-d",
        type=int,
        default=10,
        help="Benchmark duration (s) when no --csv",
    )

    # Analyze variability
    var_parser = analyze_subparsers.add_parser("variability", help="Analyze latency variability")
    var_parser.add_argument("--csv", help="CSV file with latency data (default: run a quick benchmark)")
    var_parser.add_argument("--device", "-D", help="Device to use when no --csv (e.g. nvidia_0, cpu_0)")
    var_parser.add_argument(
        "--duration",
        "-d",
        type=int,
        default=10,
        help="Benchmark duration (s) when no --csv",
    )
    var_parser.add_argument("--column", default="latency_ms", help="Column name for latency values")

    # Benchmark commands
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    bench_subparsers = bench_parser.add_subparsers(dest="bench_type")

    # Batch size benchmark
    batch_parser = bench_subparsers.add_parser("batching", help="Batch size trade-off analysis")
    batch_parser.add_argument("--batch-sizes", default="1,4,8,16,32", help="Comma-separated batch sizes")
    batch_parser.add_argument("--images", type=int, default=1000, help="Number of images")

    # LLM benchmark
    llm_parser = bench_subparsers.add_parser("llm", help="LLM inference latency")
    llm_parser.add_argument("--prompt-length", type=int, default=512, help="Prompt token count")
    llm_parser.add_argument("--output-tokens", type=int, default=256, help="Output token count")
    llm_parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs")

    # Inference benchmark (mock/vLLM backends)
    bench_inference_parser = subparsers.add_parser(
        "bench-inference",
        help="Benchmark LLM inference backend (mock or vLLM) and export JSON",
    )
    bench_inference_parser.add_argument(
        "--model",
        required=True,
        help="Model name or local path",
    )
    bench_inference_parser.add_argument(
        "--backend",
        choices=["vllm", "mock"],
        default="mock",
        help="Inference benchmark backend (default: mock)",
    )
    bench_inference_parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="Number of prompts to benchmark (default: 100)",
    )
    bench_inference_parser.add_argument(
        "--input-len",
        type=int,
        default=128,
        help="Input prompt length in tokens (default: 128)",
    )
    bench_inference_parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Output generation length in tokens (default: 128)",
    )
    bench_inference_parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output JSON path for benchmark result",
    )

    # Distributed validation
    distributed_parser = bench_subparsers.add_parser("distributed", help="Validate distributed training correctness")
    distributed_parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    distributed_parser.add_argument(
        "--processes", type=int, default=2, help="Number of processes for distributed training"
    )
    distributed_parser.add_argument("--tolerance", type=float, default=0.05, help="Loss comparison tolerance")
    distributed_parser.add_argument("--baseline", help="Baseline name for regression detection")
    distributed_parser.add_argument("--save-baseline", help="Save results as new baseline with this name")
    distributed_parser.add_argument("--output", "-o", help="Output file for results (JSON)")

    # Monitor commands
    monitor_parser = subparsers.add_parser("monitor", help="Monitor system metrics")
    monitor_subparsers = monitor_parser.add_subparsers(dest="monitor_type")

    # GPU monitor
    gpu_parser = monitor_subparsers.add_parser("gpu", help="Monitor GPU metrics")
    gpu_parser.add_argument("--duration", type=int, default=300, help="Monitor duration (seconds)")
    gpu_parser.add_argument("--interval", type=int, default=1, help="Sample interval (seconds)")

    # KV cache monitor
    cache_parser = monitor_subparsers.add_parser("kv-cache", help="Monitor KV cache")
    cache_parser.add_argument("--max-length", type=int, default=2048, help="Max sequence length")
    cache_parser.add_argument("--num-layers", type=int, default=32)
    cache_parser.add_argument("--num-heads", type=int, default=32)
    cache_parser.add_argument("--head-size", type=int, default=128)
    cache_parser.add_argument("--batch-size", type=int, default=1)
    cache_parser.add_argument(
        "--precision",
        choices=[PRECISION_FP16, PRECISION_BF16, PRECISION_FP32, PRECISION_INT8, PRECISION_INT4],
        default=PRECISION_FP16,
    )

    # Report commands
    report_parser = subparsers.add_parser("report", help="Generate performance reports")
    report_subparsers = report_parser.add_subparsers(dest="report_type")

    # HTML report
    html_parser = report_subparsers.add_parser("html", help="Generate interactive HTML report")
    html_parser.add_argument("--csv", help="CSV file with benchmark data (default: run a quick benchmark)")
    html_parser.add_argument(
        "--json",
        "-j",
        metavar="FILE",
        help="JSON file from autoperfpy run --export (benchmark results)",
    )
    html_parser.add_argument(
        "--device",
        "-D",
        help="Device to use when no --csv/--json (e.g. nvidia_0, cpu_0)",
    )
    html_parser.add_argument(
        "--duration",
        "-d",
        type=int,
        default=10,
        help="Benchmark duration (s) when no data file",
    )
    html_parser.add_argument(
        "--output",
        "-o",
        default="performance_report.html",
        help="Output HTML file path",
    )
    html_parser.add_argument(
        "--export-json",
        metavar="FILE",
        help="Also write report data as JSON (default: <output_basename>_data.json)",
    )
    html_parser.add_argument(
        "--export-csv",
        metavar="FILE",
        help="Also write report data as CSV (default: <output_basename>_data.csv)",
    )
    html_parser.add_argument("--title", default="Performance Analysis Report", help="Report title")
    html_parser.add_argument("--theme", choices=["light", "dark"], default="light", help="Color theme")
    html_parser.add_argument("--author", default="AutoPerfPy", help="Report author")

    # PDF report
    pdf_parser = report_subparsers.add_parser("pdf", help="Generate PDF report")
    pdf_parser.add_argument("--csv", help="CSV file with benchmark data (default: run a quick benchmark)")
    pdf_parser.add_argument(
        "--json",
        "-j",
        metavar="FILE",
        help="JSON file from autoperfpy run --export (benchmark results)",
    )
    pdf_parser.add_argument(
        "--device",
        "-D",
        help="Device to use when no --csv/--json (e.g. nvidia_0, cpu_0)",
    )
    pdf_parser.add_argument(
        "--duration",
        "-d",
        type=int,
        default=10,
        help="Benchmark duration (s) when no data file",
    )
    pdf_parser.add_argument("--output", "-o", default="performance_report.pdf", help="Output PDF file path")
    pdf_parser.add_argument(
        "--export-json",
        metavar="FILE",
        help="Also write report data as JSON (default: <output_basename>_data.json)",
    )
    pdf_parser.add_argument(
        "--export-csv",
        metavar="FILE",
        help="Also write report data as CSV (default: <output_basename>_data.csv)",
    )
    pdf_parser.add_argument("--title", default="Performance Analysis Report", help="Report title")
    pdf_parser.add_argument("--author", default="AutoPerfPy", help="Report author")
    pdf_parser.add_argument(
        "--pdf-backend",
        choices=list(PDF_BACKENDS),
        default=PDF_BACKEND_AUTO,
        help=("PDF backend strategy (default: auto). " "auto uses weasyprint primary with matplotlib fallback."),
    )

    # UI command (Streamlit dashboard)
    ui_parser = subparsers.add_parser("ui", help="Launch interactive Streamlit dashboard")
    ui_parser.add_argument("--data", "-d", help="Path to collector export JSON or CSV file")
    ui_parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8501,
        help="Port to run Streamlit on (default: 8501)",
    )
    ui_parser.add_argument("--host", default="localhost", help="Host to bind to (default: localhost)")
    ui_parser.add_argument(
        "--browser",
        action="store_true",
        default=True,
        help="Open browser automatically (default: True)",
    )
    ui_parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")

    return parser


def _run_default_benchmark(
    device_id: str | None = None,
    duration_seconds: int = 10,
) -> tuple[dict, str | None, str | None]:
    """Run a short benchmark and return (data_dict, temp_csv_path, temp_json_path)."""
    device = resolve_device(device_id or "cpu_0")
    if not device:
        raise HardwareNotFoundError("No devices detected. Use --device or install GPU/CPU support.")
    config = InferenceConfig(
        precision=PRECISION_FP32,
        batch_size=1,
        accelerator=device.device_id,
        streams=1,
        warmup_runs=DEFAULT_WARMUP_RUNS,
        iterations=min(100, max(20, duration_seconds * 10)),
    )
    result = run_single_benchmark(
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
        fd_csv, path_csv = tempfile.mkstemp(suffix=".csv", prefix="autoperfpy_")
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
    fd_json, path_json = tempfile.mkstemp(suffix=".json", prefix="autoperfpy_")
    try:
        os.close(fd_json)
        _save_trackiq_wrapped_json(
            path_json,
            result,
            workload_name="default_benchmark",
            workload_type="inference",
        )
    except Exception:
        if path_json and os.path.exists(path_json):
            try:
                os.unlink(path_json)
            except OSError:
                pass
        path_json = None
    return (result, path_csv, path_json)


def _output_path(args, filename: str) -> str:
    """Return path for an output file inside the output directory (create dir if needed)."""
    out_dir = getattr(args, "output_dir", None) or "output"
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, os.path.basename(filename))


def _parse_precision_list(raw: str) -> tuple[list[str], list[str]]:
    """Parse comma-separated precision list into (valid, invalid)."""
    tokens = [str(item).strip().lower() for item in str(raw or "").split(",")]
    requested = [item for item in tokens if item]
    valid: list[str] = []
    invalid: list[str] = []
    for item in requested:
        if item in PRECISIONS:
            if item not in valid:
                valid.append(item)
        else:
            invalid.append(item)
    return valid, invalid


def _safe_torch_version() -> str:
    """Best-effort PyTorch version lookup."""
    try:
        import torch

        return str(torch.__version__)
    except Exception:
        return "unknown"


def _infer_trackiq_result(
    payload: dict,
    workload_name: str = "autoperfpy_run",
    workload_type: str = "inference",
) -> TrackiqResult:
    """Convert autoperfpy payload to canonical TrackiqResult."""
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    latency = summary.get("latency", {}) if isinstance(summary, dict) else {}
    throughput = summary.get("throughput", {}) if isinstance(summary, dict) else {}
    power = summary.get("power", {}) if isinstance(summary, dict) else {}
    memory = summary.get("memory", {}) if isinstance(summary, dict) else {}
    power_profile = payload.get("power_profile", {}) if isinstance(payload, dict) else {}
    power_profile_summary = power_profile.get("summary", {}) if isinstance(power_profile, dict) else {}
    regression = payload.get("regression", {}) if isinstance(payload, dict) else {}
    platform_metadata = payload.get("platform_metadata", {}) if isinstance(payload, dict) else {}
    inference_cfg = payload.get("inference_config", {}) if isinstance(payload, dict) else {}
    kv_cache_payload = payload.get("kv_cache", {}) if isinstance(payload, dict) else {}

    status = regression.get("status")
    if status not in ("pass", "fail"):
        status = "fail" if payload.get("has_regressions") else "pass"

    return TrackiqResult(
        tool_name="autoperfpy",
        tool_version="1.0",
        timestamp=datetime.now(timezone.utc),
        platform=PlatformInfo(
            hardware_name=str(platform_metadata.get("device_name") or payload.get("collector_name") or "unknown"),
            os=str(platform_metadata.get("os") or f"{_platform.system()} {_platform.release()}"),
            framework="pytorch",
            framework_version=_safe_torch_version(),
        ),
        workload=WorkloadInfo(
            name=str(payload.get("run_label") or workload_name),
            workload_type="training" if workload_type == "training" else "inference",
            batch_size=int(inference_cfg.get("batch_size", 1)),
            steps=int(summary.get("sample_count", len(payload.get("samples", [])))),
        ),
        metrics=TrackiqMetrics(
            throughput_samples_per_sec=float(throughput.get("mean_fps", 0.0)),
            latency_p50_ms=float(latency.get("p50_ms", 0.0)),
            latency_p95_ms=float(latency.get("p95_ms", 0.0)),
            latency_p99_ms=float(latency.get("p99_ms", 0.0)),
            memory_utilization_percent=float(memory.get("mean_percent", 0.0)),
            communication_overhead_percent=None,
            power_consumption_watts=(float(power.get("mean_w")) if power.get("mean_w") is not None else None),
            energy_per_step_joules=(
                (
                    float(power_profile_summary.get("total_energy_joules"))
                    / max(1, int(summary.get("sample_count", len(payload.get("samples", [])))))
                )
                if power_profile_summary.get("total_energy_joules") is not None
                else None
            ),
            performance_per_watt=(
                float(power_profile_summary.get("performance_per_watt"))
                if power_profile_summary.get("performance_per_watt") is not None
                else None
            ),
            temperature_celsius=(
                float(power_profile_summary.get("mean_temperature_celsius"))
                if power_profile_summary.get("mean_temperature_celsius") is not None
                else None
            ),
        ),
        regression=RegressionInfo(
            baseline_id=regression.get("baseline") if isinstance(regression, dict) else None,
            delta_percent=float(regression.get("delta_percent", 0.0)) if isinstance(regression, dict) else 0.0,
            status=status,
            failed_metrics=list(regression.get("failed_metrics", [])) if isinstance(regression, dict) else [],
        ),
        kv_cache=(
            KVCacheInfo(
                estimated_size_mb=float(kv_cache_payload.get("estimated_size_mb", 0.0)),
                max_sequence_length=int(kv_cache_payload.get("max_sequence_length", 0)),
                batch_size=int(kv_cache_payload.get("batch_size", 1)),
                num_layers=int(kv_cache_payload.get("num_layers", 0)),
                num_heads=int(kv_cache_payload.get("num_heads", 0)),
                head_size=int(kv_cache_payload.get("head_size", 0)),
                precision=str(kv_cache_payload.get("precision", "unknown")),
                samples=list(kv_cache_payload.get("samples", [])),
            )
            if isinstance(kv_cache_payload, dict) and kv_cache_payload
            else None
        ),
        tool_payload=payload,
    )


def _save_trackiq_wrapped_json(
    path: str,
    payload: object,
    workload_name: str = "autoperfpy_run",
    workload_type: str = "inference",
) -> None:
    """Save payload wrapped as TrackiqResult JSON."""
    if isinstance(payload, list):
        wrapped = [
            _infer_trackiq_result(
                p if isinstance(p, dict) else {"tool_payload": p},
                workload_name=workload_name,
                workload_type=workload_type,
            ).to_dict()
            for p in payload
        ]
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(wrapped, handle, indent=2)
        return

    if isinstance(payload, dict):
        result = _infer_trackiq_result(payload, workload_name=workload_name, workload_type=workload_type)
    else:
        result = _infer_trackiq_result(
            {"tool_payload": payload},
            workload_name=workload_name,
            workload_type=workload_type,
        )
    save_trackiq_result(result, path)


def _normalize_report_input_data(data: object) -> dict[str, Any]:
    """Normalize report JSON inputs to legacy benchmark payload shape.

    Report generation historically consumed raw benchmark exports where fields
    like ``summary`` and ``samples`` lived at the top level. Newer outputs may
    be wrapped as canonical ``TrackiqResult`` objects with benchmark data in
    ``tool_payload``.
    """
    if isinstance(data, dict) and isinstance(data.get("tool_payload"), dict):
        payload = dict(data["tool_payload"])
        if "collector_name" not in payload and data.get("tool_name"):
            payload["collector_name"] = str(data.get("tool_name"))
        return payload
    if isinstance(data, dict):
        return data
    return {}


def _write_result_to_csv(result: dict, path: str) -> bool:
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


def run_analyze_latency(args, config):
    """Run latency analysis."""
    from autoperfpy.commands.analyze import run_analyze_latency as _cmd_run_analyze_latency

    return _cmd_run_analyze_latency(
        args,
        config,
        run_default_benchmark=_run_default_benchmark,
    )


def run_analyze_logs(args, config):
    """Run log analysis."""
    from autoperfpy.commands.analyze import run_analyze_logs as _cmd_run_analyze_logs

    return _cmd_run_analyze_logs(args, config)


def run_analyze_dnn_pipeline(args, config):
    """Run DNN pipeline analysis."""
    from autoperfpy.commands.analyze import run_analyze_dnn_pipeline as _cmd_run_analyze_dnn_pipeline

    return _cmd_run_analyze_dnn_pipeline(args, config)


def run_analyze_tegrastats(args, _config):
    """Run tegrastats analysis."""
    from autoperfpy.commands.analyze import run_analyze_tegrastats as _cmd_run_analyze_tegrastats

    return _cmd_run_analyze_tegrastats(args, _config)


def run_analyze_efficiency(args, config):
    """Run efficiency analysis."""
    from autoperfpy.commands.analyze import run_analyze_efficiency as _cmd_run_analyze_efficiency

    return _cmd_run_analyze_efficiency(
        args,
        config,
        run_default_benchmark=_run_default_benchmark,
    )


def run_analyze_variability(args, config):
    """Run variability analysis."""
    from autoperfpy.commands.analyze import run_analyze_variability as _cmd_run_analyze_variability

    return _cmd_run_analyze_variability(
        args,
        config,
        run_default_benchmark=_run_default_benchmark,
    )


def run_benchmark_batching(args, config):
    """Run batching trade-off benchmark."""
    from autoperfpy.commands.benchmark import run_benchmark_batching as _cmd_run_benchmark_batching

    return _cmd_run_benchmark_batching(args, config)


def run_benchmark_llm(args, config):
    """Run LLM latency benchmark."""
    from autoperfpy.commands.benchmark import run_benchmark_llm as _cmd_run_benchmark_llm

    return _cmd_run_benchmark_llm(args, config)


def run_bench_inference(args, config):
    """Run LLM inference benchmark (mock/vLLM) and export JSON."""
    from autoperfpy.commands.benchmark import run_bench_inference as _cmd_run_bench_inference

    return _cmd_run_bench_inference(args, config, output_path=_output_path)


def run_benchmark_distributed(args, config):
    """Run distributed training validation."""
    validator = DistributedValidator()

    # Create config from args
    val_config = DistributedValidationConfig(
        num_steps=getattr(args, "steps", 100),
        loss_tolerance=getattr(args, "tolerance", 0.01),
        num_processes=getattr(args, "processes", 2),
        regression_threshold=5.0,  # Default
    )

    print("\nðŸ”„ Distributed Training Validation")
    print("=" * 60)
    print(f"Steps: {val_config.num_steps}")
    print(f"Processes: {val_config.num_processes}")
    print(f"Tolerance: {val_config.loss_tolerance}")

    # Run validation
    results = validator.run_validation(val_config)

    # Handle baseline operations
    if getattr(args, "save_baseline", None):
        validator.save_baseline(args.save_baseline, results)
        print(f"\n[OK] Saved baseline: {args.save_baseline}")

    regression_results = None
    if getattr(args, "baseline", None):
        try:
            regression_results = validator.detect_regression(args.baseline, results)
            has_regression = regression_results.get("has_regressions", False)
            print(f"\n[{'FAIL' if has_regression else 'OK'}] Regression check against '{args.baseline}'")
        except Exception as e:
            print(f"\n[ERROR] Regression check failed: {e}")

    # Print summary
    summary = results["summary"]
    print("\nResults:")
    print(f"  Total Steps: {summary['total_steps']}")
    print(f"  Passed: {summary['passed_steps']}")
    print(f"  Failed: {summary['failed_steps']}")
    print(f"  Pass Rate: {summary['pass_rate']:.2%}")
    print(f"  Overall: {'PASS' if summary['overall_pass'] else 'FAIL'}")

    # Save output if requested
    if getattr(args, "output", None):
        output_path = _output_path(args, args.output)
        _save_trackiq_wrapped_json(
            output_path,
            results,
            workload_name="distributed_validation",
            workload_type="training",
        )
        print(f"\n[OK] Results saved to: {output_path}")

    return results


def run_monitor_gpu(args, config):
    """Run GPU monitoring."""
    from autoperfpy.commands.monitor import run_monitor_gpu as _cmd_run_monitor_gpu

    return _cmd_run_monitor_gpu(args, config)


def run_monitor_kv_cache(args, config):
    """Run KV cache estimation monitor."""
    from autoperfpy.commands.monitor import run_monitor_kv_cache as _cmd_run_monitor_kv_cache

    return _cmd_run_monitor_kv_cache(args, config)


# Chart building is now in trackiq.reporting.charts


def run_report_html(args, config):
    """Generate HTML report."""
    from autoperfpy.commands.report import run_report_html as _cmd_run_report_html

    return _cmd_run_report_html(
        args,
        config,
        run_default_benchmark=_run_default_benchmark,
        normalize_report_input_data=_normalize_report_input_data,
        output_path=_output_path,
        save_trackiq_wrapped_json=_save_trackiq_wrapped_json,
        write_result_to_csv=_write_result_to_csv,
    )


def run_ui(args):
    """Launch Streamlit dashboard."""
    from autoperfpy.commands.ui import run_ui as _cmd_run_ui

    return _cmd_run_ui(args, cli_file=__file__)


def run_report_pdf(args, config):
    """Generate PDF report (same content as HTML, converted to PDF)."""
    from autoperfpy.commands.report import run_report_pdf as _cmd_run_report_pdf

    return _cmd_run_report_pdf(
        args,
        config,
        run_default_benchmark=_run_default_benchmark,
        normalize_report_input_data=_normalize_report_input_data,
        output_path=_output_path,
        save_trackiq_wrapped_json=_save_trackiq_wrapped_json,
        write_result_to_csv=_write_result_to_csv,
    )


def run_compare(args):
    """Compare current run against baseline (uses trackiq_core.cli.commands.compare)."""
    from autoperfpy.commands.compare import run_compare as _cmd_run_compare

    return _cmd_run_compare(args)


def run_profiles(args):
    """Handle profiles command."""
    from autoperfpy.commands.run import run_profiles as _cmd_run_profiles

    return _cmd_run_profiles(args)


def run_with_profile(args, _config):
    """Run performance test with a profile."""
    from autoperfpy.commands.run import run_with_profile as _cmd_run_with_profile

    return _cmd_run_with_profile(
        args,
        _config,
        resolve_device_fn=_resolve_device,
        output_path=_output_path,
        save_trackiq_wrapped_json=_save_trackiq_wrapped_json,
        write_result_to_csv=_write_result_to_csv,
    )


def _resolve_device(device_id: str) -> DeviceProfile | None:
    """Resolve device ID (e.g. nvidia_0, cpu_0, 0) to a DeviceProfile."""
    return resolve_device(device_id)


def run_devices_list(args) -> int:
    """List all detected devices (uses trackiq_core.cli.commands.devices)."""
    from autoperfpy.commands.run import run_devices_list as _cmd_run_devices_list

    return _cmd_run_devices_list(args)


def run_auto_benchmarks_cli(args) -> int:
    """Run automatic benchmarks on all detected devices and configs."""
    from autoperfpy.commands.run import run_auto_benchmarks_cli as _cmd_run_auto_benchmarks_cli

    return _cmd_run_auto_benchmarks_cli(
        args,
        parse_precision_list=_parse_precision_list,
        output_path=_output_path,
        save_trackiq_wrapped_json=_save_trackiq_wrapped_json,
        write_result_to_csv=_write_result_to_csv,
        run_auto_benchmarks_fn=run_auto_benchmarks,
    )


def run_manual_single(args):
    """Run a single benchmark with manually selected device and config."""
    from autoperfpy.commands.run import run_manual_single as _cmd_run_manual_single

    return _cmd_run_manual_single(
        args,
        resolve_device_fn=_resolve_device,
        output_path=_output_path,
        save_trackiq_wrapped_json=_save_trackiq_wrapped_json,
        write_result_to_csv=_write_result_to_csv,
        run_single_benchmark_fn=run_single_benchmark,
    )


def main():
    """Main CLI entry point."""
    parser = setup_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Load configuration
    config = ConfigManager.load_or_default(args.config)

    # Backward compat: run with profile from env if no --profile and no manual/device
    if args.command == "run" and args.profile is None and getattr(args, "device", None) is None:
        env_profile = os.environ.get("AUTOPERFPY_PROFILE")
        if env_profile:
            args.profile = env_profile

    # Route to appropriate handler
    result = None
    try:
        if args.command == "profiles":
            return run_profiles(args)
        elif args.command == "devices":
            return run_devices_list(args)
        elif args.command == "run":
            if args.profile is not None:
                result = run_with_profile(args, config)
                if result is None:
                    return 1
            elif getattr(args, "manual", False) or getattr(args, "device", None) is not None:
                result = run_manual_single(args)
                if result is None:
                    return 1
            else:
                return run_auto_benchmarks_cli(args)
        elif args.command == "analyze":
            if args.analyze_type == "latency":
                result = run_analyze_latency(args, config)
            elif args.analyze_type == "logs":
                result = run_analyze_logs(args, config)
            elif args.analyze_type == "dnn-pipeline":
                result = run_analyze_dnn_pipeline(args, config)
            elif args.analyze_type == "tegrastats":
                result = run_analyze_tegrastats(args, config)
            elif args.analyze_type == "efficiency":
                result = run_analyze_efficiency(args, config)
            elif args.analyze_type == "variability":
                result = run_analyze_variability(args, config)
        elif args.command == "benchmark":
            if args.bench_type == "batching":
                result = run_benchmark_batching(args, config)
            elif args.bench_type == "llm":
                result = run_benchmark_llm(args, config)
            elif args.bench_type == "distributed":
                result = run_benchmark_distributed(args, config)
        elif args.command == "bench-inference":
            result = run_bench_inference(args, config)
            if result is None:
                return 1
        elif args.command == "monitor":
            if args.monitor_type == "gpu":
                result = run_monitor_gpu(args, config)
            elif args.monitor_type == "kv-cache":
                result = run_monitor_kv_cache(args, config)
        elif args.command == "report":
            if args.report_type == "html":
                result = run_report_html(args, config)
            elif args.report_type == "pdf":
                result = run_report_pdf(args, config)
        elif args.command == "ui":
            return run_ui(args)
        elif args.command == "compare":
            return run_compare(args)
    except (HardwareNotFoundError, DependencyError, ProfileValidationError, PdfBackendError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    # Save output if requested (report html/pdf already write to output dir; do not overwrite)
    if args.output and result and getattr(args, "command", None) not in {"report", "bench-inference"}:
        out_path = _output_path(args, args.output)
        if hasattr(result, "to_dict"):
            _save_trackiq_wrapped_json(out_path, result.to_dict())
        else:
            _save_trackiq_wrapped_json(out_path, result)
        print(f"\n[OK] Results saved to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
