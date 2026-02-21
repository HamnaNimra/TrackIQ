"""Command-line interface for minicluster.

Provides subcommands for:
- run: Execute distributed training harness
- validate: Correctness validation between single and multi-process runs
- fault-test: Fault injection testing
- baseline save: Save current run as baseline
- baseline compare: Compare run against baseline with regression detection
"""

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

from minicluster.benchmarks import run_collective_benchmark, save_collective_benchmark
from minicluster.deps import RegressionDetector, RegressionThreshold
from minicluster.monitor.cli import register_monitor_subcommand
from minicluster.reporting import (
    MiniClusterHtmlReporter,
    generate_cluster_heatmap,
    generate_fault_timeline,
    load_worker_results_from_dir,
)
from minicluster.runner import RunConfig, run_distributed, save_metrics
from trackiq_core.reporting import (
    PDF_BACKENDS,
    PdfBackendError,
    render_pdf_from_html,
    render_trackiq_result_html,
)
from trackiq_core.serializer import load_trackiq_result

try:
    MINICLUSTER_CLI_VERSION = package_version("minicluster")
except PackageNotFoundError:
    MINICLUSTER_CLI_VERSION = "0.1.0"


def _print_subcommand_help(parser: argparse.ArgumentParser, command: str) -> None:
    """Print help text for a specific top-level subcommand."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            subparser = action.choices.get(command)
            if subparser is not None:
                subparser.print_help()
                return
    parser.print_help()


def setup_run_parser(subparsers):
    """Setup 'minicluster run' subcommand."""
    parser = subparsers.add_parser(
        "run",
        help="Run distributed training harness",
        description="Execute a distributed training run on synthetic data",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of worker processes (default: 2)",
    )
    parser.add_argument(
        "--backend",
        choices=["gloo", "nccl"],
        default="gloo",
        help="Collective communication backend (default: gloo)",
    )
    parser.add_argument(
        "--workload",
        choices=["mlp", "transformer", "embedding"],
        default="mlp",
        help="Synthetic workload type (default: mlp)",
    )
    parser.add_argument(
        "--baseline-throughput",
        type=float,
        default=None,
        help="Single-worker baseline throughput for scaling efficiency calculation",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of training steps (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="Hidden layer dimension (default: 128)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of layers (default: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--tdp-watts",
        type=float,
        default=150.0,
        help="TDP used by simulated power profiler (default: 150.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./minicluster_results/run_metrics.json",
        help="Output path for metrics JSON (default: ./minicluster_results/run_metrics.json)",
    )
    parser.add_argument(
        "--health-checkpoint-path",
        type=str,
        default="./minicluster_results/health.json",
        help="Path for live health monitoring data. Read by minicluster monitor during a run.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output",
    )

    parser.set_defaults(func=cmd_run)


def setup_validate_parser(subparsers):
    """Setup 'minicluster validate' subcommand."""
    parser = subparsers.add_parser(
        "validate",
        help="Validate correctness between single and multi-process runs",
        description="Compare loss values from single-process and multi-process runs",
    )

    parser.add_argument(
        "single_run",
        help="Path to single-process metrics JSON file",
    )
    parser.add_argument(
        "multi_run",
        help="Path to multi-process metrics JSON file",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Relative tolerance for loss comparison (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for validation report JSON",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print all steps, not just failures",
    )

    parser.set_defaults(func=cmd_validate)


def setup_fault_test_parser(subparsers):
    """Setup 'minicluster fault-test' subcommand."""
    parser = subparsers.add_parser(
        "fault-test",
        help="Run fault injection tests",
        description="Test the validation framework by injecting faults",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of training steps for fault tests (default: 20)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Tolerance for fault detection (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./minicluster_results/fault_report.json",
        help="Output path for fault test report (default: ./minicluster_results/fault_report.json)",
    )

    parser.set_defaults(func=cmd_fault_test)


def setup_baseline_parser(subparsers):
    """Setup 'minicluster baseline' subcommand group."""
    parser = subparsers.add_parser(
        "baseline",
        help="Baseline management (save/compare)",
        description="Save and compare performance baselines",
    )

    baseline_subparsers = parser.add_subparsers(dest="baseline_cmd", help="Baseline subcommand")

    # baseline save
    save_parser = baseline_subparsers.add_parser(
        "save",
        help="Save current run as baseline",
    )
    save_parser.add_argument(
        "--metrics",
        type=str,
        required=True,
        help="Path to metrics JSON file to save as baseline",
    )
    save_parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Baseline name for identification",
    )
    save_parser.add_argument(
        "--baseline-dir",
        type=str,
        default=".minicluster/baselines",
        help="Directory to store baselines (default: .minicluster/baselines)",
    )
    save_parser.set_defaults(func=cmd_baseline_save)

    # baseline compare
    compare_parser = baseline_subparsers.add_parser(
        "compare",
        help="Compare run against saved baseline",
    )
    compare_parser.add_argument(
        "--metrics",
        type=str,
        required=True,
        help="Path to current metrics JSON file",
    )
    compare_parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Baseline name to compare against",
    )
    compare_parser.add_argument(
        "--baseline-dir",
        type=str,
        default=".minicluster/baselines",
        help="Directory containing baselines (default: .minicluster/baselines)",
    )
    compare_parser.add_argument(
        "--latency-threshold",
        type=float,
        default=5.0,
        help="Regression threshold for latency metrics (default: 5%%)",
    )
    compare_parser.add_argument(
        "--throughput-threshold",
        type=float,
        default=5.0,
        help="Regression threshold for throughput metrics (default: 5%%)",
    )
    compare_parser.add_argument(
        "--output",
        type=str,
        help="Output path for comparison report JSON",
    )
    compare_parser.set_defaults(func=cmd_baseline_compare)


def setup_report_parser(subparsers):
    """Setup `minicluster report` subcommands."""
    parser = subparsers.add_parser(
        "report",
        help="Generate reports from canonical TrackiqResult files",
        description="Render reports from saved run output JSON files",
    )
    report_subparsers = parser.add_subparsers(dest="report_cmd", help="Report subcommand")

    html_parser = report_subparsers.add_parser(
        "html",
        help="Generate HTML report from one or more canonical result JSON files",
    )
    html_parser.add_argument(
        "--result",
        nargs="+",
        required=True,
        help=("Path(s) to TrackiqResult JSON. " "Use multiple files to generate a consolidated multi-config report."),
    )
    html_parser.add_argument(
        "--output",
        default="./minicluster_results/report.html",
        help="Output HTML file path (default: ./minicluster_results/report.html)",
    )
    html_parser.add_argument(
        "--title",
        default="MiniCluster Performance Report",
        help="Report title",
    )
    html_parser.set_defaults(func=cmd_report_html)

    pdf_parser = report_subparsers.add_parser(
        "pdf",
        help="Generate PDF report from canonical result JSON",
    )
    pdf_parser.add_argument(
        "--result",
        required=True,
        help="Path to TrackiqResult JSON (e.g. output from `minicluster run --output ...`)",
    )
    pdf_parser.add_argument(
        "--output",
        default="./minicluster_results/report.pdf",
        help="Output PDF file path (default: ./minicluster_results/report.pdf)",
    )
    pdf_parser.add_argument(
        "--title",
        default="MiniCluster Performance Report",
        help="Report title",
    )
    pdf_parser.add_argument(
        "--pdf-backend",
        choices=list(PDF_BACKENDS),
        default="auto",
        help=("PDF backend strategy (default: auto). " "auto uses weasyprint primary with matplotlib fallback."),
    )
    pdf_parser.set_defaults(func=cmd_report_pdf)

    heatmap_parser = report_subparsers.add_parser(
        "heatmap",
        help="Generate cluster health heatmap from per-worker JSON files",
    )
    heatmap_parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing per-worker JSON files",
    )
    heatmap_parser.add_argument(
        "--metric",
        choices=[
            "allreduce_time_ms",
            "p99_allreduce_ms",
            "throughput_samples_per_sec",
            "compute_time_ms",
        ],
        default="allreduce_time_ms",
        help="Metric to visualize in heatmap (default: allreduce_time_ms)",
    )
    heatmap_parser.add_argument(
        "--output",
        default="./minicluster_results/heatmap.html",
        help="Output HTML file path (default: ./minicluster_results/heatmap.html)",
    )
    heatmap_parser.set_defaults(func=cmd_report_heatmap)

    fault_timeline_parser = report_subparsers.add_parser(
        "fault-timeline",
        help="Generate fault injection timeline HTML report",
    )
    fault_timeline_parser.add_argument(
        "--json",
        required=True,
        help="Path to fault-test report JSON (e.g. output from `minicluster fault-test`)",
    )
    fault_timeline_parser.add_argument(
        "--output",
        default="./minicluster_results/fault_timeline.html",
        help="Output HTML file path (default: ./minicluster_results/fault_timeline.html)",
    )
    fault_timeline_parser.set_defaults(func=cmd_report_fault_timeline)


def setup_bench_collective_parser(subparsers):
    """Setup 'minicluster bench-collective' subcommand."""
    parser = subparsers.add_parser(
        "bench-collective",
        help="Benchmark all-reduce bandwidth without compute overhead",
        description="Run communication-only collective benchmark",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of worker processes (default: 2)",
    )
    parser.add_argument(
        "--size-mb",
        type=float,
        default=256.0,
        help="All-reduce tensor size in MB (default: 256.0)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of all-reduce iterations (default: 50)",
    )
    parser.add_argument(
        "--backend",
        choices=["gloo", "nccl"],
        default="gloo",
        help="Collective communication backend (default: gloo)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional output path for benchmark JSON",
    )
    parser.set_defaults(func=cmd_bench_collective)


def cmd_run(args):
    """Execute 'minicluster run' command."""
    config = RunConfig(
        num_steps=args.steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_processes=args.workers,
        seed=args.seed,
        tdp_watts=args.tdp_watts,
        collective_backend=args.backend,
        baseline_throughput=args.baseline_throughput,
        workload=args.workload,
    )

    if args.verbose:
        print("Running minicluster with config:")
        print(f"  Workers: {config.num_processes}")
        print(f"  Steps: {config.num_steps}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Backend: {config.collective_backend}")
        print(f"  Workload: {config.workload}")
        if config.baseline_throughput is not None:
            print(f"  Baseline throughput: {config.baseline_throughput}")

    print("\nStarting distributed training run...")
    metrics = run_distributed(config, health_checkpoint_path=args.health_checkpoint_path)

    save_metrics(metrics, args.output)
    print("\n[OK] Run complete!")
    print(f"  Total time: {metrics.total_time_sec:.2f}s")
    print(f"  Final loss: {metrics.steps[-1].loss:.6f}")
    print(f"  Avg throughput: {metrics.to_dict()['average_throughput_samples_per_sec']:.1f} samples/sec")
    print(f"  Metrics saved to: {args.output}")


def cmd_bench_collective(args):
    """Execute 'minicluster bench-collective' command."""
    try:
        result = run_collective_benchmark(
            workers=int(args.workers),
            size_mb=float(args.size_mb),
            iterations=int(args.iterations),
            backend=args.backend,
        )
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:  # pragma: no cover - defensive runtime guard
        print(f"[ERROR] Collective benchmark failed: {e}", file=sys.stderr)
        sys.exit(2)

    if args.output:
        save_collective_benchmark(result, args.output)
        print(f"[OK] Collective benchmark written to: {args.output}")
    else:
        print(json.dumps(result, indent=2))

    print(
        f"[OK] mean={result['mean_bandwidth_gbps']:.3f} GB/s, "
        f"p99={result['p99_bandwidth_gbps']:.3f} GB/s, "
        f"min={result['min_bandwidth_gbps']:.3f} GB/s"
    )


def cmd_validate(args):
    """Execute 'minicluster validate' command."""
    try:
        from minicluster.validators.correctness_validator import CorrectnessValidator

        validator = CorrectnessValidator(tolerance=args.tolerance)
        report = validator.validate_file_pair(args.single_run, args.multi_run, output_path=args.output)

        validator.print_report(report, verbose=args.verbose)

        if args.output:
            print(f"Report saved to: {args.output}")

        sys.exit(0 if report.overall_passed else 1)

    except FileNotFoundError as e:
        print(f"[ERROR] {str(e)}", file=sys.stderr)
        sys.exit(2)
    except ValueError as e:
        print(f"[ERROR] Validation error: {str(e)}", file=sys.stderr)
        sys.exit(1)


def cmd_fault_test(args):
    """Execute 'minicluster fault-test' command."""
    try:
        from minicluster.validators.fault_injector import FaultInjector
    except Exception as exc:
        print(
            "[ERROR] fault injection requires optional ML dependencies. " 'Install with: pip install -e ".[ml]"',
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    base_config = RunConfig(num_steps=args.steps)
    injector = FaultInjector(base_config, tolerance=args.tolerance)

    print("Running fault injection tests...")
    report = injector.run_fault_injection_tests()

    injector.print_report(report)
    injector.save_report(report, args.output)

    print(f"Report saved to: {args.output}")

    sys.exit(0 if report.num_missed == 0 else 1)


def cmd_baseline_save(args):
    """Execute 'minicluster baseline save' command."""
    try:
        from minicluster.deps import load_json_file

        metrics = load_json_file(args.metrics)
        detector = RegressionDetector(baseline_dir=args.baseline_dir)

        detector.save_baseline(args.name, metrics)
        print(f"[OK] Baseline '{args.name}' saved successfully")

    except FileNotFoundError as e:
        print(f"[ERROR] {str(e)}", file=sys.stderr)
        sys.exit(1)


def cmd_baseline_compare(args):
    """Execute 'minicluster baseline compare' command."""
    try:
        from minicluster.deps import load_json_file

        current_metrics = load_json_file(args.metrics)
        detector = RegressionDetector(baseline_dir=args.baseline_dir)

        baseline_metrics = detector.load_baseline(args.name)

        thresholds = RegressionThreshold(
            latency_percent=args.latency_threshold,
            throughput_percent=args.throughput_threshold,
        )

        comparisons = detector.compare_metrics(baseline_metrics, current_metrics, thresholds)

        # Print report
        print("\n" + "=" * 80)
        print("BASELINE COMPARISON REPORT")
        print("=" * 80)
        print(f"Baseline:                    {args.name}")
        print(f"Latency threshold:           {args.latency_threshold}%")
        print(f"Throughput threshold:        {args.throughput_threshold}%")

        print("\n" + "-" * 80)
        print("Metric Comparisons:")
        print("-" * 80)

        regressions_found = False
        for metric_name, comparison in comparisons.items():
            status = "OK" if not comparison.is_regression else "REGRESS"
            if comparison.is_regression:
                regressions_found = True

            print(
                f"{metric_name:30} {comparison.baseline_value:12.6f} "
                f"â†’ {comparison.current_value:12.6f} ({comparison.percent_change:+7.2f}%) {status}"
            )

        print("=" * 80 + "\n")

        if args.output:
            report_data = {
                "baseline": args.name,
                "thresholds": {
                    "latency_percent": args.latency_threshold,
                    "throughput_percent": args.throughput_threshold,
                },
                "comparisons": {
                    k: {
                        "baseline_value": v.baseline_value,
                        "current_value": v.current_value,
                        "percent_change": v.percent_change,
                        "is_regression": v.is_regression,
                    }
                    for k, v in comparisons.items()
                },
            }
            from minicluster.deps import save_json_file

            save_json_file(args.output, report_data)
            print(f"Report saved to: {args.output}")

        sys.exit(1 if regressions_found else 0)

    except FileNotFoundError as e:
        print(f"[ERROR] {str(e)}", file=sys.stderr)
        sys.exit(1)


def cmd_report_pdf(args):
    """Execute `minicluster report pdf` command."""
    try:
        result = load_trackiq_result(args.result)
    except FileNotFoundError:
        print(f"[ERROR] Result file not found: {args.result}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:  # pragma: no cover - defensive parse guard
        print(f"[ERROR] Invalid TrackiqResult input: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        html = render_trackiq_result_html(result, title=args.title)
        outcome = render_pdf_from_html(
            html_content=html,
            output_path=args.output,
            backend=args.pdf_backend,
            title=args.title,
            author="minicluster",
        )
    except PdfBackendError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(2)

    if outcome.used_fallback:
        print("[WARN] Primary PDF backend unavailable; used matplotlib fallback.")
    print(f"[OK] PDF report generated: {outcome.output_path}")


def cmd_report_html(args):
    """Execute `minicluster report html` command."""
    results = []
    for path in args.result:
        try:
            result = load_trackiq_result(path)
        except FileNotFoundError:
            print(f"[ERROR] Result file not found: {path}", file=sys.stderr)
            sys.exit(2)
        except Exception as e:  # pragma: no cover - defensive parse guard
            print(f"[ERROR] Invalid TrackiqResult input ({path}): {e}", file=sys.stderr)
            sys.exit(2)
        results.append(result)

    try:
        output_path = MiniClusterHtmlReporter().generate(
            output_path=args.output,
            results=results,
            title=args.title,
        )
    except Exception as e:
        print(f"[ERROR] Failed to generate HTML report: {e}", file=sys.stderr)
        sys.exit(2)

    mode = "consolidated" if len(results) > 1 else "single-run"
    print(f"[OK] HTML report generated ({mode}): {output_path}")


def cmd_report_heatmap(args):
    """Execute `minicluster report heatmap` command."""
    try:
        rows = load_worker_results_from_dir(args.results_dir, args.metric)
        generate_cluster_heatmap(rows, metric=args.metric, output_path=args.output)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(2)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:  # pragma: no cover - defensive runtime guard
        print(f"[ERROR] Failed to generate heatmap report: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"[OK] Heatmap report generated: {args.output}")


def cmd_report_fault_timeline(args):
    """Execute `minicluster report fault-timeline` command."""
    try:
        with open(args.json, encoding="utf-8") as handle:
            report = json.load(handle)
        if not isinstance(report, dict):
            raise ValueError("Fault report JSON must be an object.")
        generate_fault_timeline(report, output_path=args.output)
    except FileNotFoundError:
        print(f"[ERROR] Fault report file not found: {args.json}", file=sys.stderr)
        sys.exit(2)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:  # pragma: no cover - defensive runtime guard
        print(f"[ERROR] Failed to generate fault timeline report: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"[OK] Fault timeline report generated: {args.output}")


def setup_main_parser() -> argparse.ArgumentParser:
    """Setup main argument parser with all subcommands.

    Returns:
        Configured ArgumentParser for minicluster CLI
    """
    parser = argparse.ArgumentParser(
        prog="minicluster",
        description="MiniCluster - Local distributed training validation tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run distributed training with 2 workers for 100 steps
  minicluster run --workers 2 --steps 100 --output run.json

  # Validate correctness between single and multi-process runs
  minicluster validate single_metrics.json multi_metrics.json --tolerance 0.01

  # Run fault injection tests
  minicluster fault-test --steps 50

  # Run communication-only all-reduce benchmark
  minicluster bench-collective --workers 2 --size-mb 256 --iterations 50

  # Save baseline after stable run
  minicluster baseline save --metrics run.json --name stable_v1

  # Compare current run against baseline
  minicluster baseline compare --metrics current.json --name stable_v1

  # Generate PDF report from canonical result
  minicluster report pdf --result ./minicluster_results/run_metrics.json --output ./minicluster_results/report.pdf

  # Generate consolidated HTML report from multiple configs
  minicluster report html --result run_cfg_a.json run_cfg_b.json --output ./minicluster_results/report.html

  # Generate worker heatmap from per-worker JSON files
  minicluster report heatmap --results-dir ./minicluster_results/workers --metric p99_allreduce_ms --output ./minicluster_results/heatmap.html

  # Generate fault timeline HTML report
  minicluster report fault-timeline --json ./minicluster_results/fault_report.json --output ./minicluster_results/fault_timeline.html
        """,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"minicluster {MINICLUSTER_CLI_VERSION}",
        help="Show CLI version and exit",
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")

    setup_run_parser(subparsers)
    setup_validate_parser(subparsers)
    setup_fault_test_parser(subparsers)
    setup_bench_collective_parser(subparsers)
    setup_baseline_parser(subparsers)
    setup_report_parser(subparsers)
    register_monitor_subcommand(subparsers)

    return parser


def main():
    """Main entry point for minicluster CLI."""
    parser = setup_main_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "baseline":
        if not args.baseline_cmd:
            _print_subcommand_help(parser, args.command)
            sys.exit(1)
    if args.command == "report":
        if not getattr(args, "report_cmd", None):
            _print_subcommand_help(parser, args.command)
            sys.exit(1)
    if args.command == "monitor":
        if not getattr(args, "monitor_cmd", None):
            _print_subcommand_help(parser, args.command)
            sys.exit(1)

    # Execute the selected command
    args.func(args)


if __name__ == "__main__":
    main()

