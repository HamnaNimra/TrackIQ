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
import tempfile
from pathlib import Path
from typing import Optional

from minicluster.runner import RunConfig, run_distributed, save_metrics, load_metrics
from minicluster.validators import CorrectnessValidator, FaultInjector
from minicluster.deps import RegressionDetector, RegressionThreshold


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
        "--output",
        type=str,
        default="./minicluster_results/run_metrics.json",
        help="Output path for metrics JSON (default: ./minicluster_results/run_metrics.json)",
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


def cmd_run(args):
    """Execute 'minicluster run' command."""
    config = RunConfig(
        num_steps=args.steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_workers=args.workers,
        seed=args.seed,
    )

    if args.verbose:
        print(f"Running minicluster with config:")
        print(f"  Workers: {config.num_workers}")
        print(f"  Steps: {config.num_steps}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Learning rate: {config.learning_rate}")

    print(f"\nStarting distributed training run...")
    metrics = run_distributed(config)

    save_metrics(metrics, args.output)
    print(f"\n✓ Run complete!")
    print(f"  Total time: {metrics.total_time_sec:.2f}s")
    print(f"  Final loss: {metrics.steps[-1].loss:.6f}")
    print(f"  Avg throughput: {metrics.to_dict()['average_throughput_samples_per_sec']:.1f} samples/sec")
    print(f"  Metrics saved to: {args.output}")


def cmd_validate(args):
    """Execute 'minicluster validate' command."""
    try:
        validator = CorrectnessValidator(tolerance=args.tolerance)
        report = validator.validate_file_pair(
            args.single_run, args.multi_run, output_path=args.output
        )

        validator.print_report(report, verbose=args.verbose)

        if args.output:
            print(f"Report saved to: {args.output}")

        sys.exit(0 if report.overall_passed else 1)

    except FileNotFoundError as e:
        print(f"✗ Error: {str(e)}", file=sys.stderr)
        sys.exit(2)
    except ValueError as e:
        print(f"✗ Validation error: {str(e)}", file=sys.stderr)
        sys.exit(1)


def cmd_fault_test(args):
    """Execute 'minicluster fault-test' command."""
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
        print(f"✓ Baseline '{args.name}' saved successfully")

    except FileNotFoundError as e:
        print(f"✗ Error: {str(e)}", file=sys.stderr)
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
            status = "✓" if not comparison.is_regression else "✗ REGRESS"
            if comparison.is_regression:
                regressions_found = True

            print(
                f"{metric_name:30} {comparison.baseline_value:12.6f} "
                f"→ {comparison.current_value:12.6f} ({comparison.percent_change:+7.2f}%) {status}"
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
        print(f"✗ Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


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

  # Save baseline after stable run
  minicluster baseline save --metrics run.json --name stable_v1

  # Compare current run against baseline
  minicluster baseline compare --metrics current.json --name stable_v1
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")

    setup_run_parser(subparsers)
    setup_validate_parser(subparsers)
    setup_fault_test_parser(subparsers)
    setup_baseline_parser(subparsers)

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
            parser.parse_args([args.command, "-h"])
            sys.exit(1)

    # Execute the selected command
    args.func(args)


if __name__ == "__main__":
    main()
