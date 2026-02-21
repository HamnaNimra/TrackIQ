"""Command-line interface for trackiq_compare."""

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict

from trackiq_compare.comparator import MetricComparator, SummaryGenerator
from trackiq_compare.deps import (
    RegressionDetector,
    RegressionThreshold,
    load_trackiq_result,
)
from trackiq_compare.reporters import HtmlReporter, TerminalReporter


def _metrics_for_baseline(result) -> Dict[str, float]:
    """Extract numeric metrics suitable for trackiq_core baseline comparison."""
    metrics = {}
    for key, value in asdict(result.metrics).items():
        if value is None:
            continue
        metrics[key] = float(value)
    return metrics


def run_compare(args: argparse.Namespace) -> int:
    """Execute `trackiq-compare run`."""
    result_a = load_trackiq_result(args.result_a)
    result_b = load_trackiq_result(args.result_b)

    label_a = args.label_a or result_a.platform.hardware_name or "Result A"
    label_b = args.label_b or result_b.platform.hardware_name or "Result B"

    comparator = MetricComparator(label_a=label_a, label_b=label_b)
    comparison = comparator.compare(result_a, result_b)

    summary = SummaryGenerator(
        regression_threshold_percent=args.regression_threshold
    ).generate(comparison)

    TerminalReporter(tolerance_percent=args.tolerance).render(comparison, summary)

    if args.html:
        path = HtmlReporter().generate(args.html, comparison, summary, result_a, result_b)
        print(f"\n[OK] HTML report written to: {path}")
    return 0


def save_baseline(args: argparse.Namespace) -> int:
    """Execute `trackiq-compare baseline`."""
    result = load_trackiq_result(args.result)
    baseline_name = args.name or Path(args.result).stem
    detector = RegressionDetector(baseline_dir=args.baseline_dir)
    detector.save_baseline(baseline_name, _metrics_for_baseline(result))
    print(f"[OK] Baseline saved: {baseline_name}")
    return 0


def compare_vs_baseline(args: argparse.Namespace) -> int:
    """Execute `trackiq-compare vs-baseline`."""
    result = load_trackiq_result(args.result)
    detector = RegressionDetector(baseline_dir=args.baseline_dir)
    thresholds = RegressionThreshold(
        latency_percent=args.latency_pct,
        throughput_percent=args.throughput_pct,
        p99_percent=args.p99_pct,
    )
    current_metrics = _metrics_for_baseline(result)
    report = detector.detect_regressions(args.baseline_name, current_metrics, thresholds)
    print(detector.generate_report(args.baseline_name, current_metrics, thresholds))
    return 1 if report.get("has_regressions") else 0


def build_parser() -> argparse.ArgumentParser:
    """Build top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="trackiq-compare",
        description="Compare TrackiqResult outputs across tools/platforms.",
    )
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Compare two TrackiqResult JSON files")
    run.add_argument("result_a")
    run.add_argument("result_b")
    run.add_argument("--html", help="Optional output HTML report path")
    run.add_argument("--label-a", help="Display label for result A")
    run.add_argument("--label-b", help="Display label for result B")
    run.add_argument(
        "--tolerance",
        type=float,
        default=0.5,
        help="Tolerance band (%% delta) treated as negligible",
    )
    run.add_argument(
        "--regression-threshold",
        type=float,
        default=5.0,
        help="Regression threshold (%%) for summary flagging",
    )
    run.set_defaults(func=run_compare)

    baseline = sub.add_parser("baseline", help="Save result metrics as named baseline")
    baseline.add_argument("result")
    baseline.add_argument("--name", help="Baseline name (default: file stem)")
    baseline.add_argument(
        "--baseline-dir",
        default=".trackiq/baselines",
        help="Baseline directory",
    )
    baseline.set_defaults(func=save_baseline)

    vs = sub.add_parser("vs-baseline", help="Compare result against saved baseline")
    vs.add_argument("result")
    vs.add_argument("baseline_name")
    vs.add_argument(
        "--baseline-dir",
        default=".trackiq/baselines",
        help="Baseline directory",
    )
    vs.add_argument("--latency-pct", type=float, default=5.0)
    vs.add_argument("--throughput-pct", type=float, default=5.0)
    vs.add_argument("--p99-pct", type=float, default=10.0)
    vs.set_defaults(func=compare_vs_baseline)

    return parser


def main(argv=None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

