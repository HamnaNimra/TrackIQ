"""Command-line interface for trackiq_compare."""

import argparse
import os
import sys
import tempfile
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

from trackiq_compare.comparator import MetricComparator, SummaryGenerator
from trackiq_compare.deps import (
    PDF_BACKENDS,
    PdfBackendError,
    RegressionDetector,
    RegressionThreshold,
    load_trackiq_result,
    render_pdf_from_html_file,
)
from trackiq_compare.reporters import HtmlReporter, TerminalReporter

try:
    TRACKIQ_COMPARE_CLI_VERSION = package_version("trackiq_compare")
except PackageNotFoundError:
    TRACKIQ_COMPARE_CLI_VERSION = "0.2.0"


def _print_subcommand_help(parser: argparse.ArgumentParser, command: str) -> None:
    """Print help for a specific top-level subcommand."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            subparser = action.choices.get(command)
            if subparser is not None:
                subparser.print_help()
                return
    parser.print_help()


def _metrics_for_baseline(result) -> dict[str, float]:
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

    # Make label semantics explicit: labels are display aliases, not hardware detection.
    if args.label_a:
        print(f"[INFO] label-a is display-only. Actual Result A platform: " f"{result_a.platform.hardware_name}")
    if args.label_b:
        print(f"[INFO] label-b is display-only. Actual Result B platform: " f"{result_b.platform.hardware_name}")

    comparator = MetricComparator(
        label_a=label_a,
        label_b=label_b,
        variance_threshold_percent=float(getattr(args, "variance_threshold", 25.0)),
    )
    comparison = comparator.compare(result_a, result_b)

    summary = SummaryGenerator(regression_threshold_percent=args.regression_threshold).generate(comparison)

    if result_a.workload.workload_type != result_b.workload.workload_type:
        workload_warning = (
            "Workload types differ "
            f"({result_a.workload.workload_type} vs {result_b.workload.workload_type}). "
            "Interpret metric winners cautiously for cross-workload comparisons."
        )
        print(f"[WARN] {workload_warning}")
        summary.text = f"{workload_warning} {summary.text}"

    TerminalReporter(tolerance_percent=args.tolerance).render(comparison, summary)

    if args.html:
        path = HtmlReporter().generate(args.html, comparison, summary, result_a, result_b)
        print(f"\n[OK] HTML report written to: {path}")
    return 0


def run_report_pdf(args: argparse.Namespace) -> int:
    """Execute `trackiq-compare report pdf`."""
    result_a = load_trackiq_result(args.result_a)
    result_b = load_trackiq_result(args.result_b)
    label_a = args.label_a or result_a.platform.hardware_name or "Result A"
    label_b = args.label_b or result_b.platform.hardware_name or "Result B"

    comparator = MetricComparator(
        label_a=label_a,
        label_b=label_b,
        variance_threshold_percent=float(getattr(args, "variance_threshold", 25.0)),
    )
    comparison = comparator.compare(result_a, result_b)
    summary = SummaryGenerator(regression_threshold_percent=args.regression_threshold).generate(comparison)

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as handle:
        html_path = handle.name
    try:
        HtmlReporter().generate(html_path, comparison, summary, result_a, result_b)
        outcome = render_pdf_from_html_file(
            html_path=html_path,
            output_path=args.output,
            backend=args.pdf_backend,
            title="TrackIQ Comparison Report",
            author="trackiq-compare",
        )
    except PdfBackendError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    finally:
        try:
            os.unlink(html_path)
        except OSError:
            pass

    if outcome.used_fallback:
        print("[WARN] Primary PDF backend unavailable; used matplotlib fallback.")
    print(f"[OK] PDF report written to: {outcome.output_path}")
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
    parser.add_argument(
        "--version",
        action="version",
        version=f"trackiq-compare {TRACKIQ_COMPARE_CLI_VERSION}",
        help="Show CLI version and exit",
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
    run.add_argument(
        "--variance-threshold",
        type=float,
        default=25.0,
        help="All-reduce variance regression threshold (%% increase, default: 25.0)",
    )
    run.set_defaults(func=run_compare)

    report = sub.add_parser("report", help="Generate reports from compare inputs")
    report_sub = report.add_subparsers(dest="report_type")

    report_pdf = report_sub.add_parser(
        "pdf",
        help="Generate PDF report from two TrackiqResult files",
    )
    report_pdf.add_argument("result_a")
    report_pdf.add_argument("result_b")
    report_pdf.add_argument(
        "--output",
        "-o",
        default="output/trackiq_compare_report.pdf",
        help="Output PDF report path",
    )
    report_pdf.add_argument("--label-a", help="Display label for result A")
    report_pdf.add_argument("--label-b", help="Display label for result B")
    report_pdf.add_argument(
        "--regression-threshold",
        type=float,
        default=5.0,
        help="Regression threshold (%%) for summary flagging",
    )
    report_pdf.add_argument(
        "--variance-threshold",
        type=float,
        default=25.0,
        help="All-reduce variance regression threshold (%% increase, default: 25.0)",
    )
    report_pdf.add_argument(
        "--pdf-backend",
        choices=list(PDF_BACKENDS),
        default="auto",
        help=("PDF backend strategy (default: auto). " "auto uses weasyprint primary with matplotlib fallback."),
    )
    report_pdf.set_defaults(func=run_report_pdf)

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
    if args.command == "report" and not getattr(args, "report_type", None):
        _print_subcommand_help(parser, "report")
        return 1
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
