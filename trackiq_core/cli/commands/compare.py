"""Compare command for trackiq_core CLI."""

import json
import sys

from trackiq_core.utils.compare import RegressionDetector, RegressionThreshold


def flatten_summary_for_compare(summary: dict) -> dict:
    """Flatten export summary to metric_name -> float for trackiq compare."""
    flat = {}
    for section, data in summary.items():
        if section in ("sample_count", "warmup_samples", "duration_seconds"):
            continue
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (int, float)):
                    flat[f"{section}_{k}"] = float(v)
    return flat


def run_compare(args):
    """Compare current run against baseline using trackiq comparison module."""
    baseline_dir = getattr(args, "baseline_dir", ".trackiq/baselines") or ".trackiq/baselines"
    detector = RegressionDetector(baseline_dir=baseline_dir)
    thresholds = RegressionThreshold(
        latency_percent=getattr(args, "latency_pct", 5.0),
        throughput_percent=getattr(args, "throughput_pct", 5.0),
        p99_percent=getattr(args, "p99_pct", 10.0),
    )

    with open(args.current, encoding="utf-8") as f:
        current_data = json.load(f)
    summary = current_data.get("summary", current_data.get("metrics", current_data))
    current_metrics = flatten_summary_for_compare(summary) if isinstance(summary, dict) else {}

    if getattr(args, "save_baseline", False):
        detector.save_baseline(args.baseline, current_metrics)
        print(f"Baseline '{args.baseline}' saved from {args.current}")
        return 0

    try:
        detector.load_baseline(args.baseline)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(
            "Save a baseline first: compare --save-baseline --baseline NAME --current run.json",
            file=sys.stderr,
        )
        return 1

    report = detector.generate_report(args.baseline, current_metrics, thresholds)
    print(report)
    result = detector.detect_regressions(args.baseline, current_metrics, thresholds)
    return 1 if result.get("has_regressions") else 0
