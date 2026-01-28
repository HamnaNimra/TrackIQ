#!/usr/bin/env python
"""Example: Performance Regression Detection with AutoPerfPy.

This example demonstrates how to:
1. Save performance baselines
2. Detect regressions against baselines
3. Generate regression reports
"""

from autoperfpy import RegressionDetector, RegressionThreshold


def main():
    """Run regression detection example."""
    
    print("=" * 70)
    print("AutoPerfPy Regression Detection Example")
    print("=" * 70)
    
    # Initialize regression detector
    detector = RegressionDetector(baseline_dir=".autoperfpy/baselines")
    
    # Define baseline metrics (e.g., from a successful release)
    baseline_metrics = {
        "p99_latency": 50.0,
        "p95_latency": 45.0,
        "p50_latency": 30.0,
        "mean_latency": 32.5,
        "throughput_imgs_per_sec": 1000.0,
    }
    
    print("\n1️⃣  Saving baseline metrics for 'main' branch...")
    detector.save_baseline("main", baseline_metrics)
    print("✅ Baseline saved!")
    
    # Simulate current metrics from a development branch
    current_metrics_good = {
        "p99_latency": 52.0,  # +4% increase (within 10% threshold)
        "p95_latency": 46.0,  # +2.2% increase
        "p50_latency": 31.0,  # +3.3% increase
        "mean_latency": 33.5,  # +3% increase
        "throughput_imgs_per_sec": 980.0,  # -2% decrease
    }
    
    print("\n2️⃣  Scenario A: Checking good metrics (minor changes)...")
    result_good = detector.detect_regressions(
        baseline_name="main",
        current_metrics=current_metrics_good,
        thresholds=RegressionThreshold(
            latency_percent=5.0,
            throughput_percent=5.0,
            p99_percent=10.0,
        )
    )
    
    print(f"   Regressions detected: {result_good['has_regressions']}")
    print(f"   Number of regressions: {len(result_good['regressions'])}")
    
    # Simulate worse metrics that exceed thresholds
    current_metrics_bad = {
        "p99_latency": 58.0,  # +16% increase (exceeds 10% threshold) ❌
        "p95_latency": 52.0,  # +15.6% increase (exceeds 5% threshold) ❌
        "p50_latency": 35.0,  # +16.7% increase (exceeds 5% threshold) ❌
        "mean_latency": 38.0,  # +16.9% increase (exceeds 5% threshold) ❌
        "throughput_imgs_per_sec": 850.0,  # -15% decrease (exceeds 5% threshold) ❌
    }
    
    print("\n3️⃣  Scenario B: Checking bad metrics (significant degradation)...")
    result_bad = detector.detect_regressions(
        baseline_name="main",
        current_metrics=current_metrics_bad,
        thresholds=RegressionThreshold(
            latency_percent=5.0,
            throughput_percent=5.0,
            p99_percent=10.0,
        )
    )
    
    print(f"   Regressions detected: {result_bad['has_regressions']}")
    print(f"   Number of regressions: {len(result_bad['regressions'])}")
    
    # Generate and display report
    print("\n4️⃣  Generating detailed regression report...")
    print()
    report = detector.generate_report(
        baseline_name="main",
        current_metrics=current_metrics_bad,
        thresholds=RegressionThreshold(
            latency_percent=5.0,
            throughput_percent=5.0,
            p99_percent=10.0,
        )
    )
    print(report)
    
    # Show how to access individual comparisons
    print("\n5️⃣  Accessing individual metric comparisons...")
    print()
    
    comparisons = result_bad["comparisons"]
    for metric_name, comp in comparisons.items():
        status = "🔴 REGRESSION" if comp["is_regression"] else "🟢 OK"
        print(f"   {metric_name:30} {status}")
        print(f"      Baseline: {comp['baseline_value']:10.2f}")
        print(f"      Current:  {comp['current_value']:10.2f}")
        print(f"      Change:   {comp['percent_change']:+.2f}% (threshold: {comp['threshold']:.1f}%)")
        print()
    
    # Example: List all available baselines
    print("\n6️⃣  Available baselines:")
    baselines = detector.list_baselines()
    for baseline_name in baselines:
        print(f"   - {baseline_name}")
    
    print("\n" + "=" * 70)
    print("✅ Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
