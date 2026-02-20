"""Correctness validator for distributed training runs.

Compares loss values from single-process and multi-process runs to verify
that distributed training produces consistent results. Outputs structured
reports for easy comparison.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from minicluster.deps import save_json_file, ensure_parent_dir


@dataclass
class StepComparison:
    """Comparison result for a single step."""

    step: int
    single_loss: float
    multi_loss: float
    delta: float
    delta_percent: float
    passed: bool
    tolerance: float


@dataclass
class CorrectnessReport:
    """Report of correctness validation between runs."""

    single_run_path: str
    multi_run_path: str
    tolerance: float
    num_steps_compared: int
    num_steps_passed: int
    num_steps_failed: int
    step_comparisons: List[StepComparison] = field(default_factory=list)
    overall_passed: bool = False
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "single_run_path": self.single_run_path,
            "multi_run_path": self.multi_run_path,
            "tolerance": self.tolerance,
            "num_steps_compared": self.num_steps_compared,
            "num_steps_passed": self.num_steps_passed,
            "num_steps_failed": self.num_steps_failed,
            "overall_passed": self.overall_passed,
            "summary": self.summary,
            "step_comparisons": [asdict(s) for s in self.step_comparisons],
        }


class CorrectnessValidator:
    """Validates correctness by comparing single-process and multi-process runs."""

    def __init__(self, tolerance: float = 0.01):
        """Initialize validator with tolerance.

        Args:
            tolerance: Relative tolerance for loss comparison (e.g., 0.01 for 1%)
        """
        self.tolerance = tolerance

    def compare_runs(
        self, single_metrics: Dict[str, Any], multi_metrics: Dict[str, Any]
    ) -> CorrectnessReport:
        """Compare single-process and multi-process run metrics.

        Args:
            single_metrics: Metrics dict from single-process run
            multi_metrics: Metrics dict from multi-process run

        Returns:
            CorrectnessReport with detailed step-by-step comparison

        Raises:
            ValueError: If runs have different number of steps or invalid structure
        """
        single_steps = single_metrics.get("steps", [])
        multi_steps = multi_metrics.get("steps", [])

        if len(single_steps) != len(multi_steps):
            raise ValueError(
                f"Step count mismatch: single={len(single_steps)}, multi={len(multi_steps)}"
            )

        if not single_steps:
            raise ValueError("No steps found in metrics")

        report = CorrectnessReport(
            single_run_path="single_process",
            multi_run_path="multi_process",
            tolerance=self.tolerance,
            num_steps_compared=len(single_steps),
        )

        for single_step, multi_step in zip(single_steps, multi_steps):
            single_loss = single_step["loss"]
            multi_loss = multi_step["loss"]

            # Calculate absolute and relative delta
            delta = abs(single_loss - multi_loss)
            delta_percent = (delta / abs(single_loss)) * 100 if single_loss != 0 else 0

            # Check if within tolerance (as percentage)
            passed = delta_percent <= (self.tolerance * 100)

            comparison = StepComparison(
                step=single_step["step"],
                single_loss=single_loss,
                multi_loss=multi_loss,
                delta=delta,
                delta_percent=delta_percent,
                passed=passed,
                tolerance=self.tolerance * 100,
            )

            report.step_comparisons.append(comparison)

            if passed:
                report.num_steps_passed += 1
            else:
                report.num_steps_failed += 1

        # Overall result
        report.overall_passed = report.num_steps_failed == 0

        # Generate summary
        if report.overall_passed:
            report.summary = (
                f"✓ PASSED: All {report.num_steps_compared} steps passed "
                f"correctness check within {self.tolerance*100:.2f}% tolerance"
            )
        else:
            report.summary = (
                f"✗ FAILED: {report.num_steps_failed}/{report.num_steps_compared} "
                f"steps exceeded {self.tolerance*100:.2f}% tolerance. "
                f"{report.num_steps_passed} steps passed."
            )

        return report

    def print_report(self, report: CorrectnessReport, verbose: bool = False) -> None:
        """Print human-readable validation report.

        Args:
            report: CorrectnessReport to print
            verbose: If True, print all steps; otherwise show summary and failures
        """
        print("\n" + "=" * 80)
        print("CORRECTNESS VALIDATION REPORT")
        print("=" * 80)

        print(f"Single-process run: {report.single_run_path}")
        print(f"Multi-process run:  {report.multi_run_path}")
        print(f"Tolerance:          {report.tolerance*100:.2f}%")
        print(f"Steps compared:     {report.num_steps_compared}")
        print(f"  Passed:           {report.num_steps_passed}")
        print(f"  Failed:           {report.num_steps_failed}")

        print("\n" + "-" * 80)
        print(f"RESULT: {report.summary}")
        print("-" * 80)

        if verbose or report.num_steps_failed > 0:
            print("\nStep Comparison Details:")
            print(f"{'Step':>5} {'Single Loss':>15} {'Multi Loss':>15} {'Delta':>12} {'Δ %':>8} {'Status':>8}")
            print("-" * 80)

            for cmp in report.step_comparisons:
                status = "✓ PASS" if cmp.passed else "✗ FAIL"
                if verbose or not cmp.passed:
                    print(
                        f"{cmp.step:5d} {cmp.single_loss:15.6f} {cmp.multi_loss:15.6f} "
                        f"{cmp.delta:12.6e} {cmp.delta_percent:8.3f} {status:>8}"
                    )

        print("=" * 80 + "\n")

    def save_report(self, report: CorrectnessReport, output_path: str) -> None:
        """Save validation report to JSON file.

        Args:
            report: CorrectnessReport to save
            output_path: Path to output JSON file
        """
        ensure_parent_dir(output_path)
        save_json_file(output_path, report.to_dict())

    def validate_file_pair(
        self, single_run_path: str, multi_run_path: str, output_path: Optional[str] = None
    ) -> CorrectnessReport:
        """Validate correctness using metrics files.

        Args:
            single_run_path: Path to single-process metrics JSON file
            multi_run_path: Path to multi-process metrics JSON file
            output_path: Optional path to save report

        Returns:
            CorrectnessReport

        Raises:
            FileNotFoundError: If metrics files don't exist
        """
        from minicluster.deps import load_json_file

        single_metrics = load_json_file(single_run_path)
        multi_metrics = load_json_file(multi_run_path)

        report = self.compare_runs(single_metrics, multi_metrics)
        report.single_run_path = str(single_run_path)
        report.multi_run_path = str(multi_run_path)

        if output_path:
            self.save_report(report, output_path)

        return report
