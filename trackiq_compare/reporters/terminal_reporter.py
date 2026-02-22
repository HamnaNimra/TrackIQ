"""Terminal reporter using rich output."""

import importlib
from typing import Any

from trackiq_compare.comparator.metric_comparator import ComparisonResult, MetricComparison
from trackiq_compare.comparator.summary_generator import SummaryResult


def _load_optional_module(module_name: str) -> Any:
    """Import optional dependency module, returning None if unavailable."""
    try:
        return importlib.import_module(module_name)
    except Exception:  # pragma: no cover - fallback path only when optional deps are absent
        return None


rich_console: Any = _load_optional_module("rich.console")
rich_table: Any = _load_optional_module("rich.table")


class TerminalReporter:
    """Render comparison output as a terminal table and summary."""

    def __init__(self, tolerance_percent: float = 0.5, console: Any | None = None):
        self.tolerance_percent = tolerance_percent
        self.console = console or (rich_console.Console() if rich_console is not None else None)

    def render(self, comparison: ComparisonResult, summary: SummaryResult) -> None:
        """Print comparison table and plain-English summary."""
        if rich_table is None or self.console is None:
            self._render_plain(comparison, summary)
            return

        table = rich_table.Table(title="TrackIQ Metric Comparison")
        table.add_column("Metric", style="bold")
        table.add_column(comparison.label_a)
        table.add_column(comparison.label_b)
        table.add_column("Abs Delta")
        table.add_column("% Delta")
        table.add_column("Winner")

        for metric in comparison.metrics.values():
            table.add_row(
                metric.metric_name,
                self._fmt_value(metric.value_a),
                self._fmt_value(metric.value_b),
                self._fmt_value(metric.abs_delta),
                self._fmt_percent(metric.percent_delta),
                self._winner_text(metric, comparison),
            )

        self.console.print(table)
        self.console.print("")
        self.console.print("[bold]Summary[/bold]")
        self.console.print(summary.text)
        if comparison.consistency_findings:
            self.console.print("")
            self.console.print("[bold]Consistency Analysis[/bold]")
            for finding in comparison.consistency_findings:
                self.console.print(
                    f"[yellow]{finding.code}[/yellow]: {finding.label} | "
                    f"stdev A={finding.stddev_a_ms:.4f} ms, stdev B={finding.stddev_b_ms:.4f} ms, "
                    f"increase={finding.increase_percent:+.2f}% (threshold {finding.threshold_percent:.2f}%)"
                )

    def _render_plain(self, comparison: ComparisonResult, summary: SummaryResult) -> None:
        """Fallback renderer when rich is unavailable."""
        print("TrackIQ Metric Comparison")
        print("=" * 80)
        print(
            f"{'Metric':30} {comparison.label_a:12} {comparison.label_b:12} " f"{'Abs Delta':10} {'% Delta':10} Winner"
        )
        for metric in comparison.metrics.values():
            print(
                f"{metric.metric_name:30} "
                f"{self._fmt_value(metric.value_a):12} "
                f"{self._fmt_value(metric.value_b):12} "
                f"{self._fmt_value(metric.abs_delta):10} "
                f"{self._fmt_percent(metric.percent_delta):10} "
                f"{metric.winner}"
            )
        print("\nSummary")
        print(summary.text)
        if comparison.consistency_findings:
            print("\nConsistency Analysis")
            for finding in comparison.consistency_findings:
                print(
                    f"{finding.code}: {finding.label} | "
                    f"stdev A={finding.stddev_a_ms:.4f} ms, stdev B={finding.stddev_b_ms:.4f} ms, "
                    f"increase={finding.increase_percent:+.2f}% (threshold {finding.threshold_percent:.2f}%)"
                )

    @staticmethod
    def _fmt_value(value: float | None) -> str:
        if value is None:
            return "N/A"
        return f"{value:.4f}"

    @staticmethod
    def _fmt_percent(value: float | None) -> str:
        if value is None:
            return "N/A"
        if value == float("inf"):
            return "inf"
        return f"{value:+.2f}%"

    def _winner_text(self, metric: MetricComparison, comparison: ComparisonResult) -> str:
        if not metric.comparable:
            return "[dim]N/A[/dim]"
        if metric.percent_delta is not None and abs(metric.percent_delta) <= self.tolerance_percent:
            return "[yellow]~ tie[/yellow]"
        if metric.winner == comparison.label_b:
            return f"[green]{metric.winner}[/green]"
        if metric.winner == comparison.label_a:
            return f"[red]{metric.winner}[/red]"
        return "[yellow]tie[/yellow]"
