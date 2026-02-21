"""Terminal reporter using rich output."""

from typing import Optional, Any

try:
    from rich.console import Console
    from rich.table import Table
except Exception:  # pragma: no cover - fallback path only when rich is absent
    Console = None
    Table = None

from trackiq_compare.comparator.metric_comparator import ComparisonResult, MetricComparison
from trackiq_compare.comparator.summary_generator import SummaryResult


class TerminalReporter:
    """Render comparison output as a terminal table and summary."""

    def __init__(self, tolerance_percent: float = 0.5, console: Optional[Any] = None):
        self.tolerance_percent = tolerance_percent
        self.console = console or (Console() if Console is not None else None)

    def render(self, comparison: ComparisonResult, summary: SummaryResult) -> None:
        """Print comparison table and plain-English summary."""
        if Table is None or self.console is None:
            self._render_plain(comparison, summary)
            return

        table = Table(title="TrackIQ Metric Comparison")
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

    def _render_plain(self, comparison: ComparisonResult, summary: SummaryResult) -> None:
        """Fallback renderer when rich is unavailable."""
        print("TrackIQ Metric Comparison")
        print("=" * 80)
        print(
            f"{'Metric':30} {comparison.label_a:12} {comparison.label_b:12} "
            f"{'Abs Delta':10} {'% Delta':10} Winner"
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
