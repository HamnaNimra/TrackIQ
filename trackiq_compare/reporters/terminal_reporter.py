"""Terminal reporter using rich output."""

from rich.console import Console
from rich.table import Table

from trackiq_compare.comparator.metric_comparator import ComparisonResult, MetricComparison
from trackiq_compare.comparator.summary_generator import SummaryResult


class TerminalReporter:
    """Render comparison output as a terminal table and summary."""

    def __init__(self, tolerance_percent: float = 0.5, console: Console | None = None):
        self.tolerance_percent = tolerance_percent
        self.console = console or Console()

    def render(self, comparison: ComparisonResult, summary: SummaryResult) -> None:
        """Print comparison table and plain-English summary."""
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

