"""Plain-English summary generator for metric comparisons."""

from dataclasses import dataclass, field

from .metric_comparator import ComparisonResult, ConsistencyFinding, MetricComparison


@dataclass
class SummaryResult:
    """Summary output for a comparison run."""

    overall_winner: str
    largest_deltas: list[MetricComparison] = field(default_factory=list)
    flagged_regressions: list[MetricComparison] = field(default_factory=list)
    consistency_findings: list[ConsistencyFinding] = field(default_factory=list)
    text: str = ""


class SummaryGenerator:
    """Generate plain-English summaries from comparison output."""

    def __init__(self, regression_threshold_percent: float = 5.0):
        self.regression_threshold_percent = regression_threshold_percent

    def generate(self, comparison: ComparisonResult) -> SummaryResult:
        """Build summary including winner, largest deltas, and regressions."""
        comparable = comparison.comparable_metrics
        if not comparable:
            return SummaryResult(
                overall_winner="none",
                text="No comparable metrics were found between the two results.",
            )

        wins_a = sum(1 for item in comparable if item.winner == comparison.label_a)
        wins_b = sum(1 for item in comparable if item.winner == comparison.label_b)
        if wins_b > wins_a:
            winner = comparison.label_b
        elif wins_a > wins_b:
            winner = comparison.label_a
        else:
            winner = "tie"

        largest = sorted(
            comparable,
            key=lambda item: abs(item.percent_delta or 0.0),
            reverse=True,
        )[:3]

        regressions = [
            item
            for item in comparable
            if item.winner == comparison.label_a
            and item.percent_delta is not None
            and abs(item.percent_delta) > self.regression_threshold_percent
        ]

        winner_reason = (
            f"{winner} won {max(wins_a, wins_b)} of {len(comparable)} comparable metrics."
            if winner != "tie"
            else f"No overall winner: both results won {wins_a} metrics."
        )
        largest_text = ", ".join(f"{item.metric_name} ({item.percent_delta:+.2f}%)" for item in largest)
        regression_text = (
            f"{len(regressions)} regression(s) exceeded {self.regression_threshold_percent:.1f}%."
            if regressions
            else f"No regressions exceeded {self.regression_threshold_percent:.1f}%."
        )
        consistency = comparison.consistency_findings
        consistency_text = (
            f"Consistency findings: {len(consistency)}." if consistency else "Consistency findings: none."
        )

        text = (
            f"Overall winner: {winner}. {winner_reason} "
            f"Largest deltas: {largest_text}. {regression_text} {consistency_text}"
        )

        return SummaryResult(
            overall_winner=winner,
            largest_deltas=largest,
            flagged_regressions=regressions,
            consistency_findings=consistency,
            text=text,
        )
