"""HTML reporter for TrackIQ comparisons."""

from datetime import datetime
from html import escape
from typing import List

from trackiq_compare.comparator.metric_comparator import ComparisonResult
from trackiq_compare.comparator.summary_generator import SummaryResult
from trackiq_compare.deps import TrackiqResult, ensure_parent_dir


class HtmlReporter:
    """Generate a self-contained HTML comparison report."""

    def generate(
        self,
        output_path: str,
        comparison: ComparisonResult,
        summary: SummaryResult,
        result_a: TrackiqResult,
        result_b: TrackiqResult,
    ) -> str:
        """Write HTML report to disk and return output path."""
        ensure_parent_dir(output_path)
        rows = self._metric_rows(comparison)
        platform_diff = self._platform_comparison(result_a, result_b)
        generated = datetime.utcnow().isoformat()

        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TrackIQ Compare Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .meta {{ margin-bottom: 16px; }}
    .card {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 12px; margin-bottom: 16px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    .win {{ color: #065f46; font-weight: 600; }}
    .loss {{ color: #991b1b; font-weight: 600; }}
    .warn {{ color: #92400e; font-weight: 600; }}
  </style>
</head>
<body>
  <h1>TrackIQ Comparison Report</h1>
  <p>Generated at {escape(generated)}</p>

  <div class="card">
    <h2>Summary</h2>
    <p>{escape(summary.text)}</p>
  </div>

  <div class="card">
    <h2>Result Metadata</h2>
    <div class="meta">
      <strong>{escape(comparison.label_a)}</strong>: tool={escape(result_a.tool_name)} {escape(result_a.tool_version)},
      platform={escape(result_a.platform.hardware_name)},
      framework={escape(result_a.platform.framework)} {escape(result_a.platform.framework_version)},
      workload={escape(result_a.workload.name)} ({escape(result_a.workload.workload_type)}),
      timestamp={escape(result_a.timestamp.isoformat())}
    </div>
    <div class="meta">
      <strong>{escape(comparison.label_b)}</strong>: tool={escape(result_b.tool_name)} {escape(result_b.tool_version)},
      platform={escape(result_b.platform.hardware_name)},
      framework={escape(result_b.platform.framework)} {escape(result_b.platform.framework_version)},
      workload={escape(result_b.workload.name)} ({escape(result_b.workload.workload_type)}),
      timestamp={escape(result_b.timestamp.isoformat())}
    </div>
  </div>

  <div class="card">
    <h2>Metric Comparison</h2>
    <table>
      <thead>
        <tr>
          <th>Metric</th>
          <th>{escape(comparison.label_a)}</th>
          <th>{escape(comparison.label_b)}</th>
          <th>Abs Delta</th>
          <th>% Delta</th>
          <th>Winner</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>Platform Comparison</h2>
    {platform_diff}
  </div>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(html)
        return output_path

    @staticmethod
    def _fmt(value: float | None, is_percent: bool = False) -> str:
        if value is None:
            return "N/A"
        if value == float("inf"):
            return "inf"
        return f"{value:+.2f}%" if is_percent else f"{value:.4f}"

    def _metric_rows(self, comparison: ComparisonResult) -> str:
        rows: List[str] = []
        for metric in comparison.metrics.values():
            if not metric.comparable:
                winner = '<span class="warn">N/A</span>'
            elif metric.winner == comparison.label_b:
                winner = f'<span class="win">{escape(metric.winner)}</span>'
            elif metric.winner == comparison.label_a:
                winner = f'<span class="loss">{escape(metric.winner)}</span>'
            else:
                winner = '<span class="warn">tie</span>'
            rows.append(
                "<tr>"
                f"<td>{escape(metric.metric_name)}</td>"
                f"<td>{escape(self._fmt(metric.value_a))}</td>"
                f"<td>{escape(self._fmt(metric.value_b))}</td>"
                f"<td>{escape(self._fmt(metric.abs_delta))}</td>"
                f"<td>{escape(self._fmt(metric.percent_delta, is_percent=True))}</td>"
                f"<td>{winner}</td>"
                "</tr>"
            )
        return "".join(rows)

    @staticmethod
    def _platform_comparison(result_a: TrackiqResult, result_b: TrackiqResult) -> str:
        diffs: List[str] = []
        if result_a.platform.hardware_name != result_b.platform.hardware_name:
            diffs.append(
                f"Hardware differs: {escape(result_a.platform.hardware_name)} vs {escape(result_b.platform.hardware_name)}."
            )
        if result_a.platform.framework_version != result_b.platform.framework_version:
            diffs.append(
                f"Framework version differs: {escape(result_a.platform.framework_version)} vs {escape(result_b.platform.framework_version)}."
            )
        if result_a.platform.framework != result_b.platform.framework:
            diffs.append(
                f"Framework differs: {escape(result_a.platform.framework)} vs {escape(result_b.platform.framework)}."
            )
        if not diffs:
            return "<p>No platform/framework differences detected.</p>"
        return "<ul>" + "".join(f"<li>{item}</li>" for item in diffs) + "</ul>"

