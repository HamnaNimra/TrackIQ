"""HTML reporter for polished TrackIQ comparison artifacts."""

from datetime import datetime
from html import escape
from typing import List, Optional

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
        highlighted = self._highlighted_metrics(summary)
        generated = datetime.utcnow().isoformat()

        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TrackIQ Compare Report</title>
  <style>
    :root {{
      --bg: #f7f9fc;
      --text: #1f2937;
      --muted: #6b7280;
      --card: #ffffff;
      --line: #e5e7eb;
      --accent: #0f766e;
      --accent-2: #2563eb;
      --good: #166534;
      --bad: #b91c1c;
      --warn: #a16207;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Arial", sans-serif;
      color: var(--text);
      background: radial-gradient(circle at top right, #dbeafe 0%, var(--bg) 35%);
    }}
    .wrap {{ max-width: 1150px; margin: 0 auto; padding: 28px; }}
    .hero {{
      background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
      color: #fff;
      padding: 28px;
      border-radius: 14px;
      box-shadow: 0 14px 32px rgba(15, 118, 110, 0.25);
      margin-bottom: 18px;
    }}
    .hero h1 {{ margin: 0 0 6px 0; font-size: 30px; }}
    .hero p {{ margin: 0; opacity: 0.95; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 14px;
      box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
    }}
    h2 {{ margin: 0 0 10px; font-size: 19px; }}
    .meta-line {{ margin: 6px 0; color: var(--muted); }}
    .badge {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
      margin-left: 6px;
      background: #ecfeff;
      color: #155e75;
      border: 1px solid #a5f3fc;
    }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 4px; background: #fff; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 10px 8px; text-align: left; font-size: 14px; }}
    th {{ color: #111827; background: #f8fafc; position: sticky; top: 0; }}
    tr:hover td {{ background: #f9fafb; }}
    .win {{ color: var(--good); font-weight: 700; }}
    .loss {{ color: var(--bad); font-weight: 700; }}
    .warn {{ color: var(--warn); font-weight: 700; }}
    ul {{ margin: 8px 0 0 20px; }}
    .highlights {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }}
    .pill {{
      border-radius: 999px;
      border: 1px solid #dbeafe;
      background: #eff6ff;
      color: #1d4ed8;
      font-size: 12px;
      padding: 6px 10px;
      font-weight: 600;
    }}
    @media (max-width: 900px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>TrackIQ Comparison Report</h1>
      <p>Generated at {escape(generated)}</p>
    </section>

    <section class="card">
      <h2>Executive Summary</h2>
      <p>{escape(summary.text)}</p>
      <div class="highlights">{highlighted}</div>
    </section>

    <section class="grid">
      <div class="card">
        <h2>{escape(comparison.label_a)}<span class="badge">Result A</span></h2>
        <div class="meta-line"><strong>Tool:</strong> {escape(result_a.tool_name)} {escape(result_a.tool_version)}</div>
        <div class="meta-line"><strong>Platform:</strong> {escape(result_a.platform.hardware_name)} ({escape(result_a.platform.os)})</div>
        <div class="meta-line"><strong>Framework:</strong> {escape(result_a.platform.framework)} {escape(result_a.platform.framework_version)}</div>
        <div class="meta-line"><strong>Workload:</strong> {escape(result_a.workload.name)} ({escape(result_a.workload.workload_type)})</div>
        <div class="meta-line"><strong>Timestamp:</strong> {escape(result_a.timestamp.isoformat())}</div>
      </div>
      <div class="card">
        <h2>{escape(comparison.label_b)}<span class="badge">Result B</span></h2>
        <div class="meta-line"><strong>Tool:</strong> {escape(result_b.tool_name)} {escape(result_b.tool_version)}</div>
        <div class="meta-line"><strong>Platform:</strong> {escape(result_b.platform.hardware_name)} ({escape(result_b.platform.os)})</div>
        <div class="meta-line"><strong>Framework:</strong> {escape(result_b.platform.framework)} {escape(result_b.platform.framework_version)}</div>
        <div class="meta-line"><strong>Workload:</strong> {escape(result_b.workload.name)} ({escape(result_b.workload.workload_type)})</div>
        <div class="meta-line"><strong>Timestamp:</strong> {escape(result_b.timestamp.isoformat())}</div>
      </div>
    </section>

    <section class="card">
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
    </section>

    <section class="card">
      <h2>Platform Comparison</h2>
      {platform_diff}
    </section>
  </div>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(html)
        return output_path

    @staticmethod
    def _fmt(value: Optional[float], is_percent: bool = False) -> str:
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

    @staticmethod
    def _highlighted_metrics(summary: SummaryResult) -> str:
        """Render top delta highlights for the summary card."""
        if not summary.largest_deltas:
            return '<span class="pill">No comparable metrics</span>'
        return "".join(
            f'<span class="pill">{escape(item.metric_name)}: {item.percent_delta:+.2f}%</span>'
            for item in summary.largest_deltas
            if item.percent_delta is not None and item.percent_delta != float("inf")
        )
