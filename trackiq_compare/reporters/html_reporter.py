"""HTML reporter for polished TrackIQ comparison artifacts."""

from datetime import datetime, timezone
from html import escape
from typing import Any

from trackiq_compare.comparator.metric_comparator import (
    LOWER_IS_BETTER_METRICS,
    ComparisonResult,
)
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
        normalized_rows = self._normalized_metric_delta_rows(comparison)
        family_rows = self._metric_family_delta_rows(normalized_rows, comparison.label_a, comparison.label_b)
        confidence_rows = self._metric_confidence_rows(comparison)
        platform_diff = self._platform_comparison(result_a, result_b)
        highlighted = self._highlighted_metrics(summary)
        generated = datetime.now(timezone.utc).isoformat()

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
    .section-note {{ margin: 0 0 10px; color: var(--muted); font-size: 13px; }}
    .bar-shell {{
      background: #f3f4f6;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      height: 12px;
      overflow: hidden;
      width: 100%;
      min-width: 130px;
    }}
    .bar-fill {{
      height: 100%;
      border-radius: 8px;
    }}
    .bar-fill.positive {{ background: #22c55e; }}
    .bar-fill.negative {{ background: #ef4444; }}
    .bar-fill.neutral {{ background: #9ca3af; }}
    .matrix-yes {{ color: var(--good); font-weight: 700; }}
    .matrix-no {{ color: var(--bad); font-weight: 700; }}
    .confidence-strong {{ color: var(--good); font-weight: 700; }}
    .confidence-insufficient {{ color: var(--warn); font-weight: 700; }}
    .confidence-none {{ color: var(--bad); font-weight: 700; }}
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
      <h2>Normalized Metric Deltas</h2>
      <p class="section-note">Positive values indicate an advantage for {escape(comparison.label_b)}; negative values favor {escape(comparison.label_a)}.</p>
      <table>
        <thead>
          <tr>
            <th>Metric</th>
            <th>Family</th>
            <th>Direction</th>
            <th>Raw Delta %</th>
            <th>Normalized Delta %</th>
            <th>Visual</th>
            <th>Advantage</th>
          </tr>
        </thead>
        <tbody>
          {self._render_normalized_rows(normalized_rows, comparison.label_a, comparison.label_b)}
        </tbody>
      </table>
    </section>

    <section class="card">
      <h2>Metric Family Delta Waterfall</h2>
      <p class="section-note">Family-level mean of normalized metric deltas (context-only metrics excluded).</p>
      <table>
        <thead>
          <tr>
            <th>Family</th>
            <th>Metrics</th>
            <th>Normalized Delta %</th>
            <th>Visual</th>
            <th>Winner</th>
          </tr>
        </thead>
        <tbody>
          {self._render_family_rows(family_rows, comparison.label_a, comparison.label_b)}
        </tbody>
      </table>
    </section>

    <section class="card">
      <h2>Metric Availability Confidence Matrix</h2>
      <p class="section-note">Confidence is based on metric availability in both results.</p>
      <table>
        <thead>
          <tr>
            <th>Metric</th>
            <th>Family</th>
            <th>{escape(comparison.label_a)} Available</th>
            <th>{escape(comparison.label_b)} Available</th>
            <th>Direction</th>
            <th>Confidence</th>
          </tr>
        </thead>
        <tbody>
          {self._render_confidence_rows(confidence_rows)}
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
    def _fmt(value: float | None, is_percent: bool = False) -> str:
        if value is None:
            return "N/A"
        if value == float("inf"):
            return "inf"
        return f"{value:+.2f}%" if is_percent else f"{value:.4f}"

    def _metric_rows(self, comparison: ComparisonResult) -> str:
        rows: list[str] = []
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
    def _metric_family(metric_name: str) -> str:
        if metric_name.startswith("latency_"):
            return "latency"
        if metric_name in {"throughput_samples_per_sec", "performance_per_watt"}:
            return "performance"
        if metric_name in {"power_consumption_watts", "energy_per_step_joules"}:
            return "efficiency"
        if metric_name == "memory_utilization_percent":
            return "memory"
        if metric_name == "communication_overhead_percent":
            return "communication"
        return "other"

    @staticmethod
    def _metric_direction(metric_name: str) -> str:
        if metric_name == "memory_utilization_percent":
            return "context"
        if metric_name in LOWER_IS_BETTER_METRICS:
            return "low"
        return "high"

    @classmethod
    def _normalized_delta(cls, percent_delta: float | None, direction: str) -> float | None:
        if percent_delta is None or percent_delta == float("inf"):
            return None
        if direction == "high":
            return float(percent_delta)
        if direction == "low":
            return -float(percent_delta)
        return 0.0

    def _normalized_metric_delta_rows(self, comparison: ComparisonResult) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for metric in comparison.metrics.values():
            if not metric.comparable:
                continue
            direction = self._metric_direction(metric.metric_name)
            normalized = self._normalized_delta(metric.percent_delta, direction)
            if normalized is None:
                continue
            winner = (
                comparison.label_b
                if normalized > 0
                else comparison.label_a
                if normalized < 0
                else "context"
                if direction == "context"
                else "tie"
            )
            rows.append(
                {
                    "metric": metric.metric_name,
                    "family": self._metric_family(metric.metric_name),
                    "direction": direction,
                    "raw_delta_percent": float(metric.percent_delta) if metric.percent_delta is not None else None,
                    "normalized_delta_percent": float(normalized),
                    "winner": winner,
                }
            )
        rows.sort(key=lambda item: abs(float(item["normalized_delta_percent"])), reverse=True)
        return rows

    @staticmethod
    def _metric_family_delta_rows(
        normalized_rows: list[dict[str, Any]], label_a: str, label_b: str
    ) -> list[dict[str, Any]]:
        buckets: dict[str, list[float]] = {}
        for row in normalized_rows:
            if row.get("direction") == "context":
                continue
            family = str(row.get("family", "other"))
            buckets.setdefault(family, []).append(float(row.get("normalized_delta_percent", 0.0)))

        output: list[dict[str, Any]] = []
        for family, values in buckets.items():
            if not values:
                continue
            mean_delta = sum(values) / len(values)
            winner = label_b if mean_delta > 0 else label_a if mean_delta < 0 else "tie"
            output.append(
                {
                    "family": family,
                    "metric_count": len(values),
                    "normalized_delta_percent": mean_delta,
                    "winner": winner,
                }
            )
        output.sort(key=lambda item: abs(float(item["normalized_delta_percent"])), reverse=True)
        return output

    def _metric_confidence_rows(self, comparison: ComparisonResult) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for metric in sorted(comparison.metrics.values(), key=lambda item: item.metric_name):
            a_available = metric.value_a is not None
            b_available = metric.value_b is not None
            if a_available and b_available:
                confidence = "strong"
            elif a_available or b_available:
                confidence = "insufficient"
            else:
                confidence = "none"
            rows.append(
                {
                    "metric": metric.metric_name,
                    "family": self._metric_family(metric.metric_name),
                    "direction": self._metric_direction(metric.metric_name),
                    "a_available": a_available,
                    "b_available": b_available,
                    "confidence": confidence,
                }
            )
        return rows

    @staticmethod
    def _bar_html(value: float, max_abs: float) -> str:
        if max_abs <= 0:
            width_pct = 0.0
        else:
            width_pct = min(100.0, (abs(value) / max_abs) * 100.0)
        fill_class = "positive" if value > 0 else "negative" if value < 0 else "neutral"
        return (
            '<div class="bar-shell">'
            f'<div class="bar-fill {fill_class}" style="width:{width_pct:.2f}%"></div>'
            "</div>"
        )

    def _render_normalized_rows(self, rows: list[dict[str, Any]], label_a: str, label_b: str) -> str:
        if not rows:
            return '<tr><td colspan="7">No comparable metric deltas available.</td></tr>'
        max_abs = max(abs(float(row["normalized_delta_percent"])) for row in rows)
        html_rows: list[str] = []
        for row in rows:
            winner = str(row["winner"])
            if winner == label_b:
                winner_html = f'<span class="win">{escape(winner)}</span>'
            elif winner == label_a:
                winner_html = f'<span class="loss">{escape(winner)}</span>'
            elif winner == "context":
                winner_html = '<span class="warn">context</span>'
            else:
                winner_html = '<span class="warn">tie</span>'
            html_rows.append(
                "<tr>"
                f"<td>{escape(str(row['metric']))}</td>"
                f"<td>{escape(str(row['family']))}</td>"
                f"<td>{escape(str(row['direction']))}</td>"
                f"<td>{escape(self._fmt(float(row['raw_delta_percent']), is_percent=True))}</td>"
                f"<td>{escape(self._fmt(float(row['normalized_delta_percent']), is_percent=True))}</td>"
                f"<td>{self._bar_html(float(row['normalized_delta_percent']), max_abs)}</td>"
                f"<td>{winner_html}</td>"
                "</tr>"
            )
        return "".join(html_rows)

    def _render_family_rows(self, rows: list[dict[str, Any]], label_a: str, label_b: str) -> str:
        if not rows:
            return '<tr><td colspan="5">No non-context metric families available.</td></tr>'
        max_abs = max(abs(float(row["normalized_delta_percent"])) for row in rows)
        html_rows: list[str] = []
        for row in rows:
            winner = str(row["winner"])
            if winner == label_b:
                winner_html = f'<span class="win">{escape(winner)}</span>'
            elif winner == label_a:
                winner_html = f'<span class="loss">{escape(winner)}</span>'
            else:
                winner_html = '<span class="warn">tie</span>'
            html_rows.append(
                "<tr>"
                f"<td>{escape(str(row['family']))}</td>"
                f"<td>{escape(str(row['metric_count']))}</td>"
                f"<td>{escape(self._fmt(float(row['normalized_delta_percent']), is_percent=True))}</td>"
                f"<td>{self._bar_html(float(row['normalized_delta_percent']), max_abs)}</td>"
                f"<td>{winner_html}</td>"
                "</tr>"
            )
        return "".join(html_rows)

    @staticmethod
    def _render_confidence_rows(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return '<tr><td colspan="6">No metric availability data available.</td></tr>'
        html_rows: list[str] = []
        for row in rows:
            a_html = '<span class="matrix-yes">yes</span>' if bool(row["a_available"]) else '<span class="matrix-no">no</span>'
            b_html = '<span class="matrix-yes">yes</span>' if bool(row["b_available"]) else '<span class="matrix-no">no</span>'
            confidence = str(row["confidence"])
            html_rows.append(
                "<tr>"
                f"<td>{escape(str(row['metric']))}</td>"
                f"<td>{escape(str(row['family']))}</td>"
                f"<td>{a_html}</td>"
                f"<td>{b_html}</td>"
                f"<td>{escape(str(row['direction']))}</td>"
                f"<td><span class=\"confidence-{escape(confidence)}\">{escape(confidence)}</span></td>"
                "</tr>"
            )
        return "".join(html_rows)

    @staticmethod
    def _platform_comparison(result_a: TrackiqResult, result_b: TrackiqResult) -> str:
        diffs: list[str] = []
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
