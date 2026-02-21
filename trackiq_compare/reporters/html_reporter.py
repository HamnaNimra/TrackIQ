"""HTML reporter for polished TrackIQ comparison artifacts."""

from datetime import datetime, timezone
from html import escape
from typing import Any

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency fallback
    go = None
    PLOTLY_AVAILABLE = False

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
        visual_overview = self._render_visual_overview(
            normalized_rows,
            family_rows,
            confidence_rows,
            comparison.label_a,
            comparison.label_b,
        )
        platform_diff = self._platform_comparison(result_a, result_b)
        highlighted = self._highlighted_metrics(summary)
        consistency_html = self._render_consistency_analysis(comparison)
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
    .chart-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 8px;
    }}
    .chart-card {{
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 12px;
      background: #fcfdff;
    }}
    .chart-title {{
      margin: 0 0 8px;
      font-size: 14px;
      font-weight: 700;
    }}
    .chart-sub {{
      margin: 0 0 8px;
      color: var(--muted);
      font-size: 12px;
    }}
    .figure-html {{
      width: 100%;
      min-height: 320px;
    }}
    .figure-html .plotly-graph-div {{
      width: 100% !important;
    }}
    .delta-row {{
      display: grid;
      grid-template-columns: 1.2fr 2.4fr auto;
      align-items: center;
      gap: 8px;
      margin: 6px 0;
    }}
    .delta-label {{
      font-size: 12px;
      color: #111827;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .delta-value {{
      font-size: 12px;
      font-weight: 700;
      min-width: 58px;
      text-align: right;
    }}
    .delta-track {{
      position: relative;
      height: 16px;
      border-radius: 8px;
      border: 1px solid #e5e7eb;
      background: linear-gradient(to right, #f8fafc 0%, #eef2ff 50%, #f8fafc 100%);
      overflow: hidden;
    }}
    .delta-axis {{
      position: absolute;
      left: 50%;
      top: 0;
      width: 1px;
      height: 100%;
      background: #94a3b8;
    }}
    .delta-segment {{
      position: absolute;
      top: 2px;
      height: 10px;
      border-radius: 6px;
    }}
    .delta-segment.pos {{ background: #22c55e; }}
    .delta-segment.neg {{ background: #ef4444; }}
    .delta-segment.neu {{ background: #9ca3af; }}
    .pie-wrap {{
      display: flex;
      align-items: center;
      gap: 14px;
      margin-top: 6px;
    }}
    .pie-shell {{
      position: relative;
      width: 112px;
      height: 112px;
      border-radius: 50%;
      flex: 0 0 auto;
      border: 1px solid #d1d5db;
    }}
    .pie-hole {{
      position: absolute;
      left: 50%;
      top: 50%;
      transform: translate(-50%, -50%);
      width: 56px;
      height: 56px;
      border-radius: 50%;
      background: #ffffff;
      border: 1px solid #e5e7eb;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 12px;
      font-weight: 700;
      color: #111827;
    }}
    .legend {{
      display: grid;
      gap: 4px;
      margin: 0;
      padding: 0;
      list-style: none;
      font-size: 12px;
    }}
    .legend li {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .swatch {{
      width: 10px;
      height: 10px;
      border-radius: 2px;
      display: inline-block;
      border: 1px solid rgba(15, 23, 42, 0.2);
    }}
    @media (max-width: 900px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .chart-grid {{ grid-template-columns: 1fr; }}
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
      <h2>Visual Overview</h2>
      <p class="section-note">Consolidated graph views for quick comparison validation.</p>
      {visual_overview}
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
      <h2>Consistency Analysis</h2>
      {consistency_html}
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
                else comparison.label_a if normalized < 0 else "context" if direction == "context" else "tie"
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
        return f'<div class="bar-shell"><div class="bar-fill {fill_class}" style="width:{width_pct:.2f}%"></div></div>'

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
            a_html = (
                '<span class="matrix-yes">yes</span>'
                if bool(row["a_available"])
                else '<span class="matrix-no">no</span>'
            )
            b_html = (
                '<span class="matrix-yes">yes</span>'
                if bool(row["b_available"])
                else '<span class="matrix-no">no</span>'
            )
            confidence = str(row["confidence"])
            html_rows.append(
                "<tr>"
                f"<td>{escape(str(row['metric']))}</td>"
                f"<td>{escape(str(row['family']))}</td>"
                f"<td>{a_html}</td>"
                f"<td>{b_html}</td>"
                f"<td>{escape(str(row['direction']))}</td>"
                f'<td><span class="confidence-{escape(confidence)}">{escape(confidence)}</span></td>'
                "</tr>"
            )
        return "".join(html_rows)

    def _render_visual_overview(
        self,
        normalized_rows: list[dict[str, Any]],
        family_rows: list[dict[str, Any]],
        confidence_rows: list[dict[str, Any]],
        label_a: str,
        label_b: str,
    ) -> str:
        if PLOTLY_AVAILABLE:
            return self._render_plotly_visual_overview(
                normalized_rows=normalized_rows,
                family_rows=family_rows,
                confidence_rows=confidence_rows,
                label_a=label_a,
                label_b=label_b,
            )
        return self._render_static_visual_overview(
            normalized_rows=normalized_rows,
            family_rows=family_rows,
            confidence_rows=confidence_rows,
            label_a=label_a,
            label_b=label_b,
        )

    def _render_plotly_visual_overview(
        self,
        normalized_rows: list[dict[str, Any]],
        family_rows: list[dict[str, Any]],
        confidence_rows: list[dict[str, Any]],
        label_a: str,
        label_b: str,
    ) -> str:
        if not PLOTLY_AVAILABLE:  # pragma: no cover - guarded by caller
            return self._render_static_visual_overview(
                normalized_rows=normalized_rows,
                family_rows=family_rows,
                confidence_rows=confidence_rows,
                label_a=label_a,
                label_b=label_b,
            )

        winner_counts = self._winner_counts(normalized_rows, label_a, label_b)
        confidence_counts = self._confidence_counts(confidence_rows)

        top_rows = sorted(
            normalized_rows,
            key=lambda item: abs(float(item.get("normalized_delta_percent", 0.0))),
            reverse=True,
        )[:10]
        top_metrics = [str(row.get("metric", "-")) for row in top_rows][::-1]
        top_values = [float(row.get("normalized_delta_percent", 0.0)) for row in top_rows][::-1]
        top_colors = ["#22c55e" if value > 0 else "#ef4444" if value < 0 else "#9ca3af" for value in top_values]
        fig_delta = go.Figure(
            data=[
                go.Bar(
                    x=top_values,
                    y=top_metrics,
                    orientation="h",
                    marker_color=top_colors,
                )
            ]
        )
        fig_delta.add_vline(x=0, line_dash="dash", line_color="#6b7280")
        fig_delta.update_layout(
            title=f"Top Normalized Deltas ({label_b} positive)",
            template="plotly_white",
            height=360,
            margin=dict(l=40, r=16, t=48, b=36),
            xaxis_title="Normalized Delta (%)",
            yaxis_title="Metric",
        )

        fam_labels = [str(row.get("family", "other")) for row in family_rows][::-1]
        fam_values = [float(row.get("normalized_delta_percent", 0.0)) for row in family_rows][::-1]
        fam_colors = ["#22c55e" if value > 0 else "#ef4444" if value < 0 else "#9ca3af" for value in fam_values]
        fig_family = go.Figure(
            data=[
                go.Bar(
                    x=fam_values,
                    y=fam_labels,
                    orientation="h",
                    marker_color=fam_colors,
                )
            ]
        )
        fig_family.add_vline(x=0, line_dash="dash", line_color="#6b7280")
        fig_family.update_layout(
            title="Metric Family Deltas",
            template="plotly_white",
            height=360,
            margin=dict(l=40, r=16, t=48, b=36),
            xaxis_title="Normalized Delta (%)",
            yaxis_title="Family",
        )

        winner_labels: list[str] = []
        winner_values: list[int] = []
        winner_colors: list[str] = []
        for key, color, name in [
            ("A", "#ef4444", label_a),
            ("B", "#22c55e", label_b),
            ("tie", "#94a3b8", "tie"),
            ("context", "#c084fc", "context"),
        ]:
            count = int(winner_counts.get(key, 0))
            if count <= 0:
                continue
            winner_labels.append(name)
            winner_values.append(count)
            winner_colors.append(color)
        fig_winner = go.Figure(
            data=[
                go.Pie(
                    labels=winner_labels,
                    values=winner_values,
                    marker=dict(colors=winner_colors),
                    hole=0.45,
                    textinfo="label+percent",
                )
            ]
        )
        fig_winner.update_layout(
            title="Winner Distribution",
            template="plotly_white",
            height=360,
            margin=dict(l=16, r=16, t=48, b=16),
        )

        conf_labels: list[str] = []
        conf_values: list[int] = []
        conf_colors: list[str] = []
        for key, color in [
            ("strong", "#22c55e"),
            ("insufficient", "#f59e0b"),
            ("none", "#ef4444"),
        ]:
            count = int(confidence_counts.get(key, 0))
            if count <= 0:
                continue
            conf_labels.append(key)
            conf_values.append(count)
            conf_colors.append(color)
        fig_conf = go.Figure(
            data=[
                go.Pie(
                    labels=conf_labels,
                    values=conf_values,
                    marker=dict(colors=conf_colors),
                    hole=0.45,
                    textinfo="label+percent",
                )
            ]
        )
        fig_conf.update_layout(
            title="Confidence Distribution",
            template="plotly_white",
            height=360,
            margin=dict(l=16, r=16, t=48, b=16),
        )

        include_js = True
        cards: list[str] = []
        for title, subtitle, fig in [
            ("Top Normalized Deltas", f"Positive favors {label_b}; negative favors {label_a}.", fig_delta),
            ("Metric Family Deltas", "Family-level signed summary of normalized deltas.", fig_family),
            ("Winner Distribution", "Metric-level outcome share across comparable metrics.", fig_winner),
            ("Confidence Distribution", "Data-strength breakdown for metric conclusions.", fig_conf),
        ]:
            fig_html = self._fig_to_html(fig, include_plotlyjs=include_js)
            include_js = False
            cards.append(
                '<div class="chart-card">'
                f'<p class="chart-title">{escape(title)}</p>'
                f'<p class="chart-sub">{escape(subtitle)}</p>'
                f'<div class="figure-html">{fig_html}</div>'
                "</div>"
            )

        return f'<div class="chart-grid">{"".join(cards)}</div>'

    def _render_static_visual_overview(
        self,
        normalized_rows: list[dict[str, Any]],
        family_rows: list[dict[str, Any]],
        confidence_rows: list[dict[str, Any]],
        label_a: str,
        label_b: str,
    ) -> str:
        winner_counts = self._winner_counts(normalized_rows, label_a, label_b)
        confidence_counts = self._confidence_counts(confidence_rows)
        return (
            '<div class="chart-grid">'
            f'<div class="chart-card"><p class="chart-title">Top Normalized Deltas</p>'
            f'<p class="chart-sub">Right of center favors {escape(label_b)}; left favors {escape(label_a)}.</p>'
            f"{self._render_delta_graph_rows(normalized_rows, max_items=8)}</div>"
            f'<div class="chart-card"><p class="chart-title">Metric Family Deltas</p>'
            '<p class="chart-sub">Family-level signed summary of normalized deltas.</p>'
            f"{self._render_family_graph_rows(family_rows)}</div>"
            f'<div class="chart-card"><p class="chart-title">Winner Distribution</p>'
            '<p class="chart-sub">Metric-level outcome share across comparable metrics.</p>'
            f"{self._render_pie_chart(winner_counts, [('A', '#ef4444'), ('B', '#22c55e'), ('tie', '#94a3b8'), ('context', '#c084fc')], label_a, label_b)}</div>"
            f'<div class="chart-card"><p class="chart-title">Confidence Distribution</p>'
            '<p class="chart-sub">Data-strength breakdown for metric conclusions.</p>'
            f"{self._render_pie_chart(confidence_counts, [('strong', '#22c55e'), ('insufficient', '#f59e0b'), ('none', '#ef4444')], label_a, label_b)}</div>"
            "</div>"
        )

    @staticmethod
    def _fig_to_html(fig: Any, *, include_plotlyjs: bool) -> str:
        if not PLOTLY_AVAILABLE:  # pragma: no cover - guarded by caller
            return "<p>Plotly is unavailable.</p>"
        include_mode: str | bool = "inline" if include_plotlyjs else False
        return fig.to_html(
            full_html=False,
            include_plotlyjs=include_mode,
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "responsive": True,
                "scrollZoom": True,
                "doubleClick": "reset",
            },
        )

    @staticmethod
    def _winner_counts(
        normalized_rows: list[dict[str, Any]],
        label_a: str,
        label_b: str,
    ) -> dict[str, int]:
        counts = {"A": 0, "B": 0, "tie": 0, "context": 0}
        for row in normalized_rows:
            winner = str(row.get("winner", "tie"))
            if winner == label_a:
                counts["A"] += 1
            elif winner == label_b:
                counts["B"] += 1
            elif winner == "context":
                counts["context"] += 1
            else:
                counts["tie"] += 1
        return counts

    @staticmethod
    def _confidence_counts(confidence_rows: list[dict[str, Any]]) -> dict[str, int]:
        counts = {"strong": 0, "insufficient": 0, "none": 0}
        for row in confidence_rows:
            confidence = str(row.get("confidence", "none"))
            if confidence in counts:
                counts[confidence] += 1
        return counts

    def _render_delta_graph_rows(self, rows: list[dict[str, Any]], max_items: int = 8) -> str:
        if not rows:
            return '<p class="section-note">No normalized delta data available.</p>'
        selected = rows[:max_items]
        max_abs = max(abs(float(row["normalized_delta_percent"])) for row in selected) or 1.0
        html_rows: list[str] = []
        for row in selected:
            metric = escape(str(row.get("metric", "-")))
            value = float(row.get("normalized_delta_percent", 0.0))
            value_text = escape(self._fmt(value, is_percent=True))
            if value > 0:
                width = min(50.0, (abs(value) / max_abs) * 50.0)
                segment = f'<span class="delta-segment pos" style="left:50%;width:{width:.2f}%"></span>'
            elif value < 0:
                width = min(50.0, (abs(value) / max_abs) * 50.0)
                segment = f'<span class="delta-segment neg" style="right:50%;width:{width:.2f}%"></span>'
            else:
                segment = '<span class="delta-segment neu" style="left:calc(50% - 1px);width:2px"></span>'
            html_rows.append(
                '<div class="delta-row">'
                f'<div class="delta-label" title="{metric}">{metric}</div>'
                f'<div class="delta-track"><span class="delta-axis"></span>{segment}</div>'
                f'<div class="delta-value">{value_text}</div>'
                "</div>"
            )
        return "".join(html_rows)

    def _render_family_graph_rows(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return '<p class="section-note">No family aggregation data available.</p>'
        graph_rows = [
            {"metric": row.get("family", "other"), "normalized_delta_percent": row.get("normalized_delta_percent", 0.0)}
            for row in rows
        ]
        return self._render_delta_graph_rows(graph_rows, max_items=len(graph_rows))

    @staticmethod
    def _render_pie_chart(
        counts: dict[str, int],
        legend_order: list[tuple[str, str]],
        label_a: str,
        label_b: str,
    ) -> str:
        total = sum(int(v) for v in counts.values())
        if total <= 0:
            return '<p class="section-note">No data available for this chart.</p>'

        gradient_parts: list[str] = []
        legend_rows: list[str] = []
        degrees = 0.0
        for key, color in legend_order:
            count = int(counts.get(key, 0))
            if count <= 0:
                continue
            span = (count / total) * 360.0
            start = degrees
            end = degrees + span
            gradient_parts.append(f"{color} {start:.2f}deg {end:.2f}deg")
            degrees = end
            display = key
            if key == "A":
                display = label_a
            elif key == "B":
                display = label_b
            legend_rows.append(
                f'<li><span class="swatch" style="background:{color}"></span>{escape(display)}: {count}</li>'
            )

        gradient_css = ", ".join(gradient_parts) if gradient_parts else "#d1d5db 0deg 360deg"
        return (
            '<div class="pie-wrap">'
            f'<div class="pie-shell" style="background: conic-gradient({gradient_css});">'
            f'<div class="pie-hole">{total}</div>'
            "</div>"
            f'<ul class="legend">{"".join(legend_rows)}</ul>'
            "</div>"
        )

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

    @staticmethod
    def _render_consistency_analysis(comparison: ComparisonResult) -> str:
        """Render consistency-analysis findings table or graceful fallback."""
        findings = comparison.consistency_findings
        if not findings:
            return "<p>No consistency regressions detected or all-reduce step data was unavailable.</p>"

        rows = []
        for finding in findings:
            increase = "inf" if finding.increase_percent == float("inf") else f"{finding.increase_percent:+.2f}%"
            rows.append(
                "<tr>"
                f"<td>{escape(finding.code)}</td>"
                f"<td>{escape(finding.label)}</td>"
                f"<td>{escape(finding.status)}</td>"
                f"<td>{finding.stddev_a_ms:.6f}</td>"
                f"<td>{finding.stddev_b_ms:.6f}</td>"
                f"<td>{escape(increase)}</td>"
                f"<td>{finding.threshold_percent:.2f}%</td>"
                f"<td>{escape(finding.reason)}</td>"
                "</tr>"
            )
        return (
            "<table><thead><tr>"
            "<th>Code</th><th>Label</th><th>Status</th><th>StdDev A (ms)</th><th>StdDev B (ms)</th>"
            "<th>Increase %</th><th>Threshold %</th><th>Details</th>"
            "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )
