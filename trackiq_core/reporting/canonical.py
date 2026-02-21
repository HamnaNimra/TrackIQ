"""Canonical TrackiqResult HTML report rendering helpers."""

from __future__ import annotations

from html import escape
from typing import Any


def render_trackiq_result_html(
    result: Any,
    title: str = "TrackIQ Result Report",
) -> str:
    """Render a simple HTML report from a TrackiqResult-like object."""
    metrics = [
        ("Throughput (samples/s)", _fmt_number(result.metrics.throughput_samples_per_sec)),
        ("Latency p50 (ms)", _fmt_number(result.metrics.latency_p50_ms)),
        ("Latency p95 (ms)", _fmt_number(result.metrics.latency_p95_ms)),
        ("Latency p99 (ms)", _fmt_number(result.metrics.latency_p99_ms)),
        ("Memory Utilization (%)", _fmt_number(result.metrics.memory_utilization_percent)),
        ("Communication Overhead (%)", _fmt_optional(result.metrics.communication_overhead_percent)),
        ("Power (W)", _fmt_optional(result.metrics.power_consumption_watts)),
        ("Energy/Step (J)", _fmt_optional(result.metrics.energy_per_step_joules)),
        ("Performance/Watt", _fmt_optional(result.metrics.performance_per_watt)),
        ("Temperature (C)", _fmt_optional(result.metrics.temperature_celsius)),
        ("TTFT (ms)", _fmt_optional(result.metrics.ttft_ms)),
        ("Tokens/sec", _fmt_optional(result.metrics.tokens_per_sec)),
        ("Decode TPT (ms)", _fmt_optional(result.metrics.decode_tpt_ms)),
    ]

    kv_rows = ""
    if result.kv_cache is not None:
        kv_rows = (
            f"<tr><th>Estimated Size (MB)</th><td>{escape(_fmt_number(result.kv_cache.estimated_size_mb))}</td></tr>"
            f"<tr><th>Max Sequence Length</th><td>{escape(str(result.kv_cache.max_sequence_length))}</td></tr>"
            f"<tr><th>Batch Size</th><td>{escape(str(result.kv_cache.batch_size))}</td></tr>"
            f"<tr><th>Layers</th><td>{escape(str(result.kv_cache.num_layers))}</td></tr>"
            f"<tr><th>Heads</th><td>{escape(str(result.kv_cache.num_heads))}</td></tr>"
            f"<tr><th>Head Size</th><td>{escape(str(result.kv_cache.head_size))}</td></tr>"
            f"<tr><th>Precision</th><td>{escape(str(result.kv_cache.precision))}</td></tr>"
        )

    metric_rows = "".join(f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>" for label, value in metrics)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
    h1 {{ margin-bottom: 8px; }}
    .meta {{ color: #4b5563; margin-bottom: 12px; }}
    table {{ width: 100%; border-collapse: collapse; margin-bottom: 16px; }}
    th, td {{ text-align: left; border: 1px solid #d1d5db; padding: 8px; }}
    th {{ width: 35%; background: #f8fafc; }}
    section {{ margin-bottom: 18px; }}
    .badge {{ display: inline-block; background: #e0f2fe; color: #0369a1; padding: 2px 8px; border-radius: 999px; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <div class="meta">
    <span class="badge">{escape(str(result.tool_name))}</span>
    <strong>Version:</strong> {escape(str(result.tool_version))} |
    <strong>Timestamp:</strong> {escape(result.timestamp.isoformat())}
  </div>

  <section>
    <h2>Platform</h2>
    <table>
      <tr><th>Hardware</th><td>{escape(str(result.platform.hardware_name))}</td></tr>
      <tr><th>OS</th><td>{escape(str(result.platform.os))}</td></tr>
      <tr><th>Framework</th><td>{escape(str(result.platform.framework))}</td></tr>
      <tr><th>Framework Version</th><td>{escape(str(result.platform.framework_version))}</td></tr>
    </table>
  </section>

  <section>
    <h2>Workload</h2>
    <table>
      <tr><th>Name</th><td>{escape(str(result.workload.name))}</td></tr>
      <tr><th>Type</th><td>{escape(str(result.workload.workload_type))}</td></tr>
      <tr><th>Batch Size</th><td>{escape(str(result.workload.batch_size))}</td></tr>
      <tr><th>Steps</th><td>{escape(str(result.workload.steps))}</td></tr>
    </table>
  </section>

  <section>
    <h2>Metrics</h2>
    <table>
      {metric_rows}
    </table>
  </section>

  {"<section><h2>KV Cache</h2><table>" + kv_rows + "</table></section>" if kv_rows else ""}
</body>
</html>"""


def _fmt_optional(value: Any) -> str:
    if value is None:
        return "N/A"
    return _fmt_number(value)


def _fmt_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.4f}"


__all__ = ["render_trackiq_result_html"]
