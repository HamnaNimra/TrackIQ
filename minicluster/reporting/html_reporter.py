"""HTML reporting for MiniCluster result files."""

from __future__ import annotations

from html import escape
from typing import Any

from minicluster.deps import ensure_parent_dir

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency fallback
    go = None
    PLOTLY_AVAILABLE = False


class MiniClusterHtmlReporter:
    """Render MiniCluster result HTML, including consolidated multi-run visuals."""

    def generate(self, output_path: str, results: list[Any], title: str = "MiniCluster Performance Report") -> str:
        """Write single-run or consolidated HTML report."""
        if not results:
            raise ValueError("At least one result is required.")
        ensure_parent_dir(output_path)
        if len(results) == 1:
            html = self._render_single_result_html(results[0], title=title)
        else:
            html = self._render_multi_result_html(results, title=title)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(html)
        return output_path

    def _render_single_result_html(self, result: Any, title: str) -> str:
        run = self._run_data(result, default_label="run-1")
        metric_rows = self._single_metric_rows(result, run)
        config_rows = self._config_rows(run["config"])
        if PLOTLY_AVAILABLE:
            loss_chart, throughput_chart, timing_chart = self._single_plotly_charts(run)
        else:
            loss_chart = self._line_chart_svg(
                run["step_points"],
                title="Loss by Step",
                y_label="Loss",
                color="#dc2626",
                show_points=True,
            )
            throughput_chart = self._line_chart_svg(
                run["throughput_points"],
                title="Throughput by Step",
                y_label="Samples/sec",
                color="#2563eb",
                show_points=True,
            )
            timing_chart = self._stacked_timing_svg(
                run["timing_points"],
                title="Per-Step Timing (Compute + AllReduce)",
            )

        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escape(title)}</title>
  {self._style_block()}
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>{escape(title)}</h1>
      <p>MiniCluster HTML Report</p>
    </section>

    <section class="grid kpi">
      <div class="card"><h3>Throughput</h3><p>{self._fmt(run["throughput"])}</p></div>
      <div class="card"><h3>Final Loss</h3><p>{self._fmt(run["final_loss"])}</p></div>
      <div class="card"><h3>Total Time (s)</h3><p>{self._fmt(run["total_time_sec"])}</p></div>
      <div class="card"><h3>Power (W)</h3><p>{self._fmt(run["power_w"])}</p></div>
    </section>

    <section class="card">
      <h2>Run Metadata</h2>
      <table>
        <tbody>
          <tr><th>Tool</th><td>{escape(str(result.tool_name))} {escape(str(result.tool_version))}</td></tr>
          <tr><th>Timestamp</th><td>{escape(result.timestamp.isoformat())}</td></tr>
          <tr><th>Hardware</th><td>{escape(str(result.platform.hardware_name))}</td></tr>
          <tr><th>Framework</th><td>{escape(str(result.platform.framework))} {escape(str(result.platform.framework_version))}</td></tr>
          <tr><th>Workload</th><td>{escape(str(result.workload.name))} ({escape(str(result.workload.workload_type))})</td></tr>
        </tbody>
      </table>
    </section>

    <section class="card">
      <h2>Configuration</h2>
      <table>
        <tbody>
          {config_rows}
        </tbody>
      </table>
    </section>

    <section class="card">
      <h2>Metrics</h2>
      <table>
        <tbody>
          {metric_rows}
        </tbody>
      </table>
    </section>

    <section class="card">
      <h2>Training Graphs</h2>
      <div class="charts-grid">
        <div>{loss_chart}</div>
        <div>{throughput_chart}</div>
      </div>
      <div class="chart-full">{timing_chart}</div>
    </section>
  </div>
</body>
</html>"""

    def _render_multi_result_html(self, results: list[Any], title: str) -> str:
        run_data = [self._run_data(result, default_label=f"run-{idx + 1}") for idx, result in enumerate(results)]
        table_rows = self._consolidated_rows(run_data)
        if PLOTLY_AVAILABLE:
            throughput_chart, loss_chart, runtime_chart, winner_chart, overlay_chart = self._multi_plotly_charts(run_data)
        else:
            throughput_chart = self._bar_chart_svg(
                [run["label"] for run in run_data],
                [run["throughput"] for run in run_data],
                title="Throughput by Config",
                y_label="Samples/sec",
                color="#2563eb",
            )
            loss_chart = self._bar_chart_svg(
                [run["label"] for run in run_data],
                [run["final_loss"] for run in run_data],
                title="Final Loss by Config (lower is better)",
                y_label="Loss",
                color="#dc2626",
            )
            runtime_chart = self._bar_chart_svg(
                [run["label"] for run in run_data],
                [run["total_time_sec"] for run in run_data],
                title="Total Runtime by Config",
                y_label="Seconds",
                color="#0891b2",
            )
            overlay_chart = self._multi_line_chart_svg(
                [(run["label"], run["step_points"]) for run in run_data],
                title="Loss Curves Overlay",
                y_label="Loss",
            )
            winners = self._winner_share(run_data)
            winner_chart = self._pie_div(
                winners,
                title="Key-Metric Winner Share",
                labels={run["label"]: run["label"] for run in run_data},
            )

        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escape(title)}</title>
  {self._style_block()}
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>{escape(title)}</h1>
      <p>MiniCluster consolidated HTML report for {len(run_data)} configs</p>
    </section>

    <section class="card">
      <h2>Configuration Comparison</h2>
      <table>
        <thead>
          <tr>
            <th>Run Label</th>
            <th>Workers</th>
            <th>Steps</th>
            <th>Batch</th>
            <th>Learning Rate</th>
            <th>Hidden</th>
            <th>Layers</th>
            <th>Throughput</th>
            <th>Final Loss</th>
            <th>Total Time (s)</th>
            <th>Power (W)</th>
            <th>Perf/Watt</th>
          </tr>
        </thead>
        <tbody>
          {table_rows}
        </tbody>
      </table>
    </section>

    <section class="card">
      <h2>Consolidated Graphs</h2>
      <div class="charts-grid">
        <div>{throughput_chart}</div>
        <div>{loss_chart}</div>
      </div>
      <div class="charts-grid">
        <div>{runtime_chart}</div>
        <div>{winner_chart}</div>
      </div>
      <div class="chart-full">{overlay_chart}</div>
    </section>
  </div>
</body>
</html>"""

    @staticmethod
    def _style_block() -> str:
        return """
<style>
  :root {
    --bg: #f8fafc;
    --card: #ffffff;
    --line: #d1d5db;
    --text: #1f2937;
    --muted: #6b7280;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: "Segoe UI", Arial, sans-serif;
    color: var(--text);
    background: radial-gradient(circle at top right, #dbeafe 0%, var(--bg) 40%);
  }
  .wrap { max-width: 1200px; margin: 0 auto; padding: 22px; }
  .hero {
    background: linear-gradient(135deg, #0f766e 0%, #2563eb 100%);
    color: #fff;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 14px;
  }
  .hero h1 { margin: 0; font-size: 28px; }
  .hero p { margin: 6px 0 0; opacity: 0.95; }
  .card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 12px;
    box-shadow: 0 4px 10px rgba(15, 23, 42, 0.05);
  }
  h2 { margin: 0 0 10px; font-size: 20px; }
  h3 { margin: 0 0 6px; font-size: 14px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
  table { width: 100%; border-collapse: collapse; }
  th, td { border-bottom: 1px solid #e5e7eb; text-align: left; padding: 8px; font-size: 13px; }
  th { background: #f8fafc; position: sticky; top: 0; }
  .grid { display: grid; gap: 10px; }
  .kpi { grid-template-columns: repeat(4, minmax(120px, 1fr)); }
  .kpi p { margin: 0; font-size: 22px; font-weight: 700; color: #0f172a; }
  .charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .chart-full { margin-top: 12px; }
  .figure-html { width: 100%; min-height: 320px; }
  .figure-html .plotly-graph-div { width: 100% !important; }
  .pie-wrap { display: flex; align-items: center; gap: 14px; }
  .pie-shell {
    width: 118px;
    height: 118px;
    border-radius: 50%;
    border: 1px solid #d1d5db;
    position: relative;
    flex: 0 0 auto;
  }
  .pie-hole {
    width: 58px;
    height: 58px;
    border-radius: 50%;
    background: #fff;
    border: 1px solid #e5e7eb;
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 12px;
  }
  .legend { margin: 0; padding: 0; list-style: none; display: grid; gap: 4px; font-size: 12px; }
  .legend li { display: flex; align-items: center; gap: 6px; }
  .swatch { width: 10px; height: 10px; border-radius: 2px; border: 1px solid rgba(15, 23, 42, 0.25); display: inline-block; }
  .chart-note { color: var(--muted); font-size: 12px; margin: 0 0 8px; }
  @media (max-width: 980px) {
    .kpi { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
    .charts-grid { grid-template-columns: 1fr; }
  }
</style>
"""

    def _single_plotly_charts(self, run: dict[str, Any]) -> tuple[str, str, str]:
        """Build Plotly chart fragments for single-run report."""
        if not PLOTLY_AVAILABLE:  # pragma: no cover - guarded by caller
            return (
                self._line_chart_svg(run["step_points"], title="Loss by Step", y_label="Loss", color="#dc2626", show_points=True),
                self._line_chart_svg(
                    run["throughput_points"],
                    title="Throughput by Step",
                    y_label="Samples/sec",
                    color="#2563eb",
                    show_points=True,
                ),
                self._stacked_timing_svg(run["timing_points"], title="Per-Step Timing (Compute + AllReduce)"),
            )

        steps = [float(step) for step, value in run["step_points"] if value is not None]
        losses = [float(value) for _, value in run["step_points"] if value is not None]
        thr_steps = [float(step) for step, value in run["throughput_points"] if value is not None]
        throughputs = [float(value) for _, value in run["throughput_points"] if value is not None]

        include_js = True

        if len(losses) >= 2:
            fig_loss = go.Figure()
            fig_loss.add_trace(
                go.Scatter(
                    x=steps,
                    y=losses,
                    mode="lines+markers",
                    name="loss",
                    line=dict(color="#dc2626", width=2),
                )
            )
            fig_loss.update_layout(
                title="Loss by Step",
                template="plotly_white",
                height=340,
                margin=dict(l=40, r=16, t=48, b=36),
                xaxis_title="Step",
                yaxis_title="Loss",
            )
            loss_chart = self._wrap_plotly_chart(
                "Loss by Step",
                self._fig_to_html(fig_loss, include_plotlyjs=include_js),
            )
            include_js = False
        else:
            loss_chart = self._line_chart_svg(
                run["step_points"],
                title="Loss by Step",
                y_label="Loss",
                color="#dc2626",
                show_points=True,
            )

        if len(throughputs) >= 2:
            fig_thr = go.Figure()
            fig_thr.add_trace(
                go.Scatter(
                    x=thr_steps,
                    y=throughputs,
                    mode="lines+markers",
                    name="throughput",
                    line=dict(color="#2563eb", width=2),
                )
            )
            fig_thr.update_layout(
                title="Throughput by Step",
                template="plotly_white",
                height=340,
                margin=dict(l=40, r=16, t=48, b=36),
                xaxis_title="Step",
                yaxis_title="Samples/sec",
            )
            throughput_chart = self._wrap_plotly_chart(
                "Throughput by Step",
                self._fig_to_html(fig_thr, include_plotlyjs=include_js),
            )
            include_js = False
        else:
            throughput_chart = self._line_chart_svg(
                run["throughput_points"],
                title="Throughput by Step",
                y_label="Samples/sec",
                color="#2563eb",
                show_points=True,
            )

        timing_rows = [(step, compute, allreduce) for step, compute, allreduce in run["timing_points"] if compute is not None and allreduce is not None]
        if timing_rows:
            timing_x = [float(step) for step, _, _ in timing_rows]
            compute_vals = [float(compute) for _, compute, _ in timing_rows]
            allreduce_vals = [float(allreduce) for _, _, allreduce in timing_rows]
            if any((compute + allreduce) > 0 for compute, allreduce in zip(compute_vals, allreduce_vals)):
                fig_timing = go.Figure()
                fig_timing.add_trace(
                    go.Bar(
                        x=timing_x,
                        y=compute_vals,
                        name="compute_ms",
                        marker_color="#2563eb",
                    )
                )
                fig_timing.add_trace(
                    go.Bar(
                        x=timing_x,
                        y=allreduce_vals,
                        name="allreduce_ms",
                        marker_color="#dc2626",
                    )
                )
                fig_timing.update_layout(
                    title="Per-Step Timing (Compute + AllReduce)",
                    template="plotly_white",
                    height=360,
                    margin=dict(l=40, r=16, t=48, b=36),
                    xaxis_title="Step",
                    yaxis_title="Time (ms)",
                    barmode="stack",
                )
                timing_chart = self._wrap_plotly_chart(
                    "Per-Step Timing (Compute + AllReduce)",
                    self._fig_to_html(fig_timing, include_plotlyjs=include_js),
                )
                include_js = False
            else:
                timing_chart = "<p>No positive timing values for Per-Step Timing (Compute + AllReduce).</p>"
        else:
            timing_chart = self._stacked_timing_svg(
                run["timing_points"],
                title="Per-Step Timing (Compute + AllReduce)",
            )

        return loss_chart, throughput_chart, timing_chart

    def _multi_plotly_charts(
        self,
        run_data: list[dict[str, Any]],
    ) -> tuple[str, str, str, str, str]:
        """Build Plotly chart fragments for multi-run report."""
        if not PLOTLY_AVAILABLE:  # pragma: no cover - guarded by caller
            labels = [run["label"] for run in run_data]
            return (
                self._bar_chart_svg(labels, [run["throughput"] for run in run_data], title="Throughput by Config", y_label="Samples/sec", color="#2563eb"),
                self._bar_chart_svg(labels, [run["final_loss"] for run in run_data], title="Final Loss by Config (lower is better)", y_label="Loss", color="#dc2626"),
                self._bar_chart_svg(labels, [run["total_time_sec"] for run in run_data], title="Total Runtime by Config", y_label="Seconds", color="#0891b2"),
                self._pie_div(self._winner_share(run_data), title="Key-Metric Winner Share", labels={run["label"]: run["label"] for run in run_data}),
                self._multi_line_chart_svg([(run["label"], run["step_points"]) for run in run_data], title="Loss Curves Overlay", y_label="Loss"),
            )

        labels = [str(run["label"]) for run in run_data]
        include_js = True

        fig_thr = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=[run["throughput"] if run["throughput"] is not None else 0.0 for run in run_data],
                    marker_color="#2563eb",
                )
            ]
        )
        fig_thr.update_layout(
            title="Throughput by Config",
            template="plotly_white",
            height=340,
            margin=dict(l=40, r=16, t=48, b=48),
            xaxis_title="Config",
            yaxis_title="Samples/sec",
        )
        throughput_chart = self._wrap_plotly_chart(
            "Throughput by Config",
            self._fig_to_html(fig_thr, include_plotlyjs=include_js),
        )
        include_js = False

        fig_loss = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=[run["final_loss"] if run["final_loss"] is not None else 0.0 for run in run_data],
                    marker_color="#dc2626",
                )
            ]
        )
        fig_loss.update_layout(
            title="Final Loss by Config (lower is better)",
            template="plotly_white",
            height=340,
            margin=dict(l=40, r=16, t=48, b=48),
            xaxis_title="Config",
            yaxis_title="Loss",
        )
        loss_chart = self._wrap_plotly_chart(
            "Final Loss by Config (lower is better)",
            self._fig_to_html(fig_loss, include_plotlyjs=include_js),
        )

        fig_runtime = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=[run["total_time_sec"] if run["total_time_sec"] is not None else 0.0 for run in run_data],
                    marker_color="#0891b2",
                )
            ]
        )
        fig_runtime.update_layout(
            title="Total Runtime by Config",
            template="plotly_white",
            height=340,
            margin=dict(l=40, r=16, t=48, b=48),
            xaxis_title="Config",
            yaxis_title="Seconds",
        )
        runtime_chart = self._wrap_plotly_chart(
            "Total Runtime by Config",
            self._fig_to_html(fig_runtime, include_plotlyjs=False),
        )

        winners = self._winner_share(run_data)
        pie_labels = [label for label in labels if winners.get(label, 0) > 0]
        pie_values = [int(winners.get(label, 0)) for label in pie_labels]
        pie_colors = ["#2563eb", "#dc2626", "#0891b2", "#16a34a", "#7c3aed", "#ea580c"][: len(pie_labels)]
        if pie_labels:
            fig_winner = go.Figure(
                data=[
                    go.Pie(
                        labels=pie_labels,
                        values=pie_values,
                        hole=0.45,
                        marker=dict(colors=pie_colors),
                        textinfo="label+percent",
                    )
                ]
            )
            fig_winner.update_layout(
                title="Key-Metric Winner Share",
                template="plotly_white",
                height=340,
                margin=dict(l=16, r=16, t=48, b=16),
            )
            winner_chart = self._wrap_plotly_chart(
                "Key-Metric Winner Share",
                self._fig_to_html(fig_winner, include_plotlyjs=False),
            )
        else:
            winner_chart = "<p>No data available for Key-Metric Winner Share.</p>"

        fig_overlay = go.Figure()
        for idx, run in enumerate(run_data):
            points = [(step, value) for step, value in run["step_points"] if value is not None]
            if len(points) < 2:
                continue
            fig_overlay.add_trace(
                go.Scatter(
                    x=[float(step) for step, _ in points],
                    y=[float(value) for _, value in points],
                    mode="lines+markers",
                    name=str(run["label"]),
                )
            )
        if fig_overlay.data:
            fig_overlay.update_layout(
                title="Loss Curves Overlay",
                template="plotly_white",
                height=360,
                margin=dict(l=40, r=16, t=48, b=36),
                xaxis_title="Step",
                yaxis_title="Loss",
            )
            overlay_chart = self._wrap_plotly_chart(
                "Loss Curves Overlay",
                self._fig_to_html(fig_overlay, include_plotlyjs=False),
            )
        else:
            overlay_chart = "<p>No overlay data available for Loss Curves Overlay.</p>"

        return throughput_chart, loss_chart, runtime_chart, winner_chart, overlay_chart

    @staticmethod
    def _wrap_plotly_chart(title: str, figure_html: str) -> str:
        return (
            f"<h3>{escape(title)}</h3>"
            f"<div class='figure-html'>{figure_html}</div>"
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

    def _single_metric_rows(self, result: Any, run: dict[str, Any]) -> str:
        metrics = [
            ("Throughput (samples/sec)", result.metrics.throughput_samples_per_sec),
            ("Final Loss", run["final_loss"]),
            ("Total Time (s)", run["total_time_sec"]),
            ("Power (W)", result.metrics.power_consumption_watts),
            ("Performance/Watt", result.metrics.performance_per_watt),
            ("Energy/Step (J)", result.metrics.energy_per_step_joules),
            ("Temperature (C)", result.metrics.temperature_celsius),
            ("Communication Overhead (%)", result.metrics.communication_overhead_percent),
        ]
        return "".join(f"<tr><th>{escape(name)}</th><td>{escape(self._fmt(value))}</td></tr>" for name, value in metrics)

    def _config_rows(self, config: dict[str, Any]) -> str:
        keys = [
            "num_processes",
            "num_steps",
            "batch_size",
            "learning_rate",
            "hidden_size",
            "num_layers",
            "seed",
            "tdp_watts",
            "loss_tolerance",
            "regression_threshold",
        ]
        rows: list[str] = []
        for key in keys:
            if key in config:
                rows.append(f"<tr><th>{escape(key)}</th><td>{escape(self._fmt(config.get(key)))}</td></tr>")
        if not rows:
            return "<tr><td colspan='2'>No config available in tool payload.</td></tr>"
        return "".join(rows)

    def _consolidated_rows(self, runs: list[dict[str, Any]]) -> str:
        rows: list[str] = []
        for run in runs:
            config = run["config"]
            rows.append(
                "<tr>"
                f"<td>{escape(run['label'])}</td>"
                f"<td>{escape(self._fmt(config.get('num_processes', run.get('num_workers'))))}</td>"
                f"<td>{escape(self._fmt(config.get('num_steps')))}</td>"
                f"<td>{escape(self._fmt(config.get('batch_size')))}</td>"
                f"<td>{escape(self._fmt(config.get('learning_rate')))}</td>"
                f"<td>{escape(self._fmt(config.get('hidden_size')))}</td>"
                f"<td>{escape(self._fmt(config.get('num_layers')))}</td>"
                f"<td>{escape(self._fmt(run['throughput']))}</td>"
                f"<td>{escape(self._fmt(run['final_loss']))}</td>"
                f"<td>{escape(self._fmt(run['total_time_sec']))}</td>"
                f"<td>{escape(self._fmt(run['power_w']))}</td>"
                f"<td>{escape(self._fmt(run['perf_per_watt']))}</td>"
                "</tr>"
            )
        return "".join(rows)

    def _run_data(self, result: Any, default_label: str) -> dict[str, Any]:
        payload = result.tool_payload if isinstance(result.tool_payload, dict) else {}
        config = payload.get("config", {}) if isinstance(payload.get("config"), dict) else {}
        steps = payload.get("steps", []) if isinstance(payload.get("steps"), list) else []
        parsed_steps: list[dict[str, float]] = []
        for index, item in enumerate(steps):
            if not isinstance(item, dict):
                continue
            parsed_steps.append(
                {
                    "step": float(item.get("step", index)),
                    "loss": self._to_float(item.get("loss")),
                    "throughput": self._to_float(item.get("throughput_samples_per_sec")),
                    "allreduce_ms": self._to_float(item.get("allreduce_time_ms")),
                    "compute_ms": self._to_float(item.get("compute_time_ms")),
                }
            )

        if parsed_steps:
            final_loss = parsed_steps[-1]["loss"]
        else:
            final_loss = self._to_float(payload.get("final_loss"))
        total_time_sec = self._to_float(payload.get("total_time_sec"))

        step_points = [(point["step"], point["loss"]) for point in parsed_steps]
        throughput_points = [(point["step"], point["throughput"]) for point in parsed_steps]
        timing_points = [(point["step"], point["compute_ms"], point["allreduce_ms"]) for point in parsed_steps]

        workers = config.get("num_processes", payload.get("num_workers"))
        batch = config.get("batch_size", result.workload.batch_size)
        num_steps = config.get("num_steps", result.workload.steps)
        lr = config.get("learning_rate")
        label = f"w{workers}-b{batch}-s{num_steps}-lr{lr}" if workers is not None else default_label

        return {
            "label": str(label),
            "config": config,
            "num_workers": payload.get("num_workers"),
            "throughput": self._to_float(result.metrics.throughput_samples_per_sec),
            "final_loss": final_loss,
            "total_time_sec": total_time_sec,
            "power_w": self._to_float(result.metrics.power_consumption_watts),
            "perf_per_watt": self._to_float(result.metrics.performance_per_watt),
            "step_points": step_points,
            "throughput_points": throughput_points,
            "timing_points": timing_points,
        }

    def _winner_share(self, runs: list[dict[str, Any]]) -> dict[str, int]:
        scores = {run["label"]: 0 for run in runs}

        throughput_candidates = [(run["label"], run["throughput"]) for run in runs if run["throughput"] is not None]
        if throughput_candidates:
            scores[max(throughput_candidates, key=lambda item: item[1])[0]] += 1

        loss_candidates = [(run["label"], run["final_loss"]) for run in runs if run["final_loss"] is not None]
        if loss_candidates:
            scores[min(loss_candidates, key=lambda item: item[1])[0]] += 1

        runtime_candidates = [(run["label"], run["total_time_sec"]) for run in runs if run["total_time_sec"] is not None]
        if runtime_candidates:
            scores[min(runtime_candidates, key=lambda item: item[1])[0]] += 1

        perf_candidates = [(run["label"], run["perf_per_watt"]) for run in runs if run["perf_per_watt"] is not None]
        if perf_candidates:
            scores[max(perf_candidates, key=lambda item: item[1])[0]] += 1

        return scores

    def _bar_chart_svg(
        self,
        labels: list[str],
        values: list[float | None],
        *,
        title: str,
        y_label: str,
        color: str,
    ) -> str:
        rows = [(label, value) for label, value in zip(labels, values) if value is not None]
        if not rows:
            return f"<p>No data available for {escape(title)}.</p>"
        use_labels = [label for label, _ in rows]
        use_values = [float(value) for _, value in rows]

        width = 560
        height = 260
        margin_left = 48
        margin_bottom = 52
        chart_w = width - margin_left - 16
        chart_h = height - 20 - margin_bottom
        max_value = max(use_values) if max(use_values) > 0 else 1.0
        bar_gap = 8.0
        bar_width = max(10.0, (chart_w / max(1, len(use_values))) - bar_gap)

        parts: list[str] = [
            f"<p class='chart-note'>{escape(y_label)}</p>",
            f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' "
            "xmlns='http://www.w3.org/2000/svg'>",
            f"<line x1='{margin_left}' y1='20' x2='{margin_left}' y2='{20 + chart_h}' stroke='#6b7280'/>",
            f"<line x1='{margin_left}' y1='{20 + chart_h}' x2='{margin_left + chart_w}' y2='{20 + chart_h}' stroke='#6b7280'/>",
        ]
        for idx, value in enumerate(use_values):
            scaled = (value / max_value) * chart_h
            x = margin_left + idx * (bar_width + bar_gap)
            y = 20 + (chart_h - scaled)
            label = escape(use_labels[idx])
            parts.append(
                f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_width:.1f}' height='{scaled:.1f}' fill='{color}' opacity='0.85'/>"
            )
            parts.append(
                f"<text x='{x + bar_width / 2:.1f}' y='{max(12.0, y - 3.0):.1f}' text-anchor='middle' font-size='10'>{self._fmt(value)}</text>"
            )
            parts.append(
                f"<text x='{x + bar_width / 2:.1f}' y='{height - 22}' text-anchor='middle' font-size='10'>{label}</text>"
            )
        parts.append("</svg>")
        return f"<h3>{escape(title)}</h3>{''.join(parts)}"

    def _line_chart_svg(
        self,
        points: list[tuple[float, float | None]],
        *,
        title: str,
        y_label: str,
        color: str,
        show_points: bool,
    ) -> str:
        usable = [(x, float(y)) for x, y in points if y is not None]
        if len(usable) < 2:
            return f"<p>No line data available for {escape(title)}.</p>"

        width = 560
        height = 260
        margin_left = 48
        margin_bottom = 34
        chart_w = width - margin_left - 18
        chart_h = height - 20 - margin_bottom

        x_min = min(x for x, _ in usable)
        x_max = max(x for x, _ in usable)
        y_min = min(y for _, y in usable)
        y_max = max(y for _, y in usable)
        x_span = (x_max - x_min) or 1.0
        y_span = (y_max - y_min) or 1.0

        points_px: list[tuple[float, float]] = []
        for x, y in usable:
            px = margin_left + ((x - x_min) / x_span) * chart_w
            py = 20 + chart_h - ((y - y_min) / y_span) * chart_h
            points_px.append((px, py))
        polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points_px)

        parts: list[str] = [
            f"<h3>{escape(title)}</h3>",
            f"<p class='chart-note'>{escape(y_label)}</p>",
            f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>",
            f"<line x1='{margin_left}' y1='20' x2='{margin_left}' y2='{20 + chart_h}' stroke='#6b7280'/>",
            f"<line x1='{margin_left}' y1='{20 + chart_h}' x2='{margin_left + chart_w}' y2='{20 + chart_h}' stroke='#6b7280'/>",
            f"<polyline points='{polyline}' fill='none' stroke='{color}' stroke-width='2'/>",
        ]
        if show_points:
            for px, py in points_px:
                parts.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='2.5' fill='{color}'/>")
        parts.append("</svg>")
        return "".join(parts)

    def _multi_line_chart_svg(
        self,
        series: list[tuple[str, list[tuple[float, float | None]]]],
        *,
        title: str,
        y_label: str,
    ) -> str:
        colors = ["#2563eb", "#dc2626", "#0891b2", "#16a34a", "#7c3aed", "#ea580c"]
        usable_series: list[tuple[str, list[tuple[float, float]]]] = []
        for label, points in series:
            clean = [(x, float(y)) for x, y in points if y is not None]
            if len(clean) >= 2:
                usable_series.append((label, clean))
        if not usable_series:
            return f"<p>No overlay data available for {escape(title)}.</p>"

        width = 1120
        height = 300
        margin_left = 52
        margin_bottom = 36
        chart_w = width - margin_left - 18
        chart_h = height - 20 - margin_bottom
        all_x = [x for _, points in usable_series for x, _ in points]
        all_y = [y for _, points in usable_series for _, y in points]
        x_min = min(all_x)
        x_max = max(all_x)
        y_min = min(all_y)
        y_max = max(all_y)
        x_span = (x_max - x_min) or 1.0
        y_span = (y_max - y_min) or 1.0

        parts: list[str] = [
            f"<h3>{escape(title)}</h3>",
            f"<p class='chart-note'>{escape(y_label)}</p>",
            f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>",
            f"<line x1='{margin_left}' y1='20' x2='{margin_left}' y2='{20 + chart_h}' stroke='#6b7280'/>",
            f"<line x1='{margin_left}' y1='{20 + chart_h}' x2='{margin_left + chart_w}' y2='{20 + chart_h}' stroke='#6b7280'/>",
        ]
        legend_rows: list[str] = []
        for idx, (label, points) in enumerate(usable_series):
            color = colors[idx % len(colors)]
            coords: list[str] = []
            for x, y in points:
                px = margin_left + ((x - x_min) / x_span) * chart_w
                py = 20 + chart_h - ((y - y_min) / y_span) * chart_h
                coords.append(f"{px:.1f},{py:.1f}")
            parts.append(
                f"<polyline points='{' '.join(coords)}' fill='none' stroke='{color}' stroke-width='2' opacity='0.9'/>"
            )
            legend_rows.append(
                f"<li><span class='swatch' style='background:{color}'></span>{escape(label)}</li>"
            )
        parts.append("</svg>")
        return "".join(parts) + f"<ul class='legend'>{''.join(legend_rows)}</ul>"

    def _stacked_timing_svg(self, points: list[tuple[float, float | None, float | None]], *, title: str) -> str:
        usable = [(step, compute, allreduce) for step, compute, allreduce in points if compute is not None and allreduce is not None]
        if not usable:
            return f"<p>No timing data available for {escape(title)}.</p>"

        labels = [str(int(step)) for step, _, _ in usable]
        totals = [float(compute) + float(allreduce) for _, compute, allreduce in usable]
        if max(totals) <= 0:
            return f"<p>No positive timing values for {escape(title)}.</p>"

        width = 1120
        height = 280
        margin_left = 52
        margin_bottom = 50
        chart_w = width - margin_left - 16
        chart_h = height - 20 - margin_bottom
        bar_gap = 6.0
        bar_width = max(8.0, (chart_w / max(1, len(usable))) - bar_gap)
        max_total = max(totals)

        parts: list[str] = [
            f"<h3>{escape(title)}</h3>",
            "<p class='chart-note'>Compute (blue) + AllReduce (red), stacked per step.</p>",
            f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>",
            f"<line x1='{margin_left}' y1='20' x2='{margin_left}' y2='{20 + chart_h}' stroke='#6b7280'/>",
            f"<line x1='{margin_left}' y1='{20 + chart_h}' x2='{margin_left + chart_w}' y2='{20 + chart_h}' stroke='#6b7280'/>",
        ]
        for idx, (_, compute, allreduce) in enumerate(usable):
            x = margin_left + idx * (bar_width + bar_gap)
            comp_h = (float(compute) / max_total) * chart_h
            all_h = (float(allreduce) / max_total) * chart_h
            base_y = 20 + chart_h
            comp_y = base_y - comp_h
            all_y = comp_y - all_h
            parts.append(
                f"<rect x='{x:.1f}' y='{comp_y:.1f}' width='{bar_width:.1f}' height='{comp_h:.1f}' fill='#2563eb' opacity='0.85'/>"
            )
            parts.append(
                f"<rect x='{x:.1f}' y='{all_y:.1f}' width='{bar_width:.1f}' height='{all_h:.1f}' fill='#dc2626' opacity='0.85'/>"
            )
            parts.append(
                f"<text x='{x + bar_width / 2:.1f}' y='{height - 22}' text-anchor='middle' font-size='9'>{escape(labels[idx])}</text>"
            )
        parts.append("</svg>")
        parts.append(
            "<ul class='legend'>"
            "<li><span class='swatch' style='background:#2563eb'></span>compute_ms</li>"
            "<li><span class='swatch' style='background:#dc2626'></span>allreduce_ms</li>"
            "</ul>"
        )
        return "".join(parts)

    def _pie_div(self, counts: dict[str, int], *, title: str, labels: dict[str, str]) -> str:
        total = sum(counts.values())
        if total <= 0:
            return f"<p>No data available for {escape(title)}.</p>"

        palette = ["#2563eb", "#dc2626", "#0891b2", "#16a34a", "#7c3aed", "#ea580c"]
        gradient_parts: list[str] = []
        legend_parts: list[str] = []
        start = 0.0
        for idx, key in enumerate(sorted(counts.keys())):
            count = counts[key]
            if count <= 0:
                continue
            color = palette[idx % len(palette)]
            span = (count / total) * 360.0
            end = start + span
            gradient_parts.append(f"{color} {start:.2f}deg {end:.2f}deg")
            legend_parts.append(
                "<li>"
                f"<span class='swatch' style='background:{color}'></span>"
                f"{escape(labels.get(key, key))}: {count}"
                "</li>"
            )
            start = end

        gradient = ", ".join(gradient_parts) if gradient_parts else "#d1d5db 0deg 360deg"
        return (
            f"<h3>{escape(title)}</h3>"
            "<div class='pie-wrap'>"
            f"<div class='pie-shell' style='background: conic-gradient({gradient});'>"
            f"<div class='pie-hole'>{total}</div>"
            "</div>"
            f"<ul class='legend'>{''.join(legend_parts)}</ul>"
            "</div>"
        )

    @staticmethod
    def _fmt(value: Any) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, str):
            return value
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        return f"{number:.4f}"

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
