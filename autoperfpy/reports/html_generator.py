"""HTML report generation for performance analysis."""

import base64
import io
import os
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt


class HTMLReportGenerator:
    """Generate interactive HTML reports from performance analysis."""

    def __init__(
        self,
        title: str = "Performance Analysis Report",
        author: str = "AutoPerfPy",
        theme: str = "light",
    ):
        """Initialize HTML report generator.

        Args:
            title: Report title
            author: Report author name
            theme: Color theme ('light' or 'dark')
        """
        self.title = title
        self.author = author
        self.theme = theme
        self.figures: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {}
        self.sections: list[dict[str, Any]] = []
        self.summary_items: list[dict[str, Any]] = []
        self.tables: list[dict[str, Any]] = []
        self.interactive_charts: list[dict[str, Any]] = []
        self.html_figures: list[dict[str, Any]] = []
        self._chart_id_counter = 0

    def add_html_figure(
        self,
        html_content: str,
        caption: str = "",
        section: str = "General",
        description: str = "",
    ) -> None:
        """Add an HTML figure fragment (e.g. Plotly chart) to the report.

        Use this to embed charts that match the UI (e.g. Plotly.to_html()).
        Plotly.js will be included automatically when any HTML figures are present.

        Args:
            html_content: Raw HTML fragment (div + script from e.g. fig.to_html())
            caption: Caption for the figure
            section: Section name to group the figure under
            description: Additional description text
        """
        self.html_figures.append(
            {
                "html": html_content,
                "caption": caption,
                "section": section,
                "description": description,
            }
        )

    def add_multi_run_comparison(
        self,
        runs: list[dict[str, Any]],
        run_names: list[str] | None = None,
        section: str = "Comparative Analysis",
        description: str = "Side-by-side comparison across multiple runs.",
        chart_engine: str = "plotly",
    ) -> None:
        """Add multi-run comparison charts and summary table to the report.

        Args:
            runs: List of collector export dictionaries (each with summary data)
            run_names: Optional list of display names for runs
            section: Section name to group comparison outputs
            description: Section description text
            chart_engine: "plotly" (default) or "chartjs"
        """
        if not runs or len(runs) < 2:
            return

        if run_names is None:
            run_names = [r.get("run_label") or r.get("collector_name") or f"Run {i+1}" for i, r in enumerate(runs)]
        elif len(run_names) != len(runs):
            padded = list(run_names)
            for i in range(len(padded), len(runs)):
                padded.append(f"Run {i+1}")
            run_names = padded[: len(runs)]

        def _float_or_none(value: Any) -> float | None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def _format_cell(value: Any, digits: int | None = None) -> str:
            if value is None or value == "":
                return "-"
            try:
                num = float(value)
            except (TypeError, ValueError):
                return str(value)
            if digits is None:
                if num.is_integer():
                    return str(int(num))
                return str(num)
            return f"{num:.{digits}f}"

        latency_p50: list[float | None] = []
        latency_p95: list[float | None] = []
        latency_p99: list[float | None] = []
        throughput_mean: list[float | None] = []
        comparison_rows: list[list[Any]] = []

        for i, run in enumerate(runs):
            summary = run.get("summary", {}) or {}
            pm = run.get("platform_metadata") or {}
            inf = run.get("inference_config") or {}
            lat = summary.get("latency", {}) or {}
            thr = summary.get("throughput", {}) or {}
            pwr = summary.get("power", {}) or {}
            temp = summary.get("temperature", {}) or {}

            latency_p50.append(_float_or_none(lat.get("p50_ms")))
            latency_p95.append(_float_or_none(lat.get("p95_ms")))
            latency_p99.append(_float_or_none(lat.get("p99_ms")))
            throughput_mean.append(_float_or_none(thr.get("mean_fps")))

            comparison_rows.append(
                [
                    run_names[i],
                    pm.get("device_name") or inf.get("accelerator") or "",
                    inf.get("precision") or "",
                    _format_cell(inf.get("batch_size")),
                    _format_cell(summary.get("sample_count")),
                    _format_cell(summary.get("duration_seconds"), 1),
                    _format_cell(lat.get("p99_ms"), 2),
                    _format_cell(thr.get("mean_fps"), 2),
                    _format_cell(pwr.get("mean_w"), 2),
                    _format_cell(temp.get("max_c"), 1),
                ]
            )

        section_added = False

        def _ensure_section() -> None:
            nonlocal section_added
            if not section_added:
                self.add_section(section, description)
                section_added = True

        def _plotly_to_html(fig) -> str:
            fig.update_layout(
                autosize=True,
                height=380,
                margin=dict(l=50, r=50, t=50, b=50),
            )
            plotly_id = f"plotly_{self._get_next_chart_id()}"
            return fig.to_html(
                full_html=False,
                include_plotlyjs=False,
                div_id=plotly_id,
                config={
                    "responsive": True,
                    "displayModeBar": True,
                    "displaylogo": False,
                    "scrollZoom": True,
                },
            )

        chart_engine = chart_engine.lower().strip()
        used_plotly = False
        if chart_engine != "chartjs":
            try:
                from autoperfpy.reports import charts as shared_charts
            except ImportError:
                shared_charts = None

            if shared_charts and shared_charts.is_available():
                fig = shared_charts.create_latency_comparison_bar(runs, run_names)
                if fig:
                    _ensure_section()
                    self.add_html_figure(
                        _plotly_to_html(fig),
                        caption="Latency Percentiles Across Runs",
                        section=section,
                        description="P50/P95/P99 latency per run.",
                    )
                    used_plotly = True
                fig = shared_charts.create_throughput_comparison_bar(runs, run_names)
                if fig:
                    _ensure_section()
                    self.add_html_figure(
                        _plotly_to_html(fig),
                        caption="Throughput Comparison",
                        section=section,
                        description="Mean throughput per run.",
                    )
                    used_plotly = True

        if chart_engine == "chartjs" or not used_plotly:
            has_latency = any(v is not None for v in latency_p50 + latency_p95 + latency_p99)
            has_thr = any(v is not None for v in throughput_mean)
            if has_latency:
                _ensure_section()
                self.add_interactive_bar_chart(
                    labels=run_names,
                    datasets=[
                        {"label": "P50", "data": latency_p50},
                        {"label": "P95", "data": latency_p95},
                        {"label": "P99", "data": latency_p99},
                    ],
                    title="Latency Percentiles Across Runs",
                    section=section,
                    description="P50/P95/P99 latency by run.",
                    x_label="Run",
                    y_label="Latency (ms)",
                )
            if has_thr:
                _ensure_section()
                self.add_interactive_bar_chart(
                    labels=run_names,
                    datasets=[{"label": "Mean Throughput", "data": throughput_mean}],
                    title="Throughput Comparison",
                    section=section,
                    description="Mean throughput per run.",
                    x_label="Run",
                    y_label="Throughput (FPS)",
                )

        if comparison_rows:
            _ensure_section()
            self.add_table(
                title="Summary Comparison",
                headers=[
                    "Run",
                    "Device",
                    "Precision",
                    "Batch",
                    "Samples",
                    "Duration (s)",
                    "P99 Latency (ms)",
                    "Mean Throughput (FPS)",
                    "Mean Power (W)",
                    "Max Temp (C)",
                ],
                rows=comparison_rows,
                section=section,
            )

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to report.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value

    def add_figure(
        self,
        fig: plt.Figure,
        caption: str = "",
        section: str = "General",
        description: str = "",
    ) -> None:
        """Add a figure to the report.

        Args:
            fig: Matplotlib figure to add
            caption: Caption for the figure
            section: Section name to group the figure under
            description: Additional description text
        """
        self.figures.append(
            {
                "figure": fig,
                "caption": caption,
                "section": section,
                "description": description,
            }
        )

    def add_figures_from_visualizer(
        self,
        visualizer,
        captions: list[str] | None = None,
        section: str = "General",
    ) -> None:
        """Add all figures from a PerformanceVisualizer.

        Args:
            visualizer: PerformanceVisualizer instance
            captions: Optional list of captions for figures
            section: Section name for all figures
        """
        for i, fig in enumerate(visualizer.figures):
            caption = captions[i] if captions and i < len(captions) else f"Graph {i+1}"
            self.add_figure(fig, caption, section)

    def add_section(self, name: str, description: str = "") -> None:
        """Add a section to the report.

        Args:
            name: Section name
            description: Section description
        """
        self.sections.append({"name": name, "description": description})

    def add_summary_item(
        self,
        label: str,
        value: Any,
        unit: str = "",
        status: str = "neutral",
    ) -> None:
        """Add an item to the executive summary.

        Args:
            label: Item label
            value: Item value
            unit: Unit of measurement
            status: Status indicator ('good', 'warning', 'critical', 'neutral')
        """
        self.summary_items.append(
            {
                "label": label,
                "value": value,
                "unit": unit,
                "status": status,
            }
        )

    def add_table(
        self,
        title: str,
        headers: list[str],
        rows: list[list[Any]],
        section: str = "General",
    ) -> None:
        """Add a data table to the report.

        Args:
            title: Table title
            headers: Column headers
            rows: Table data rows
            section: Section to place the table in
        """
        self.tables.append(
            {
                "title": title,
                "headers": headers,
                "rows": rows,
                "section": section,
            }
        )

    def _get_next_chart_id(self) -> str:
        """Generate unique chart ID."""
        self._chart_id_counter += 1
        return f"chart_{self._chart_id_counter}"

    def add_interactive_line_chart(
        self,
        labels: list[str],
        datasets: list[dict[str, Any]],
        title: str = "",
        section: str = "General",
        description: str = "",
        x_label: str = "",
        y_label: str = "",
        enable_zoom: bool = True,
    ) -> None:
        """Add an interactive line chart with hover values, zoom, and filter.

        Args:
            labels: X-axis labels
            datasets: List of dataset dicts with 'label', 'data', and optional 'color'
            title: Chart title
            section: Section name
            description: Chart description
            x_label: X-axis label
            y_label: Y-axis label
            enable_zoom: Enable zoom/pan functionality
        """
        self.interactive_charts.append(
            {
                "id": self._get_next_chart_id(),
                "type": "line",
                "labels": labels,
                "datasets": datasets,
                "title": title,
                "section": section,
                "description": description,
                "x_label": x_label,
                "y_label": y_label,
                "enable_zoom": enable_zoom,
            }
        )

    def add_interactive_bar_chart(
        self,
        labels: list[str],
        datasets: list[dict[str, Any]],
        title: str = "",
        section: str = "General",
        description: str = "",
        x_label: str = "",
        y_label: str = "",
        horizontal: bool = False,
        stacked: bool = False,
    ) -> None:
        """Add an interactive bar chart with hover values and filter.

        Args:
            labels: Category labels
            datasets: List of dataset dicts with 'label', 'data', and optional 'color'
            title: Chart title
            section: Section name
            description: Chart description
            x_label: X-axis label
            y_label: Y-axis label
            horizontal: Use horizontal bars
            stacked: Stack bars
        """
        self.interactive_charts.append(
            {
                "id": self._get_next_chart_id(),
                "type": "bar",
                "labels": labels,
                "datasets": datasets,
                "title": title,
                "section": section,
                "description": description,
                "x_label": x_label,
                "y_label": y_label,
                "horizontal": horizontal,
                "stacked": stacked,
            }
        )

    def add_interactive_scatter_chart(
        self,
        datasets: list[dict[str, Any]],
        title: str = "",
        section: str = "General",
        description: str = "",
        x_label: str = "",
        y_label: str = "",
        enable_zoom: bool = True,
    ) -> None:
        """Add an interactive scatter chart with hover values and zoom.

        Args:
            datasets: List of dataset dicts with 'label', 'data' (list of {x, y} points),
                     and optional 'color'
            title: Chart title
            section: Section name
            description: Chart description
            x_label: X-axis label
            y_label: Y-axis label
            enable_zoom: Enable zoom/pan functionality
        """
        self.interactive_charts.append(
            {
                "id": self._get_next_chart_id(),
                "type": "scatter",
                "datasets": datasets,
                "title": title,
                "section": section,
                "description": description,
                "x_label": x_label,
                "y_label": y_label,
                "enable_zoom": enable_zoom,
            }
        )

    def add_interactive_pie_chart(
        self,
        labels: list[str],
        data: list[float],
        title: str = "",
        section: str = "General",
        description: str = "",
        colors: list[str] | None = None,
        doughnut: bool = False,
    ) -> None:
        """Add an interactive pie/doughnut chart with hover values.

        Args:
            labels: Slice labels
            data: Slice values
            title: Chart title
            section: Section name
            description: Chart description
            colors: Optional list of colors for slices
            doughnut: Use doughnut style instead of pie
        """
        self.interactive_charts.append(
            {
                "id": self._get_next_chart_id(),
                "type": "doughnut" if doughnut else "pie",
                "labels": labels,
                "data": data,
                "title": title,
                "section": section,
                "description": description,
                "colors": colors,
            }
        )

    def _fig_to_base64(self, fig: plt.Figure, format: str = "png", dpi: int = 150) -> str:
        """Convert matplotlib figure to base64 string.

        Args:
            fig: Matplotlib figure
            format: Image format (png, svg)
            dpi: Resolution for PNG

        Returns:
            Base64 encoded image string
        """
        buf = io.BytesIO()
        fig.savefig(buf, format=format, dpi=dpi, bbox_inches="tight", facecolor="white")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        return img_base64

    def _get_css_styles(self) -> str:
        """Get CSS styles based on theme.

        Returns:
            CSS stylesheet string
        """
        if self.theme == "dark":
            bg_color = "#1a1a2e"
            card_bg = "#16213e"
            text_color = "#eaeaea"
            border_color = "#0f3460"
            accent_color = "#0ea5e9"
            accent_secondary = "#14b8a6"
            secondary_color = "#0f3460"
            table_header_bg = "#0f3460"
            table_row_hover = "#1a1a4e"
        else:
            bg_color = "#f5f7fa"
            card_bg = "#ffffff"
            text_color = "#2d3748"
            border_color = "#e2e8f0"
            accent_color = "#0f6feb"
            accent_secondary = "#0ea5a4"
            secondary_color = "#edf2f7"
            table_header_bg = "#0f6feb"
            table_row_hover = "#f7fafc"

        return f"""
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

        :root {{
            --bg-color: {bg_color};
            --card-bg: {card_bg};
            --text-color: {text_color};
            --border-color: {border_color};
            --accent-color: {accent_color};
            --accent-secondary: {accent_secondary};
            --secondary-color: {secondary_color};
            --table-header-bg: {table_header_bg};
            --table-row-hover: {table_row_hover};
            --surface-strong: rgba(255, 255, 255, 0.14);
            --surface-soft: rgba(255, 255, 255, 0.08);
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            background-image:
                radial-gradient(circle at 2% 10%, rgba(15, 111, 235, 0.14) 0%, transparent 35%),
                radial-gradient(circle at 98% 5%, rgba(14, 165, 164, 0.12) 0%, transparent 30%);
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 28px;
        }}

        /* Header */
        .header {{
            background:
                linear-gradient(135deg, rgba(0, 0, 0, 0.05), rgba(255, 255, 255, 0.05)),
                linear-gradient(125deg, var(--accent-color), var(--accent-secondary));
            color: white;
            padding: 40px;
            border-radius: 18px;
            margin-bottom: 30px;
            box-shadow: 0 18px 36px rgba(15, 23, 42, 0.18);
            position: relative;
            overflow: hidden;
        }}

        .header::after {{
            content: '';
            position: absolute;
            width: 340px;
            height: 340px;
            right: -120px;
            top: -220px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.18);
            pointer-events: none;
        }}

        .header-top {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 16px;
            position: relative;
            z-index: 1;
        }}

        .header h1 {{
            font-family: 'Space Grotesk', 'IBM Plex Sans', sans-serif;
            font-size: 2.5rem;
            margin-bottom: 6px;
            letter-spacing: 0.01em;
        }}

        .header .subtitle {{
            font-size: 1.1rem;
            opacity: 0.94;
            max-width: 760px;
        }}

        .header .meta {{
            margin-top: 22px;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            position: relative;
            z-index: 1;
        }}

        .header .meta-item {{
            background: var(--surface-soft);
            border: 1px solid rgba(255, 255, 255, 0.22);
            padding: 8px 14px;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 500;
            backdrop-filter: blur(4px);
        }}

        .header-kpi-grid {{
            margin-top: 18px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 10px;
            position: relative;
            z-index: 1;
        }}

        .header-kpi-card {{
            background: var(--surface-strong);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 10px 12px;
            backdrop-filter: blur(4px);
            min-height: 72px;
        }}

        .header-kpi-label {{
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            opacity: 0.84;
        }}

        .header-kpi-value {{
            font-size: 1.2rem;
            font-weight: 700;
            line-height: 1.15;
            margin-top: 4px;
            word-break: break-word;
        }}

        .platform-strip {{
            margin-top: 16px;
            padding: 14px;
            border-radius: 14px;
            background: var(--surface-soft);
            border: 1px solid rgba(255, 255, 255, 0.2);
            position: relative;
            z-index: 1;
        }}

        .platform-strip h3 {{
            margin: 0 0 10px 0;
            font-family: 'Space Grotesk', 'IBM Plex Sans', sans-serif;
            font-size: 1rem;
            letter-spacing: 0.01em;
        }}

        .platform-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
        }}

        .platform-card {{
            background: var(--surface-strong);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.18);
            padding: 8px 10px;
        }}

        .platform-card .platform-label {{
            display: block;
            font-size: 0.72rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            opacity: 0.82;
        }}

        .platform-card .platform-value {{
            display: block;
            margin-top: 2px;
            font-size: 0.95rem;
            font-weight: 600;
            line-height: 1.25;
            word-break: break-word;
        }}

        /* Navigation */
        .nav {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 15px 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            position: sticky;
            top: 10px;
            z-index: 100;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 15px;
        }}

        .nav ul {{
            list-style: none;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 0;
            padding: 0;
        }}

        .nav a {{
            color: var(--text-color);
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 6px;
            transition: all 0.2s;
            font-size: 0.9rem;
        }}

        .nav a:hover {{
            background: var(--accent-color);
            color: white;
        }}

        /* Print Button */
        .print-btn {{
            display: flex;
            align-items: center;
            gap: 8px;
            background: var(--accent-color);
            color: white;
            border: none;
            padding: 10px 18px;
            border-radius: 6px;
            font-size: 0.9rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .print-btn:hover {{
            background: #0b5ec7;
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(15, 111, 235, 0.4);
        }}

        .print-btn svg {{
            width: 18px;
            height: 18px;
        }}

        /* Summary Cards */
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .summary-card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            border-left: 4px solid var(--accent-color);
        }}

        .summary-card.good {{
            border-left-color: #48bb78;
        }}

        .summary-card.warning {{
            border-left-color: #ecc94b;
        }}

        .summary-card.critical {{
            border-left-color: #f56565;
        }}

        .summary-card .label {{
            font-size: 0.85rem;
            color: var(--text-color);
            opacity: 0.7;
            margin-bottom: 5px;
        }}

        .summary-card .value {{
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--text-color);
        }}

        .summary-card .unit {{
            font-size: 0.9rem;
            opacity: 0.7;
            margin-left: 4px;
        }}

        /* Sections */
        .section {{
            background: var(--card-bg);
            border-radius: 12px;
            margin-bottom: 30px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }}

        .section-header {{
            background: var(--secondary-color);
            padding: 20px 25px;
            border-bottom: 1px solid var(--border-color);
        }}

        .section-header h2 {{
            font-size: 1.4rem;
            margin-bottom: 5px;
        }}

        .section-header p {{
            font-size: 0.9rem;
            opacity: 0.7;
        }}

        .section-content {{
            padding: 25px;
        }}

        /* Figures */
        .figure-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 25px;
        }}

        @media (max-width: 1200px) {{
            .figure-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .figure-card {{
            background: var(--secondary-color);
            border-radius: 8px;
            overflow: hidden;
        }}

        .figure-card img {{
            width: 100%;
            height: auto;
            display: block;
        }}

        .figure-html {{
            width: 100%;
            min-height: 400px;
            height: 400px;
            padding: 10px;
            background: #ffffff;
        }}

        .figure-html > div {{
            width: 100% !important;
            height: 100% !important;
        }}

        .figure-html .plotly, 
        .figure-html .js-plotly-plot {{
            width: 100% !important;
            height: 100% !important;
        }}

        .figure-html .plot-container {{
            width: 100% !important;
            height: 100% !important;
        }}

        .figure-html .main-svg {{
            width: 100% !important;
            height: 100% !important;
        }}

        .figure-html .svg-container {{
            width: 100% !important;
            height: 100% !important;
        }}

        /* Ensure Plotly modebar is visible */
        .figure-html .modebar {{
            top: 0 !important;
            right: 0 !important;
        }}

        .figure-html .modebar-container {{
            position: absolute !important;
            top: 5px !important;
            right: 5px !important;
        }}

        .figure-caption {{
            padding: 15px;
            text-align: center;
            background: rgba(0,0,0,0.03);
        }}

        .figure-caption h4 {{
            font-size: 1rem;
            margin-bottom: 5px;
            font-weight: 600;
        }}

        .figure-caption p {{
            font-size: 0.85rem;
            opacity: 0.7;
        }}

        /* Tables */
        .table-container {{
            overflow-x: auto;
            margin: 20px 0;
        }}

        .table-title {{
            font-size: 1.1rem;
            margin-bottom: 15px;
            font-weight: 600;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}

        th {{
            background: var(--table-header-bg);
            color: white;
            padding: 12px 15px;
            text-align: left;
            font-weight: 600;
        }}

        td {{
            padding: 12px 15px;
            border-bottom: 1px solid var(--border-color);
        }}

        tr:hover {{
            background: var(--table-row-hover);
        }}

        /* Footer */
        .footer {{
            text-align: center;
            padding: 30px;
            color: var(--text-color);
            opacity: 0.7;
            font-size: 0.9rem;
        }}

        /* Print Styles - Ensures PDF matches HTML appearance */
        @media print {{
            * {{
                -webkit-print-color-adjust: exact !important;
                print-color-adjust: exact !important;
                color-adjust: exact !important;
            }}
            
            body {{
                background: white !important;
                font-size: 11pt !important;
            }}
            
            .nav {{
                display: none !important;
            }}
            
            .container {{
                max-width: 100% !important;
                padding: 10px !important;
            }}
            
            .header {{
                background: linear-gradient(135deg, #0f6feb 0%, #0ea5a4 100%) !important;
                -webkit-print-color-adjust: exact !important;
                print-color-adjust: exact !important;
                padding: 25px !important;
                margin-bottom: 20px !important;
            }}
            
            .header h1 {{
                font-size: 1.8rem !important;
            }}
            
            .section {{
                break-inside: avoid;
                page-break-inside: avoid;
                margin-bottom: 15px !important;
            }}
            
            .section-header {{
                padding: 12px 15px !important;
            }}
            
            .section-content {{
                padding: 15px !important;
            }}
            
            .figure-card, .chart-card, .summary-card, .table-card {{
                break-inside: avoid;
                page-break-inside: avoid;
                box-shadow: none !important;
                border: 1px solid #ddd !important;
            }}
            
            .figure-grid, .chart-grid {{
                grid-template-columns: 1fr !important;
                gap: 15px !important;
            }}
            
            .figure-html {{
                min-height: 300px !important;
                width: 100% !important;
            }}
            
            .figure-html .plotly {{
                width: 100% !important;
                min-height: 300px !important;
            }}
            
            /* Ensure Plotly charts are visible in print */
            .js-plotly-plot, .plotly {{
                break-inside: avoid;
                page-break-inside: avoid;
            }}
            
            .footer {{
                display: none !important;
            }}
            
            /* Summary cards in print */
            .summary-grid {{
                display: grid !important;
                grid-template-columns: repeat(4, 1fr) !important;
                gap: 10px !important;
            }}
            
            .summary-card {{
                padding: 12px !important;
            }}
            
            .summary-card .value {{
                font-size: 1.4rem !important;
            }}
            
            /* Ensure table is readable */
            table {{
                font-size: 9pt !important;
            }}
            
            th, td {{
                padding: 8px 10px !important;
            }}
            
            /* Page breaks for major sections */
            .section:not(:first-of-type) {{
                page-break-before: auto;
            }}
            
            /* Hide chart controls in print */
            .chart-controls {{
                display: none !important;
            }}
            
            /* Ensure metadata section prints nicely */
            .metadata-grid {{
                grid-template-columns: repeat(2, 1fr) !important;
            }}
        }}

        /* Responsive */
        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 1.8rem;
            }}
            .figure-grid {{
                grid-template-columns: 1fr;
            }}
            .summary-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
            .chart-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        /* Interactive Charts */
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 25px;
            margin-top: 25px;
        }}

        .chart-card {{
            background: var(--secondary-color);
            border-radius: 8px;
            overflow: hidden;
            padding: 20px;
        }}

        .chart-card .chart-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            flex-wrap: wrap;
            gap: 10px;
        }}

        .chart-card .chart-title {{
            font-size: 1rem;
            font-weight: 600;
            margin: 0;
        }}

        .chart-card .chart-controls {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}

        .chart-card .chart-btn {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            color: var(--text-color);
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .chart-card .chart-btn:hover {{
            background: var(--accent-color);
            color: white;
            border-color: var(--accent-color);
        }}

        .chart-card .chart-btn.active {{
            background: var(--accent-color);
            color: white;
            border-color: var(--accent-color);
        }}

        .chart-container {{
            position: relative;
            height: 350px;
            width: 100%;
        }}

        .chart-card .chart-description {{
            font-size: 0.85rem;
            opacity: 0.7;
            margin-top: 12px;
            text-align: center;
        }}

        .chart-card .zoom-info {{
            font-size: 0.75rem;
            opacity: 0.6;
            text-align: center;
            margin-top: 8px;
        }}
        """

    def _generate_nav_html(self, section_names: list[str]) -> str:
        """Generate navigation HTML.

        Args:
            section_names: List of section names

        Returns:
            Navigation HTML string
        """
        nav_items = ['<li><a href="#summary">Summary</a></li>']
        for name in section_names:
            anchor = name.lower().replace(" ", "-")
            nav_items.append(f'<li><a href="#{anchor}">{name}</a></li>')
        nav_items.append('<li><a href="#metadata">Report Info</a></li>')

        return f"""
        <nav class="nav">
            <ul>
                {''.join(nav_items)}
            </ul>
            <button class="print-btn" onclick="window.print()" title="Print or save as PDF">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="6 9 6 2 18 2 18 9"></polyline>
                    <path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"></path>
                    <rect x="6" y="14" width="12" height="8"></rect>
                </svg>
                Print / Save as PDF
            </button>
        </nav>
        """

    def _format_header_value(self, value: Any) -> str:
        """Format values for compact display in header cards/chips."""
        if value is None:
            return ""
        if isinstance(value, float):
            if value.is_integer():
                return str(int(value))
            return f"{value:.3f}".rstrip("0").rstrip(".")
        text = str(value).strip()
        return text if text else ""

    def _metadata_lookup(self) -> dict[str, tuple[str, str]]:
        """Return lowercase metadata index: lower-key -> (display key, display value)."""
        lookup: dict[str, tuple[str, str]] = {}
        for key, value in self.metadata.items():
            display_key = str(key).strip()
            display_value = self._format_header_value(value)
            if not display_key or not display_value:
                continue
            lookup[display_key.lower()] = (display_key, display_value)
        return lookup

    def _build_header_meta_items(self, timestamp: str) -> list[tuple[str, str]]:
        """Select key metadata chips for the top hero panel."""
        lookup = self._metadata_lookup()
        selected: list[tuple[str, str]] = []
        used: set[str] = set()

        preferred_keys = [
            "Collector",
            "Run Label",
            "Data Source",
            "Profile",
            "Device",
            "Accelerator",
            "Precision",
            "Batch Size",
            "Run Count",
            "Run Labels",
        ]
        for key in preferred_keys:
            hit = lookup.get(key.lower())
            if not hit:
                continue
            marker = hit[0].lower()
            if marker in used:
                continue
            selected.append(hit)
            used.add(marker)

        for key, value in self.metadata.items():
            key_text = str(key).strip()
            marker = key_text.lower()
            value_text = self._format_header_value(value)
            if not key_text or not value_text or marker in used:
                continue
            selected.append((key_text, value_text))
            used.add(marker)
            if len(selected) >= 10:
                break

        selected.append(("Generated", timestamp))
        return selected

    def _build_header_kpis(self) -> list[tuple[str, str]]:
        """Build compact top KPI cards using the same summary cards as Streamlit report."""
        if not self.summary_items:
            return []

        def _render_summary_value(item: dict[str, Any]) -> str:
            value_text = self._format_header_value(item.get("value"))
            if not value_text:
                return ""
            unit = self._format_header_value(item.get("unit"))
            return f"{value_text} {unit}".strip()

        by_label: dict[str, dict[str, Any]] = {}
        for item in self.summary_items:
            label = self._format_header_value(item.get("label"))
            if label:
                by_label[label] = item

        preferred = [
            "Samples",
            "Runs",
            "P99 Latency",
            "Mean Throughput",
            "Mean Power",
            "Duration",
            "Total Samples",
        ]
        cards: list[tuple[str, str]] = []
        used_labels: set[str] = set()
        for label in preferred:
            item = by_label.get(label)
            if not item:
                continue
            value = _render_summary_value(item)
            if not value:
                continue
            cards.append((label, value))
            used_labels.add(label)
            if len(cards) >= 6:
                return cards

        for item in self.summary_items:
            label = self._format_header_value(item.get("label"))
            if not label or label in used_labels:
                continue
            value = _render_summary_value(item)
            if not value:
                continue
            cards.append((label, value))
            used_labels.add(label)
            if len(cards) >= 6:
                break
        return cards

    def _build_platform_snapshot(self) -> list[tuple[str, str]]:
        """Extract platform-oriented metadata fields for the top metadata strip."""
        lookup = self._metadata_lookup()

        def _pick(*keys: str) -> str:
            for key in keys:
                hit = lookup.get(key.lower())
                if hit:
                    return hit[1]
            return ""

        entries = [
            ("Device", _pick("Device")),
            ("Accelerator", _pick("Accelerator")),
            ("GPU", _pick("GPU")),
            ("CPU", _pick("CPU")),
            ("Precision", _pick("Precision")),
            ("Batch", _pick("Batch Size")),
            ("SoC", _pick("SoC")),
            ("Power Mode", _pick("Power Mode")),
            ("OS", _pick("OS")),
        ]
        return [(label, value) for label, value in entries if value]

    def _generate_summary_html(self) -> str:
        """Generate summary section HTML.

        Returns:
            Summary HTML string
        """
        if not self.summary_items:
            return ""

        cards_html = []
        for item in self.summary_items:
            status_class = item.get("status", "neutral")
            unit_html = f'<span class="unit">{item["unit"]}</span>' if item.get("unit") else ""
            cards_html.append(f"""
            <div class="summary-card {status_class}">
                <div class="label">{item['label']}</div>
                <div class="value">{item['value']}{unit_html}</div>
            </div>
            """)

        return f"""
        <section class="section" id="summary">
            <div class="section-header">
                <h2>Executive Summary</h2>
                <p>Key performance metrics at a glance</p>
            </div>
            <div class="section-content">
                <div class="summary-grid">
                    {''.join(cards_html)}
                </div>
            </div>
        </section>
        """

    def _generate_figures_html(self, figures: list[dict], section_name: str) -> str:
        """Generate figures HTML for a section.

        Args:
            figures: List of figure dictionaries
            section_name: Section name

        Returns:
            Figures HTML string
        """
        if not figures:
            return ""

        figure_cards = []
        for fig_data in figures:
            img_base64 = self._fig_to_base64(fig_data["figure"])
            caption = fig_data.get("caption", "")
            description = fig_data.get("description", "")

            desc_html = f"<p>{description}</p>" if description else ""
            figure_cards.append(f"""
            <div class="figure-card">
                <img src="data:image/png;base64,{img_base64}" alt="{caption}">
                <div class="figure-caption">
                    <h4>{caption}</h4>
                    {desc_html}
                </div>
            </div>
            """)

        return f"""
        <div class="figure-grid">
            {''.join(figure_cards)}
        </div>
        """

    def _generate_html_figures_html(self, html_figures_list: list[dict]) -> str:
        """Generate HTML for pre-rendered HTML figure fragments (e.g. Plotly).

        Args:
            html_figures_list: List of dicts with html, caption, description

        Returns:
            Figure grid HTML string
        """
        if not html_figures_list:
            return ""

        figure_cards = []
        for fig_data in html_figures_list:
            html_content = fig_data.get("html", "")
            caption = fig_data.get("caption", "")
            description = fig_data.get("description", "")
            desc_html = f"<p>{description}</p>" if description else ""
            figure_cards.append(f"""
            <div class="figure-card">
                <div class="figure-html">
                    {html_content}
                </div>
                <div class="figure-caption">
                    <h4>{caption}</h4>
                    {desc_html}
                </div>
            </div>
            """)

        return f"""
        <div class="figure-grid">
            {''.join(figure_cards)}
        </div>
        """

    def _generate_tables_html(self, tables: list[dict]) -> str:
        """Generate tables HTML.

        Args:
            tables: List of table dictionaries

        Returns:
            Tables HTML string
        """
        if not tables:
            return ""

        tables_html = []
        for table in tables:
            headers_html = "".join(f"<th>{h}</th>" for h in table["headers"])
            rows_html = ""
            for row in table["rows"]:
                cells = "".join(f"<td>{cell}</td>" for cell in row)
                rows_html += f"<tr>{cells}</tr>"

            tables_html.append(f"""
            <div class="table-container">
                <h4 class="table-title">{table['title']}</h4>
                <table>
                    <thead>
                        <tr>{headers_html}</tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
            """)

        return "".join(tables_html)

    def _generate_metadata_html(self) -> str:
        """Generate metadata section HTML.

        Returns:
            Metadata HTML string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        meta_items = [
            f"<tr><td><strong>Generated</strong></td><td>{timestamp}</td></tr>",
            f"<tr><td><strong>Author</strong></td><td>{self.author}</td></tr>",
            f"<tr><td><strong>Total Figures</strong></td><td>{len(self.figures)}</td></tr>",
        ]

        for key, value in self.metadata.items():
            meta_items.append(f"<tr><td><strong>{key}</strong></td><td>{value}</td></tr>")

        return f"""
        <section class="section" id="metadata">
            <div class="section-header">
                <h2>Report Information</h2>
                <p>Report metadata and generation details</p>
            </div>
            <div class="section-content">
                <table>
                    <tbody>
                        {''.join(meta_items)}
                    </tbody>
                </table>
            </div>
        </section>
        """

    def _generate_interactive_charts_html(self, charts: list[dict]) -> str:
        """Generate HTML for interactive charts in a section.

        Args:
            charts: List of chart dictionaries

        Returns:
            Charts HTML string
        """
        if not charts:
            return ""

        chart_cards = []
        for chart in charts:
            chart_id = chart["id"]
            title = chart.get("title", "")
            description = chart.get("description", "")
            enable_zoom = chart.get("enable_zoom", False)

            desc_html = f'<p class="chart-description">{description}</p>' if description else ""

            # Add zoom controls for supported chart types
            zoom_controls = ""
            zoom_info = ""
            if enable_zoom and chart["type"] in ("line", "scatter"):
                zoom_controls = f"""
                <div class="chart-controls">
                    <button class="chart-btn" onclick="resetZoom('{chart_id}')">Reset Zoom</button>
                </div>
                """
                zoom_info = '<p class="zoom-info">Scroll to zoom, drag to pan</p>'

            chart_cards.append(f"""
            <div class="chart-card">
                <div class="chart-header">
                    <h4 class="chart-title">{title}</h4>
                    {zoom_controls}
                </div>
                <div class="chart-container">
                    <canvas id="{chart_id}"></canvas>
                </div>
                {desc_html}
                {zoom_info}
            </div>
            """)

        return f"""
        <div class="chart-grid">
            {''.join(chart_cards)}
        </div>
        """

    def _get_chart_color_palette(self) -> list[str]:
        """Get color palette for charts based on theme."""
        if self.theme == "dark":
            return [
                "rgba(233, 69, 96, 0.8)",  # Red
                "rgba(78, 205, 196, 0.8)",  # Teal
                "rgba(255, 209, 102, 0.8)",  # Yellow
                "rgba(149, 117, 205, 0.8)",  # Purple
                "rgba(100, 181, 246, 0.8)",  # Blue
                "rgba(129, 199, 132, 0.8)",  # Green
                "rgba(255, 138, 101, 0.8)",  # Orange
                "rgba(240, 98, 146, 0.8)",  # Pink
            ]
        else:
            return [
                "rgba(66, 153, 225, 0.8)",  # Blue
                "rgba(72, 187, 120, 0.8)",  # Green
                "rgba(237, 137, 54, 0.8)",  # Orange
                "rgba(159, 122, 234, 0.8)",  # Purple
                "rgba(245, 101, 101, 0.8)",  # Red
                "rgba(56, 178, 172, 0.8)",  # Teal
                "rgba(236, 201, 75, 0.8)",  # Yellow
                "rgba(237, 100, 166, 0.8)",  # Pink
            ]

    def _generate_chartjs_config(self, chart: dict) -> str:
        """Generate Chart.js configuration for a chart.

        Args:
            chart: Chart dictionary

        Returns:
            JavaScript configuration object string
        """
        import json

        chart_type = chart["type"]
        colors = self._get_chart_color_palette()

        # Common options
        common_opts = {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {
                "legend": {
                    "display": True,
                    "position": "top",
                    "labels": {
                        "usePointStyle": True,
                        "padding": 15,
                    },
                    "onClick": None,  # Will be replaced with function
                },
                "tooltip": {
                    "enabled": True,
                    "mode": "index",
                    "intersect": False,
                    "backgroundColor": "rgba(0, 0, 0, 0.8)",
                    "titleFont": {"size": 13},
                    "bodyFont": {"size": 12},
                    "padding": 12,
                    "cornerRadius": 6,
                },
            },
        }

        if chart_type == "line":
            datasets = []
            for i, ds in enumerate(chart.get("datasets", [])):
                color = ds.get("color", colors[i % len(colors)])
                datasets.append(
                    {
                        "label": ds.get("label", f"Series {i+1}"),
                        "data": ds.get("data", []),
                        "borderColor": color,
                        "backgroundColor": color.replace("0.8", "0.2"),
                        "borderWidth": 2,
                        "tension": 0.3,
                        "pointRadius": 4,
                        "pointHoverRadius": 6,
                        "fill": False,
                    }
                )

            config = {
                "type": "line",
                "data": {
                    "labels": chart.get("labels", []),
                    "datasets": datasets,
                },
                "options": {
                    **common_opts,
                    "scales": {
                        "x": {
                            "title": {
                                "display": bool(chart.get("x_label")),
                                "text": chart.get("x_label", ""),
                            },
                            "grid": {"display": True, "color": "rgba(0,0,0,0.1)"},
                        },
                        "y": {
                            "title": {
                                "display": bool(chart.get("y_label")),
                                "text": chart.get("y_label", ""),
                            },
                            "grid": {"display": True, "color": "rgba(0,0,0,0.1)"},
                        },
                    },
                },
            }

            if chart.get("enable_zoom"):
                config["options"]["plugins"]["zoom"] = {
                    "zoom": {
                        "wheel": {"enabled": True},
                        "pinch": {"enabled": True},
                        "mode": "xy",
                    },
                    "pan": {
                        "enabled": True,
                        "mode": "xy",
                    },
                }

        elif chart_type == "bar":
            datasets = []
            for i, ds in enumerate(chart.get("datasets", [])):
                color = ds.get("color", colors[i % len(colors)])
                datasets.append(
                    {
                        "label": ds.get("label", f"Series {i+1}"),
                        "data": ds.get("data", []),
                        "backgroundColor": color,
                        "borderColor": color.replace("0.8", "1"),
                        "borderWidth": 1,
                    }
                )

            config = {
                "type": "bar",
                "data": {
                    "labels": chart.get("labels", []),
                    "datasets": datasets,
                },
                "options": {
                    **common_opts,
                    "indexAxis": "y" if chart.get("horizontal") else "x",
                    "scales": {
                        "x": {
                            "stacked": chart.get("stacked", False),
                            "title": {
                                "display": bool(chart.get("x_label")),
                                "text": chart.get("x_label", ""),
                            },
                            "grid": {"display": True, "color": "rgba(0,0,0,0.1)"},
                        },
                        "y": {
                            "stacked": chart.get("stacked", False),
                            "title": {
                                "display": bool(chart.get("y_label")),
                                "text": chart.get("y_label", ""),
                            },
                            "grid": {"display": True, "color": "rgba(0,0,0,0.1)"},
                        },
                    },
                },
            }

        elif chart_type == "scatter":
            datasets = []
            for i, ds in enumerate(chart.get("datasets", [])):
                color = ds.get("color", colors[i % len(colors)])
                datasets.append(
                    {
                        "label": ds.get("label", f"Series {i+1}"),
                        "data": ds.get("data", []),
                        "backgroundColor": color,
                        "borderColor": color.replace("0.8", "1"),
                        "pointRadius": 6,
                        "pointHoverRadius": 8,
                    }
                )

            config = {
                "type": "scatter",
                "data": {"datasets": datasets},
                "options": {
                    **common_opts,
                    "scales": {
                        "x": {
                            "title": {
                                "display": bool(chart.get("x_label")),
                                "text": chart.get("x_label", ""),
                            },
                            "grid": {"display": True, "color": "rgba(0,0,0,0.1)"},
                        },
                        "y": {
                            "title": {
                                "display": bool(chart.get("y_label")),
                                "text": chart.get("y_label", ""),
                            },
                            "grid": {"display": True, "color": "rgba(0,0,0,0.1)"},
                        },
                    },
                },
            }

            if chart.get("enable_zoom"):
                config["options"]["plugins"]["zoom"] = {
                    "zoom": {
                        "wheel": {"enabled": True},
                        "pinch": {"enabled": True},
                        "mode": "xy",
                    },
                    "pan": {
                        "enabled": True,
                        "mode": "xy",
                    },
                }

        elif chart_type in ("pie", "doughnut"):
            chart_colors = chart.get("colors") or colors[: len(chart.get("data", []))]

            config = {
                "type": chart_type,
                "data": {
                    "labels": chart.get("labels", []),
                    "datasets": [
                        {
                            "data": chart.get("data", []),
                            "backgroundColor": chart_colors,
                            "borderWidth": 2,
                            "borderColor": ("#fff" if self.theme == "light" else "#16213e"),
                        }
                    ],
                },
                "options": {
                    **common_opts,
                    "plugins": {
                        **common_opts["plugins"],
                        "legend": {
                            "display": True,
                            "position": "right",
                            "labels": {
                                "usePointStyle": True,
                                "padding": 15,
                            },
                        },
                    },
                },
            }

        else:
            config = {"type": "bar", "data": {}, "options": {}}

        return json.dumps(config, indent=2)

    def _get_chartjs_scripts(self) -> str:
        """Generate Chart.js initialization scripts.

        Returns:
            JavaScript string for chart initialization
        """
        if not self.interactive_charts:
            return ""

        # CDN links for Chart.js and zoom plugin
        scripts = """
    <!-- Chart.js library -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1/dist/chartjs-plugin-zoom.min.js"></script>
    <script>
        // Store chart instances for interaction
        const chartInstances = {};

        // Reset zoom function
        function resetZoom(chartId) {
            if (chartInstances[chartId]) {
                chartInstances[chartId].resetZoom();
            }
        }

        // Toggle dataset visibility (filter)
        function toggleDataset(chartId, datasetIndex) {
            const chart = chartInstances[chartId];
            if (chart) {
                const meta = chart.getDatasetMeta(datasetIndex);
                meta.hidden = !meta.hidden;
                chart.update();
            }
        }

        // Initialize charts on DOM load
        document.addEventListener('DOMContentLoaded', function() {
"""

        # Generate initialization code for each chart
        for chart in self.interactive_charts:
            chart_id = chart["id"]
            config = self._generate_chartjs_config(chart)

            # Replace null onClick with actual function for legend filtering
            config_with_legend = config.replace(
                '"onClick": null',
                """\"onClick\": function(e, legendItem, legend) {
                    const index = legendItem.datasetIndex;
                    const chart = legend.chart;
                    const meta = chart.getDatasetMeta(index);
                    meta.hidden = !meta.hidden;
                    chart.update();
                }""",
            )

            scripts += f"""
            // Initialize {chart_id}
            (function() {{
                const ctx = document.getElementById('{chart_id}');
                if (ctx) {{
                    const config = {config_with_legend};
                    chartInstances['{chart_id}'] = new Chart(ctx, config);
                }}
            }})();
"""

        scripts += """
        });
    </script>
"""
        return scripts

    def generate_html(
        self,
        output_path: str,
        include_summary: bool = True,
        embed_images: bool = True,
    ) -> str:
        """Generate HTML report.

        Args:
            output_path: Path to save HTML file
            include_summary: Whether to include summary section
            embed_images: Whether to embed images as base64 (vs external files)

        Returns:
            Path to generated HTML file
        """
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )

        # Organize figures by section
        section_figures: dict[str, list[dict]] = {}
        for fig in self.figures:
            section = fig.get("section", "General")
            if section not in section_figures:
                section_figures[section] = []
            section_figures[section].append(fig)

        # Organize HTML figures by section (e.g. Plotly)
        section_html_figures: dict[str, list[dict]] = {}
        for fig in self.html_figures:
            section = fig.get("section", "General")
            if section not in section_html_figures:
                section_html_figures[section] = []
            section_html_figures[section].append(fig)

        # Organize tables by section
        section_tables: dict[str, list[dict]] = {}
        for table in self.tables:
            section = table.get("section", "General")
            if section not in section_tables:
                section_tables[section] = []
            section_tables[section].append(table)

        # Organize interactive charts by section
        section_charts: dict[str, list[dict]] = {}
        for chart in self.interactive_charts:
            section = chart.get("section", "General")
            if section not in section_charts:
                section_charts[section] = []
            section_charts[section].append(chart)

        # Get all unique section names
        all_sections = (
            set(section_figures.keys())
            | set(section_tables.keys())
            | set(section_charts.keys())
            | set(section_html_figures.keys())
        )
        for s in self.sections:
            all_sections.add(s["name"])

        # Define preferred section order for consistent layout
        preferred_order = [
            "Latency",
            "Memory",
            "Power & Thermal",
            "Throughput",
            "Utilization",
            "Summary Statistics",
            "Comparative Analysis",
            "General",
        ]

        # Sort sections: preferred order first, then alphabetical for any extras
        def section_sort_key(name):
            try:
                return (0, preferred_order.index(name))
            except ValueError:
                return (1, name)

        section_names = sorted(all_sections, key=section_sort_key)

        # Generate sections HTML
        sections_html = []
        for section_name in section_names:
            anchor = section_name.lower().replace(" ", "-")

            # Find section description
            desc = ""
            for s in self.sections:
                if s["name"] == section_name:
                    desc = s.get("description", "")
                    break

            desc_html = f"<p>{desc}</p>" if desc else ""

            figures_html = self._generate_figures_html(
                section_figures.get(section_name, []),
                section_name,
            )
            html_figures_html = self._generate_html_figures_html(section_html_figures.get(section_name, []))
            tables_html = self._generate_tables_html(section_tables.get(section_name, []))
            charts_html = self._generate_interactive_charts_html(section_charts.get(section_name, []))

            if figures_html or html_figures_html or tables_html or charts_html:
                sections_html.append(f"""
                <section class="section" id="{anchor}">
                    <div class="section-header">
                        <h2>{section_name}</h2>
                        {desc_html}
                    </div>
                    <div class="section-content">
                        {figures_html}
                        {html_figures_html}
                        {charts_html}
                        {tables_html}
                    </div>
                </section>
                """)

        # Generate meta items for header
        meta_html = ""
        for key, value in list(self.metadata.items())[:4]:  # Show first 4 in header
            meta_html += f'<span class="meta-item"><strong>{key}:</strong> {value}</span>'

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Complete HTML document
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="author" content="{self.author}">
    <meta name="generator" content="AutoPerfPy HTML Report Generator">
    <title>{self.title}</title>
    <style>
        {self._get_css_styles()}
    </style>
    {('<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>' if self.html_figures else '')}
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>{self.title}</h1>
            <p class="subtitle">Performance Analysis Report</p>
            <div class="meta">
                {meta_html}
                <span class="meta-item"><strong>Generated:</strong> {timestamp}</span>
            </div>
        </header>

        {self._generate_nav_html(section_names)}

        {self._generate_summary_html() if include_summary else ''}

        {''.join(sections_html)}

        {self._generate_metadata_html()}

        <footer class="footer">
            <p>Generated by AutoPerfPy HTML Report Generator</p>
            <p>© {datetime.now().year} {self.author}</p>
        </footer>
    </div>

    <script>
        // Smooth scrolling for navigation
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
            anchor.addEventListener('click', function (e) {{
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {{
                    target.scrollIntoView({{
                        behavior: 'smooth',
                        block: 'start'
                    }});
                }}
            }});
        }});

        // Highlight current section in navigation
        const sections = document.querySelectorAll('.section');
        const navLinks = document.querySelectorAll('.nav a');

        window.addEventListener('scroll', () => {{
            let current = '';
            sections.forEach(section => {{
                const sectionTop = section.offsetTop;
                if (scrollY >= sectionTop - 100) {{
                    current = section.getAttribute('id');
                }}
            }});

            navLinks.forEach(link => {{
                link.style.background = '';
                link.style.color = '';
                if (link.getAttribute('href') === '#' + current) {{
                    link.style.background = 'var(--accent-color)';
                    link.style.color = 'white';
                }}
            }});
        }});

        // Resize Plotly charts on window load and resize
        if (typeof Plotly !== 'undefined') {{
            window.addEventListener('load', function() {{
                document.querySelectorAll('.js-plotly-plot').forEach(function(gd) {{
                    Plotly.Plots.resize(gd);
                }});
            }});
            window.addEventListener('resize', function() {{
                document.querySelectorAll('.js-plotly-plot').forEach(function(gd) {{
                    Plotly.Plots.resize(gd);
                }});
            }});
        }}
    </script>
    {self._get_chartjs_scripts()}
</body>
</html>
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        return output_path

    def clear(self) -> None:
        """Clear all figures, tables, summary items, and interactive charts."""
        self.figures = []
        self.tables = []
        self.summary_items = []
        self.sections = []
        self.metadata = {}
        self.interactive_charts = []
        self.html_figures = []
        self._chart_id_counter = 0


__all__ = ["HTMLReportGenerator"]
