"""HTML report generation for performance analysis."""

import base64
import io
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

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
        self.figures: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        self.sections: List[Dict[str, Any]] = []
        self.summary_items: List[Dict[str, Any]] = []
        self.tables: List[Dict[str, Any]] = []
        self.interactive_charts: List[Dict[str, Any]] = []
        self._chart_id_counter = 0

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
        self.figures.append({
            "figure": fig,
            "caption": caption,
            "section": section,
            "description": description,
        })

    def add_figures_from_visualizer(
        self,
        visualizer,
        captions: Optional[List[str]] = None,
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
        self.summary_items.append({
            "label": label,
            "value": value,
            "unit": unit,
            "status": status,
        })

    def add_table(
        self,
        title: str,
        headers: List[str],
        rows: List[List[Any]],
        section: str = "General",
    ) -> None:
        """Add a data table to the report.

        Args:
            title: Table title
            headers: Column headers
            rows: Table data rows
            section: Section to place the table in
        """
        self.tables.append({
            "title": title,
            "headers": headers,
            "rows": rows,
            "section": section,
        })

    def _get_next_chart_id(self) -> str:
        """Generate unique chart ID."""
        self._chart_id_counter += 1
        return f"chart_{self._chart_id_counter}"

    def add_interactive_line_chart(
        self,
        labels: List[str],
        datasets: List[Dict[str, Any]],
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
        self.interactive_charts.append({
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
        })

    def add_interactive_bar_chart(
        self,
        labels: List[str],
        datasets: List[Dict[str, Any]],
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
        self.interactive_charts.append({
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
        })

    def add_interactive_scatter_chart(
        self,
        datasets: List[Dict[str, Any]],
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
        self.interactive_charts.append({
            "id": self._get_next_chart_id(),
            "type": "scatter",
            "datasets": datasets,
            "title": title,
            "section": section,
            "description": description,
            "x_label": x_label,
            "y_label": y_label,
            "enable_zoom": enable_zoom,
        })

    def add_interactive_pie_chart(
        self,
        labels: List[str],
        data: List[float],
        title: str = "",
        section: str = "General",
        description: str = "",
        colors: Optional[List[str]] = None,
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
        self.interactive_charts.append({
            "id": self._get_next_chart_id(),
            "type": "doughnut" if doughnut else "pie",
            "labels": labels,
            "data": data,
            "title": title,
            "section": section,
            "description": description,
            "colors": colors,
        })

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
            accent_color = "#e94560"
            secondary_color = "#0f3460"
            table_header_bg = "#0f3460"
            table_row_hover = "#1a1a4e"
        else:
            bg_color = "#f5f7fa"
            card_bg = "#ffffff"
            text_color = "#2d3748"
            border_color = "#e2e8f0"
            accent_color = "#4299e1"
            secondary_color = "#edf2f7"
            table_header_bg = "#4299e1"
            table_row_hover = "#f7fafc"

        return f"""
        :root {{
            --bg-color: {bg_color};
            --card-bg: {card_bg};
            --text-color: {text_color};
            --border-color: {border_color};
            --accent-color: {accent_color};
            --secondary-color: {secondary_color};
            --table-header-bg: {table_header_bg};
            --table-row-hover: {table_row_hover};
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}

        /* Header */
        .header {{
            background: linear-gradient(135deg, var(--accent-color), #805ad5);
            color: white;
            padding: 40px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}

        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
        }}

        .header .subtitle {{
            font-size: 1.1rem;
            opacity: 0.9;
        }}

        .header .meta {{
            margin-top: 20px;
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }}

        .header .meta-item {{
            background: rgba(255, 255, 255, 0.2);
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 0.9rem;
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
        }}

        .nav ul {{
            list-style: none;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
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
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 25px;
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

        .figure-caption {{
            padding: 15px;
            text-align: center;
        }}

        .figure-caption h4 {{
            font-size: 1rem;
            margin-bottom: 5px;
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

        /* Print Styles */
        @media print {{
            .nav {{
                display: none;
            }}
            .section {{
                break-inside: avoid;
            }}
            .figure-card {{
                break-inside: avoid;
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
        }}
        """

    def _generate_nav_html(self, section_names: List[str]) -> str:
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
        </nav>
        """

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

    def _generate_figures_html(self, figures: List[Dict], section_name: str) -> str:
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

    def _generate_tables_html(self, tables: List[Dict]) -> str:
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
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        # Organize figures by section
        section_figures: Dict[str, List[Dict]] = {}
        for fig in self.figures:
            section = fig.get("section", "General")
            if section not in section_figures:
                section_figures[section] = []
            section_figures[section].append(fig)

        # Organize tables by section
        section_tables: Dict[str, List[Dict]] = {}
        for table in self.tables:
            section = table.get("section", "General")
            if section not in section_tables:
                section_tables[section] = []
            section_tables[section].append(table)

        # Get all unique section names
        all_sections = set(section_figures.keys()) | set(section_tables.keys())
        for s in self.sections:
            all_sections.add(s["name"])
        section_names = sorted(all_sections)

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
            tables_html = self._generate_tables_html(
                section_tables.get(section_name, [])
            )

            if figures_html or tables_html:
                sections_html.append(f"""
                <section class="section" id="{anchor}">
                    <div class="section-header">
                        <h2>{section_name}</h2>
                        {desc_html}
                    </div>
                    <div class="section-content">
                        {figures_html}
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
    </script>
</body>
</html>
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        return output_path

    def clear(self) -> None:
        """Clear all figures, tables, and summary items."""
        self.figures = []
        self.tables = []
        self.summary_items = []
        self.sections = []
        self.metadata = {}


__all__ = ["HTMLReportGenerator"]
