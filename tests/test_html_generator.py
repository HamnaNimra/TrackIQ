"""Tests for HTMLReportGenerator."""

# Use non-interactive backend so tests run without Tk (e.g. on Windows without tk/tcl)
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from autoperfpy.reporting import HTMLReportGenerator, PerformanceVisualizer

try:
    from autoperfpy.reports import charts as shared_charts

    CHARTS_AVAILABLE = shared_charts.is_available()
except ImportError:
    CHARTS_AVAILABLE = False
    shared_charts = None


class TestHTMLReportGenerator:
    """Tests for HTMLReportGenerator class."""

    @pytest.fixture
    def report(self):
        """Create a basic HTML report generator."""
        return HTMLReportGenerator(
            title="Test Report",
            author="Test Author",
            theme="light",
        )

    @pytest.fixture
    def sample_figure(self):
        """Create a sample matplotlib figure."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        ax.set_title("Test Plot")
        yield fig
        plt.close(fig)

    def test_init_default_values(self):
        """Test default initialization values."""
        report = HTMLReportGenerator()
        assert report.title == "Performance Analysis Report"
        assert report.author == "AutoPerfPy"
        assert report.theme == "light"
        assert report.figures == []
        assert report.metadata == {}

    def test_init_custom_values(self):
        """Test custom initialization values."""
        report = HTMLReportGenerator(
            title="Custom Title",
            author="Custom Author",
            theme="dark",
        )
        assert report.title == "Custom Title"
        assert report.author == "Custom Author"
        assert report.theme == "dark"

    def test_add_metadata(self, report):
        """Test adding metadata."""
        report.add_metadata("key1", "value1")
        report.add_metadata("key2", 123)

        assert report.metadata["key1"] == "value1"
        assert report.metadata["key2"] == 123

    def test_add_figure(self, report, sample_figure):
        """Test adding a figure."""
        report.add_figure(sample_figure, caption="Test Caption", section="Test Section")

        assert len(report.figures) == 1
        assert report.figures[0]["caption"] == "Test Caption"
        assert report.figures[0]["section"] == "Test Section"
        assert report.figures[0]["figure"] is sample_figure

    def test_add_figure_default_section(self, report, sample_figure):
        """Test adding a figure with default section."""
        report.add_figure(sample_figure, caption="Test Caption")

        assert report.figures[0]["section"] == "General"

    def test_add_section(self, report):
        """Test adding sections."""
        report.add_section("Section 1", "Description 1")
        report.add_section("Section 2", "Description 2")

        assert len(report.sections) == 2
        assert report.sections[0]["name"] == "Section 1"
        assert report.sections[0]["description"] == "Description 1"

    def test_add_summary_item(self, report):
        """Test adding summary items."""
        report.add_summary_item("Latency", "25.5", "ms", "good")
        report.add_summary_item("Errors", "5", "", "critical")

        assert len(report.summary_items) == 2
        assert report.summary_items[0]["label"] == "Latency"
        assert report.summary_items[0]["value"] == "25.5"
        assert report.summary_items[0]["unit"] == "ms"
        assert report.summary_items[0]["status"] == "good"

    def test_add_table(self, report):
        """Test adding data tables."""
        headers = ["Col1", "Col2", "Col3"]
        rows = [
            ["A", "B", "C"],
            ["D", "E", "F"],
        ]
        report.add_table("Test Table", headers, rows, "Test Section")

        assert len(report.tables) == 1
        assert report.tables[0]["title"] == "Test Table"
        assert report.tables[0]["headers"] == headers
        assert report.tables[0]["rows"] == rows
        assert report.tables[0]["section"] == "Test Section"

    def test_add_html_figure(self, report):
        """Test adding HTML figure fragments (e.g. Plotly)."""
        report.add_html_figure(
            "<div id='plotly-1'></div>",
            caption="Plotly Chart",
            section="Charts",
            description="A chart",
        )
        assert len(report.html_figures) == 1
        assert report.html_figures[0]["caption"] == "Plotly Chart"
        assert report.html_figures[0]["section"] == "Charts"
        assert "plotly-1" in report.html_figures[0]["html"]

    def test_add_interactive_line_chart(self, report):
        """Test adding Chart.js line chart."""
        report.add_interactive_line_chart(
            labels=["t1", "t2", "t3"],
            datasets=[
                {"label": "P50", "data": [22, 24, 23]},
                {"label": "P99", "data": [35, 38, 36]},
            ],
            title="Latency Over Time",
            section="Latency",
            description="Zoom enabled",
            x_label="Time",
            y_label="ms",
            enable_zoom=True,
        )
        assert len(report.interactive_charts) == 1
        ch = report.interactive_charts[0]
        assert ch["type"] == "line"
        assert ch["title"] == "Latency Over Time"
        assert ch["section"] == "Latency"
        assert ch["enable_zoom"] is True
        assert len(ch["datasets"]) == 2
        assert ch["labels"] == ["t1", "t2", "t3"]

    def test_add_interactive_bar_chart(self, report):
        """Test adding Chart.js bar chart."""
        report.add_interactive_bar_chart(
            labels=["A", "B", "C"],
            datasets=[{"label": "Val", "data": [10, 20, 15]}],
            title="Bar Chart",
            section="Compare",
            stacked=True,
        )
        assert len(report.interactive_charts) == 1
        ch = report.interactive_charts[0]
        assert ch["type"] == "bar"
        assert ch["stacked"] is True

    def test_fig_to_base64(self, report, sample_figure):
        """Test figure to base64 conversion."""
        base64_str = report._fig_to_base64(sample_figure)

        assert isinstance(base64_str, str)
        assert len(base64_str) > 0
        # Base64 encoded PNG should start with specific characters
        assert base64_str.startswith("iVBOR") or len(base64_str) > 100

    def test_get_css_styles_light(self, report):
        """Test CSS generation for light theme."""
        css = report._get_css_styles()

        assert "background-color" in css
        assert "#f5f7fa" in css  # Light theme background
        assert "body" in css
        assert ".header" in css

    def test_get_css_styles_dark(self):
        """Test CSS generation for dark theme."""
        report = HTMLReportGenerator(theme="dark")
        css = report._get_css_styles()

        assert "#1a1a2e" in css  # Dark theme background

    def test_generate_nav_html(self, report):
        """Test navigation HTML generation."""
        nav_html = report._generate_nav_html(["Section 1", "Section 2"])

        assert '<nav class="nav">' in nav_html
        assert 'href="#summary"' in nav_html
        assert 'href="#section-1"' in nav_html
        assert 'href="#section-2"' in nav_html
        assert 'href="#metadata"' in nav_html

    def test_generate_summary_html_empty(self, report):
        """Test summary HTML generation with no items."""
        summary_html = report._generate_summary_html()
        assert summary_html == ""

    def test_generate_summary_html_with_items(self, report):
        """Test summary HTML generation with items."""
        report.add_summary_item("Test", "100", "units", "good")
        summary_html = report._generate_summary_html()

        assert 'class="summary-card good"' in summary_html
        assert "Test" in summary_html
        assert "100" in summary_html

    def test_generate_html_creates_file(self, report, sample_figure, tmp_path):
        """Test that generate_html creates a file."""
        report.add_metadata("Test", "Value")
        report.add_figure(sample_figure, caption="Test Figure")

        output_path = tmp_path / "report.html"
        result = report.generate_html(str(output_path))

        assert result == str(output_path)
        assert output_path.exists()

    def test_generate_html_content(self, report, sample_figure, tmp_path):
        """Test generated HTML content."""
        report.add_metadata("Platform", "Test Platform")
        report.add_summary_item("Metric", "42", "units", "good")
        report.add_section("Test Section", "Test Description")
        report.add_figure(sample_figure, caption="Test Figure", section="Test Section")

        output_path = tmp_path / "report.html"
        report.generate_html(str(output_path))

        content = output_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "<title>Test Report</title>" in content
        assert "Test Author" in content
        assert "Platform" in content
        assert "Test Platform" in content
        assert "Test Section" in content
        assert "Test Figure" in content
        assert "data:image/png;base64," in content

    def test_generate_html_without_summary(self, report, sample_figure, tmp_path):
        """Test HTML generation without summary section."""
        report.add_figure(sample_figure, caption="Test Figure")

        output_path = tmp_path / "report.html"
        report.generate_html(str(output_path), include_summary=False)

        content = output_path.read_text(encoding="utf-8")
        assert 'id="summary"' not in content

    def test_generate_html_with_tables(self, report, tmp_path):
        """Test HTML generation with tables."""
        report.add_table(
            "Test Table",
            ["A", "B", "C"],
            [["1", "2", "3"], ["4", "5", "6"]],
            "Data Section",
        )

        output_path = tmp_path / "report.html"
        report.generate_html(str(output_path))

        content = output_path.read_text(encoding="utf-8")
        assert "<table>" in content
        assert "<th>A</th>" in content
        assert "<td>1</td>" in content
        assert "Test Table" in content

    def test_clear(self, report, sample_figure):
        """Test clearing all data."""
        report.add_metadata("key", "value")
        report.add_figure(sample_figure, caption="Test")
        report.add_summary_item("label", "value", "", "neutral")
        report.add_section("Section", "Desc")
        report.add_table("Table", ["A"], [["1"]], "Section")
        report.add_html_figure("<div>plot</div>", "Plot", "Section")
        report.add_interactive_line_chart(
            labels=["a", "b"],
            datasets=[{"label": "X", "data": [1, 2]}],
            title="Line",
            section="Section",
        )

        report.clear()

        assert report.figures == []
        assert report.tables == []
        assert report.summary_items == []
        assert report.sections == []
        assert report.metadata == {}
        assert report.interactive_charts == []
        assert report.html_figures == []

    def test_add_figures_from_visualizer(self, report):
        """Test adding figures from PerformanceVisualizer."""
        viz = PerformanceVisualizer()

        # Create a simple figure using visualizer
        data = {"Workload1": {"P50": 25.0, "P95": 30.0, "P99": 35.0}}
        viz.plot_latency_percentiles(data)

        report.add_figures_from_visualizer(
            viz,
            captions=["Latency Graph"],
            section="Performance",
        )

        assert len(report.figures) == 1
        assert report.figures[0]["caption"] == "Latency Graph"
        assert report.figures[0]["section"] == "Performance"

        viz.close_all()

    def test_generate_metadata_html(self, report):
        """Test metadata section HTML generation."""
        report.add_metadata("System", "Test System")
        report.add_metadata("Version", "1.0")

        html = report._generate_metadata_html()

        assert 'id="metadata"' in html
        assert "System" in html
        assert "Test System" in html
        assert "Version" in html
        assert "1.0" in html

    def test_nested_directory_creation(self, report, sample_figure, tmp_path):
        """Test that nested directories are created."""
        report.add_figure(sample_figure, caption="Test")

        output_path = tmp_path / "nested" / "deep" / "report.html"
        report.generate_html(str(output_path))

        assert output_path.exists()

    def test_javascript_included(self, report, sample_figure, tmp_path):
        """Test that JavaScript for navigation is included."""
        report.add_figure(sample_figure, caption="Test")

        output_path = tmp_path / "report.html"
        report.generate_html(str(output_path))

        content = output_path.read_text(encoding="utf-8")
        assert "<script>" in content
        assert "scrollIntoView" in content
        assert "addEventListener" in content

    def test_print_styles_included(self, report, sample_figure, tmp_path):
        """Test that print-friendly CSS is included."""
        report.add_figure(sample_figure, caption="Test")

        output_path = tmp_path / "report.html"
        report.generate_html(str(output_path))

        content = output_path.read_text(encoding="utf-8")
        assert "@media print" in content

    def test_responsive_styles_included(self, report, sample_figure, tmp_path):
        """Test that responsive CSS is included."""
        report.add_figure(sample_figure, caption="Test")

        output_path = tmp_path / "report.html"
        report.generate_html(str(output_path))

        content = output_path.read_text(encoding="utf-8")
        assert "@media (max-width:" in content

    def test_generate_html_includes_chartjs_when_interactive_charts(self, report, tmp_path):
        """Test that Chart.js and chart canvas are included when using interactive charts."""
        report.add_interactive_line_chart(
            labels=["a", "b"],
            datasets=[{"label": "X", "data": [1, 2]}],
            title="Test Chart",
            section="Charts",
        )
        output_path = tmp_path / "report.html"
        report.generate_html(str(output_path))
        content = output_path.read_text(encoding="utf-8")
        assert "chart.js" in content.lower() or "jsdelivr" in content.lower()
        assert "chart-container" in content
        assert "canvas" in content
        assert "Test Chart" in content

    def test_generate_html_includes_plotly_when_html_figures(self, report, tmp_path):
        """Test that Plotly script is included when adding HTML figures."""
        report.add_html_figure(
            "<div id='plotly-div'>placeholder</div>",
            caption="Plot",
            section="Charts",
        )
        output_path = tmp_path / "report.html"
        report.generate_html(str(output_path))
        content = output_path.read_text(encoding="utf-8")
        assert "plotly" in content.lower()
        assert "plotly-div" in content or "placeholder" in content


class TestHTMLReportGeneratorIntegration:
    """Integration tests for HTMLReportGenerator with full workflow."""

    def test_full_workflow(self, tmp_path):
        """Test complete report generation workflow."""
        report = HTMLReportGenerator(
            title="Integration Test Report",
            author="Test Suite",
            theme="light",
        )

        # Add comprehensive data
        report.add_metadata("Platform", "Test Platform")
        report.add_metadata("Date", "2024-01-15")

        report.add_summary_item("Samples", "1000", "", "neutral")
        report.add_summary_item("Mean Latency", "25.5", "ms", "good")
        report.add_summary_item("P99 Latency", "45.2", "ms", "warning")
        report.add_summary_item("Errors", "0", "", "good")

        report.add_section("Latency Analysis", "Detailed latency metrics")
        report.add_section("Power Analysis", "Power consumption data")

        # Create and add figures
        viz = PerformanceVisualizer()

        latency_data = {
            "ResNet50": {"P50": 25.0, "P95": 30.0, "P99": 35.0},
            "YOLO": {"P50": 30.0, "P95": 38.0, "P99": 45.0},
        }
        fig1 = viz.plot_latency_percentiles(latency_data)
        report.add_figure(fig1, "Latency Percentiles", "Latency Analysis")

        fig2 = viz.plot_latency_throughput_tradeoff([1, 2, 4, 8], [20, 25, 35, 50], [50, 80, 115, 160])
        report.add_figure(fig2, "Batch Scaling", "Latency Analysis")

        # Add table
        report.add_table(
            "Configuration Results",
            ["Config", "Latency", "Throughput"],
            [
                ["Batch=1", "20ms", "50 FPS"],
                ["Batch=4", "35ms", "115 FPS"],
            ],
            "Latency Analysis",
        )

        # Generate report
        output_path = tmp_path / "full_report.html"
        report.generate_html(str(output_path))

        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")

        # Verify all components are present
        assert "Integration Test Report" in content
        assert "Test Platform" in content
        assert "Latency Analysis" in content
        assert "Power Analysis" in content
        assert "1000" in content
        assert "Configuration Results" in content
        assert "data:image/png;base64," in content

        viz.close_all()

    def test_dark_theme_integration(self, tmp_path):
        """Test dark theme styling is properly applied."""
        report = HTMLReportGenerator(
            title="Dark Theme Test",
            theme="dark",
        )

        report.add_summary_item("Test", "Value", "", "neutral")

        output_path = tmp_path / "dark_report.html"
        report.generate_html(str(output_path))

        content = output_path.read_text(encoding="utf-8")
        # Dark theme specific colors
        assert "#1a1a2e" in content  # Dark background
        assert "#eaeaea" in content  # Light text


@pytest.mark.skipif(not CHARTS_AVAILABLE, reason="trackiq.reporting.charts (pandas/plotly) not available")
class TestChartsReportIntegration:
    """Integration tests for add_charts_to_html_report and add_interactive_charts_to_html_report."""

    def _minimal_df_summary(self):
        """Minimal DataFrame and summary matching collector export shape."""
        df = pd.DataFrame(
            {
                "timestamp": [1000.0 + i * 0.5 for i in range(20)],
                "elapsed_seconds": [i * 0.5 for i in range(20)],
                "latency_ms": [22.0 + (i % 3) for i in range(20)],
                "cpu_percent": [40.0 + i for i in range(20)],
                "gpu_percent": [70.0 + (i % 10) for i in range(20)],
                "power_w": [15.0 + i * 0.2 for i in range(20)],
                "temperature_c": [45.0 + i * 0.5 for i in range(20)],
                "memory_used_mb": [4000.0 + i * 2 for i in range(20)],
                "memory_total_mb": [16384.0] * 20,
                "throughput_fps": [1000.0 / (22.0 + (i % 3)) for i in range(20)],
                "is_warmup": [i < 2 for i in range(20)],
            }
        )
        summary = {
            "sample_count": 20,
            "warmup_samples": 2,
            "latency": {"mean_ms": 23.0, "p50_ms": 23.0, "p95_ms": 25.0, "p99_ms": 25.0},
            "throughput": {"mean_fps": 44.0, "min_fps": 40.0, "max_fps": 50.0},
            "cpu": {"mean_percent": 49.5, "max_percent": 59.0},
            "gpu": {"mean_percent": 74.5, "max_percent": 79.0},
            "power": {"mean_w": 17.0, "max_w": 19.0},
            "temperature": {"mean_c": 49.5, "max_c": 54.0},
            "memory": {"mean_mb": 4019.0, "max_mb": 4038.0},
        }
        return df, summary

    def test_add_charts_to_html_report_chartjs(self, tmp_path):
        """add_charts_to_html_report with chart_engine=chartjs adds interactive charts and table."""
        df, summary = self._minimal_df_summary()
        report = HTMLReportGenerator(title="Chart.js Report", author="Test", theme="light")
        report.add_metadata("Source", "unit test")
        report.add_summary_item("Samples", summary["sample_count"], "", "neutral")

        shared_charts.add_charts_to_html_report(report, df, summary, chart_engine="chartjs")

        assert len(report.interactive_charts) > 0
        assert len(report.html_figures) == 0
        assert any(t["section"] == "Summary Statistics" for t in report.tables)

        out = tmp_path / "chartjs_report.html"
        report.generate_html(str(out))
        content = out.read_text(encoding="utf-8")
        assert "chart.js" in content.lower() or "jsdelivr" in content.lower()
        assert 'id="latency"' in content or 'id="summary-statistics"' in content

    def test_add_charts_to_html_report_plotly(self, tmp_path):
        """add_charts_to_html_report with chart_engine=plotly adds Plotly HTML figures."""
        df, summary = self._minimal_df_summary()
        report = HTMLReportGenerator(title="Plotly Report", author="Test", theme="light")

        shared_charts.add_charts_to_html_report(report, df, summary, chart_engine="plotly")

        assert len(report.html_figures) > 0
        out = tmp_path / "plotly_report.html"
        report.generate_html(str(out))
        content = out.read_text(encoding="utf-8")
        assert "plotly" in content.lower()

    def test_add_interactive_charts_to_html_report(self, tmp_path):
        """add_interactive_charts_to_html_report adds Chart.js charts and Summary Statistics."""
        df, summary = self._minimal_df_summary()
        report = HTMLReportGenerator(title="Interactive Report", author="Test", theme="light")

        shared_charts.add_interactive_charts_to_html_report(report, df, summary)

        assert len(report.interactive_charts) > 0
        assert any(t["section"] == "Summary Statistics" for t in report.tables)
        out = tmp_path / "interactive_report.html"
        report.generate_html(str(out))
        content = out.read_text(encoding="utf-8")
        assert "Latency" in content
        assert "Summary Statistics" in content or "Key metrics" in content

    def test_warmup_only_samples_fallback_keeps_charts_non_empty(self):
        """When all samples are warmup, chart builders should fall back to full data."""
        df, summary = self._minimal_df_summary()
        df["is_warmup"] = True

        # Plotly paths should still return figures instead of empty charts.
        assert shared_charts.create_latency_histogram(df, summary, exclude_warmup=True) is not None
        assert shared_charts.create_throughput_timeline(df, summary, exclude_warmup=True) is not None

        # Chart.js report path should still include throughput chart.
        report = HTMLReportGenerator(title="Warmup Fallback", author="Test", theme="light")
        shared_charts.add_interactive_charts_to_html_report(report, df, summary)
        titles = [chart.get("title") for chart in report.interactive_charts]
        assert "Throughput Over Time" in titles
