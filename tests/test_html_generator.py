"""Tests for HTMLReportGenerator."""

import os
import tempfile
from pathlib import Path

# Use non-interactive backend so tests run without Tk (e.g. on Windows without tk/tcl)
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from autoperfpy.reporting import HTMLReportGenerator, PerformanceVisualizer


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

        report.clear()

        assert report.figures == []
        assert report.tables == []
        assert report.summary_items == []
        assert report.sections == []
        assert report.metadata == {}

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

        fig2 = viz.plot_latency_throughput_tradeoff(
            [1, 2, 4, 8], [20, 25, 35, 50], [50, 80, 115, 160]
        )
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
        result = report.generate_html(str(output_path))

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
