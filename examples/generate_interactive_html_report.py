#!/usr/bin/env python3
"""
Example: Generate an interactive HTML report with hover, zoom, and filter features.

This example demonstrates the interactive charting capabilities of the
HTMLReportGenerator using Chart.js for client-side rendering.
"""

import numpy as np
from autoperfpy.reporting import HTMLReportGenerator, PerformanceVisualizer


def main():
    """Generate interactive HTML report with Chart.js visualizations."""

    print("Generating Interactive HTML Report...")
    print("=" * 60)

    # Initialize the report generator
    report = HTMLReportGenerator(
        title="Interactive Performance Analysis",
        author="AutoPerfPy",
        theme="light",  # or "dark"
    )

    # Also use matplotlib visualizer for static charts
    viz = PerformanceVisualizer()

    # Add metadata
    report.add_metadata("System", "NVIDIA Drive Orin AGX")
    report.add_metadata("Date", "2024-01-15")

    # Add summary items
    report.add_summary_item("Mean Latency", "22.5", "ms", "good")
    report.add_summary_item("P99 Latency", "42.5", "ms", "warning")
    report.add_summary_item("Throughput", "1,250", "inf/s", "good")
    report.add_summary_item("GPU Util", "87", "%", "good")

    # =========================================================================
    # Section 1: Interactive Line Charts
    # =========================================================================
    report.add_section("Latency Trends", "Real-time latency monitoring")

    # Line chart with zoom and pan
    print("1. Adding interactive line chart (with zoom)...")
    report.add_interactive_line_chart(
        labels=[f"T{i}" for i in range(30)],
        datasets=[
            {
                "label": "P50 Latency",
                "data": [22 + np.sin(i/3) * 2 + np.random.normal(0, 0.5) for i in range(30)],
            },
            {
                "label": "P95 Latency",
                "data": [28 + np.sin(i/3) * 3 + np.random.normal(0, 0.8) for i in range(30)],
            },
            {
                "label": "P99 Latency",
                "data": [35 + np.sin(i/3) * 4 + np.random.normal(0, 1.2) for i in range(30)],
            },
        ],
        title="Latency Percentiles Over Time",
        section="Latency Trends",
        description="Scroll to zoom in/out. Drag to pan. Click legend items to show/hide series.",
        x_label="Time",
        y_label="Latency (ms)",
        enable_zoom=True,
    )

    # =========================================================================
    # Section 2: Interactive Bar Charts
    # =========================================================================
    report.add_section("Model Comparison", "Comparing different models")

    # Grouped bar chart
    print("2. Adding interactive bar chart...")
    report.add_interactive_bar_chart(
        labels=["ResNet50", "YOLO V8", "Segformer", "EfficientDet", "MobileNet"],
        datasets=[
            {"label": "P50 Latency", "data": [22.5, 30.2, 18.5, 15.2, 12.1]},
            {"label": "P95 Latency", "data": [25.3, 35.8, 20.3, 17.5, 14.2]},
            {"label": "P99 Latency", "data": [28.1, 42.5, 23.8, 19.8, 16.5]},
        ],
        title="Latency by Model",
        section="Model Comparison",
        description="Hover for exact values. Click legend to filter.",
        x_label="Model",
        y_label="Latency (ms)",
    )

    # Horizontal bar chart
    print("3. Adding horizontal bar chart...")
    report.add_interactive_bar_chart(
        labels=["TensorRT", "ONNX Runtime", "PyTorch", "TFLite", "OpenVINO"],
        datasets=[
            {"label": "Relative Performance", "data": [100, 85, 65, 78, 82]},
        ],
        title="Framework Comparison",
        section="Model Comparison",
        description="Normalized to TensorRT (100%)",
        x_label="Performance (%)",
        y_label="Framework",
        horizontal=True,
    )

    # Stacked bar chart
    print("4. Adding stacked bar chart...")
    report.add_interactive_bar_chart(
        labels=["Config A", "Config B", "Config C", "Config D"],
        datasets=[
            {"label": "Preprocessing", "data": [5, 6, 4, 7]},
            {"label": "Inference", "data": [20, 18, 22, 15]},
            {"label": "Postprocessing", "data": [3, 4, 3, 5]},
        ],
        title="Execution Time Breakdown",
        section="Model Comparison",
        description="Stacked view of execution phases",
        x_label="Configuration",
        y_label="Time (ms)",
        stacked=True,
    )

    # =========================================================================
    # Section 3: Interactive Scatter Charts
    # =========================================================================
    report.add_section("Trade-off Analysis", "Power vs Performance trade-offs")

    print("5. Adding interactive scatter chart (with zoom)...")
    report.add_interactive_scatter_chart(
        datasets=[
            {
                "label": "GPU Models",
                "data": [
                    {"x": 65.2, "y": 44.4},
                    {"x": 82.5, "y": 33.1},
                    {"x": 55.3, "y": 54.1},
                    {"x": 48.1, "y": 65.8},
                    {"x": 35.6, "y": 81.2},
                ],
            },
            {
                "label": "DLA Models",
                "data": [
                    {"x": 25.2, "y": 38.4},
                    {"x": 32.5, "y": 28.1},
                    {"x": 22.3, "y": 45.1},
                ],
            },
        ],
        title="Power vs Throughput",
        section="Trade-off Analysis",
        description="Scroll to zoom. Each point represents a model configuration.",
        x_label="Power (W)",
        y_label="Throughput (inf/s)",
        enable_zoom=True,
    )

    # =========================================================================
    # Section 4: Interactive Pie/Doughnut Charts
    # =========================================================================
    report.add_section("Resource Distribution", "System resource breakdown")

    print("6. Adding interactive pie chart...")
    report.add_interactive_pie_chart(
        labels=["GPU Compute", "Memory Bandwidth", "DLA", "CPU", "Other"],
        data=[45, 25, 15, 10, 5],
        title="Power Distribution",
        section="Resource Distribution",
        description="Hover for percentages and values",
    )

    print("7. Adding interactive doughnut chart...")
    report.add_interactive_pie_chart(
        labels=["GPU", "DLA0", "DLA1", "CPU"],
        data=[60, 20, 15, 5],
        title="Compute Distribution",
        section="Resource Distribution",
        description="Workload distribution across compute units",
        doughnut=True,
    )

    # =========================================================================
    # Add a static matplotlib figure for comparison
    # =========================================================================
    print("8. Adding static matplotlib figure...")
    latency_data = {
        "ResNet50": {"P50": 22.5, "P95": 25.3, "P99": 28.1},
        "YOLO V8": {"P50": 30.2, "P95": 35.8, "P99": 42.5},
    }
    fig = viz.plot_latency_percentiles(latency_data)
    report.add_figure(fig, "Static Chart (matplotlib)", section="Resource Distribution",
                     description="Traditional static image for comparison")

    # =========================================================================
    # Add a data table
    # =========================================================================
    report.add_table(
        title="Summary Statistics",
        headers=["Model", "P50 (ms)", "P95 (ms)", "P99 (ms)", "Throughput"],
        rows=[
            ["ResNet50", "22.5", "25.3", "28.1", "44.4"],
            ["YOLO V8", "30.2", "35.8", "42.5", "33.1"],
            ["Segformer", "18.5", "20.3", "23.8", "54.1"],
            ["EfficientDet", "15.2", "17.5", "19.8", "65.8"],
        ],
        section="Resource Distribution",
    )

    # Generate the report
    print("\nGenerating HTML report...")
    output_path = report.generate_html("interactive_report.html")

    print("=" * 60)
    print(f"Report generated: {output_path}")
    print(f"  Interactive charts: {len(report.interactive_charts)}")
    print(f"  Static figures: {len(report.figures)}")
    print(f"  Tables: {len(report.tables)}")
    print("\nOpen the HTML file in a browser to explore:")
    print("  - Hover over data points for values")
    print("  - Click legend items to filter series")
    print("  - Scroll to zoom (on line/scatter charts)")
    print("  - Drag to pan (on zoomed charts)")
    print("  - Click 'Reset Zoom' to restore original view")

    # Cleanup
    viz.close_all()

    return output_path


if __name__ == "__main__":
    main()
