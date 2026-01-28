#!/usr/bin/env python3
"""
Example: Generate comprehensive performance report with graphs and PDF.

Demonstrates the reporting and visualization modules.
"""

import numpy as np
from autoperfpy.reporting import PerformanceVisualizer, PDFReportGenerator


def main():
    """Generate performance report with visualizations."""

    # Load configuration

    print("📊 Generating Performance Report...")
    print("=" * 70)

    # Initialize visualizer and PDF generator
    viz = PerformanceVisualizer()
    pdf_gen = PDFReportGenerator(title="Performance Analysis Report", author="AutoPerfPy")

    # Add metadata
    pdf_gen.add_metadata("Date", "2024-01-15")
    pdf_gen.add_metadata("System", "NVIDIA Drive Orin AGX")
    pdf_gen.add_metadata("Analysis Type", "Latency & Throughput")

    # 1. Latency Percentiles
    print("\n1️⃣  Creating latency percentiles graph...")
    latency_data = {
        "ResNet50": {"P50": 22.5, "P95": 25.3, "P99": 28.1},
        "YOLO V8": {"P50": 30.2, "P95": 35.8, "P99": 42.5},
        "Segformer": {"P50": 18.5, "P95": 20.3, "P99": 23.8},
        "EfficientDet": {"P50": 15.2, "P95": 17.5, "P99": 19.8},
    }
    fig1 = viz.plot_latency_percentiles(latency_data)
    pdf_gen.add_figure(fig1, "Latency Percentiles (P50, P95, P99) across workloads")

    # 2. Latency vs Throughput Trade-off
    print("2️⃣  Creating latency vs throughput trade-off graph...")
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    latencies = [15.0, 12.5, 10.8, 9.5, 8.8, 8.5, 8.3]
    throughputs = [66.7, 160.0, 370.4, 842.1, 1136.4, 1882.4, 2409.6]

    fig2 = viz.plot_latency_throughput_tradeoff(
        batch_sizes, latencies, throughputs, title="Batch Size Trade-off Analysis"
    )
    pdf_gen.add_figure(fig2, "Latency vs Throughput: The fundamental trade-off in batching")

    # 3. Power vs Performance
    print("3️⃣  Creating power vs performance graph...")
    workloads = ["ResNet50", "YOLO V8", "Segformer", "EfficientDet", "MobileNet"]
    power_values = [65.2, 82.5, 55.3, 48.1, 35.6]
    performance_values = [44.4, 33.1, 54.1, 65.8, 81.2]

    fig3 = viz.plot_power_vs_performance(
        workloads, power_values, performance_values, title="Power Consumption vs Inference Performance"
    )
    pdf_gen.add_figure(fig3, "Power Efficiency: Workload comparison")

    # 4. GPU Memory Timeline
    print("4️⃣  Creating GPU memory timeline graph...")
    timestamps = np.arange(0, 60, 1)
    memory_used = 2500 + 150 * np.sin(timestamps / 20) + np.random.normal(0, 50, len(timestamps))
    memory_total = [8000] * len(timestamps)

    fig4 = viz.plot_gpu_memory_timeline(
        timestamps.tolist(), memory_used.tolist(), memory_total, title="GPU Memory Usage During LLM Inference"
    )
    pdf_gen.add_figure(fig4, "Real-time GPU memory monitoring (60 second window)")

    # 5. Relative Performance Comparison
    print("5️⃣  Creating relative performance comparison graph...")
    relative_perf = {
        "Baseline": {"latency": 25.5, "throughput": 100, "power": 65},
        "Optimized V1": {"latency": 22.3, "throughput": 115, "power": 62},
        "Optimized V2": {"latency": 20.8, "throughput": 130, "power": 60},
        "Quantized": {"latency": 18.5, "throughput": 145, "power": 55},
    }

    fig5 = viz.plot_relative_performance("Baseline", relative_perf, title="Optimization Impact: Relative to Baseline")
    pdf_gen.add_figure(fig5, "Relative Performance: Quantization and optimization improvements")

    # 6. Distribution Comparison
    print("6️⃣  Creating distribution comparison graph...")
    np.random.seed(42)
    dist_data = {
        "Baseline": np.random.normal(25, 3, 1000),
        "Optimized": np.random.normal(22, 2.5, 1000),
        "Quantized": np.random.normal(20, 2, 1000),
    }

    fig6 = viz.plot_distribution(dist_data, title="Latency Distribution Comparison", bins=40)
    pdf_gen.add_figure(fig6, "Distribution analysis: Latency variance across optimization levels")

    # Generate PDF report
    print("\n📄 Generating consolidated PDF report...")
    output_file = "performance_report.pdf"
    pdf_path = pdf_gen.generate_pdf(output_file, include_summary=True)

    print(f"✅ Report generated: {pdf_path}")
    print("\n" + "=" * 70)
    print("Report Summary:")
    print(f"  • Total graphs: {len(pdf_gen.figures)}")
    print(f"  • Output file: {output_file}")
    print(f"  • File size: {os.path.getsize(output_file) / 1024:.1f} KB")
    print("=" * 70)

    # Optional: Display individual graphs
    print("\n💡 Tip: View graphs interactively with:")
    print("   python examples/generate_performance_report.py --show")

    viz.close_all()

    return pdf_path


if __name__ == "__main__":
    import os
    import sys

    # Check if --show flag provided for interactive display
    if "--show" in sys.argv:
        import matplotlib.pyplot as plt

        pdf_path = main()
        print("\nDisplaying graphs interactively (close window to exit)...")
        plt.show()
    else:
        main()
