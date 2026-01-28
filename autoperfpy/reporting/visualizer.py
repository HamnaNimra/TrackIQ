"""Performance visualization module for creating graphs and charts."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
# pandas not required here; remove unused import


class PerformanceVisualizer:
    """Create performance graphs and visualizations."""

    def __init__(self, style: str = "seaborn-v0_8-darkgrid"):
        """Initialize visualizer.

        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use("default")

        self.figures = []

    def plot_latency_percentiles(
        self, latencies_by_workload: Dict[str, Dict[str, float]], title: str = "Latency Percentiles Comparison"
    ) -> plt.Figure:
        """Create percentile latency comparison graph.

        Args:
            latencies_by_workload: Dict with workload names and their P50/P95/P99 values
            title: Graph title

        Returns:
            Matplotlib figure

        Example:
            data = {
                'ResNet50': {'P50': 25.5, 'P95': 28.3, 'P99': 32.1},
                'YOLO': {'P50': 30.2, 'P95': 35.8, 'P99': 42.5}
            }
            fig = viz.plot_latency_percentiles(data)
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        workloads = list(latencies_by_workload.keys())
        x = np.arange(len(workloads))
        width = 0.25

        p50_values = [latencies_by_workload[w].get("P50", 0) for w in workloads]
        p95_values = [latencies_by_workload[w].get("P95", 0) for w in workloads]
        p99_values = [latencies_by_workload[w].get("P99", 0) for w in workloads]

        bars1 = ax.bar(x - width, p50_values, width, label="P50 (Median)", alpha=0.8)
        bars2 = ax.bar(x, p95_values, width, label="P95", alpha=0.8)
        bars3 = ax.bar(x + width, p99_values, width, label="P99", alpha=0.8)

        ax.set_xlabel("Workload", fontsize=12, fontweight="bold")
        ax.set_ylabel("Latency (ms)", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(workloads, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{height:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        plt.tight_layout()
        self.figures.append(fig)
        return fig

    def plot_latency_throughput_tradeoff(
        self,
        batch_sizes: List[int],
        latencies: List[float],
        throughputs: List[float],
        title: str = "Latency vs Throughput Trade-off",
    ) -> plt.Figure:
        """Create latency vs throughput trade-off graph.

        Args:
            batch_sizes: List of batch sizes tested
            latencies: Corresponding latency values in ms
            throughputs: Corresponding throughput values (items/sec)
            title: Graph title

        Returns:
            Matplotlib figure
        """
        fig, ax1 = plt.subplots(figsize=(12, 6))

        color = "tab:blue"
        ax1.set_xlabel("Batch Size", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Latency (ms)", color=color, fontsize=12, fontweight="bold")
        line1 = ax1.plot(batch_sizes, latencies, color=color, marker="o", linewidth=2.5, markersize=8, label="Latency")
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.grid(alpha=0.3)

        ax2 = ax1.twinx()
        color = "tab:orange"
        ax2.set_ylabel("Throughput (items/sec)", color=color, fontsize=12, fontweight="bold")
        line2 = ax2.plot(
            batch_sizes, throughputs, color=color, marker="s", linewidth=2.5, markersize=8, label="Throughput"
        )
        ax2.tick_params(axis="y", labelcolor=color)

        fig.suptitle(title, fontsize=14, fontweight="bold")

        # Combined legend
        lines = line1 + line2
        labels = [ln.get_label() for ln in lines]
        ax1.legend(lines, labels, loc="upper left")

        plt.tight_layout()
        self.figures.append(fig)
        return fig

    def plot_power_vs_performance(
        self,
        workloads: List[str],
        power_values: List[float],
        performance_values: List[float],
        title: str = "Power vs Performance",
    ) -> plt.Figure:
        """Create power consumption vs performance scatter plot.

        Args:
            workloads: List of workload names
            power_values: Power consumption in Watts
            performance_values: Performance metric (throughput, FPS, etc.)
            title: Graph title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.Set3(np.linspace(0, 1, len(workloads)))

        ax.scatter(power_values, performance_values, s=300, alpha=0.6, c=colors)

        for i, workload in enumerate(workloads):
            ax.annotate(
                workload,
                (power_values[i], performance_values[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
            )

        ax.set_xlabel("Power Consumption (W)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Performance", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self.figures.append(fig)
        return fig

    def plot_gpu_memory_timeline(
        self,
        timestamps: List[float],
        memory_used: List[float],
        memory_total: Optional[List[float]] = None,
        title: str = "GPU Memory Usage Over Time",
    ) -> plt.Figure:
        """Create GPU memory usage timeline.

        Args:
            timestamps: Time values in seconds
            memory_used: GPU memory used in MB
            memory_total: Optional total GPU memory for reference
            title: Graph title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.fill_between(timestamps, 0, memory_used, alpha=0.4, label="Used Memory")
        ax.plot(timestamps, memory_used, color="blue", linewidth=2, marker="o", markersize=4, label="Memory Used (MB)")

        if memory_total:
            ax.plot(timestamps, memory_total, color="red", linewidth=2, linestyle="--", label="Total Memory")

        ax.set_xlabel("Time (seconds)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Memory (MB)", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self.figures.append(fig)
        return fig

    def plot_relative_performance(
        self,
        baseline_name: str,
        metrics_data: Dict[str, Dict[str, float]],
        title: str = "Relative Performance Comparison",
    ) -> plt.Figure:
        """Create relative performance comparison (normalized to baseline).

        Args:
            baseline_name: Name of baseline configuration
            metrics_data: Dict with config names and metric dicts
                         Example: {
                             'Config1': {'latency': 25.5, 'throughput': 100},
                             'Config2': {'latency': 28.3, 'throughput': 95},
                         }
            title: Graph title

        Returns:
            Matplotlib figure
        """
        if baseline_name not in metrics_data:
            raise ValueError(f"Baseline '{baseline_name}' not found in metrics")

        baseline = metrics_data[baseline_name]
        fig, ax = plt.subplots(figsize=(12, 6))

        configs = list(metrics_data.keys())
        metrics = list(baseline.keys())

        x = np.arange(len(configs))
        width = 0.15

        for i, metric in enumerate(metrics):
            baseline_value = baseline[metric]
            relative_values = [metrics_data[cfg].get(metric, baseline_value) / baseline_value * 100 for cfg in configs]

            ax.bar(x + (i - len(metrics) / 2) * width, relative_values, width, label=metric, alpha=0.8)

        # Reference line at 100%
        ax.axhline(y=100, color="red", linestyle="--", linewidth=2, label="Baseline")

        ax.set_ylabel("Relative Performance (%)", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        self.figures.append(fig)
        return fig

    def plot_distribution(
        self, data_dict: Dict[str, List[float]], title: str = "Distribution Comparison", bins: int = 30
    ) -> plt.Figure:
        """Create histogram distribution comparison.

        Args:
            data_dict: Dict with names and lists of values
            title: Graph title
            bins: Number of histogram bins

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        for name, data in data_dict.items():
            ax.hist(data, bins=bins, alpha=0.6, label=name)

        ax.set_xlabel("Value", fontsize=12, fontweight="bold")
        ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        self.figures.append(fig)
        return fig

    def save_figure(self, fig: plt.Figure, filepath: str, dpi: int = 300) -> str:
        """Save a figure to file.

        Args:
            fig: Matplotlib figure object
            filepath: Path to save to
            dpi: Resolution in dots per inch

        Returns:
            Path to saved file
        """
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
        return filepath

    def close_all(self) -> None:
        """Close all figures."""
        plt.close("all")
        self.figures = []


__all__ = ["PerformanceVisualizer"]
