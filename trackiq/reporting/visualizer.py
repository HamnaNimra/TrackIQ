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

    # ==================== DNN Pipeline Visualizations ====================

    def plot_layer_timings(
        self,
        layers: List[Dict[str, any]],
        title: str = "DNN Layer Execution Times",
        top_n: int = 10,
    ) -> plt.Figure:
        """Create horizontal bar chart of layer execution times.

        Args:
            layers: List of layer dicts with 'name', 'time_ms', and optionally 'device'
            title: Graph title
            top_n: Number of top layers to show

        Returns:
            Matplotlib figure

        Example:
            layers = [
                {'name': 'conv1', 'time_ms': 2.5, 'device': 'GPU'},
                {'name': 'conv2', 'time_ms': 1.8, 'device': 'DLA0'},
            ]
            fig = viz.plot_layer_timings(layers)
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Sort by time and take top N
        sorted_layers = sorted(layers, key=lambda x: x.get("time_ms", 0), reverse=True)[:top_n]
        sorted_layers = sorted_layers[::-1]  # Reverse for horizontal bar

        names = [l.get("name", "unknown") for l in sorted_layers]
        times = [l.get("time_ms", 0) for l in sorted_layers]
        devices = [l.get("device", "GPU") for l in sorted_layers]

        # Color by device
        colors = ["#FF6B6B" if d.startswith("DLA") else "#4ECDC4" for d in devices]

        bars = ax.barh(names, times, color=colors, alpha=0.8)

        ax.set_xlabel("Execution Time (ms)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Layer Name", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        # Add value labels
        for bar, time in zip(bars, times):
            ax.text(
                bar.get_width() + 0.1,
                bar.get_y() + bar.get_height() / 2,
                f"{time:.2f}ms",
                va="center",
                fontsize=9,
            )

        # Legend for device colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#4ECDC4", label="GPU"),
            Patch(facecolor="#FF6B6B", label="DLA"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")

        plt.tight_layout()
        self.figures.append(fig)
        return fig

    def plot_device_split(
        self,
        gpu_time_ms: float,
        dla_time_ms: float,
        title: str = "DLA vs GPU Execution Split",
    ) -> plt.Figure:
        """Create pie chart showing DLA vs GPU execution time split.

        Args:
            gpu_time_ms: Total GPU execution time in ms
            dla_time_ms: Total DLA execution time in ms
            title: Graph title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        sizes = [gpu_time_ms, dla_time_ms]
        labels = [f"GPU\n{gpu_time_ms:.1f}ms", f"DLA\n{dla_time_ms:.1f}ms"]
        colors = ["#4ECDC4", "#FF6B6B"]
        explode = (0.02, 0.02)

        total = gpu_time_ms + dla_time_ms
        if total > 0:
            wedges, texts, autotexts = ax.pie(
                sizes,
                labels=labels,
                colors=colors,
                explode=explode,
                autopct=lambda pct: f"{pct:.1f}%",
                startangle=90,
                textprops={"fontsize": 12},
            )
            for autotext in autotexts:
                autotext.set_fontweight("bold")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)

        ax.set_title(title, fontsize=14, fontweight="bold")

        plt.tight_layout()
        self.figures.append(fig)
        return fig

    def plot_memory_transfer_timeline(
        self,
        transfers: List[Dict[str, any]],
        title: str = "Memory Transfer Timeline",
    ) -> plt.Figure:
        """Create bar chart of memory transfer times.

        Args:
            transfers: List of transfer dicts with 'transfer_type', 'duration_ms', 'size_bytes'
            title: Graph title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        if not transfers:
            ax.text(0.5, 0.5, "No memory transfers", ha="center", va="center", fontsize=14)
            ax.set_title(title, fontsize=14, fontweight="bold")
            plt.tight_layout()
            self.figures.append(fig)
            return fig

        h2d_times = [t.get("duration_ms", 0) for t in transfers if t.get("transfer_type") == "H2D"]
        d2h_times = [t.get("duration_ms", 0) for t in transfers if t.get("transfer_type") == "D2H"]

        x = np.arange(max(len(h2d_times), len(d2h_times)))
        width = 0.35

        if h2d_times:
            ax.bar(x[:len(h2d_times)] - width/2, h2d_times, width, label="H2D (Host→Device)", color="#3498DB", alpha=0.8)
        if d2h_times:
            ax.bar(x[:len(d2h_times)] + width/2, d2h_times, width, label="D2H (Device→Host)", color="#E74C3C", alpha=0.8)

        ax.set_xlabel("Transfer Index", fontsize=12, fontweight="bold")
        ax.set_ylabel("Duration (ms)", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        self.figures.append(fig)
        return fig

    def plot_batch_scaling(
        self,
        batch_metrics: List[Dict[str, any]],
        title: str = "Batch Size Scaling Analysis",
    ) -> plt.Figure:
        """Create dual-axis plot showing latency and throughput vs batch size.

        Args:
            batch_metrics: List of dicts with 'batch_size', 'avg_latency_ms', 'avg_throughput_fps'
            title: Graph title

        Returns:
            Matplotlib figure
        """
        fig, ax1 = plt.subplots(figsize=(12, 6))

        batch_sizes = [m.get("batch_size", 0) for m in batch_metrics]
        latencies = [m.get("avg_latency_ms", 0) for m in batch_metrics]
        throughputs = [m.get("avg_throughput_fps", 0) for m in batch_metrics]

        color1 = "#3498DB"
        ax1.set_xlabel("Batch Size", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Latency (ms)", color=color1, fontsize=12, fontweight="bold")
        line1 = ax1.plot(batch_sizes, latencies, color=color1, marker="o", linewidth=2.5, markersize=10, label="Latency")
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.grid(alpha=0.3)

        ax2 = ax1.twinx()
        color2 = "#E74C3C"
        ax2.set_ylabel("Throughput (FPS)", color=color2, fontsize=12, fontweight="bold")
        line2 = ax2.plot(batch_sizes, throughputs, color=color2, marker="s", linewidth=2.5, markersize=10, label="Throughput")
        ax2.tick_params(axis="y", labelcolor=color2)

        # Add efficiency line (throughput/latency)
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 60))
        color3 = "#2ECC71"
        efficiency = [t/l if l > 0 else 0 for t, l in zip(throughputs, latencies)]
        line3 = ax3.plot(batch_sizes, efficiency, color=color3, marker="^", linewidth=2, markersize=8, linestyle="--", label="Efficiency")
        ax3.set_ylabel("Efficiency (FPS/ms)", color=color3, fontsize=10)
        ax3.tick_params(axis="y", labelcolor=color3)

        fig.suptitle(title, fontsize=14, fontweight="bold")

        lines = line1 + line2 + line3
        labels = [ln.get_label() for ln in lines]
        ax1.legend(lines, labels, loc="upper left")

        plt.tight_layout()
        self.figures.append(fig)
        return fig

    # ==================== Tegrastats Visualizations ====================

    def plot_tegrastats_overview(
        self,
        metrics: Dict[str, any],
        title: str = "Tegrastats System Overview",
    ) -> plt.Figure:
        """Create multi-panel overview of Tegrastats metrics.

        Args:
            metrics: Dict with 'cpu', 'gpu', 'memory', 'thermal' keys
            title: Graph title

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # CPU Utilization
        ax1 = axes[0, 0]
        cpu_data = metrics.get("cpu", {})
        if cpu_data:
            cores = list(cpu_data.keys())
            utilizations = [cpu_data[c].get("utilization", 0) for c in cores]
            colors = ["#E74C3C" if u > 80 else "#F39C12" if u > 50 else "#2ECC71" for u in utilizations]
            ax1.bar(cores, utilizations, color=colors, alpha=0.8)
            ax1.axhline(y=80, color="red", linestyle="--", alpha=0.5, label="High (80%)")
            ax1.set_ylim(0, 100)
        ax1.set_xlabel("CPU Core", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Utilization (%)", fontsize=10, fontweight="bold")
        ax1.set_title("CPU Utilization by Core", fontsize=12, fontweight="bold")
        ax1.grid(axis="y", alpha=0.3)

        # GPU Utilization
        ax2 = axes[0, 1]
        gpu_util = metrics.get("gpu", {}).get("utilization", 0)
        gpu_freq = metrics.get("gpu", {}).get("frequency_mhz", 0)
        ax2.barh(["Utilization"], [gpu_util], color="#9B59B6", alpha=0.8, height=0.4)
        ax2.set_xlim(0, 100)
        ax2.set_xlabel("Percentage / Value", fontsize=10, fontweight="bold")
        ax2.set_title(f"GPU: {gpu_util}% @ {gpu_freq}MHz", fontsize=12, fontweight="bold")
        ax2.grid(axis="x", alpha=0.3)

        # Memory Usage
        ax3 = axes[1, 0]
        mem = metrics.get("memory", {})
        mem_used = mem.get("used_mb", 0)
        mem_total = mem.get("total_mb", 1)
        mem_pct = (mem_used / mem_total * 100) if mem_total > 0 else 0
        ax3.barh(["RAM"], [mem_pct], color="#3498DB", alpha=0.8, height=0.4)
        ax3.set_xlim(0, 100)
        ax3.set_xlabel("Usage (%)", fontsize=10, fontweight="bold")
        ax3.set_title(f"Memory: {mem_used:.0f}MB / {mem_total:.0f}MB ({mem_pct:.1f}%)", fontsize=12, fontweight="bold")
        ax3.grid(axis="x", alpha=0.3)

        # Thermal
        ax4 = axes[1, 1]
        thermal = metrics.get("thermal", {})
        if thermal:
            zones = list(thermal.keys())
            temps = [thermal[z] for z in zones]
            colors = ["#E74C3C" if t > 80 else "#F39C12" if t > 60 else "#2ECC71" for t in temps]
            ax4.bar(zones, temps, color=colors, alpha=0.8)
            ax4.axhline(y=80, color="red", linestyle="--", alpha=0.5, label="Critical (80°C)")
            ax4.axhline(y=60, color="orange", linestyle="--", alpha=0.5, label="Warning (60°C)")
        ax4.set_xlabel("Thermal Zone", fontsize=10, fontweight="bold")
        ax4.set_ylabel("Temperature (°C)", fontsize=10, fontweight="bold")
        ax4.set_title("Thermal Status", fontsize=12, fontweight="bold")
        ax4.grid(axis="y", alpha=0.3)
        ax4.tick_params(axis="x", rotation=45)

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        self.figures.append(fig)
        return fig

    def plot_tegrastats_timeline(
        self,
        timeline_data: List[Dict[str, any]],
        title: str = "Tegrastats Timeline",
    ) -> plt.Figure:
        """Create timeline plot of Tegrastats metrics over time.

        Args:
            timeline_data: List of metric snapshots with timestamps
            title: Graph title

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        if not timeline_data:
            for ax in axes:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
            plt.tight_layout()
            self.figures.append(fig)
            return fig

        timestamps = range(len(timeline_data))

        # CPU utilization over time
        ax1 = axes[0]
        cpu_utils = [d.get("cpu_avg_utilization", 0) for d in timeline_data]
        ax1.fill_between(timestamps, 0, cpu_utils, alpha=0.3, color="#3498DB")
        ax1.plot(timestamps, cpu_utils, color="#3498DB", linewidth=2, label="CPU Avg")
        ax1.set_ylabel("CPU (%)", fontsize=10, fontweight="bold")
        ax1.set_ylim(0, 100)
        ax1.legend(loc="upper right")
        ax1.grid(alpha=0.3)

        # GPU utilization over time
        ax2 = axes[1]
        gpu_utils = [d.get("gpu_utilization", 0) for d in timeline_data]
        ax2.fill_between(timestamps, 0, gpu_utils, alpha=0.3, color="#9B59B6")
        ax2.plot(timestamps, gpu_utils, color="#9B59B6", linewidth=2, label="GPU")
        ax2.set_ylabel("GPU (%)", fontsize=10, fontweight="bold")
        ax2.set_ylim(0, 100)
        ax2.legend(loc="upper right")
        ax2.grid(alpha=0.3)

        # Temperature over time
        ax3 = axes[2]
        temps = [d.get("max_temperature", 0) for d in timeline_data]
        ax3.fill_between(timestamps, 0, temps, alpha=0.3, color="#E74C3C")
        ax3.plot(timestamps, temps, color="#E74C3C", linewidth=2, label="Max Temp")
        ax3.axhline(y=80, color="red", linestyle="--", alpha=0.5)
        ax3.set_xlabel("Sample Index", fontsize=10, fontweight="bold")
        ax3.set_ylabel("Temp (°C)", fontsize=10, fontweight="bold")
        ax3.legend(loc="upper right")
        ax3.grid(alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        self.figures.append(fig)
        return fig

    # ==================== Efficiency Visualizations ====================

    def plot_efficiency_metrics(
        self,
        efficiency_data: Dict[str, Dict[str, float]],
        title: str = "Efficiency Metrics Comparison",
    ) -> plt.Figure:
        """Create multi-bar chart comparing efficiency metrics across workloads.

        Args:
            efficiency_data: Dict with workload names and their efficiency metrics
                Example: {'ResNet': {'perf_per_watt': 10.5, 'energy_per_inference_j': 0.5}}
            title: Graph title

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        workloads = list(efficiency_data.keys())
        x = np.arange(len(workloads))

        # Performance per Watt
        ax1 = axes[0]
        ppw = [efficiency_data[w].get("perf_per_watt", 0) for w in workloads]
        bars1 = ax1.bar(x, ppw, color="#2ECC71", alpha=0.8)
        ax1.set_xlabel("Workload", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Perf/Watt (infer/s/W)", fontsize=10, fontweight="bold")
        ax1.set_title("Performance per Watt", fontsize=12, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(workloads, rotation=45, ha="right")
        ax1.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars1, ppw):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val:.2f}", ha="center", va="bottom", fontsize=9)

        # Energy per Inference
        ax2 = axes[1]
        epi = [efficiency_data[w].get("energy_per_inference_j", 0) for w in workloads]
        bars2 = ax2.bar(x, epi, color="#E74C3C", alpha=0.8)
        ax2.set_xlabel("Workload", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Energy (Joules)", fontsize=10, fontweight="bold")
        ax2.set_title("Energy per Inference", fontsize=12, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(workloads, rotation=45, ha="right")
        ax2.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars2, epi):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        # Throughput
        ax3 = axes[2]
        throughput = [efficiency_data[w].get("throughput_fps", 0) for w in workloads]
        bars3 = ax3.bar(x, throughput, color="#3498DB", alpha=0.8)
        ax3.set_xlabel("Workload", fontsize=10, fontweight="bold")
        ax3.set_ylabel("Throughput (FPS)", fontsize=10, fontweight="bold")
        ax3.set_title("Throughput", fontsize=12, fontweight="bold")
        ax3.set_xticks(x)
        ax3.set_xticklabels(workloads, rotation=45, ha="right")
        ax3.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars3, throughput):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val:.1f}", ha="center", va="bottom", fontsize=9)

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        self.figures.append(fig)
        return fig

    def plot_pareto_frontier(
        self,
        workloads: List[str],
        latencies: List[float],
        throughputs: List[float],
        power_values: List[float],
        title: str = "Pareto Frontier: Latency vs Throughput",
    ) -> plt.Figure:
        """Create scatter plot with Pareto frontier for efficiency analysis.

        Args:
            workloads: List of workload/config names
            latencies: Latency values (ms)
            throughputs: Throughput values (FPS)
            power_values: Power consumption (W) for color coding
            title: Graph title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        scatter = ax.scatter(
            latencies, throughputs,
            c=power_values, cmap="RdYlGn_r",
            s=200, alpha=0.7, edgecolors="black", linewidth=1
        )

        for i, workload in enumerate(workloads):
            ax.annotate(
                workload,
                (latencies[i], throughputs[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

        # Add colorbar for power
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Power (W)", fontsize=10, fontweight="bold")

        ax.set_xlabel("Latency (ms)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Throughput (FPS)", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Ideal direction annotation
        ax.annotate(
            "← Lower latency\n↑ Higher throughput\nIdeal direction",
            xy=(0.02, 0.98), xycoords="axes fraction",
            fontsize=9, ha="left", va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        )

        plt.tight_layout()
        self.figures.append(fig)
        return fig

    # ==================== Variability Visualizations ====================

    def plot_variability_metrics(
        self,
        variability_data: Dict[str, Dict[str, float]],
        title: str = "Latency Variability Analysis",
    ) -> plt.Figure:
        """Create variability metrics comparison chart.

        Args:
            variability_data: Dict with workload names and variability metrics
                Example: {'ResNet': {'cv_percent': 5.2, 'jitter_ms': 1.5, 'iqr_ms': 3.2}}
            title: Graph title

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        workloads = list(variability_data.keys())
        x = np.arange(len(workloads))

        # Coefficient of Variation
        ax1 = axes[0]
        cv = [variability_data[w].get("cv_percent", 0) for w in workloads]
        colors1 = ["#E74C3C" if c > 20 else "#F39C12" if c > 10 else "#2ECC71" for c in cv]
        ax1.bar(x, cv, color=colors1, alpha=0.8)
        ax1.axhline(y=10, color="orange", linestyle="--", alpha=0.5, label="Moderate (10%)")
        ax1.axhline(y=20, color="red", linestyle="--", alpha=0.5, label="High (20%)")
        ax1.set_xlabel("Workload", fontsize=10, fontweight="bold")
        ax1.set_ylabel("CV (%)", fontsize=10, fontweight="bold")
        ax1.set_title("Coefficient of Variation", fontsize=12, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(workloads, rotation=45, ha="right")
        ax1.legend(fontsize=8)
        ax1.grid(axis="y", alpha=0.3)

        # Jitter
        ax2 = axes[1]
        jitter = [variability_data[w].get("jitter_ms", 0) for w in workloads]
        ax2.bar(x, jitter, color="#9B59B6", alpha=0.8)
        ax2.set_xlabel("Workload", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Jitter (ms)", fontsize=10, fontweight="bold")
        ax2.set_title("Latency Jitter", fontsize=12, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(workloads, rotation=45, ha="right")
        ax2.grid(axis="y", alpha=0.3)

        # IQR
        ax3 = axes[2]
        iqr = [variability_data[w].get("iqr_ms", 0) for w in workloads]
        ax3.bar(x, iqr, color="#3498DB", alpha=0.8)
        ax3.set_xlabel("Workload", fontsize=10, fontweight="bold")
        ax3.set_ylabel("IQR (ms)", fontsize=10, fontweight="bold")
        ax3.set_title("Interquartile Range", fontsize=12, fontweight="bold")
        ax3.set_xticks(x)
        ax3.set_xticklabels(workloads, rotation=45, ha="right")
        ax3.grid(axis="y", alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        self.figures.append(fig)
        return fig

    def plot_consistency_rating(
        self,
        workloads: List[str],
        ratings: List[str],
        cv_values: List[float],
        title: str = "Consistency Rating Overview",
    ) -> plt.Figure:
        """Create visual consistency rating chart.

        Args:
            workloads: List of workload names
            ratings: Consistency ratings ('very_consistent', 'consistent', etc.)
            cv_values: Coefficient of variation values
            title: Graph title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Map ratings to colors and numeric values
        rating_map = {
            "very_consistent": (4, "#2ECC71"),
            "consistent": (3, "#3498DB"),
            "moderately_consistent": (2, "#F39C12"),
            "high_variability": (1, "#E74C3C"),
        }

        y_vals = [rating_map.get(r, (0, "#95A5A6"))[0] for r in ratings]
        colors = [rating_map.get(r, (0, "#95A5A6"))[1] for r in ratings]

        bars = ax.barh(workloads, y_vals, color=colors, alpha=0.8)

        # Add CV values as text
        for bar, cv in zip(bars, cv_values):
            ax.text(
                bar.get_width() + 0.1,
                bar.get_y() + bar.get_height() / 2,
                f"CV: {cv:.1f}%",
                va="center",
                fontsize=10,
            )

        ax.set_xlim(0, 5)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(["High Var", "Moderate", "Consistent", "Very Consistent"])
        ax.set_xlabel("Consistency Rating", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        self.figures.append(fig)
        return fig

    def plot_outlier_analysis(
        self,
        latencies: List[float],
        workload_name: str = "Workload",
        title: str = "Outlier Analysis",
    ) -> plt.Figure:
        """Create box plot with outlier visualization.

        Args:
            latencies: List of latency values
            workload_name: Name of the workload
            title: Graph title

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Box plot
        ax1 = axes[0]
        bp = ax1.boxplot(latencies, patch_artist=True)
        bp["boxes"][0].set_facecolor("#3498DB")
        bp["boxes"][0].set_alpha(0.7)
        ax1.set_ylabel("Latency (ms)", fontsize=12, fontweight="bold")
        ax1.set_title(f"{workload_name} - Box Plot", fontsize=12, fontweight="bold")
        ax1.set_xticklabels([workload_name])
        ax1.grid(axis="y", alpha=0.3)

        # Histogram with outlier threshold
        ax2 = axes[1]
        q1 = np.percentile(latencies, 25)
        q3 = np.percentile(latencies, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        ax2.hist(latencies, bins=30, color="#3498DB", alpha=0.7, edgecolor="black")
        ax2.axvline(x=lower_bound, color="red", linestyle="--", label=f"Lower bound: {lower_bound:.1f}")
        ax2.axvline(x=upper_bound, color="red", linestyle="--", label=f"Upper bound: {upper_bound:.1f}")
        ax2.axvline(x=np.median(latencies), color="green", linestyle="-", linewidth=2, label=f"Median: {np.median(latencies):.1f}")

        outliers = [l for l in latencies if l < lower_bound or l > upper_bound]
        ax2.set_xlabel("Latency (ms)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Frequency", fontsize=12, fontweight="bold")
        ax2.set_title(f"Distribution ({len(outliers)} outliers)", fontsize=12, fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(axis="y", alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        self.figures.append(fig)
        return fig


__all__ = ["PerformanceVisualizer"]
