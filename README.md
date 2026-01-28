# AutoPerfPy
## Performance Analysis & Optimization Toolkit

A comprehensive collection of Python scripts and shell utilities for **performance benchmarking, monitoring, and analysis** across automotive AI, LLM inference, and edge computing workloads.

> **⚠️ Disclaimer**: All data in this repository is **synthetic** and created for **practice/demonstration purposes only**. Results are not representative of real hardware performance.

**Focus**: Performance optimization for machine learning inference and edge computing workloads

---

## 📋 Quick Links

- [Quick Start](#-quick-start)
- [Features](#-features)
- [Installation](#-setup--installation)
- [Usage Examples](#-usage-examples)
- [Testing](#-testing)
- [Changelog](CHANGELOG.md)
- [Learning Path](#-learning-path)
- [Troubleshooting](#-troubleshooting)

---

## ⚡ Quick Start

### **3-in-1 Repository Structure**

This repository provides three ways to use the tools:

#### 1️⃣ **Legacy Scripts** (Quick CLI, no setup)
```bash
# Run standalone scripts directly - no dependencies beyond Python
python scripts/calculate_p99_latency.py scripts/data/automotive_benchmark_data.csv
python benchmarks/batching_tradeoff.py --images 1000 --batch-sizes 1 4 8 16
python monitoring/gpu_monitor.py --window 10 --interval 2
```
✅ **Best for**: Quick analysis, no environment setup needed  
✅ **Why keep**: Standalone tools work immediately, great for learning

#### 2️⃣ **AutoPerfPy Package + CLI** (Programmatic + CLI)
```bash
# Install package (one-time setup)
pip install -e .

# Use via CLI
autoperfpy analyze latency --csv scripts/data/automotive_benchmark_data.csv
autoperfpy benchmark batching --batch-sizes 1,4,8,16
autoperfpy monitor gpu --duration 60

# Or import as library
python examples/analyze_percentiles.py
python examples/benchmark_batching.py
```
✅ **Best for**: Production code, reusable components, integration  
✅ **Why use**: Better abstraction, configuration, error handling

#### 3️⃣ **Import Directly as Library** (Programmatic only)
```python
from autoperfpy import PercentileLatencyAnalyzer, BatchingTradeoffBenchmark, ConfigManager

# Load configuration
config = ConfigManager.load_or_default("config.yaml")

# Use analyzers
analyzer = PercentileLatencyAnalyzer(config)
result = analyzer.analyze("benchmark.csv")

# Use benchmarks
benchmark = BatchingTradeoffBenchmark(config)
results = benchmark.run(batch_sizes=[1, 4, 8, 16, 32])
```
✅ **Best for**: Embedding in other Python projects  
✅ **Why use**: Full abstraction, extensible via inheritance

---

## �️ Architecture: Legacy Scripts + Modern Package

**Why both coexist:**

| Aspect | Legacy Scripts | AutoPerfPy Package |
|--------|----------------|-------------------|
| **Setup** | None - run directly | `pip install -e .` (one-time) |
| **Use Case** | Quick analysis, learning | Production code, integration |
| **Dependencies** | Minimal (numpy, pandas) | Managed via setup.py |
| **Reusability** | Low - copy/paste | High - import and extend |
| **Error Handling** | Basic | Comprehensive |
| **Testing** | Manual | Can be automated |
| **Documentation** | Comments in code | Docstrings + examples |
| **When to Use** | Fast prototyping | Team projects, deployment |

**📌 Design Philosophy**: 
- **Scripts** = Direct learning tools (see implementation immediately)
- **Package** = Abstracted, reusable components (better for scaling)
- **Examples** = Bridge showing how to use the package
### ℹ️ Intentional Duplication

Some logic exists in both scripts and package (e.g., percentile calculation, GPU monitoring):

✅ **This is intentional and beneficial** because:
- Scripts can be run **without any setup** → lower barrier to entry
- Package provides **better abstraction** for production use
- Each serves a different audience (students/learners vs engineers)
- You can update logic in one place and use whichever version fits your need

📝 **If you modify percentile logic**, update both:
1. `scripts/calculate_p99_latency.py::PercentileCalculator.calculate_percentiles()`
2. `autoperfpy/core/utils.py::LatencyStats.calculate_percentiles()`
---

## �🎯 Features

### Legacy Scripts (Direct, Standalone)
| Feature | Module | Purpose | Run As |
|---------|--------|---------|--------|
| **Automotive Log Analysis** | `scripts/automotive_parser_log.py` | Extract and analyze latency events from performance logs | `python scripts/automotive_parser_log.py` |
| **Percentile Latency** | `scripts/calculate_p99_latency.py` | Calculate P99, P95, P50 statistics for benchmarks | `python scripts/calculate_p99_latency.py <csv>` |
| **LLM Benchmarking** | `benchmarks/llm_latency_benchmark.py` | Measure TTFT, time-per-token, and throughput metrics | `python benchmarks/llm_latency_benchmark.py` |
| **Batching Trade-offs** | `benchmarks/batching_tradeoff.py` | Analyze latency vs throughput with different batch sizes | `python benchmarks/batching_tradeoff.py --images 1000` |
| **LLM Memory Monitoring** | `monitoring/llm_monitor.py` | Track GPU memory and KV cache usage during inference | `python monitoring/llm_monitor.py` |
| **GPU Monitoring** | `monitoring/gpu_monitor.py` | Real-time GPU utilization and metrics tracking | `python monitoring/gpu_monitor.py` |
| **Process Management** | `tools/process_monitor.py` | Monitor and prevent zombie processes | `python tools/process_monitor.py` |
| **Process Termination** | `tools/detect_hung_proc.sh` | Detect and kill hung processes gracefully | `bash tools/detect_hung_proc.sh` |
| **Performance Graphs & PDF** | `scripts/generate_performance_graphs.py` | Generate consolidated performance report PDF | `python scripts/generate_performance_graphs.py` |

### AutoPerfPy Package (Abstracted, Reusable)
| Component | Location | Purpose | Import |
|-----------|----------|---------|--------|
| **Percentile Analyzer** | `autoperfpy/analyzers/latency.py` | OOP-based percentile analysis from CSV | `from autoperfpy import PercentileLatencyAnalyzer` |
| **DNN Pipeline Analyzer** | `autoperfpy/analyzers/dnn_pipeline.py` | TensorRT/DriveWorks DNN inference analysis | `from autoperfpy import DNNPipelineAnalyzer` |
| **Tegrastats Analyzer** | `autoperfpy/analyzers/tegrastats.py` | NVIDIA Jetson tegrastats analysis | `from autoperfpy import TegrastatsAnalyzer` |
| **Efficiency Analyzer** | `autoperfpy/analyzers/efficiency.py` | Power efficiency and Perf/Watt analysis | `from autoperfpy import EfficiencyAnalyzer` |
| **Variability Analyzer** | `autoperfpy/analyzers/variability.py` | Latency jitter and consistency analysis | `from autoperfpy import VariabilityAnalyzer` |
| **Batching Benchmark** | `autoperfpy/benchmarks/latency.py` | Batch size impact analysis with optimization hints | `from autoperfpy import BatchingTradeoffBenchmark` |
| **LLM Benchmark** | `autoperfpy/benchmarks/latency.py` | TTFT & time-per-token measurement | `from autoperfpy import LLMLatencyBenchmark` |
| **GPU Monitor** | `autoperfpy/monitoring/gpu.py` | Real-time GPU memory monitoring | `from autoperfpy import GPUMemoryMonitor` |
| **KV Cache Monitor** | `autoperfpy/monitoring/gpu.py` | LLM KV cache estimation & tracking | `from autoperfpy import LLMKVCacheMonitor` |
| **Performance Visualizer** | `autoperfpy/reporting/visualizer.py` | Create graphs: latency, throughput, power, memory, DNN layers | `from autoperfpy import PerformanceVisualizer` |
| **PDF Report Generator** | `autoperfpy/reporting/pdf_generator.py` | Consolidate graphs into professional PDF reports | `from autoperfpy import PDFReportGenerator` |
| **HTML Report Generator** | `autoperfpy/reporting/html_generator.py` | Create interactive HTML reports with navigation | `from autoperfpy import HTMLReportGenerator` |
| **Configuration System** | `autoperfpy/config/` | YAML/JSON-based config management | `from autoperfpy import ConfigManager` |
| **CLI Interface** | `autoperfpy/cli.py` | Unified command-line interface | `autoperfpy <command> [options]` |

---

## 📁 Project Structure

```
AutoPerfPy/
│
├── 📄 README.md                          # Documentation
├── 📄 setup.py                           # Package installation
├── 📄 config.yaml                        # Default configuration
├── 📄 requirements.txt                   # Python dependencies
│
├── 📁 autoperfpy/                        # Main package (NEW!)
│   ├── __init__.py                       # Package exports
│   ├── cli.py                            # Command-line interface
│   │
│   ├── 📁 config/                        # Configuration system
│   │   ├── __init__.py
│   │   ├── defaults.py                   # Default settings
│   │   └── config.py                     # Config management
│   │
│   ├── 📁 core/                          # Core abstractions & utilities
│   │   ├── __init__.py
│   │   ├── base.py                       # Base classes
│   │   └── utils.py                      # Utilities & helpers
│   │
│   ├── 📁 analyzers/                     # Analysis modules
│   │   ├── __init__.py
│   │   └── latency.py                    # Latency analyzers
│   │
│   ├── 📁 benchmarks/                    # Benchmarking modules
│   │   ├── __init__.py
│   │   └── latency.py                    # Latency benchmarks
│   │
│   └── 📁 monitoring/                    # Monitoring modules
│       ├── __init__.py
│       └── gpu.py                        # GPU monitoring
│
├── 📁 examples/                          # Example usage scripts
│   ├── analyze_percentiles.py            # Latency analysis example
│   ├── benchmark_batching.py             # Batching example
│   └── monitor_gpu.py                    # GPU monitoring example
│
    ├── 📁 scripts/                           # Legacy analysis scripts
│   ├── automotive_parser_log.py
│   ├── calculate_p99_latency.py
│   └── data/
│       └── automotive_benchmark_data.csv
│
├── 📁 benchmarks/                        # Legacy benchmark scripts
│   ├── llm_latency_benchmark.py
│   └── batching_tradeoff.py
│
└── 📁 tools/                             # Legacy utility scripts
    ├── process_monitor.py
    ├── llm_memory_calculator.py
    └── service_benchmarks.py
```

**New Architecture**: The `autoperfpy/` package provides object-oriented abstractions for all functionality, making it easy to:
- Use as a library in your Python code
- Extend with custom analyzers/benchmarks
- Configure via YAML files
- Use via command-line interface
│
└── 📁 tools/                             # System utilities
    ├── detect_hung_proc.sh               # Hung process detector
    ├── tensorrt_build.sh                 # TensorRT build helper
    ├── process_monitor.py                # Process lifecycle manager
    ├── llm_memory_calculator.py          # Memory requirement calculator
    └── service_benchmarks.py             # Service performance testing
```

---

## 🛠️ Setup & Installation

### Prerequisites

- **Python**: 3.8 or higher
- **OS**: Linux/Unix environment (tested on Ubuntu 20.04+)
- **Shell**: Bash (for shell scripts)
- **NVIDIA**: Optional - CUDA/cuDNN for GPU acceleration

### Installation Steps

```bash
# 1. Clone repository
git clone <repository-url>
cd AutoPerfPy

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Make shell scripts executable (optional)
chmod +x tools/*.sh

# 4. Verify installation
python -c "import numpy, pandas; print('✓ Dependencies installed')"
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | Latest | Numerical computing & array operations |
| `pandas` | Latest | Data analysis & CSV processing |
| `requests` | Latest | HTTP library for API calls |
| `fastapi` | Latest | Web API framework |
| `uvicorn` | Latest | ASGI server for FastAPI |
| `pytest` | Latest | Testing framework |

See [requirements.txt](requirements.txt) for exact versions.

---

## 📚 Module Documentation

### 🔧 Core Scripts

#### 1. **automotive_parser_log.py**
Parses automotive performance logs and extracts latency events.

**Purpose**: Identify and analyze performance anomalies in automotive inference pipelines

**Features**:
- ✓ Extract latency events exceeding thresholds
- ✓ Parse NVIDIA Drive Orin AGX logs
- ✓ Frame-level metrics (inference vs end-to-end latency)
- ✓ Timestamp correlation

**Usage**:
```bash
python scripts/automotive_parser_log.py <log_file> [threshold_ms]
```

**Example**:
```bash
python scripts/automotive_parser_log.py performance.log 50
# Output: Timestamped events with frame IDs and latencies
```

**Expected Input Format**:
```
[2024-01-15 10:30:45.123] Frame 0001 | Inference: 25.5ms | E2E: 28.3ms
[2024-01-15 10:30:46.234] Frame 0002 | Inference: 26.1ms | E2E: 29.1ms
```

**Output**: Summary of events exceeding threshold with timestamps

---

#### 2. **calculate_p99_latency.py**
Comprehensive benchmark analysis with percentile latency calculations.

**Purpose**: Analyze performance distributions and identify tail latencies (P99, P95, etc.)

**Features**:
- ✓ Calculate P99, P95, P50 (median) percentiles
- ✓ Multi-workload analysis (YOLO, ResNet, etc.)
- ✓ Batch size performance breakdown
- ✓ Power consumption correlation
- ✓ Statistical summaries (mean, std dev, min, max)

**Usage**:
```bash
python scripts/calculate_p99_latency.py <benchmark_csv>
```

**Example**:
```bash
python scripts/calculate_p99_latency.py scripts/data/automotive_benchmark_data.csv
```

**Expected CSV Format**:
```csv
timestamp,workload,batch_size,latency_ms,power_w
2024-01-15 10:30:45,yolo_v8,1,25.5,65.2
2024-01-15 10:30:46,yolo_v8,1,26.1,64.8
2024-01-15 10:30:47,yolo_v8,4,22.3,85.5
2024-01-15 10:30:48,resnet50,1,15.2,45.3
```

**Output Example**:
```
╔════════════════════════════════════════╗
║        YOLO V8 (Batch Size: 1)        ║
╠════════════════════════════════════════╣
║ P99:     28.50 ms                     ║
║ P95:     27.80 ms                     ║
║ P50:     25.50 ms (Median)            ║
║ Mean:    25.67 ms ± 1.23 ms           ║
║ Range:   [23.1, 29.5] ms              ║
║ Power:   65.2 ± 2.1 W                 ║
╚════════════════════════════════════════╝
```

---

#### 3. **pytorch_operations.py**
PyTorch operation profiling and optimization.

**Purpose**: Benchmark and analyze PyTorch operations for optimization

---

#### 4. **tensorrt_llm_interface.py**
TensorRT LLM integration and optimization interface.

**Purpose**: Manage TensorRT LLM inference pipelines

---

### 🧠 DNN Pipeline Analysis (NEW!)

#### **DNNPipelineAnalyzer (autoperfpy.analyzers)**
Analyze TensorRT/DriveWorks DNN inference performance.

**Purpose**: Profile neural network inference on NVIDIA platforms with GPU and DLA accelerators.

**Features**:
- ✓ Layer-by-layer inference timing analysis
- ✓ DLA vs GPU execution split tracking
- ✓ Memory transfer overhead (H2D/D2H) analysis
- ✓ Batch scaling analysis
- ✓ Automatic optimization recommendations

**CLI Usage**:
```bash
# Analyze from CSV layer timings
autoperfpy analyze dnn-pipeline --csv layer_times.csv --batch-size 4

# Analyze from profiler output
autoperfpy analyze dnn-pipeline --profiler profiler_output.txt
```

**Programmatic Usage**:
```python
from autoperfpy import DNNPipelineAnalyzer

analyzer = DNNPipelineAnalyzer(config={"top_n_layers": 10})

# From profiler output
result = analyzer.analyze_profiler_output(profiler_text)

# From CSV
result = analyzer.analyze_layer_csv("layer_times.csv", batch_size=4)

# From raw data
result = analyzer.analyze_from_data(
    layer_timings=[
        {"name": "conv1", "layer_type": "Conv", "execution_time_ms": 2.0, "device": "GPU"},
        {"name": "conv2", "layer_type": "Conv", "execution_time_ms": 1.5, "device": "DLA0"},
    ],
    batch_size=4
)

# Get recommendations
print(result.metrics["recommendations"])

# Compare two engine configurations
comparison = analyzer.compare_engines(baseline_metrics, optimized_metrics)
print(f"Speedup: {comparison['latency_improvement_percent']:.1f}%")
```

**Output Metrics**:
- Timing breakdown (total, GPU, DLA, memory overhead)
- Throughput (FPS)
- Device split percentages
- Slowest layers list
- Batch scaling analysis
- DLA vs GPU comparison

---

### 📊 Tegrastats Analysis (NEW!)

#### **TegrastatsAnalyzer (autoperfpy.analyzers)**
Analyze NVIDIA Jetson tegrastats output for system health monitoring.

**Purpose**: Monitor CPU, GPU, memory, and thermal metrics on Jetson platforms.

**Features**:
- ✓ Per-core CPU utilization tracking
- ✓ GPU utilization and frequency monitoring
- ✓ Memory pressure detection
- ✓ Thermal throttling detection
- ✓ Health status assessment (healthy/warning/critical)

**CLI Usage**:
```bash
autoperfpy analyze tegrastats --log tegrastats.log --throttle-threshold 85
```

**Programmatic Usage**:
```python
from autoperfpy import TegrastatsAnalyzer

analyzer = TegrastatsAnalyzer(throttle_temp_threshold=85.0)

# Analyze log file
result = analyzer.analyze("tegrastats.log")

# Or analyze raw lines
result = analyzer.analyze_lines(log_lines)

# Check health status
health = result.metrics["health"]
if health["status"] != "healthy":
    print("Warnings:", health["warnings"])
```

---

### ⚡ Efficiency Analysis (NEW!)

#### **EfficiencyAnalyzer (autoperfpy.analyzers)**
Analyze power efficiency metrics for inference workloads.

**Purpose**: Calculate Performance/Watt and energy consumption metrics.

**Features**:
- ✓ Performance per Watt calculation
- ✓ Energy per inference (Joules)
- ✓ Cost per inference estimation
- ✓ Pareto-optimal configuration identification

**CLI Usage**:
```bash
autoperfpy analyze efficiency --csv benchmark_data.csv
```

**Programmatic Usage**:
```python
from autoperfpy import EfficiencyAnalyzer

analyzer = EfficiencyAnalyzer()
result = analyzer.analyze("benchmark_data.csv")

for workload, metrics in result.metrics.items():
    print(f"{workload}: {metrics['perf_per_watt']:.2f} infer/s/W")
```

---

### 📈 Variability Analysis (NEW!)

#### **VariabilityAnalyzer (autoperfpy.analyzers)**
Analyze latency variability and consistency.

**Purpose**: Measure jitter, coefficient of variation, and identify outliers.

**Features**:
- ✓ Coefficient of Variation (CV) calculation
- ✓ Jitter measurement (std dev)
- ✓ Interquartile Range (IQR)
- ✓ Outlier detection and counting
- ✓ Consistency rating (very_consistent → high_variability)

**CLI Usage**:
```bash
autoperfpy analyze variability --csv latency_data.csv --column latency_ms
```

**Programmatic Usage**:
```python
from autoperfpy import VariabilityAnalyzer

analyzer = VariabilityAnalyzer()
result = analyzer.analyze("latency_data.csv")

print(f"CV: {result.metrics['cv_percent']:.2f}%")
print(f"Jitter: {result.metrics['jitter_ms']:.2f}ms")
print(f"Rating: {result.metrics['consistency_rating']}")
```

---

### 📊 Benchmarking Modules

#### **llm_latency_benchmark.py**
Measures LLM inference latency metrics.

**Purpose**: Profile LLM performance for real-time applications

**Metrics Measured**:
- **TTFT** (Time-To-First-Token): Prefill phase latency
- **Time-Per-Token**: Token generation speed
- **Throughput**: Tokens per second
- **Percentiles**: P50, P90, P99 latencies

**Use Cases**:
- Chatbot responsiveness optimization
- Code generation throughput analysis
- Content creation system profiling

**Key Concepts**:
```
Prefill Phase: Process input tokens → generate first output token
  └─ High latency due to KV cache population
  └─ Critical for user experience (time before response appears)

Decode Phase: Generate remaining output tokens one-by-one
  └─ Typically lower latency than prefill
  └─ Affects streaming response speed
```

---

#### **batching_tradeoff.py**
Analyzes batch size impact on latency and throughput.

**Purpose**: Find optimal batch size for your inference requirements

**Key Concept**:
```
Latency-Throughput Trade-off:
┌─────────────────────────────────────────┐
│ Larger Batch Size:                      │
│ ✓ Higher throughput (images/sec)        │
│ ✗ Higher latency per image              │
│                                         │
│ Small Batch Size:                       │
│ ✓ Lower latency                         │
│ ✗ Lower throughput                      │
└─────────────────────────────────────────┘
```

**Usage**:
```bash
python benchmarks/batching_tradeoff.py [--images NUM] [--batch-sizes LIST]
```

---

### 🖥️ Monitoring Modules

#### **llm_monitor.py**
GPU memory monitoring during LLM inference.

**Purpose**: Track memory usage and predict OOM conditions

**Monitors**:
- GPU memory utilization
- KV cache size growth per token
- Memory per batch/sequence length
- OOM risk detection

**Configuration Example**:
```python
# Model: 7B LLM with 32 layers
num_layers = 32
num_heads = 32
head_size = 128
max_sequence_length = 2048
precision = "fp16"  # or "fp32"
```

---

#### **gpu_monitor.py**
Real-time GPU metrics tracking.

**Purpose**: Monitor GPU utilization during workload execution

---

### 🔌 System Tools

#### **process_monitor.py**
Process lifecycle management and zombie prevention.

**Purpose**: Monitor subprocess execution and prevent zombie processes

**Key Concepts**:
- Zombie Process: Child process terminated but parent didn't call `wait()`
- Solution: Always call `wait()` or `communicate()` on child processes

**Features**:
- ✓ Subprocess timeout handling
- ✓ Signal management (SIGTERM, SIGKILL)
- ✓ Zombie detection and cleanup

**Usage**:
```bash
python tools/process_monitor.py <command> [--timeout SECONDS]
python tools/process_monitor.py "sleep 100" --timeout 30
```

---

#### **detect_hung_proc.sh**
Detect and terminate hung processes.

**Purpose**: Automatically kill processes exceeding runtime thresholds

**Features**:
- ✓ Pattern-based process matching
- ✓ Graceful termination with escalation
- ✓ Structured logging with timestamps
- ✓ Zombie process detection

**Configuration**:
```bash
# Edit these variables in the script:
PROCESS_PATTERN="trtexec"              # Process name to match
MAX_RUNTIME_MINUTES=30                 # Timeout in minutes
LOG_FILE="/var/log/process_monitor.log" # Log file path
```

---

### 📊 Reporting & Visualization (NEW!)

#### **generate_performance_graphs.py**
Generate consolidated performance analysis PDF with graphs.

**Purpose**: Create professional reports with multiple performance visualizations

**Included Graphs**:
1. **Latency Percentiles** - Compare P50, P95, P99 across workloads
2. **Latency vs Throughput** - Show fundamental batching trade-off
3. **Power vs Performance** - Efficiency analysis
4. **GPU Memory Timeline** - Track memory usage over time
5. **Relative Performance** - Compare optimization improvements
6. **Distribution Analysis** - Understand latency variance

**Features**:
- ✓ Professional PDF output with title page and metadata
- ✓ Automatic graph layout and formatting
- ✓ Captions and summary page
- ✓ Customizable metrics and data

**Usage**:
```bash
# Generate standalone report (no package setup)
python scripts/generate_performance_graphs.py --output my_report.pdf

# Or use the package example
python examples/generate_performance_report.py
python examples/generate_performance_report.py --show  # View interactively
```

**Output**:
- Consolidated PDF with title page, summary, and 6 graphs
- Suitable for presentations and documentation
- File size: ~500KB per report

---

#### **PerformanceVisualizer (autoperfpy.reporting)**
Python class for creating performance graphs programmatically.

**Available Graphs**:
```python
from autoperfpy import PerformanceVisualizer

viz = PerformanceVisualizer()

# === Core Latency Graphs ===
fig1 = viz.plot_latency_percentiles(workload_data)
fig2 = viz.plot_latency_throughput_tradeoff(batch_sizes, latencies, throughputs)
fig3 = viz.plot_power_vs_performance(workloads, power, perf)
fig4 = viz.plot_gpu_memory_timeline(timestamps, memory_used)
fig5 = viz.plot_relative_performance(baseline, configs_data)
fig6 = viz.plot_distribution(data_dict)

# === DNN Pipeline Graphs ===
fig7 = viz.plot_layer_timings(layers)              # Layer execution times
fig8 = viz.plot_device_split(gpu_time, dla_time)   # DLA vs GPU pie chart
fig9 = viz.plot_memory_transfer_timeline(transfers) # H2D/D2H transfers
fig10 = viz.plot_batch_scaling(batch_metrics)       # Batch size analysis

# === Tegrastats Graphs ===
fig11 = viz.plot_tegrastats_overview(metrics)       # CPU/GPU/Memory/Thermal
fig12 = viz.plot_tegrastats_timeline(timeline_data) # Metrics over time

# === Efficiency Graphs ===
fig13 = viz.plot_efficiency_metrics(efficiency_data) # Perf/Watt, Energy/Inference
fig14 = viz.plot_pareto_frontier(workloads, latencies, throughputs, power)

# === Variability Graphs ===
fig15 = viz.plot_variability_metrics(variability_data) # CV, Jitter, IQR
fig16 = viz.plot_consistency_rating(workloads, ratings, cv_values)
fig17 = viz.plot_outlier_analysis(latencies, workload_name)

# Save any figure
viz.save_figure(fig1, "graph.png", dpi=300)
```

---

#### **PDFReportGenerator (autoperfpy.reporting)**
Consolidate multiple graphs into professional PDF reports.

**Features**:
```python
from autoperfpy import PDFReportGenerator

pdf_gen = PDFReportGenerator(
    title="My Performance Report",
    author="Data Analysis Team"
)

# Add metadata
pdf_gen.add_metadata("System", "GPU Inference Platform")
pdf_gen.add_metadata("Date", "2024-01-15")

# Add figures from visualizer
pdf_gen.add_figures_from_visualizer(viz, captions=[
    "Latency percentiles",
    "Batching trade-off",
    "Power efficiency",
])

# Generate PDF
pdf_gen.generate_pdf("report.pdf", include_summary=True)
```

**Output**:
- Title page with metadata
- Summary page with table of contents
- Graph pages with captions
- PDF metadata (title, author, creation date)

---

#### **HTMLReportGenerator (autoperfpy.reporting)**
Generate interactive HTML reports with navigation, theming, and executive summaries.

**Features**:
- ✓ Light and dark theme support
- ✓ Interactive navigation with smooth scrolling
- ✓ Executive summary cards with status indicators
- ✓ Data tables with styling
- ✓ Section-based organization
- ✓ Embedded images (base64) or external files
- ✓ Print-friendly CSS
- ✓ Responsive design for mobile

**CLI Usage**:
```bash
# Generate HTML report from CSV data
autoperfpy report html --csv data.csv --output report.html --title "My Analysis"

# Use dark theme
autoperfpy report html --csv data.csv --output report.html --theme dark

# Generate demo report (no data file)
autoperfpy report html --output demo_report.html
```

**Programmatic Usage**:
```python
from autoperfpy import HTMLReportGenerator, PerformanceVisualizer

# Create report with dark theme
report = HTMLReportGenerator(
    title="DNN Performance Analysis",
    author="ML Team",
    theme="dark"
)

# Add metadata
report.add_metadata("Platform", "NVIDIA Jetson AGX Orin")
report.add_metadata("Model", "ResNet50 INT8")

# Add executive summary items with status
report.add_summary_item("Mean Latency", "25.3", "ms", status="good")
report.add_summary_item("P99 Latency", "42.1", "ms", status="warning")
report.add_summary_item("Throughput", "156", "FPS", status="good")
report.add_summary_item("CV", "15.2", "%", status="warning")

# Create sections
report.add_section("Latency Analysis", "Percentile breakdown by workload")
report.add_section("Power Efficiency", "Performance per Watt metrics")

# Add figures with section assignment
viz = PerformanceVisualizer()
fig1 = viz.plot_latency_percentiles(latency_data)
report.add_figure(fig1, "Latency Percentiles", section="Latency Analysis")

fig2 = viz.plot_power_vs_performance(workloads, power, perf)
report.add_figure(fig2, "Power vs Performance", section="Power Efficiency")

# Add data tables
report.add_table(
    title="Top Configurations",
    headers=["Config", "Latency (ms)", "Throughput (FPS)", "Power (W)"],
    rows=[
        ["Batch=4, FP16", "28.5", "140", "15.2"],
        ["Batch=8, INT8", "25.3", "156", "14.8"],
    ],
    section="Latency Analysis"
)

# Generate HTML file
report.generate_html("performance_report.html")
```

**Output Features**:
- Responsive header with metadata badges
- Sticky navigation bar with section links
- Summary cards with color-coded status (good/warning/critical)
- Embedded graphs organized by section
- Styled data tables
- Footer with generation timestamp

---

**Termination Flow**:
```
1. Find processes matching pattern
2. Check if runtime exceeds threshold
3. Send SIGTERM (graceful termination)
4. Wait up to 10 seconds for exit
5. If still running → Send SIGKILL
6. Log all actions and results
```

**Usage**:
```bash
# Manual execution
./tools/detect_hung_proc.sh

# Setup as cron job (every 5 minutes)
(crontab -l 2>/dev/null; echo "*/5 * * * * /path/to/detect_hung_proc.sh") | crontab -

# May require sudo for permission
sudo ./tools/detect_hung_proc.sh
```

---

#### **tensorrt_build.sh**
TensorRT model building utilities.

**Purpose**: Streamline TensorRT engine compilation

---

#### **llm_memory_calculator.py**
Estimate memory requirements for LLM inference.

**Purpose**: Predict GPU memory needed before deployment

**Calculates**:
- Model weight memory
- KV cache memory
- Activation memory
- Batch size limitations

---

#### **service_benchmarks.py**
Service-level performance testing.

**Purpose**: Benchmark inference service response times and throughput

---

## 📖 Usage Examples

### Example 1: Analyze Performance Logs
```bash
# Parse automotive logs and find events exceeding 50ms latency
python scripts/automotive_parser_log.py performance.log 50
```

### Example 2: Calculate Percentile Latencies
```bash
# Analyze benchmark data
python scripts/calculate_p99_latency.py scripts/data/automotive_benchmark_data.csv

# Output includes P99, P95, P50, mean, std dev for each workload
```

### Example 3: Benchmark LLM Inference
```bash
# Measure TTFT, time-per-token, and throughput
python benchmarks/llm_latency_benchmark.py
```

### Example 4: Analyze Batch Size Trade-offs
```bash
# Test different batch sizes
python benchmarks/batching_tradeoff.py --images 1000 --batch-sizes "1,4,8,16,32"
```

### Example 5: Monitor GPU Memory
```bash
# Track GPU during inference
python monitoring/llm_monitor.py --duration 300 --interval 1
```

### Example 6: Manage Hung Processes
```bash
# Monitor and kill processes exceeding 30 minutes
./tools/detect_hung_proc.sh

# Or monitor a custom command
python tools/process_monitor.py "python scripts/long_running_benchmark.py" --timeout 1800
```

### Example 7: Generate Performance Report (NEW!)

#### Using Standalone Script
```bash
# Generate consolidated PDF with 6 performance graphs
python scripts/generate_performance_graphs.py --output report.pdf
```

#### Using Package & Examples
```bash
# Install package first
pip install -e .

# Run example to generate report with visualization
python examples/generate_performance_report.py

# View graphs interactively
python examples/generate_performance_report.py --show
```

#### Programmatic Usage
```python
from autoperfpy import PerformanceVisualizer, PDFReportGenerator
import numpy as np

# Create visualizations
viz = PerformanceVisualizer()

# Example: Latency percentiles
latency_data = {
    'ResNet50': {'P50': 22.5, 'P95': 25.3, 'P99': 28.1},
    'YOLO V8': {'P50': 30.2, 'P95': 35.8, 'P99': 42.5},
}
fig1 = viz.plot_latency_percentiles(latency_data)

# Example: Trade-off analysis
batch_sizes = [1, 4, 8, 16, 32]
latencies = [15.0, 10.8, 9.5, 8.8, 8.5]
throughputs = [66.7, 370.4, 842.1, 1136.4, 1882.4]
fig2 = viz.plot_latency_throughput_tradeoff(batch_sizes, latencies, throughputs)

# Example: GPU memory timeline
timestamps = np.arange(0, 60, 1)
memory_used = [2500 + 150 * np.sin(t/20) for t in timestamps]
fig3 = viz.plot_gpu_memory_timeline(timestamps, memory_used)

# Generate consolidated PDF report
pdf_gen = PDFReportGenerator("My Performance Report")
pdf_gen.add_metadata("System", "GPU Inference Platform")
pdf_gen.add_figure(fig1, "Latency Percentiles")
pdf_gen.add_figure(fig2, "Latency vs Throughput Trade-off")
pdf_gen.add_figure(fig3, "GPU Memory Timeline")

pdf_gen.generate_pdf("my_report.pdf")
```

### Example 8: Performance Regression Detection (NEW!)

#### Basic Usage
```python
from autoperfpy import RegressionDetector, RegressionThreshold

# Create detector
detector = RegressionDetector()

# Save baseline metrics (e.g., from main branch)
detector.save_baseline("main", {
    "p99_latency": 50.0,
    "throughput": 1000.0,
})

# Compare current metrics against baseline
result = detector.detect_regressions(
    baseline_name="main",
    current_metrics={
        "p99_latency": 56.0,  # 12% increase
        "throughput": 950.0,  # 5% decrease
    },
    thresholds=RegressionThreshold(
        p99_percent=10.0,          # P99 can increase 10%
        throughput_percent=5.0,    # Throughput can decrease 5%
    )
)

# Check results
if result["has_regressions"]:
    print("⚠️ Performance regression detected!")
    for metric, comp in result["regressions"].items():
        print(f"  {metric}: {comp['percent_change']:+.2f}%")

# Generate human-readable report
print(detector.generate_report("main", current_metrics))
```

#### List and Manage Baselines
```python
# List all available baselines
baselines = detector.list_baselines()
print(f"Available baselines: {baselines}")

# Load a baseline for comparison
baseline_metrics = detector.load_baseline("v1.0")
```

---

## 🧪 Testing

### Installation with Test Dependencies
```bash
pip install -e ".[test]"
```

### Running Tests
```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=autoperfpy --cov-report=html

# Run specific test file
pytest tests/test_core.py -v

# Run specific test class
pytest tests/test_core.py::TestRegressionDetector -v
```

### Test Coverage
- **42 Total Tests** across 4 test files
- **Core Utilities** (15 tests): DataLoader, LatencyStats, RegressionDetector
- **Analyzers** (11 tests): Percentile analysis, log analysis
- **Benchmarks** (11 tests): Batching trade-offs, LLM inference
- **CLI** (5 tests): Argument parsing and execution

### Using Make Commands
```bash
make help              # Show all available commands
make install-test      # Install with test dependencies
make test              # Run all tests
make test-verbose      # Run with verbose output
make test-coverage     # Generate coverage report
make lint              # Run linting
make format            # Format code
```

---

## 📊 Sample Data

Location: [scripts/data/automotive_benchmark_data.csv](scripts/data/automotive_benchmark_data.csv)

**Contents**: Synthetic benchmark data including:
- Timestamps
- Workload types (YOLO V8, ResNet50, MobileNet)
- Batch sizes (1, 4, 8, 16)
- Latency measurements (milliseconds)
- Power consumption (watts)

**Format**:
```csv
timestamp,workload,batch_size,latency_ms,power_w
2024-01-15 10:30:45,yolo_v8,1,25.5,65.2
2024-01-15 10:30:46,yolo_v8,1,26.1,64.8
```

---

## 🤝 Contributing

### Code Style
- Use type hints in function signatures
- Include docstrings explaining purpose and usage
- Follow PEP 8 naming conventions
- Add command-line examples in comments

### Adding New Scripts
1. Create in appropriate directory (`scripts/`, `benchmarks/`, `monitoring/`, or `tools/`)
2. Add `#!/usr/bin/env python3` shebang for Python scripts
3. Include comprehensive module docstring
4. Document usage with examples
5. Update this README with new section

### Updating Documentation
- Keep examples up-to-date and tested
- Include expected input/output formats
- Explain key concepts for educational clarity
- Add usage examples for new features

---

## Author

Hamna Nimra

---

## 📖 Learning Path

New to performance optimization? Follow this learning path:

1. **Start Here**: Read this README and understand the project structure
2. **Basic Concepts**: Run `batching_tradeoff.py` to understand latency-throughput trade-offs
3. **Log Analysis**: Use `automotive_parser_log.py` to parse performance data
4. **Statistical Analysis**: Run `calculate_p99_latency.py` to learn about percentiles
5. **LLM Concepts**: Run `llm_latency_benchmark.py` to understand TTFT and time-per-token
6. **Memory Management**: Use `llm_monitor.py` to track GPU memory
7. **Process Management**: Explore `process_monitor.py` for subprocess handling

---

## 🔗 Resources

### NVIDIA Documentation
- [TensorRT Deployment](https://docs.nvidia.com/deeplearning/tensorrt/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Drive Orin AGX Platform](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/drive-orin/)

### Performance Concepts
- [P99 Latency in Production](https://www.1ms.io/)
- [Batch Processing Trade-offs](https://cs231n.github.io/)
- [GPU Memory Best Practices](https://docs.nvidia.com/deeplearning/cudnn/)

---

## 🐛 Troubleshooting

### Import Errors
```bash
# Missing dependencies
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.8+
```

### Permission Errors (Shell Scripts)
```bash
# Make script executable
chmod +x tools/detect_hung_proc.sh

# Run with sudo if needed
sudo ./tools/detect_hung_proc.sh
```

### CSV Not Found
```bash
# Verify file exists
ls -la scripts/data/automotive_benchmark_data.csv

# Use correct path
python scripts/calculate_p99_latency.py scripts/data/automotive_benchmark_data.csv
```

### GPU Not Available (for monitoring scripts)
```bash
# Check GPU availability
nvidia-smi

# Scripts will still run with CPU fallback
python monitoring/gpu_monitor.py --cpu-only
```

---

## ⚠️ Important Disclaimer

**All data in this repository is SYNTHETIC and created for PRACTICE and DEMONSTRATION purposes only.**

- ❌ Results are **NOT** representative of real hardware performance
- ❌ Performance metrics are simulated
- ❌ Do NOT use these results for production decisions
- ✅ Use this codebase to **learn** performance engineering concepts
- ✅ Use as a **template** for your own benchmarking

For real performance benchmarking, use:
- NVIDIA's official benchmarking tools
- Your actual hardware and inference pipelines
- Validated datasets and workloads

---

## 📧 Support

For questions or issues:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Review the [Learning Path](#-learning-path)
3. Examine script docstrings: `python scripts/script_name.py --help`

---

**Happy Learning! 🚀**

