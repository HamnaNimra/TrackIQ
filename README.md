# TrackIQ
## Performance Analysis & Optimization Toolkit

A production-ready Python toolkit for **performance benchmarking, monitoring, and analysis** of ML inference workloads on edge devices and automotive platforms (NVIDIA Jetson, DRIVE, GPUs, and CPUs).

> **⚠️ Disclaimer**: All data in this repository is **synthetic** for demonstration purposes only.

---

## 🏗️ Architecture

**Two-layer design**: Reusable core library + Application layer

| Layer | Package | Purpose |
|-------|---------|---------|
| **Core** | `trackiq_core` | Reusable library: collectors (NVML, psutil, synthetic), benchmark runners, config management, data schemas, regression detection, comparison logic |
| **App** | `autoperfpy` | CLI tool, Streamlit interactive dashboard, automotive/edge profiles, specialized analyzers (DNN pipeline, Tegrastats), HTML/PDF reports |

This architecture enables you to:
- **Use as a library**: Import `trackiq_core` for programmatic benchmarking
- **Use as a CLI**: Run `autoperfpy` for interactive performance testing
- **Extend easily**: Add custom collectors, analyzers, and profiles

---

## ⚡ Quick Start

```bash
# 1. Install
pip install -e .

# 2. List available devices (GPUs, CPU)
autoperfpy devices --list

# 3. Run benchmarks - Multiple options:

# Option A: Auto mode (all devices, multiple configs)
autoperfpy run --auto --duration 30 --export results.json

# Option B: Use a predefined profile
autoperfpy run --profile automotive_safety --export-csv results.csv

# Option C: Manual single run
autoperfpy run --manual --device nvidia_0 --precision fp16 --batch-size 4

# 4. Analyze results
autoperfpy analyze latency --csv results.csv
autoperfpy analyze efficiency --csv results.csv

# 5. Generate interactive HTML report (auto-runs benchmark if no data)
autoperfpy report html --csv results.csv --output report.html --theme dark

# 6. Launch Streamlit dashboard
autoperfpy ui
```

### Using as a Library

```python
from autoperfpy import PercentileLatencyAnalyzer
from autoperfpy.benchmarks import BatchingTradeoffBenchmark
from trackiq_core.utils.compare import RegressionDetector

# Analyze benchmark data
analyzer = PercentileLatencyAnalyzer()
result = analyzer.analyze("benchmark.csv")
print(f"P99: {result.metrics['default']['p99']}ms")

# Run batch size analysis
benchmark = BatchingTradeoffBenchmark()
results = benchmark.run(batch_sizes=[1, 4, 8, 16])

# Detect performance regressions
detector = RegressionDetector()
detector.save_baseline("main", {"p99_ms": 45.0, "throughput_fps": 30.0})
regression = detector.detect_regressions("main", {"p99_ms": 52.0, "throughput_fps": 28.0})
print(detector.generate_report("main", {"p99_ms": 52.0}))
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Linux/Unix environment (tested on Ubuntu 20.04+)
- Optional: NVIDIA GPU with CUDA for GPU monitoring (requires `nvidia-ml-py`)
- Optional: NVIDIA Jetson/DRIVE for Tegrastats monitoring

### Install

```bash
# Clone repository
git clone https://github.com/HamnaNimra/trackiq.git
cd trackiq

# Install package (includes all dependencies)
pip install -e .

# Verify installation
autoperfpy --help

# Optional: Install PDF generation support (choose one)
pip install playwright && playwright install chromium  # Best quality
# OR
pip install pdfkit  # Alternative (requires wkhtmltopdf system package)
```

### Optional Dependencies

```bash
# For NVIDIA GPU monitoring
pip install nvidia-ml-py

# For system metrics collection
pip install psutil

# For testing and development
pip install -e ".[test]"  # Includes pytest, pytest-cov, pytest-mock
pip install -e ".[dev]"   # Includes black, flake8, isort
```

---

## 🎯 Key Features

### 🚀 Automated Benchmarking

- **Auto Mode**: Detect all devices and run benchmarks across multiple configs (precisions, batch sizes)
- **Profile-Based**: Use predefined profiles (`automotive_safety`, `edge_max_perf`, `edge_low_power`, `ci_smoke`)
- **Manual Mode**: Single device runs with custom configuration
- **Multi-device Support**: GPU (CUDA), CPU, NVIDIA Jetson/DRIVE (DLA/NPU)

### 📊 Advanced Analyzers

- **Percentile Latency Analyzer**: P50, P95, P99 calculations from benchmark data
- **DNN Pipeline Analyzer**: Layer-by-layer TensorRT/DriveWorks inference profiling with bottleneck identification
- **Tegrastats Analyzer**: NVIDIA Jetson/DRIVE metrics (CPU, GPU, memory, thermal, throttling detection)
- **Efficiency Analyzer**: Performance/Watt, energy per inference, power consumption analysis
- **Variability Analyzer**: Latency jitter, coefficient of variation, consistency ratings
- **Log Analyzer**: Performance log parsing with spike detection

### 🔬 Regression Detection & CI/CD Integration

- **Baseline Management**: Save and compare performance baselines
- **Automated Regression Detection**: Configurable thresholds for latency, throughput, and P99
- **CI/CD Ready**: `autoperfpy compare` command for build validation
- **Intelligent Metrics**: Automatically detects whether higher/lower is better for each metric

### 📈 Visualization & Reporting

- **Interactive HTML Reports**: Plotly-based charts with light/dark themes
- **PDF Reports**: Professional reports with embedded visualizations
- **Auto-Run Benchmarks**: Reports can run benchmarks automatically if no data provided
- **Streamlit Dashboard**: Interactive UI for running benchmarks and exploring results
- **Multi-Run Comparisons**: Compare multiple benchmark runs with charts and tables

### 🧪 Benchmarking Tools

- **Batching Trade-offs**: Analyze latency vs throughput across batch sizes (1, 4, 8, 16, 32)
- **LLM Benchmarking**: TTFT (Time-To-First-Token), time-per-token, token throughput
- **GPU Memory Monitoring**: Real-time GPU memory and KV cache tracking

---

## 💻 CLI Usage

### Device Detection

```bash
# List all detected devices (GPUs, CPU, Jetson/DRIVE)
autoperfpy devices --list
```

### Run Benchmarks

```bash
# Auto mode: Run on all detected devices with multiple configs
autoperfpy run --auto --duration 30 --export results.json --export-csv results.csv

# Auto mode with filters
autoperfpy run --auto --devices nvidia_0,cpu_0 --precisions fp16,fp32 --batch-sizes 1,4,8

# Profile mode: Use predefined automotive/edge profiles
autoperfpy run --profile automotive_safety --export results.json
autoperfpy run --profile ci_smoke --duration 10

# Manual mode: Single device with custom config
autoperfpy run --manual --device nvidia_0 --precision fp16 --batch-size 4 --duration 60
autoperfpy run --manual --device cpu_0 --precision fp32 --iterations 1000

# Validate profile without running
autoperfpy run --profile automotive_safety --validate-only
```

### Profiles

```bash
# List all available profiles
autoperfpy profiles --list

# Show detailed profile information
autoperfpy profiles --info automotive_safety
```

Available profiles:
- `automotive_safety` - Strict latency (<33.3ms P99), 50W power budget, 80°C thermal limit
- `edge_max_perf` - High throughput focus, minimal latency constraints
- `edge_low_power` - Power-constrained environments
- `ci_smoke` - Quick CI/CD validation

### Compare & Regression Detection

```bash
# Save current results as baseline
autoperfpy compare --baseline main --current results.json --save-baseline

# Compare against baseline (detect regressions)
autoperfpy compare --baseline main --current new_results.json \
  --latency-pct 5 --throughput-pct 5 --p99-pct 10

# Custom baseline directory
autoperfpy compare --baseline-dir .ci/baselines --baseline release-1.0 --current results.json
```

### Analyze Data

```bash
# Latency analysis (auto-runs benchmark if no --csv)
autoperfpy analyze latency --csv benchmark.csv
autoperfpy analyze latency --device nvidia_0 --duration 15

# DNN pipeline analysis
autoperfpy analyze dnn-pipeline --csv layer_times.csv --batch-size 4
autoperfpy analyze dnn-pipeline --profiler profiler_output.txt --top-layers 10

# Tegrastats analysis (NVIDIA Jetson/DRIVE)
autoperfpy analyze tegrastats --log tegrastats.log --throttle-threshold 85

# Efficiency analysis
autoperfpy analyze efficiency --csv benchmark.csv
autoperfpy analyze efficiency --device nvidia_0 --duration 20

# Variability analysis (jitter, consistency)
autoperfpy analyze variability --csv benchmark.csv --column latency_ms
autoperfpy analyze variability --duration 30

# Log analysis (spike detection)
autoperfpy analyze logs --log performance.log --threshold 50
```

### Benchmarking

```bash
# Batch size trade-off analysis
autoperfpy benchmark batching --batch-sizes 1,4,8,16,32 --images 1000

# LLM inference benchmarking
autoperfpy benchmark llm --prompt-length 512 --output-tokens 256 --runs 10
```

### Monitoring

```bash
# Real-time GPU monitoring
autoperfpy monitor gpu --duration 300 --interval 1

# KV cache monitoring
autoperfpy monitor kv-cache --max-length 2048
```

### Generate Reports

```bash
# HTML report (auto-runs benchmark if no data)
autoperfpy report html --output report.html --theme dark
autoperfpy report html --csv data.csv --output report.html --title "Performance Report"
autoperfpy report html --json results.json --output report.html --author "Team"

# PDF report
autoperfpy report pdf --csv data.csv --output report.pdf
autoperfpy report pdf --device nvidia_0 --duration 15 --output report.pdf

# Reports auto-export data files alongside the report
# Example: report.html → report_data.json, report_data.csv
```

### Launch Streamlit UI

```bash
# Launch interactive dashboard
autoperfpy ui

# Launch with pre-loaded data
autoperfpy ui --data results.json --port 8501

# Launch without auto-opening browser
autoperfpy ui --no-browser
```

### Dashboard

Launch the shared TrackIQ dashboard for an AutoPerfPy canonical result:

```bash
python dashboard.py --tool autoperfpy --result output/autoperf_power.json
```

---

## 📁 Project Structure

```
AutoPerfPy/
├── autoperfpy/              # Application layer
│   ├── cli.py              # CLI interface
│   ├── analyzers/          # App-specific analyzers (DNN, Tegrastats)
│   ├── benchmarks/         # Benchmark implementations
│   ├── collectors/         # Data collectors
│   ├── monitoring/         # Monitoring tools
│   ├── reports/            # Report generators (HTML, PDF, charts)
│   ├── ui/                 # Streamlit UI
│   └── profiles/           # Device profiles
│
├── trackiq_core/           # Core library
│   ├── collectors/         # Base collectors (synthetic, psutil, NVML)
│   ├── runners/            # Benchmark runners
│   ├── configs/            # Configuration management
│   ├── schemas/            # Data schemas
│   ├── utils/              # Core analyzers and utilities
│   └── compare/            # Regression detection
│
├── examples/               # Usage examples
├── tests/                  # Test suite (42 tests)
├── output/                 # Sample outputs and default report directory
├── config.yaml            # Configuration file
└── README.md              # This file
```

---

## 🧪 Testing

```bash
# Install with test dependencies
pip install -e ".[test]"

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=autoperfpy --cov=trackiq_core --cov-report=html

# Run specific test file
pytest tests/test_core.py -v
pytest tests/test_analyzers.py -v
pytest tests/test_integration.py -v
```

**Test Coverage**: 42+ comprehensive unit and integration tests

| Test File | Focus Area | Tests |
|-----------|------------|-------|
| `test_core.py` | Core utilities (LatencyStats, DataLoader, RegressionDetector) | 15 |
| `test_analyzers.py` | Percentile, Log, Efficiency, Variability analyzers | 11 |
| `test_benchmarks.py` | Batching, LLM benchmarks | 6 |
| `test_dnn_pipeline.py` | DNN/TensorRT pipeline analysis | 10 |
| `test_tegrastats.py` | Tegrastats parsing and analysis | 10+ |
| `test_html_generator.py` | HTML/PDF report generation | 10+ |
| `test_integration.py` | End-to-end workflows | 5+ |
| `test_trackiq_*.py` | trackiq_core library (collectors, config, compare) | 15+ |

---

## 📦 Examples

The `examples/` directory contains 7 working examples demonstrating all major features:

1. **`regression_detection_example.py`** - Baseline management and regression detection workflow
2. **`analyze_percentiles.py`** - Percentile latency analysis from CSV data
3. **`benchmark_batching.py`** - Batch size trade-off analysis
4. **`collect_synthetic_metrics.py`** - Synthetic data collection with configurable patterns
5. **`generate_performance_report.py`** - HTML/PDF report generation
6. **`generate_interactive_html_report.py`** - Advanced interactive reports with Plotly
7. **`monitor_gpu.py`** - Real-time GPU memory monitoring

Run any example:
```bash
python examples/regression_detection_example.py
python examples/benchmark_batching.py
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [CONCEPTS.md](CONCEPTS.md) | Core performance engineering concepts: latency, percentiles, batching, LLM metrics, regression detection, automotive requirements |
| [CHANGELOG.md](CHANGELOG.md) | Version history, release notes, and recent improvements |
| [TODO.md](TODO.md) | Planned features and roadmap (accelerator support, additional precisions, multi-run comparisons) |

### Key Concepts Covered

**Performance Metrics**
- Latency vs Throughput: Trade-offs between response time and requests per second
- Percentiles (P50, P95, P99): Why P99 matters more than averages for user experience
- Coefficient of Variation: Measuring latency consistency and jitter

**Optimization Techniques**
- Batching Trade-offs: How batch size affects latency and throughput
- Power Efficiency: Performance/Watt calculations and energy per inference
- Thermal Management: Throttling detection and thermal limits

**LLM-Specific Metrics**
- TTFT (Time-To-First-Token): Prefill phase latency
- Time-per-Token: Decode phase performance
- KV Cache: Memory consumption in transformer models

**Regression Detection**
- Baseline Management: Saving and loading performance baselines
- Threshold Configuration: Per-metric regression thresholds
- CI/CD Integration: Automated performance validation

**Automotive & Edge**
- Safety-Critical Requirements: Deterministic latency for ADAS
- Power Budgets: 50W constraints for automotive platforms
- Thermal Limits: 80°C operational constraints

See [CONCEPTS.md](CONCEPTS.md) for detailed explanations with examples.

---

## 🔧 Configuration

AutoPerfPy uses YAML configuration files for custom settings. Example [config.yaml](config.yaml):

```yaml
benchmark:
  duration: 60  # seconds
  sample_interval: 1.0  # seconds
  warmup_iterations: 10
  test_iterations: 100

device:
  type: cuda  # cuda, cpu, dla, npu
  id: 0

inference:
  precision: fp16  # fp32, fp16, int8
  batch_size: 4
  streams: 1

monitoring:
  sample_interval_ms: 100
  enable_gpu_metrics: true
  enable_power_metrics: true

output:
  directory: output/
  format: json  # json, csv, html
  export_csv: true

regression:
  baseline_dir: .trackiq/baselines
  thresholds:
    latency_pct: 5.0
    throughput_pct: 5.0
    p99_pct: 10.0
```

### Environment Variables

```bash
# Default profile for runs
export AUTOPERFPY_PROFILE=automotive_safety

# Default config file
export AUTOPERFPY_CONFIG=config.yaml

# Default collector type
export AUTOPERFPY_COLLECTOR=synthetic  # synthetic, nvml, tegrastats, psutil
```

---

## ✨ What Makes AutoPerfPy Production-Ready?

1. **Two-Layer Architecture**: Separation between reusable core (`trackiq_core`) and application layer (`autoperfpy`) enables flexibility
2. **Comprehensive Testing**: 42+ unit and integration tests with pytest coverage
3. **CI/CD Integration**: Built-in regression detection with configurable thresholds
4. **Multiple Interfaces**: CLI, Python library, and interactive Streamlit dashboard
5. **Automotive Focus**: Profiles for ADAS, edge computing, and safety-critical systems
6. **Multi-Device Support**: GPU, CPU, Jetson/DRIVE with automatic detection
7. **Rich Reporting**: Interactive HTML and PDF reports with Plotly visualizations
8. **Extensible Design**: Easy to add custom collectors, analyzers, and profiles
9. **Production Monitoring**: Real-time GPU monitoring, thermal tracking, and throttling detection
10. **Well-Documented**: Comprehensive README, CONCEPTS guide, CHANGELOG, and inline docstrings

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Adding New Features

1. **Core library changes**: Add to `trackiq_core/` for reusable components (collectors, runners, analyzers)
2. **App-specific features**: Add to `autoperfpy/` for CLI commands, UI components, or specialized analyzers
3. **Tests**: Add unit tests in `tests/` (use pytest). Aim for >80% coverage
4. **Documentation**: Update README, CONCEPTS.md, and add docstrings

### Development Workflow

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Check code style
black --check .
flake8 autoperfpy/ trackiq_core/

# Format code
black .
isort .
```

### Code Style

- Use type hints for all function signatures
- Include comprehensive docstrings (Google style)
- Follow PEP 8 conventions
- Add CLI usage examples in command docstrings
- Write unit tests for new features

### Pull Request Checklist

- [ ] Tests pass (`pytest tests/`)
- [ ] Code formatted (`black .` and `isort .`)
- [ ] Docstrings added for new functions/classes
- [ ] README updated if adding new features
- [ ] CHANGELOG.md updated with changes

---

## 🐛 Troubleshooting

### Import Errors

```bash
# Missing dependencies
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.8+
```

### GPU Not Available

```bash
# Check GPU availability
nvidia-smi

# Run with CPU fallback
autoperfpy run --device cpu
```

### Permission Errors

```bash
# Make scripts executable
chmod +x tools/*.sh
```

---

## 📖 Learning Path

New to performance optimization? Start here:

1. **Understand Concepts**: Read [CONCEPTS.md](CONCEPTS.md) to learn about latency, percentiles, batching, LLM metrics, and regression detection
2. **Check Hardware**: Run `autoperfpy devices --list` to see available devices
3. **Simple Benchmark**: Try `autoperfpy run --auto --duration 10`
4. **Analyze Results**: Run `autoperfpy analyze latency --csv output/*.csv`
5. **Visualize**: Generate an interactive report with `autoperfpy report html --csv output/*.csv --theme dark`
6. **Interactive Dashboard**: Launch `autoperfpy ui` for hands-on exploration
7. **Regression Detection**: Set up baselines with `autoperfpy compare --save-baseline main --current results.json`
8. **Advanced**: Explore DNN pipeline analysis, Tegrastats monitoring, and custom profiles
9. **Library Usage**: Import modules from `trackiq_core` and `autoperfpy` for programmatic use

---

## 🔌 Using as a Library

AutoPerfPy can be used programmatically in your Python code:

### Core Library (`trackiq_core`)

```python
from trackiq_core.collectors import SyntheticCollector, NVMLCollector
from trackiq_core.runners import BenchmarkRunner
from trackiq_core.utils.compare import RegressionDetector, RegressionThreshold

# Use collectors
collector = SyntheticCollector(config={"latency_mean_ms": 25.0})
collector.start()
metrics = collector.sample(time.time())
collector.stop()

# Regression detection
detector = RegressionDetector(baseline_dir=".trackiq/baselines")
detector.save_baseline("main", {"p99_ms": 45.0, "throughput_fps": 30.0})

thresholds = RegressionThreshold(latency_pct=5.0, throughput_pct=5.0, p99_pct=10.0)
result = detector.detect_regressions("main", {"p99_ms": 52.0}, thresholds)

if result.has_regressions:
    print(detector.generate_report("main", {"p99_ms": 52.0}, thresholds))
```

### Application Layer (`autoperfpy`)

```python
from autoperfpy.analyzers import (
    PercentileLatencyAnalyzer,
    EfficiencyAnalyzer,
    DNNPipelineAnalyzer,
    TegrastatsAnalyzer,
)
from autoperfpy.benchmarks import BatchingTradeoffBenchmark, LLMLatencyBenchmark
from autoperfpy.reports import HTMLReportGenerator, PerformanceVisualizer

# Analyze data
latency_analyzer = PercentileLatencyAnalyzer()
result = latency_analyzer.analyze("benchmark.csv")

# Run benchmarks
benchmark = BatchingTradeoffBenchmark()
results = benchmark.run(batch_sizes=[1, 4, 8, 16], num_images=1000)

# Generate reports
report = HTMLReportGenerator(title="Performance Report", theme="dark")
report.add_metadata("Device", "NVIDIA RTX 4090")
report.add_summary_item("P99 Latency", "45.2", "ms", "good")
report.generate_html("output/report.html")
```

---

## 🆕 Recent Improvements (v0.2.0)

### trackiq Core Library Refactor
- Separated reusable components into `trackiq_core` library
- Backward-compatible API with existing `autoperfpy` code
- Enables programmatic use without CLI overhead

### Enhanced CLI
- `autoperfpy run --auto`: Multi-device automatic benchmarking
- `autoperfpy run --manual`: Single device with custom config
- `autoperfpy compare`: Performance regression detection with baselines
- `autoperfpy devices --list`: Device detection and enumeration
- Explicit errors when hardware/dependencies missing (no silent fallbacks)

### Interactive Dashboard
- Streamlit UI with benchmark execution from browser
- Device and precision configuration interface
- Platform metadata display (device name, SoC, power mode)
- Data visualization and exploration tools

### Regression Detection System
- `RegressionDetector` class for baseline management
- `RegressionThreshold` for configurable per-metric thresholds
- JSON-based baseline storage
- Human-readable regression reports
- CI/CD integration ready

### Report Enhancements
- Interactive HTML reports with Plotly charts
- PDF generation with embedded visualizations
- Light/dark themes for reports
- Multi-run comparison charts
- Auto-runs benchmarks if no data provided
- Exports JSON and CSV alongside reports

### Testing & Quality
- 42+ comprehensive unit and integration tests
- pytest configuration with coverage reporting
- Test fixtures for CSV, logs, and metrics
- Development tools (black, flake8, isort)

See [CHANGELOG.md](CHANGELOG.md) for complete version history.

---

## 📧 Support

For questions or issues:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Review [CONCEPTS.md](CONCEPTS.md) for performance concepts
3. Examine CLI help: `autoperfpy --help` or `autoperfpy <command> --help`
4. Check the [examples/](examples/) directory for working code samples
5. Review [CHANGELOG.md](CHANGELOG.md) for recent changes

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👤 Author

**Hamna Nimra**
- GitHub: [@HamnaNimra](https://github.com/HamnaNimra)
- Repository: [trackiq](https://github.com/HamnaNimra/trackiq)

---

## 🙏 Acknowledgments

- NVIDIA for GPU monitoring APIs (nvidia-ml-py)
- Plotly for interactive visualizations
- Streamlit for the dashboard framework
- The open-source Python community

---

**Happy Benchmarking! 🚀**

*Performance matters. Measure it. Optimize it. Track it.*
