# AutoPerfPy
## Performance Analysis & Optimization Toolkit

A Python toolkit for **performance benchmarking, monitoring, and analysis** of ML inference workloads on edge devices and automotive platforms.

> **⚠️ Disclaimer**: All data in this repository is **synthetic** for demonstration purposes only.

---

## 🏗️ Architecture

**Two-layer design**: Core library + Application layer

| Layer | Package | Purpose |
|-------|---------|---------|
| **Core** | `trackiq_core` | Reusable library: collectors, runners, config, schemas, comparison logic, analyzers |
| **App** | `autoperfpy` | CLI tool, Streamlit UI, TensorRT/automotive benchmarks, reports |

---

## ⚡ Quick Start

```bash
# 1. Install
pip install -e .

# 2. Run a benchmark
autoperfpy run --device cuda:0 --precision fp16 --duration 10

# 3. View results in Streamlit UI
autoperfpy ui

# 4. Generate HTML report
autoperfpy report html --csv output/benchmark.csv --output report.html
```

### Using as a Library

```python
from autoperfpy import PercentileLatencyAnalyzer

# Analyze benchmark data
analyzer = PercentileLatencyAnalyzer()
result = analyzer.analyze("benchmark.csv")

print(f"P99: {result.metrics['p99']}ms")
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Linux/Unix environment (tested on Ubuntu 20.04+)
- Optional: NVIDIA GPU with CUDA for GPU monitoring

### Install

```bash
# Clone repository
git clone <repository-url>
cd AutoPerfPy

# Install dependencies
pip install -r requirements.txt

# Install package (editable mode for development)
pip install -e .

# Verify installation
autoperfpy --help
```

---

## 🎯 Key Features

### Core Analyzers

- **Percentile Latency Analyzer**: Calculate P50, P95, P99 from benchmark data
- **DNN Pipeline Analyzer**: Profile TensorRT/DriveWorks inference with layer-by-layer timing
- **Tegrastats Analyzer**: Monitor NVIDIA Jetson CPU, GPU, memory, and thermal metrics
- **Efficiency Analyzer**: Calculate Performance/Watt and energy consumption
- **Variability Analyzer**: Measure latency jitter and consistency

### Benchmarking

- **Batching Trade-offs**: Analyze latency vs throughput across batch sizes
- **LLM Benchmarking**: Measure Time-To-First-Token (TTFT) and time-per-token
- **Multi-device Support**: Run benchmarks on GPU (CUDA), NPU/DLA, or CPU

### Monitoring & Reporting

- **GPU Memory Monitor**: Real-time GPU memory and KV cache tracking
- **Performance Visualizer**: Generate graphs for latency, throughput, power, memory
- **HTML/PDF Reports**: Professional reports with interactive charts and navigation
- **Regression Detection**: Compare runs against baselines with configurable thresholds

---

## 💻 CLI Usage

### Run Benchmarks

```bash
# Run with specific device and precision
autoperfpy run --device cuda:0 --precision fp16 --duration 30

# Run with automatic device detection
autoperfpy run --auto

# Export results to JSON
autoperfpy run --device cuda:0 --export output/results.json
```

### Compare Results

```bash
# Save a baseline
autoperfpy compare --save-baseline main --csv baseline.csv

# Compare current run against baseline
autoperfpy compare --baseline main --current current.csv --latency-pct 5 --throughput-pct 5
```

### Generate Reports

```bash
# HTML report with dark theme
autoperfpy report html --csv data.csv --output report.html --theme dark

# PDF report
autoperfpy report pdf --csv data.csv --output report.pdf
```

### Analyze Data

```bash
# Analyze latency percentiles
autoperfpy analyze latency --csv benchmark.csv

# Analyze DNN pipeline performance
autoperfpy analyze dnn-pipeline --csv layer_times.csv --batch-size 4

# Analyze power efficiency
autoperfpy analyze efficiency --csv benchmark.csv
```

### Launch Streamlit UI

```bash
# Launch interactive dashboard
autoperfpy ui

# Launch with pre-loaded data
autoperfpy ui --data results.json
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
pytest tests/ --cov=autoperfpy --cov-report=html

# Run specific test file
pytest tests/test_core.py -v
```

**Test Coverage**: 42 tests across core utilities, analyzers, benchmarks, and CLI

---

## 📚 Documentation

- [CONCEPTS.md](CONCEPTS.md) - Core performance engineering concepts (latency, percentiles, batching, LLM metrics)
- [CHANGELOG.md](CHANGELOG.md) - Version history and recent changes
- [TODO.md](TODO.md) - Planned features and improvements

### Key Concepts

**Latency vs Throughput**: Understanding the trade-off between response time and requests per second

**Percentiles (P50, P95, P99)**: Why P99 matters more than averages for user experience

**Batching Trade-offs**: How batch size affects latency and throughput

**LLM Inference**: TTFT (Time-To-First-Token) and time-per-token metrics

**KV Cache**: Memory consumption in transformer models

See [CONCEPTS.md](CONCEPTS.md) for detailed explanations.

---

## 🔧 Configuration

AutoPerfPy uses YAML configuration files. Example [config.yaml](config.yaml):

```yaml
benchmark:
  duration: 60  # seconds
  sample_interval: 1.0  # seconds

device:
  type: cuda  # cuda, cpu, dla
  id: 0

inference:
  precision: fp16  # fp32, fp16, int8
  batch_size: 1

output:
  directory: output/
  format: json  # json, csv, html
```

---

## 🤝 Contributing

### Adding New Features

1. **Core library changes**: Add to `trackiq_core/` for reusable components
2. **App-specific features**: Add to `autoperfpy/` for CLI, UI, or specialized analyzers
3. **Tests**: Add unit tests in `tests/` (use pytest)
4. **Documentation**: Update README and CONCEPTS.md as needed

### Code Style

- Use type hints
- Include docstrings
- Follow PEP 8
- Add CLI examples in docstrings

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

1. **Basics**: Read [CONCEPTS.md](CONCEPTS.md) to understand latency, percentiles, and batching
2. **Simple Analysis**: Run `autoperfpy analyze latency --csv scripts/data/automotive_benchmark_data.csv`
3. **Benchmarking**: Try `autoperfpy run --device cpu --duration 10`
4. **Visualization**: Generate a report with `autoperfpy report html`
5. **Advanced**: Explore DNN pipeline analysis and Tegrastats monitoring

---

## 📧 Support

For questions or issues:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Review [CONCEPTS.md](CONCEPTS.md) for performance concepts
3. Examine CLI help: `autoperfpy --help` or `autoperfpy <command> --help`

---

## Author

Hamna Nimra

---

**Happy Benchmarking! 🚀**
