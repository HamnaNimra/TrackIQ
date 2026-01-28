# AutoPerfPy
## Performance Analysis & Utilities

A collection of Python scripts and shell utilities for automotive performance analysis, benchmarking, and system monitoring. Focused on edge AI performance engineering for NVIDIA platforms.

> **Note**: All data in this repository is synthetic and for practice/demonstration purposes only.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Scripts Documentation](#scripts-documentation)
- [Tools Documentation](#tools-documentation)
- [Usage Examples](#usage-examples)
- [Data](#data)

---

## Project Overview

This project contains tools for:

1. **Performance Monitoring** - Detect and manage long-running processes
2. **Log Analysis** - Parse automotive performance logs with latency metrics
3. **Benchmarking** - Calculate percentile latency statistics (P99, P95, etc.) from benchmark data
4. **Practice** - Learning and experimentation with performance engineering concepts

**Target Platform**: NVIDIA Edge AI / Automotive (Drive Orin AGX)

---

## Project Structure

```
├── README.md                      # This file
├── requirements.txt               # Python package dependencies
├── scripts/
│   ├── automotive_parser_log.py   # Parse automotive performance logs
│   ├── calculate_p99_latency.py   # Calculate percentile latency statistics
│   ├── practice.py                # Learning/experimentation script
│   └── data/
│       └── automotive benchmark.csv  # Sample benchmark data
└── tools/
    └── detech_hung_proc.sh        # Detect and kill hung processes
```

---

## Setup & Installation

### Prerequisites

- Python 3.7+
- Bash shell
- Linux/Unix environment

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Scratchpad
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Make scripts executable** (optional)
   ```bash
   chmod +x scripts/*.py
   chmod +x tools/*.sh
   ```

### Dependencies

See [requirements.txt](requirements.txt) for a complete list:
- `requests` - HTTP library
- `numpy` - Numerical computing
- `pandas` - Data analysis
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `pytest` - Testing framework

---

## Scripts Documentation

### 1. automotive_parser_log.py

**Purpose**: Extract and analyze automotive performance logs from NVIDIA Drive Orin AGX platform

**Features**:
- Parses timestamps and latency events
- Filters events exceeding specified latency thresholds
- Supports frame-level performance metrics (inference vs end-to-end latency)

**Usage**:
```bash
python scripts/automotive_parser_log.py [log_file] [threshold_ms]
```

**Example**:
```bash
python scripts/automotive_parser_log.py performance.log 50
```

**Output**:
- Filtered latency events with timestamps
- Frame IDs and latency measurements
- Summary statistics

---

### 2. calculate_p99_latency.py

**Purpose**: Comprehensive benchmark analysis with percentile latency calculations

**Features**:
- Calculates P99, P95, P50 (median) latencies
- Supports multiple workloads and batch sizes
- Provides statistical summaries (mean, std dev, min, max)
- Tracks power consumption metrics

**Usage**:
```bash
python scripts/calculate_p99_latency.py <benchmark_csv>
```

**Example**:
```bash
python scripts/calculate_p99_latency.py scripts/data/automotive\ benchmark.csv
```

**Input Format** (CSV):
```
timestamp,workload,batch_size,latency_ms,power_w
2024-01-15 10:30:45,yolo_v8,1,25.5,65.2
2024-01-15 10:30:46,yolo_v8,1,26.1,64.8
```

**Output**:
- Percentile statistics (P99, P95, P50, min, max)
- Mean and standard deviation
- Per-workload breakdowns
- Power efficiency metrics

---

### 3. practice.py

**Purpose**: Learning and experimentation script for performance engineering concepts

**Use Cases**:
- Testing new analysis techniques
- Prototyping performance utilities
- Learning Python data analysis

---

## Tools Documentation

### detech_hung_proc.sh

**Purpose**: Detect and gracefully kill processes that exceed runtime thresholds

**Features**:
- Pattern-based process matching (default: "trtexec")
- Graceful termination with SIGTERM, escalates to SIGKILL if needed
- Zombie process detection
- Structured logging with timestamps

**Configuration**:
```bash
PROCESS_PATTERN="trtexec"        # Process name to monitor
MAX_RUNTIME_MINUTES=30            # Timeout threshold
LOG_FILE="/var/log/process_monitor.log"  # Output log location
```

**Usage**:
```bash
./tools/detech_hung_proc.sh
```

**Process Termination Flow**:
1. Send SIGTERM (graceful termination)
2. Wait up to 10 seconds for process to exit
3. If still running, send SIGKILL (forceful termination)
4. Log all actions and zombies detected

**Permissions**:
May require `sudo` depending on process ownership:
```bash
sudo ./tools/detech_hung_proc.sh
```

---

## Usage Examples

### Example 1: Analyze Automotive Performance Logs

```bash
# Parse logs and filter events with >50ms latency
python scripts/automotive_parser_log.py performance.log 50

# Output: Timestamped latency spikes with frame IDs
```

### Example 2: Calculate P99 Latency from Benchmarks

```bash
# Analyze benchmark CSV
python scripts/calculate_p99_latency.py scripts/data/automotive\ benchmark.csv

# Output: P99 latency, percentiles, and power metrics
```

### Example 3: Monitor and Clean Hung Processes

```bash
# Setup cron job to monitor trtexec processes
(crontab -l; echo "*/5 * * * * /workspaces/Scratchpad/tools/detech_hung_proc.sh") | crontab -

# Or run manually
./tools/detech_hung_proc.sh
```

---

## Data

### Sample Benchmark Data

Location: [scripts/data/automotive benchmark.csv](scripts/data/automotive%20benchmark.csv)

Contains synthetic performance data including:
- Timestamps
- Workload types (YOLO, ResNet, etc.)
- Batch sizes
- Latency measurements (milliseconds)
- Power consumption (watts)

---

## Contributing

When adding new scripts or tools:

1. Include docstrings explaining purpose and usage
2. Add command-line examples in comments
3. Update this README with new sections
4. Use type hints in Python code
5. Include error handling and logging

---

## Author

Hamna Nimra

---

## Disclaimer

This project uses **synthetic data for practice and demonstration purposes only**. All performance metrics and measurements are simulated and not representative of real hardware performance.

---

