# MiniCluster: Local Distributed Training Validation Tool

**MiniCluster** simulates a distributed AI training cluster locally using PyTorch's distributed training framework with the CPU-only Gloo backend. It validates distributed training workloads the same way a production cluster validation engineer would: **correctness first, performance second, fault tolerance third**.

## Overview

MiniCluster provides a complete validation framework for distributed training that runs entirely on CPU without requiring GPUs, making it suitable for CI/CD pipelines, local development, and quick validation iterations. It automates the three critical validation phases:

1. **Correctness**: Compare single-process vs multi-process training convergence
2. **Performance**: Track throughput and all-reduce timing  
3. **Fault Tolerance**: Inject failures and verify detection capability

## How It Fits Into TrackIQ Ecosystem

MiniCluster is a specialized validator within the TrackIQ framework that focuses on distributed AI training workloads. While [AutoPerfPy](../autoperfpy) validates edge inference performance, **MiniCluster validates distributed training correctness** using the same baseline management, regression detection, and reporting infrastructure from `trackiq_core`:

- **Shared Baseline Management**: Uses `RegressionDetector` from `trackiq_core` to save/load training baselines
- **Consistent Reporting**: Outputs metrics in trackiq_core schema for unified comparison across tools
- **Regression Detection**: Uses configurable thresholds to flag training divergence
- **Modular Architecture**: All trackiq_core imports isolated in `deps.py` for future standalone pip package

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- Linux/macOS or Windows (with appropriate torch distribution)

### Install from source
```bash
cd AutoPerfPy
pip install -e .
pip install -e .[minicluster]  # If there's a minicluster extras group define
```

Or directly include minicluster in your Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/AutoPerfPy"
python -m minicluster --help
```

## Usage

### 1. Run Distributed Training

Execute a training run with configurable number of workers:

```bash
# Single-worker baseline
python -m minicluster run \
  --workers 1 \
  --steps 100 \
  --batch-size 32 \
  --output single_process.json

# Multi-worker training
python -m minicluster run \
  --workers 2 \
  --steps 100 \
  --batch-size 32 \
  --output multi_process.json
```

**Example Output:**
```
Starting distributed training run...

✓ Run complete!
  Total time: 42.15s
  Final loss: 0.082345
  Avg throughput: 152.3 samples/sec
  Metrics saved to: multi_process.json
```

**Output Metrics** (in JSON):
- Per-step loss values
- Throughput (samples/sec)
- All-reduce time (ms) - timing for gradient synchronization
- Compute time (ms) - time for forward/backward passes
- Total training time

### 2. Validate Correctness

Compare single-process baseline against multi-process run:

```bash
python -m minicluster validate \
  single_process.json \
  multi_process.json \
  --tolerance 0.01 \
  --output validation_report.json
```

**Example Output:**
```
================================================================================
CORRECTNESS VALIDATION REPORT
================================================================================
Single-process run: single_process.json
Multi-process run:  multi_process.json
Tolerance:          1.00%
Steps compared:     100
  Passed:           100
  Failed:           0

--------------------------------------------------------------------------------
RESULT: ✓ PASSED: All 100 steps passed correctness check within 1.00% tolerance
--------------------------------------------------------------------------------

================================================================================
```

**Report Contents**:
- Per-step losses from both runs
- Delta between runs (absolute and percentage)
- Pass/fail status per step
- Overall pass/fail with summary

### 3. Run Fault Injection Tests

Test the validation framework's ability to detect failures:

```bash
python -m minicluster fault-test \
  --steps 50 \
  --tolerance 0.01 \
  --output fault_report.json
```

**Example Output:**
```
================================================================================
FAULT INJECTION TEST REPORT
================================================================================
Total faults tested:  3
Faults detected:      3
Faults missed:        0
Detection rate:       100.0%

--------------------------------------------------------------------------------
Fault Detection Results:

SLOW_WORKER: ✓ DETECTED
  Affected rank: 0
  Reason: Detected via throughput deviation

GRADIENT_SYNC_ANOMALY: ✓ DETECTED
  Affected rank: 0
  Reason: Detected via loss divergence

WORKER_TIMEOUT: ✓ DETECTED
  Affected rank: 0
  Reason: Detected via step mismatch

--------------------------------------------------------------------------------
SUMMARY: Fault injection testing complete: 3/3 faults detected, 0 missed
================================================================================
```

**Fault Types Tested**:

1. **Slow Worker**: One rank sleeps 0.5s per step -> reduced throughput
   - **Why it matters**: Stragglers in production clusters cause all-reduce to stall, effectively creating a bottleneck. Detecting slow workers is critical for SLA compliance.
   
2. **Gradient Sync Anomaly**: Injected noise into one rank's gradients before all-reduce
   - **Why it matters**: Silent gradient corruption (e.g., from memory errors) causes divergent loss curves. Production clusters must detect this before model divergence becomes severe.
   
3. **Worker Timeout**: One rank stops producing steps mid-training
   - **Why it matters**: Network partitions, software crashes, or resource exhaustion cause worker timeouts. The validation framework must catch incomplete runs that would corrupt model checkpoints.

### 4. Save Baseline

Store successful run as a reference baseline:

```bash
python -m minicluster baseline save \
  --metrics multi_process.json \
  --name stable_v1 \
  --baseline-dir .minicluster/baselines
```

**Output:**
```
✓ Baseline 'stable_v1' saved successfully
```

Baselines are stored as JSON files in `.minicluster/baselines/`:
```
.minicluster/baselines/
├── stable_v1.json
└── staging_v2.json
```

### 5. Compare Against Baseline

Detect regressions by comparing current run against saved baseline:

```bash
python -m minicluster baseline compare \
  --metrics current_run.json \
  --name stable_v1 \
  --latency-threshold 5.0 \
  --throughput-threshold 5.0 \
  --output comparison_report.json
```

**Example Output:**
```
================================================================================
BASELINE COMPARISON REPORT
================================================================================
Baseline:                    stable_v1
Latency threshold:           5.0%
Throughput threshold:        5.0%

--------------------------------------------------------------------------------
Metric Comparisons:
--------------------------------------------------------------------------------
average_loss                     0.082000 → 0.085200 (+4.15%)      ✓
final_loss                       0.045000 → 0.046800 (+4.00%)      ✓
average_throughput_samples_per_sec 152.500000 → 149.800000 (-1.77%) ✓
total_allreduce_time_ms       25300.000000 → 26850.000000 (+6.30%) ✗ REGRESS

================================================================================
```

**Exit codes**:
- `0`: No regressions detected
- `1`: Regressions found (exceeds thresholds)
- `2`: Error (missing baseline, file not found)

### Dashboard

Launch the shared TrackIQ dashboard for a MiniCluster canonical result:

```bash
python dashboard.py --tool minicluster --result minicluster_power.json
```

## Architecture

### Module Structure

```
minicluster/
├── deps.py                          # Centralized trackiq_core imports
├── cli.py                           # Command-line interface (argparse)
├── __init__.py                      # Package exports
├── runner/
│   ├── __init__.py
│   └── distributed_runner.py        # Training harness using torch.distributed
├── validators/
│   ├── __init__.py
│   ├── correctness_validator.py     # Single vs multi-process comparison
│   └── fault_injector.py            # Fault injection and detection testing
└── tests/
    ├── __init__.py
    ├── test_distributed_runner.py   # Runner tests
    ├── test_correctness_validator.py # Validator tests
    └── test_fault_injector.py       # Fault injector tests
```

### Component Responsibilities

- **deps.py**: Single source of truth for trackiq_core imports, enables future pip package migration
- **distributed_runner.py**: Orchestrates torch.distributed training, records per-step metrics in trackiq_core schema
- **correctness_validator.py**: Compares runs within tolerance, generates pass/fail reports
- **fault_injector.py**: Injects faults, validates detection, produces structured reports
- **cli.py**: argparse-based CLI with 5 main subcommands (run, validate, fault-test, baseline save/compare)

## Mapping to Production

This tool directly mirrors the workflow of a production cluster validation engineer at AMD:

| MiniCluster Component | Production Equivalent | What It Validates |
|---|---|---|
| `distributed_runner.py` | Running benchmark workloads on ROCm cluster | Does training complete without crashes? Are loss curves smooth? |
| `correctness_validator.py` | Comparing single-node vs multi-node reference runs | Do gradients sync correctly across nodes? |
| `fault_injector.py` → slow worker | Injecting network latency on one node | Can monitoring detect stragglers? |
| `fault_injector.py` → gradient anomaly | Memory bit-flip simulation or gradient clipping test | Does the framework catch divergent training? |
| `fault_injector.py` → worker timeout | Killing a process mid-training | Can orchestration recover? Are checkpoints valid? |
| `baseline save` | Recording golden run metrics | Do we have a reference point for regression tracking? |
| `baseline compare` | CI/CD regression detection | Did this cluster change break convergence? |

### Typical Production Workflow

1. **Baseline Phase**: Run initial training, record metrics with `baseline save`
2. **Validation Phase**: Run new training, use `validate` to check convergence matches single-process
3. **Regression Detection**: Use `baseline compare` in CI pipeline - fail if throughput drops >5%
4. **Fault Resilience**: Periodically run `fault-test` to ensure monitoring catches degradation
5. **Incident Root Cause**: Use fault injection to reproduce customer-reported issues

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest minicluster/tests/ -v

# Run specific test module
pytest minicluster/tests/test_distributed_runner.py -v

# Run specific test
pytest minicluster/tests/test_correctness_validator.py::TestCorrectnessValidator::test_compare_identical_runs -v
```

**Test Coverage**:
- ✓ Single-process training completes with correct metrics
- ✓ Multi-process training (via single-worker wrapper) completes
- ✓ Correctness validator passes for identical runs
- ✓ Correctness validator fails when loss diverges beyond tolerance
- ✓ Fault injector detects slow worker via throughput
- ✓ Fault injector detects gradient anomaly via loss divergence
- ✓ Fault injector detects timeout via step mismatch
- ✓ Baseline save/load round-trips correctly
- ✓ Baseline comparison flags regressions
- ✓ CLI subcommands execute without errors

## Technical Details

### Training Configuration

Default model and training setup:
- **Model**: SimpleMLP (configurable 2+ layer feedforward network)
- **Optimizer**: Adam (lr=0.01 default)
- **Loss Function**: MSELoss (regression on synthetic data)
- **Dataset**: Deterministic synthetic regression dataset
- **Distributed Backend**: Gloo (CPU-only, no GPU required)
- **Sampler**: DistributedSampler for multi-worker data sharding

### Metrics Recorded

Per-step metrics for each training step:
```json
{
  "step": 0,
  "loss": 2.34567,
  "throughput_samples_per_sec": 145.2,
  "allreduce_time_ms": 12.5,
  "compute_time_ms": 3.2
}
```

Aggregated metrics for entire run:
```json
{
  "num_workers": 2,
  "num_steps": 100,
  "total_time_sec": 42.15,
  "total_allreduce_time_ms": 1240.5,
  "total_compute_time_ms": 315.2,
  "average_loss": 0.12345,
  "average_throughput_samples_per_sec": 152.3,
  "final_loss": 0.0823,
  "start_timestamp": "2024-01-15T10:30:00",
  "end_timestamp": "2024-01-15T10:30:42"
}
```

### Reproducibility

All training is fully deterministic:
- Synthetic dataset uses fixed seed (default 42)
- Model initialized with seed
- NumPy/PyTorch seeding for all random operations
- Same input always produces same loss values (within floating-point precision)

Use `--seed` parameter to change base seed for different data distributions:
```bash
python -m minicluster run --workers 2 --steps 100 --seed 999
```

## Common Use Cases

### CI/CD Integration

Block deployment if training regresses:
```bash
#!/bin/bash
set -e

# Baseline stable version
minicluster baseline save --metrics stable.json --name main_branch

# New feature branch
minicluster run --workers 2 --steps 100 --output feature.json
minicluster baseline compare --metrics feature.json --name main_branch \
  --latency-threshold 5.0 --throughput-threshold 5.0
echo "✓ No regressions detected"
```

### Local Development

Quick validation before pushing:
```bash
# Single-worker baseline
minicluster run --workers 1 --steps 50 --output single.json

# Multi-worker comparison
minicluster run --workers 4 --steps 50 --output multi.json

# Validate
minicluster validate single.json multi.json --tolerance 0.02
```

### Fault Resilience Testing

Verify monitoring catches degradation:
```bash
minicluster fault-test --steps 100 --output fault_results.json

# Should show 100% detection rate
jq '.num_detected / .num_faults' fault_results.json
```

## Troubleshooting

### "Address already in use" error
Multiple distributed jobs using same port. Each job tries port 29500 by default.
```bash
# Use different port by restarting (each job gets auto-assigned port)
# Or run jobs sequentially
```

### "RuntimeError: An error occurred in the collective function"
Ensure all workers run exactly the same number of steps. Check for timeouts or crashes.

### Metrics file not found
Ensure `--output` path is created:
```bash
python -m minicluster run --output ./results/metrics.json
# Creates ./results/ if needed with 'ensure_parent_dir'
```

### Tolerance too strict
Start with 0.05 (5%) tolerance, tighten gradually:
```bash
# Loose tolerance
minicluster validate s.json m.json --tolerance 0.05

# Tighter tolerance  
minicluster validate s.json m.json --tolerance 0.01
```

## Contributing

To extend MiniCluster:

1. **Add new fault types**: Extend `FaultType` enum and implement in `fault_injector.py`
2. **Add new metrics**: Extend `StepMetrics` dataclass and collection in `distributed_runner.py`
3. **Add CLI commands**: Add subparser in `cli.py` and corresponding handler function

Ensure all imports from trackiq_core go through `deps.py` for clean decoupling.

## License

Same as AutoPerfPy (see LICENSE in root)

## References

- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html)
- [TrackIQ Core Documentation](../../trackiq_core/README.md)
- [AutoPerfPy](../autoperfpy/README.md)
