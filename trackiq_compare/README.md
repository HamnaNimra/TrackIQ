# trackiq-compare

`trackiq-compare` compares two canonical `TrackiqResult` JSON files and highlights which run performed better per metric. It produces both terminal-friendly and HTML-friendly outputs with regression flagging and baseline workflows.

## How It Fits

`trackiq-compare` sits beside `autoperfpy` and `minicluster` and consumes their shared `TrackiqResult` outputs from `trackiq_core`.

```text
trackiq/
├── trackiq_core/     -> shared schema + serializer + baseline utilities
├── autoperfpy/       -> inference tool (emits TrackiqResult)
├── minicluster/      -> training tool (emits TrackiqResult)
└── trackiq_compare/  -> compares TrackiqResult outputs
```

## Install

From repo root:

```bash
pip install -e .
```

Run as module:

```bash
python -m trackiq_compare --help
```

## Usage

### 1) Compare two results in terminal

```bash
python -m trackiq_compare run result_a.json result_b.json
```

Example output:

```text
TrackIQ Metric Comparison
Metric                      Result A    Result B    Abs Delta   % Delta    Winner
throughput_samples_per_sec  98.5000     102.1000    3.6000      +3.65%     Result B
latency_p99_ms              13.1000     14.0000     0.9000      +6.87%     Result A
...
Summary
Overall winner: Result B. Result B won 4 of 6 comparable metrics. Largest deltas: latency_p99_ms (+6.87%), throughput_samples_per_sec (+3.65%), ...
```

### 2) Compare and generate HTML report

```bash
python -m trackiq_compare run result_a.json result_b.json --html compare_report.html
```

### 3) Compare with custom labels

```bash
python -m trackiq_compare run result_a.json result_b.json \
  --label-a "AMD MI300X" --label-b "NVIDIA A100"
```

### 4) Save baseline from a result

```bash
python -m trackiq_compare baseline result.json --name release_v1
```

Example output:

```text
[OK] Baseline saved: release_v1
```

### 5) Compare a result against baseline

```bash
python -m trackiq_compare vs-baseline result.json release_v1
```

Example output:

```text
======================================================================
PERFORMANCE REGRESSION REPORT
======================================================================
Baseline: release_v1
...
NO REGRESSIONS DETECTED
```

## Real World Use Cases

### 1) Compare AMD vs NVIDIA benchmark outputs

Use `run` with labels `--label-a "AMD MI300X"` and `--label-b "NVIDIA A100"` to identify which platform wins per metric (throughput, latency, memory, optional comm/power).

### 2) Compare two ROCm driver versions

Run the same workload before and after a ROCm upgrade, save both as `TrackiqResult`, and use `run` to detect which metrics improved/regressed and by how much.

### 3) Compare edge inference against cluster inference for same model

Compare `autoperfpy` inference output from edge hardware against a cluster run represented in `TrackiqResult` to quantify tradeoffs in throughput, latency, and memory.

## Extending the Schema

To add new metrics and have them appear in comparisons:

1. Add the new field to `trackiq_core/schema.py` in `Metrics`.
2. Ensure producing tools (`autoperfpy`, `minicluster`, others) populate that field.
3. `trackiq-compare` will include it automatically because metric comparison iterates all metric fields from `TrackiqResult.metrics`.
4. If a new metric needs special winner logic (higher-is-better vs lower-is-better), update `LOWER_IS_BETTER_METRICS` in `trackiq_compare/comparator/metric_comparator.py`.
