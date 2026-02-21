# TrackIQ Monorepo

(Formerly Autoperfpy)  
TrackIQ is a multi-tool performance validation repository.  
It contains one shared library (`trackiq_core`) and three tool applications (`autoperfpy`, `minicluster`, `trackiq_compare`) built on top of that library.

## Naming and Scope

Use these names consistently:

- `trackiq`:
  - The repository/monorepo name.
  - Contains all packages and tools.
- `trackiq_core`:
  - The shared Python library.
  - Defines canonical result schema (`TrackiqResult`), serialization/validation, hardware and power integrations, UI base components, baseline/report utilities.
- `autoperfpy`:
  - Edge and inference benchmarking tool.
  - Produces `TrackiqResult` JSON outputs.
- `minicluster`:
  - Local distributed training validation tool.
  - Produces `TrackiqResult` JSON outputs and health checkpoint data.
- `trackiq-compare` (`trackiq_compare` package):
  - Result-to-result comparison tool.
  - Compares any two `TrackiqResult` files regardless of source tool.

## Repository Layout

```text
trackiq/
├── trackiq_core/      # shared library
├── autoperfpy/        # inference / edge benchmarking tool
├── minicluster/       # distributed training validation tool
├── trackiq_compare/   # comparison tool
├── dashboard.py       # unified dashboard launcher
└── tests/             # repo-level tests
```

## Installation

```bash
# from repo root
py -3.12 -m pip install -e .
```

If your shell entry points are stale on Windows, use module execution directly:

```bash
py -3.12 -m autoperfpy.cli --help
py -3.12 -m minicluster.cli --help
py -3.12 -m trackiq_compare.cli --help
```

## Core Workflow

1. Run a producer tool (`autoperfpy` or `minicluster`) to generate canonical `TrackiqResult` JSON.
2. Compare results with `trackiq-compare`.
3. View results in dashboards (tool-specific Streamlit apps or unified launcher).

## Tool Workflows

### AutoPerfPy

Main use: edge/inference benchmark runs and analysis.

```bash
autoperfpy devices --list
autoperfpy run --auto --duration 30 --export output/autoperf_result.json
autoperfpy analyze latency --csv output/results.csv
autoperfpy report html --json output/autoperf_result.json --output output/autoperf_report.html
```

Dashboard options:

```bash
# tool-owned app
streamlit run autoperfpy/ui/streamlit_app.py

# unified launcher
python dashboard.py --tool autoperfpy --result output/autoperf_result.json
```

### MiniCluster

Main use: distributed training correctness/performance/fault validation.

```bash
minicluster run --workers 2 --steps 50 --output minicluster_results/run_metrics.json
minicluster run --workers 2 --steps 50 --health-checkpoint-path ./minicluster_results/health.json
minicluster monitor status --checkpoint ./minicluster_results/health.json
```

Dashboard options:

```bash
# tool-owned app
streamlit run minicluster/ui/streamlit_app.py

# unified launcher
python dashboard.py --tool minicluster --result minicluster_results/run_metrics.json
```

### TrackIQ Compare

Main use: compare two canonical results and generate terminal/HTML reports.

```bash
trackiq-compare run output/autoperf_result.json minicluster_results/run_metrics.json
trackiq-compare run output/autoperf_result.json minicluster_results/run_metrics.json --html output/comparison.html
trackiq-compare run output/autoperf_result.json minicluster_results/run_metrics.json --label-a "AMD MI300X" --label-b "NVIDIA A100"
```

Baseline flows:

```bash
trackiq-compare baseline output/autoperf_result.json --name edge_baseline
trackiq-compare vs-baseline output/autoperf_result.json edge_baseline
```

Dashboard options:

```bash
# tool-owned app
streamlit run trackiq_compare/ui/streamlit_app.py

# unified launcher
python dashboard.py --tool compare --result-a output/autoperf_result.json --result-b minicluster_results/run_metrics.json --label-a "AMD MI300X" --label-b "NVIDIA A100"
```

## Canonical Result Contract

All tool outputs should conform to:

```python
from trackiq_core.schema import TrackiqResult
```

This is the single schema contract across the ecosystem.
`trackiq_compare` and dashboards assume `TrackiqResult`-compatible JSON.

## Hardware and Power Notes

- `trackiq_core` includes multi-platform detection and metrics paths for NVIDIA, AMD, Intel, Apple Silicon, and CPU (best effort depending on available system tools).
- Power profiling readers support ROCm SMI, tegrastats, and simulation fallback.

## Testing

```bash
py -3.12 -m pytest -q
```

## Where to Go Next

- `trackiq_core/ui/USAGE.md`: library-first dashboard API usage.
- `minicluster/monitor/README.md`: live health monitoring pipeline and anomaly model.
- `trackiq_compare/README.md`: comparison semantics, CLI options, and report behavior.

## License

MIT. See `LICENSE`.