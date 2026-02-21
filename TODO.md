# TrackIQ Monorepo TODO

## High Priority

### Add Support for Accelerator Categories

Support different accelerator types commonly used in automotive and edge systems:

- GPU: flexible, high power, broad precision support (FP32/FP16/INT8)
- NPU/DLA: lower power, usually constrained precision support
- DSP: strong for signal processing workloads
- FPGA: customizable hardware paths

Implementation tasks:

1. Extend accelerator detection in `autoperfpy/device_config.py`.
2. Add accelerator-specific collectors under `autoperfpy/collectors/`.
3. Add accelerator-aware profiles under `autoperfpy/profiles/`.
4. Update benchmark paths to respect accelerator capability constraints.
5. Add CLI filters/selectors for accelerator categories.

## Medium Priority

### Add More Precision Support

Current precision coverage is FP32, FP16, and INT8. Expand for newer hardware:

- BF16
- FP8 variants
- INT4
- Mixed precision modes

Reference:

- `trackiq_core/inference/config.py`

### Complete Multi-Run Comparison Charts in Reports

Integrate multi-run chart generation into HTML reporting flow.

References:

- `autoperfpy/reports/charts.py`
- `autoperfpy/reports/html_generator.py`

### Continue CLI Modularization

`autoperfpy/cli.py` is still large; continue splitting command groups into focused modules.

## Low Priority

### Performance Optimization

- Profile collector overhead.
- Optimize serialization for large runs.
- Evaluate streaming output for long-running benchmarks.

### Testing Expansion

- Add integration tests for accelerator-specific collectors.
- Add multi-accelerator scenario tests.
- Add validation around power and thermal metric accuracy.

