# TrackIQ & AutoPerfPy TODO

## High Priority

### Add Support for Accelerator Categories
Support different accelerator types commonly used in automotive systems:

- **GPU**: Flexible, high power, handles any precision (FP32/FP16/INT8)
- **NPU/DLA**: Rigid, low power, handles INT8/FP16 only (NVIDIA, Qualcomm, etc.)
- **DSP**: Great for audio/signal processing (common in Qualcomm Automotive)
- **FPGA**: Customizable silicon (common in AMD/Xilinx Automotive)

**Implementation Tasks:**
1. Extend device detection in `autoperfpy/device_config.py` to identify accelerator types
2. Add accelerator-specific collectors for NPU/DLA, DSP, and FPGA
   - Similar to how tegrastats handles Jetson/DRIVE
   - Create collectors in `autoperfpy/collectors/`
3. Create accelerator-specific profiles in `autoperfpy/profiles/`
   - Define precision constraints (e.g., NPU/DLA limited to INT8/FP16)
   - Define power/thermal characteristics
   - Define typical workloads (vision on GPU/NPU, sensor fusion on DSP, etc.)
4. Update benchmarks to respect accelerator capabilities
   - Skip unsupported precisions for NPU/DLA
   - Add accelerator-specific metrics
5. Update CLI commands to filter/select by accelerator type
   - `autoperfpy devices --list --type npu`
   - `autoperfpy run --auto --accelerator-types gpu,npu`

**Files to Modify:**
- `autoperfpy/device_config.py` - Add accelerator detection
- `autoperfpy/collectors/` - Add new collector modules
- `autoperfpy/profiles/profiles.py` - Add accelerator-aware profiles
- `autoperfpy/benchmarks/` - Update to respect accelerator constraints
- `autoperfpy/cli.py` - Add CLI options for accelerator filtering

## Medium Priority

### Add More Precision Support
**Issue:** Currently supports FP32, FP16, INT8. Need to add more precisions as hardware support expands.

**Potential additions:**
- BF16 (bfloat16) - Common in modern accelerators
- FP8 (E4M3, E5M2) - NVIDIA Hopper, AMD CDNA3
- INT4 - Emerging quantization format
- Mixed precision modes

**Files:**
- `trackiq_core/inference/config.py:12` - Has TODO to add more precisions

### Implement Multi-Run Comparison Chart
**Issue:** Multi-Run Comparison Chart from charts.py is not implemented yet in HTML reports.

**Implementation:**
- Add chart generation in `autoperfpy/reports/charts.py`
- Integrate into `autoperfpy/reports/html_generator.py`

**Files:**
- `autoperfpy/reports/html_generator.py:12` - Has TODO noting missing implementation

### Further CLI Modularization
The `autoperfpy/cli.py` file is still 2030+ lines. Consider splitting into:
- `autoperfpy/cli/commands/run.py` - Run commands
- `autoperfpy/cli/commands/analyze.py` - Analyze commands
- `autoperfpy/cli/commands/benchmark.py` - Benchmark commands
- `autoperfpy/cli/commands/monitor.py` - Monitor commands
- `autoperfpy/cli/commands/report.py` - Report commands
- `autoperfpy/cli/commands/profiles.py` - Profiles commands

Keep as lower priority since current structure works well.

### Documentation
- Add architecture documentation explaining trackiq_core vs autoperfpy separation
- Document the accelerator categories feature once implemented
- Add examples for automotive-specific use cases

## Low Priority

### Performance Optimizations
- Profile collector overhead
- Optimize data serialization for large benchmark runs
- Consider streaming results for long-running benchmarks

### Testing
- Add integration tests for accelerator-specific collectors
- Add tests for multi-accelerator scenarios
- Test power/thermal monitoring accuracy

## Completed ✓
- ✓ Refactored CLI to separate generic (trackiq_core) and automotive-specific (autoperfpy) code
- ✓ Moved generic modules to trackiq_core (inference, runners, monitoring, benchmarks)
- ✓ Merged autoperf_app into autoperfpy
- ✓ Created backward-compatible wrappers
- ✓ Resolved Config duplication - autoperfpy.config.Config now extends trackiq_core.configs.Config
