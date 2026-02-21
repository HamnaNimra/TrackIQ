# TrackIQ Hardware Support

## Supported Platforms

| Device Type Constant | Detection Function | Live Metrics Function | Required Tool/Library | Fallback Behavior |
|---|---|---|---|---|
| `DEVICE_TYPE_NVIDIA_GPU` | `detect_nvidia_gpus()` | `get_memory_metrics()`, `get_performance_metrics()` | `nvidia-smi` or `nvidia-ml-py` | Returns empty detection list or `None` metrics when unavailable |
| `DEVICE_TYPE_AMD_GPU` | `detect_amd_gpus()` | `get_amd_gpu_metrics()` | `rocm-smi` | Returns empty detection list or `None` metrics when unavailable |
| `DEVICE_TYPE_APPLE_SILICON` | `detect_apple_silicon()` | `get_apple_silicon_metrics()` | `sysctl`, optional `powermetrics`, `psutil` | Returns empty detection list off macOS arm; metrics fall back to psutil on macOS |
| `DEVICE_TYPE_INTEL_GPU` | `detect_intel_gpus()` | `get_intel_gpu_metrics()` | `intel_gpu_top` (optional), Linux sysfs (optional) | Returns empty detection list or `None` metrics when unavailable |
| `DEVICE_TYPE_NVIDIA_JETSON` | `detect_tegrastats_platforms()` | Device panel uses tegrastats guidance message | tegrastats-capable platform | Detection succeeds when platform markers exist; live metrics are delegated to tegrastats integration |
| `DEVICE_TYPE_NVIDIA_DRIVE` | `detect_tegrastats_platforms()` | Device panel uses tegrastats guidance message | tegrastats-capable platform | Detection succeeds when platform markers exist; live metrics are delegated to tegrastats integration |
| `DEVICE_TYPE_CPU` | `detect_cpu()` | `get_cpu_metrics()` | `psutil` | Always detected and always returns metrics dict |

## How To Add A New Platform

1. Add a new device type constant in `trackiq_core/hardware/devices.py`.
2. Add a detection function in `trackiq_core/hardware/devices.py` that returns `List[DeviceProfile]`.
3. Add a live metrics function in `trackiq_core/hardware/gpu.py` that returns `Optional[Dict[str, float]]` (or a dict for always-on metrics like CPU).
4. Register the device type in `trackiq_core/ui/components/device_panel.py` by adding an entry to `METRICS_DISPATCH`.

## Known Limitations

- Apple `powermetrics` typically requires `sudo`; non-interactive sudo may fail in CI or developer shells.
- Intel sysfs fallback metrics are Linux-only and depend on kernel/driver paths being available.
- Tegrastats live metrics require physical Jetson or DRIVE hardware and are not available on standard desktop hosts.
