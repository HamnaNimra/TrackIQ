"""Hardware device detection for TrackIQ.

Generic detection for edge AI and embedded platforms: NVIDIA GPUs (nvidia-ml-py or
nvidia-smi), Intel GPUs/integrated, NVIDIA Jetson and DRIVE (tegrastats-capable),
and CPUs (psutil + platform). Builds DeviceProfile dicts with platform metadata.
No remote execution or networking. Applications (e.g. autoperfpy) map device_type
to collectors; trackiq does not depend on any specific collector.
"""

from dataclasses import dataclass, field
import json
import platform
import subprocess
import sys
from typing import Any, Dict, List
try:
    import pynvml  # provided by nvidia-ml-py
except Exception:  # pragma: no cover - optional dependency
    pynvml = None


# Device type constants (generic; apps map these to collectors)
DEVICE_TYPE_NVIDIA_GPU = "NVIDIA_GPU"
DEVICE_TYPE_AMD_GPU = "AMD_GPU"
DEVICE_TYPE_INTEL_GPU = "INTEL_GPU"
DEVICE_TYPE_CPU = "CPU"
DEVICE_TYPE_APPLE_SILICON = "Apple_Silicon"
# Tegrastats-capable embedded / automotive platforms (Jetson, DRIVE Orin/Thor, etc.)
DEVICE_TYPE_NVIDIA_JETSON = "NVIDIA_Jetson"
DEVICE_TYPE_NVIDIA_DRIVE = "NVIDIA_Drive"


@dataclass
class DeviceProfile:
    """Profile for a detected device.

    Attributes:
        device_id: Unique id (e.g. "nvidia_0", "cpu_0", "tegrastats_0")
        device_type: One of nvidia_gpu, intel_gpu, cpu, nvidia_jetson, nvidia_drive
        device_name: Human-readable name
        gpu_model: GPU model string if applicable
        cpu_model: CPU model string (filled for platform info)
        soc: SoC or board identifier (e.g. Jetson AGX Orin, DRIVE Thor)
        power_mode: Power / perf mode if known
        index: Device index for collectors (e.g. GPU index)
        metadata: Optional OS, driver_version, cuda_version, etc.
    """

    device_id: str
    device_type: str
    device_name: str
    gpu_model: str = ""
    cpu_model: str = ""
    soc: str = ""
    power_mode: str = ""
    index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Export as dict for JSON and UI."""
        return {
            "device_id": self.device_id,
            "device_type": self.device_type,
            "device_name": self.device_name,
            "gpu_model": self.gpu_model,
            "cpu_model": self.cpu_model,
            "soc": self.soc,
            "power_mode": self.power_mode,
            "index": self.index,
            "metadata": self.metadata,
        }


def _get_cpu_info() -> str:
    """Get CPU identifier using platform (and psutil if available)."""
    try:
        import platform as plat

        proc = plat.processor() or plat.machine() or ""
        if proc:
            return proc.strip()
    except Exception:
        pass
    try:
        import psutil

        # Fallback: count cores as minimal identifier
        return f"CPU ({psutil.cpu_count()} cores)"
    except Exception:
        pass
    return "Unknown CPU"


def _get_os_info() -> str:
    """Get OS name and version."""
    try:
        import platform as plat

        return f"{plat.system()} {plat.release()}"
    except Exception:
        return "Unknown OS"


def detect_nvidia_gpus() -> List[DeviceProfile]:
    """Detect NVIDIA GPUs using nvidia-ml-py (pynvml module) if available, else nvidia-smi."""
    devices: List[DeviceProfile] = []
    cpu_model = _get_cpu_info()
    os_info = _get_os_info()
    driver_version = ""
    cuda_version = ""

    try:
        if pynvml is None:
            raise ImportError("pynvml unavailable")
        pynvml.nvmlInit()
        try:
            driver_version = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver_version, bytes):
                driver_version = driver_version.decode("utf-8")
        except Exception:
            pass
        try:
            cuda_version = str(pynvml.nvmlSystemGetCudaDriverVersion())
        except Exception:
            pass
        n = pynvml.nvmlDeviceGetCount()
        for i in range(n):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            name = name or f"NVIDIA GPU {i}"
            devices.append(
                DeviceProfile(
                    device_id=f"nvidia_{i}",
                    device_type=DEVICE_TYPE_NVIDIA_GPU,
                    device_name=name,
                    gpu_model=name,
                    cpu_model=cpu_model,
                    soc="",
                    power_mode="",
                    index=i,
                    metadata={
                        "os": os_info,
                        "driver_version": driver_version,
                        "cuda_version": cuda_version,
                    },
                )
            )
        pynvml.nvmlShutdown()
        return devices
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: nvidia-smi
    try:
        from .gpu import query_nvidia_smi

        out = query_nvidia_smi(["name"], timeout=3)
        if out:
            names = [s.strip() for s in out.split("\n")]
            for i, name in enumerate(names):
                if name:
                    devices.append(
                        DeviceProfile(
                            device_id=f"nvidia_{i}",
                            device_type=DEVICE_TYPE_NVIDIA_GPU,
                            device_name=name,
                            gpu_model=name,
                            cpu_model=cpu_model,
                            soc="",
                            power_mode="",
                            index=i,
                            metadata={"os": os_info},
                        )
                    )
    except Exception:
        pass

    return devices


def detect_amd_gpus() -> List[DeviceProfile]:
    """Detect AMD GPUs via `rocm-smi` JSON output with plain-text fallback."""
    devices: List[DeviceProfile] = []
    cpu_model = _get_cpu_info()
    os_info = _get_os_info()

    def _rocm_version() -> str:
        try:
            out = subprocess.run(
                ["rocm-smi", "--version"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            return (out.stdout or out.stderr or "").strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return ""

    rocm_version = _rocm_version()

    try:
        result = subprocess.run(
            ["rocm-smi", "--showallinfo", "--json"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        raw = (result.stdout or "").strip()
        if raw:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                for idx, (_, data) in enumerate(payload.items()):
                    if not isinstance(data, dict):
                        continue
                    name = ""
                    for key, value in data.items():
                        low = key.lower()
                        if "product" in low or "card series" in low or "gpu" in low:
                            name = str(value).strip()
                            if name:
                                break
                    gpu_name = name or f"AMD GPU {idx}"
                    devices.append(
                        DeviceProfile(
                            device_id=f"amd_{idx}",
                            device_type=DEVICE_TYPE_AMD_GPU,
                            device_name=gpu_name,
                            gpu_model=gpu_name,
                            cpu_model=cpu_model,
                            soc="",
                            power_mode="",
                            index=idx,
                            metadata={
                                "os": os_info,
                                "rocm_version": rocm_version,
                            },
                        )
                    )
            if devices:
                return devices
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return []
    except Exception:
        pass

    try:
        fallback = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        lines = [line.strip() for line in (fallback.stdout or "").splitlines() if line.strip()]
        for idx, line in enumerate(lines):
            if "card series" not in line.lower() and "product name" not in line.lower():
                continue
            parts = line.split(":", 1)
            gpu_name = parts[1].strip() if len(parts) == 2 else line
            devices.append(
                DeviceProfile(
                    device_id=f"amd_{idx}",
                    device_type=DEVICE_TYPE_AMD_GPU,
                    device_name=gpu_name,
                    gpu_model=gpu_name,
                    cpu_model=cpu_model,
                    soc="",
                    power_mode="",
                    index=idx,
                    metadata={
                        "os": os_info,
                        "rocm_version": rocm_version,
                    },
                )
            )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    except Exception:
        pass
    return devices


def detect_apple_silicon() -> List[DeviceProfile]:
    """Detect Apple Silicon platform and expose it as a single device profile."""
    devices: List[DeviceProfile] = []
    machine = (platform.machine() or "").lower()
    if sys.platform != "darwin" or "arm" not in machine:
        return devices

    chip_name = "Apple Silicon"
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        raw = (out.stdout or "").strip()
        if raw:
            chip_name = raw
    except Exception:
        pass

    cpu_model = _get_cpu_info()
    os_info = _get_os_info()
    devices.append(
        DeviceProfile(
            device_id="apple_0",
            device_type=DEVICE_TYPE_APPLE_SILICON,
            device_name=chip_name,
            gpu_model=chip_name,
            cpu_model=cpu_model,
            soc=chip_name,
            power_mode="",
            index=0,
            metadata={"os": os_info},
        )
    )
    return devices


def detect_cpu() -> DeviceProfile:
    """Detect CPU using psutil and platform."""
    cpu_model = _get_cpu_info()
    os_info = _get_os_info()
    return DeviceProfile(
        device_id="cpu_0",
        device_type=DEVICE_TYPE_CPU,
        device_name=cpu_model,
        gpu_model="",
        cpu_model=cpu_model,
        soc="",
        power_mode="",
        index=0,
        metadata={"os": os_info},
    )


def detect_intel_gpus() -> List[DeviceProfile]:
    """Detect Intel GPUs / integrated accelerators if identifiable.

    Uses platform and optional env/probe; no remote or network.
    """
    devices: List[DeviceProfile] = []
    cpu_model = _get_cpu_info()
    os_info = _get_os_info()

    try:
        import platform as plat

        # Heuristic: Intel integrated often reported in machine or processor
        mach = (plat.machine() or "").lower()
        proc = (plat.processor() or "").lower()
        if "intel" in proc or "intel" in mach:
            devices.append(
                DeviceProfile(
                    device_id="intel_0",
                    device_type=DEVICE_TYPE_INTEL_GPU,
                    device_name="Intel (integrated)",
                    gpu_model="Intel (integrated)",
                    cpu_model=cpu_model,
                    soc="",
                    power_mode="",
                    index=0,
                    metadata={"os": os_info},
                )
            )
    except Exception:
        pass

    return devices


def detect_tegrastats_platforms() -> List[DeviceProfile]:
    """Detect NVIDIA Jetson or DRIVE (tegrastats-capable) platforms.

    Uses /etc/nv_tegra_release when present; no subprocess or network.
    Returns at most one device (single SoC per system). Generic for edge AI
    and automotive; applications map device_type to TegrastatsCollector or equivalent.
    """
    devices: List[DeviceProfile] = []
    cpu_model = _get_cpu_info()
    os_info = _get_os_info()
    tegra_release_path = "/etc/nv_tegra_release"
    try:
        import os

        if not os.path.isfile(tegra_release_path):
            return devices
        with open(tegra_release_path, "r", encoding="utf-8", errors="replace") as f:
            first_line = (f.readline() or "").strip()
        if not first_line:
            return devices
        # Infer Jetson vs DRIVE from content (e.g. "DRIVE" in string -> DRIVE)
        upper = first_line.upper()
        if "DRIVE" in upper or "DRIVEOS" in upper:
            device_type = DEVICE_TYPE_NVIDIA_DRIVE
            device_id = "nvidia_drive_0"
            soc = "DRIVE"
        else:
            device_type = DEVICE_TYPE_NVIDIA_JETSON
            device_id = "nvidia_jetson_0"
            soc = "Jetson"
        # Use first line as device name (e.g. "# R35.4.1" or similar)
        device_name = first_line[:80] if len(first_line) > 80 else first_line
        if device_name.startswith("#"):
            device_name = device_name[1:].strip() or f"NVIDIA {soc}"
        devices.append(
            DeviceProfile(
                device_id=device_id,
                device_type=device_type,
                device_name=device_name,
                gpu_model="",
                cpu_model=cpu_model,
                soc=soc,
                power_mode="",
                index=0,
                metadata={"os": os_info, "tegra_release": first_line},
            )
        )
    except (OSError, IOError):
        pass
    except Exception:
        pass
    return devices


def get_all_devices(
    include_nvidia: bool = True,
    include_amd: bool = True,
    include_apple: bool = True,
    include_intel: bool = True,
    include_cpu: bool = True,
    include_tegrastats: bool = True,
) -> List[DeviceProfile]:
    """Detect all available devices and return DeviceProfile list.

    Order: NVIDIA GPUs (by index), Tegrastats platforms (Jetson/DRIVE), Intel GPUs, CPU.
    """
    result: List[DeviceProfile] = []
    if include_nvidia:
        result.extend(detect_nvidia_gpus())
    if include_amd:
        result.extend(detect_amd_gpus())
    if include_apple:
        result.extend(detect_apple_silicon())
    if include_tegrastats:
        result.extend(detect_tegrastats_platforms())
    if include_intel:
        result.extend(detect_intel_gpus())
    # CPU is always included as the baseline platform.
    _ = include_cpu
    result.append(detect_cpu())
    return result


def get_platform_metadata_for_device(device: DeviceProfile) -> Dict[str, Any]:
    """Build platform_metadata dict for a device (for run exports)."""
    return {
        "device_name": device.device_name,
        "device_id": device.device_id,
        "device_type": device.device_type,
        "soc": device.soc,
        "gpu_model": device.gpu_model or None,
        "cpu_model": device.cpu_model or None,
        "power_mode": device.power_mode or None,
        **{k: v for k, v in device.metadata.items() if v},
    }
