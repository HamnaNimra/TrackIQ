"""GPU detection and metrics via nvidia-smi for TrackIQ platform."""

import json
import os
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional

DEFAULT_NVIDIA_SMI_TIMEOUT = 5


def query_nvidia_smi(
    query_fields: List[str],
    timeout: int = DEFAULT_NVIDIA_SMI_TIMEOUT,
) -> Optional[str]:
    """Execute nvidia-smi query and return raw output."""
    try:
        query_string = ",".join(query_fields)
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query_string}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode == 0:
            return result.stdout.strip()
        return None

    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def query_rocm_smi(args: List[str], timeout: int = 3) -> Optional[str]:
    """Execute rocm-smi with args and return raw output or None on failure."""
    try:
        result = subprocess.run(
            ["rocm-smi", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def parse_gpu_metrics(
    output: str,
    field_names: List[str],
    separator: str = ",",
) -> Optional[Dict[str, float]]:
    """Parse nvidia-smi CSV output into a dictionary."""
    try:
        values = [v.strip() for v in output.split(separator)]

        if len(values) != len(field_names):
            return None

        return {name: float(value) for name, value in zip(field_names, values)}

    except (ValueError, IndexError):
        return None


def get_memory_metrics(
    timeout: int = DEFAULT_NVIDIA_SMI_TIMEOUT,
) -> Optional[Dict[str, Any]]:
    """Get GPU memory metrics."""
    output = query_nvidia_smi(
        ["memory.used", "memory.total", "utilization.gpu"],
        timeout=timeout,
    )

    if output is None:
        return None

    parsed = parse_gpu_metrics(
        output,
        ["gpu_memory_used_mb", "gpu_memory_total_mb", "gpu_utilization_percent"],
    )

    if parsed is None:
        return None

    if parsed["gpu_memory_total_mb"] > 0:
        parsed["gpu_memory_percent"] = (
            parsed["gpu_memory_used_mb"] / parsed["gpu_memory_total_mb"] * 100
        )
    else:
        parsed["gpu_memory_percent"] = 0.0

    return parsed


def get_performance_metrics(
    timeout: int = DEFAULT_NVIDIA_SMI_TIMEOUT,
) -> Optional[Dict[str, float]]:
    """Get GPU performance metrics (utilization, temperature, power)."""
    output = query_nvidia_smi(
        ["utilization.gpu", "temperature.gpu", "power.draw"],
        timeout=timeout,
    )

    if output is None:
        return None

    return parse_gpu_metrics(
        output,
        ["utilization", "temperature", "power"],
    )


def _extract_float(text: str) -> Optional[float]:
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def get_amd_gpu_metrics(index: int = 0) -> Optional[Dict[str, float]]:
    """Get AMD GPU metrics from rocm-smi output."""
    out_temp = query_rocm_smi(["--showtemp"])
    out_power = query_rocm_smi(["--showpower"])
    out_use = query_rocm_smi(["--showuse"])
    out_mem = query_rocm_smi(["--showmeminfo", "vram"])
    if not any([out_temp, out_power, out_use, out_mem]):
        return None

    temperature = _extract_float(out_temp or "")
    power = _extract_float(out_power or "")
    utilization = _extract_float(out_use or "")
    used = None
    total = None
    if out_mem:
        lines = [line.strip() for line in out_mem.splitlines() if line.strip()]
        if len(lines) >= 2:
            nums = [_extract_float(line) for line in lines]
            nums = [n for n in nums if n is not None]
            if len(nums) >= 2:
                used = float(nums[0]) / 1024.0 if nums[0] > 2048 else float(nums[0])
                total = float(nums[1]) / 1024.0 if nums[1] > 2048 else float(nums[1])
    memory_percent = (used / total * 100.0) if used is not None and total else None
    return {
        "utilization": utilization if utilization is not None else 0.0,
        "temperature": temperature if temperature is not None else 0.0,
        "power": power if power is not None else 0.0,
        "gpu_memory_used_mb": used if used is not None else 0.0,
        "gpu_memory_total_mb": total if total is not None else 0.0,
        "gpu_memory_percent": memory_percent if memory_percent is not None else 0.0,
    }


def get_intel_gpu_metrics() -> Optional[Dict[str, float]]:
    """Get Intel GPU metrics from intel_gpu_top JSON or Linux sysfs fallback."""
    if shutil.which("intel_gpu_top"):
        try:
            output = subprocess.run(
                ["intel_gpu_top", "-J", "-s", "100", "-n", "1"],
                capture_output=True,
                text=True,
                timeout=1,
                check=False,
            )
            raw = (output.stdout or "").strip()
            if raw:
                payload = json.loads(raw)
                utilization = None
                power = None
                if isinstance(payload, dict):
                    engines = payload.get("engines") or payload.get("clients") or {}
                    if isinstance(engines, dict):
                        values = []
                        for value in engines.values():
                            if isinstance(value, dict):
                                busy = value.get("busy") or value.get("busy%")
                                if isinstance(busy, (int, float)):
                                    values.append(float(busy))
                        if values:
                            utilization = sum(values) / len(values)
                    power_val = payload.get("power")
                    if isinstance(power_val, (int, float)):
                        power = float(power_val)
                if utilization is not None or power is not None:
                    return {
                        "utilization": utilization,  # type: ignore[typeddict-item]
                        "power": power,  # type: ignore[typeddict-item]
                    }
        except Exception:
            pass

    # Linux sysfs fallback.
    try:
        base = "/sys/class/drm/card0"
        if not os.path.exists(base):
            return None
        utilization = None
        power = None
        busy_path = os.path.join(base, "gt_busy_percent")
        if os.path.exists(busy_path):
            with open(busy_path, "r", encoding="utf-8") as handle:
                utilization = _extract_float(handle.read())
        hwmon_base = os.path.join(base, "device/hwmon")
        if os.path.exists(hwmon_base):
            for name in os.listdir(hwmon_base):
                power_path = os.path.join(hwmon_base, name, "power1_average")
                if os.path.exists(power_path):
                    with open(power_path, "r", encoding="utf-8") as handle:
                        value = _extract_float(handle.read())
                        if value is not None:
                            power = value / 1_000_000.0
                            break
        if utilization is None and power is None:
            return None
        return {
            "utilization": utilization,  # type: ignore[typeddict-item]
            "power": power,  # type: ignore[typeddict-item]
        }
    except Exception:
        return None


def get_apple_silicon_metrics() -> Optional[Dict[str, float]]:
    """Get Apple Silicon metrics via powermetrics with psutil fallback."""
    if os.sys.platform != "darwin":
        return None

    cpu_util = None
    mem_percent = None
    power = None
    gpu_util = None
    try:
        import psutil

        cpu_util = float(psutil.cpu_percent(interval=0.1))
        mem_percent = float(psutil.virtual_memory().percent)
    except Exception:
        cpu_util = 0.0
        mem_percent = 0.0

    try:
        result = subprocess.run(
            ["sudo", "-n", "powermetrics", "--samplers", "gpu_power", "-n", "1"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        raw = (result.stdout or "") + "\n" + (result.stderr or "")
        for line in raw.splitlines():
            low = line.lower()
            if "gpu power" in low:
                val = _extract_float(line)
                if val is not None:
                    power = val
            if "gpu active" in low or "gpu utilization" in low:
                val = _extract_float(line)
                if val is not None:
                    gpu_util = val
    except Exception:
        pass

    return {
        "gpu_utilization": gpu_util,  # type: ignore[typeddict-item]
        "cpu_utilization": cpu_util if cpu_util is not None else 0.0,
        "memory_percent": mem_percent if mem_percent is not None else 0.0,
        "power": power,  # type: ignore[typeddict-item]
    }


def get_cpu_metrics() -> Optional[Dict[str, float]]:
    """Get CPU metrics; always returns a dictionary."""
    try:
        import psutil

        cpu_util = float(psutil.cpu_percent(interval=0.1))
        mem_percent = float(psutil.virtual_memory().percent)
        temperature = None
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for entries in temps.values():
                    if entries:
                        current = getattr(entries[0], "current", None)
                        if current is not None:
                            temperature = float(current)
                            break
        except Exception:
            temperature = None
        return {
            "cpu_utilization": cpu_util,
            "memory_percent": mem_percent,
            "temperature": temperature,  # type: ignore[typeddict-item]
        }
    except Exception:
        return {"cpu_utilization": 0.0, "memory_percent": 0.0, "temperature": None}  # type: ignore[typeddict-item]
