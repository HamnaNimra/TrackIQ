"""GPU detection and metrics via nvidia-smi for TrackIQ platform."""

import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any

DEFAULT_NVIDIA_SMI_TIMEOUT = 5


def query_nvidia_smi(
    query_fields: list[str],
    timeout: int = DEFAULT_NVIDIA_SMI_TIMEOUT,
) -> str | None:
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


def query_rocm_smi(args: list[str], timeout: int = 3) -> str | None:
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
    field_names: list[str],
    separator: str = ",",
) -> dict[str, float] | None:
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
) -> dict[str, Any] | None:
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
        parsed["gpu_memory_percent"] = parsed["gpu_memory_used_mb"] / parsed["gpu_memory_total_mb"] * 100
    else:
        parsed["gpu_memory_percent"] = 0.0

    return parsed


def get_performance_metrics(
    timeout: int = DEFAULT_NVIDIA_SMI_TIMEOUT,
) -> dict[str, float] | None:
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


def _extract_float(text: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def _get_windows_cpu_temperature() -> float | None:
    """Best-effort Windows CPU temperature using optional WMI providers."""
    # Option 1: python wmi package (optional dependency).
    try:
        import wmi  # type: ignore

        client = wmi.WMI(namespace="root\\wmi")
        zones = client.MSAcpi_ThermalZoneTemperature()
        if zones:
            raw = getattr(zones[0], "CurrentTemperature", None)
            if raw is not None:
                # Deci-Kelvin to Celsius.
                return float(raw) / 10.0 - 273.15
    except Exception:
        pass

    # Option 2: WMIC fallback when available.
    try:
        result = subprocess.run(
            [
                "wmic",
                "/namespace:\\\\root\\wmi",
                "PATH",
                "MSAcpi_ThermalZoneTemperature",
                "get",
                "CurrentTemperature",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        values = [_extract_float(line) for line in (result.stdout or "").splitlines()]
        values = [v for v in values if v is not None and v > 0]
        if values:
            return float(values[0]) / 10.0 - 273.15
    except Exception:
        pass
    return None


def _get_macos_cpu_temperature() -> float | None:
    """Best-effort macOS CPU temperature via powermetrics or osx-cpu-temp."""
    try:
        result = subprocess.run(
            ["sudo", "-n", "powermetrics", "--samplers", "smc", "-n", "1"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        raw = (result.stdout or "") + "\n" + (result.stderr or "")
        for line in raw.splitlines():
            low = line.lower()
            if "cpu die temperature" in low or "cpu temperature" in low:
                value = _extract_float(line)
                if value is not None:
                    return value
    except Exception:
        pass

    try:
        if shutil.which("osx-cpu-temp") is None:
            return None
        out = subprocess.run(
            ["osx-cpu-temp"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        value = _extract_float(out.stdout or "")
        if value is not None:
            return value
    except Exception:
        pass
    return None


def get_amd_gpu_metrics(index: int = 0) -> dict[str, float] | None:
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


def get_intel_gpu_metrics() -> dict[str, float] | None:
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
            with open(busy_path, encoding="utf-8") as handle:
                utilization = _extract_float(handle.read())
        hwmon_base = os.path.join(base, "device/hwmon")
        if os.path.exists(hwmon_base):
            for name in os.listdir(hwmon_base):
                power_path = os.path.join(hwmon_base, name, "power1_average")
                if os.path.exists(power_path):
                    with open(power_path, encoding="utf-8") as handle:
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


def get_apple_silicon_metrics() -> dict[str, float] | None:
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


def get_cpu_metrics() -> dict[str, float] | None:
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

        if temperature is None and sys.platform.startswith("win"):
            temperature = _get_windows_cpu_temperature()
        if temperature is None and sys.platform == "darwin":
            temperature = _get_macos_cpu_temperature()
        return {
            "cpu_utilization": cpu_util,
            "memory_percent": mem_percent,
            "temperature": temperature,  # type: ignore[typeddict-item]
        }
    except Exception:
        return {"cpu_utilization": 0.0, "memory_percent": 0.0, "temperature": None}  # type: ignore[typeddict-item]
