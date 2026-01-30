"""Environment and hardware availability checks."""

import os
import subprocess
from typing import List, Optional

# Common nvidia-smi search paths (used by tools and monitoring)
NVIDIA_SMI_PATHS = [
    "/usr/bin/nvidia-smi",
    "/usr/local/bin/nvidia-smi",
    "/opt/nvidia/bin/nvidia-smi",
]


def command_available(cmd_name: str, timeout: float = 5.0) -> bool:
    """Check if a command is available on the system (e.g. which cmd).

    Args:
        cmd_name: Command name (e.g. "tegrastats", "nvidia-smi")
        timeout: Timeout in seconds for the check

    Returns:
        True if the command is found and executable
    """
    try:
        result = subprocess.run(
            ["which", cmd_name],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return False


def nvidia_smi_available(paths: Optional[List[str]] = None) -> bool:
    """Check if nvidia-smi is available (file exists in common paths).

    Args:
        paths: Optional list of paths to check; default is NVIDIA_SMI_PATHS

    Returns:
        True if nvidia-smi exists at one of the paths
    """
    for path in paths or NVIDIA_SMI_PATHS:
        if os.path.exists(path):
            return True
    return False


def find_nvidia_smi_path(paths: Optional[List[str]] = None) -> Optional[str]:
    """Return first path where nvidia-smi exists, or None."""
    for path in paths or NVIDIA_SMI_PATHS:
        if os.path.exists(path):
            return path
    return None
