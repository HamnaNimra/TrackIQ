#!/usr/bin/env python3
"""
This example attempts to run a GPU benchmark using nvidia-smi and includes
robust error handling and logging to ensure it works correctly as a systemd service.
Usage:
    python service_benchmarks.py


Disclaimer: This code is for educational purposes only.
It demonstrates how to run GPU benchmarks with error handling and logging.
Author: Hamna
Target: NVIDIA Edge AI / Automotive Performance Engineering
"""

import subprocess
import os
import sys
import logging
from pathlib import Path

try:
    from trackiq_core.env import nvidia_smi_available, NVIDIA_SMI_PATHS
except ImportError:
    nvidia_smi_available = None
    NVIDIA_SMI_PATHS = [
        "/usr/bin/nvidia-smi",
        "/usr/local/bin/nvidia-smi",
        "/opt/nvidia/bin/nvidia-smi",
    ]

# Configure logging to file (systemd will capture this)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/var/log/gpu_benchmark.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


def check_gpu_available() -> bool:
    """Check if NVIDIA GPU and drivers are available."""
    if nvidia_smi_available is not None:
        if nvidia_smi_available():
            for path in NVIDIA_SMI_PATHS:
                if os.path.exists(path):
                    logger.info(f"Found nvidia-smi at: {path}")
                    break
            return True
        logger.error("nvidia-smi not found in any common location")
        logger.error(f"Searched: {NVIDIA_SMI_PATHS}")
        return False
    for path in NVIDIA_SMI_PATHS:
        if os.path.exists(path):
            logger.info(f"Found nvidia-smi at: {path}")
            return True
    logger.error("nvidia-smi not found in any common location")
    logger.error(f"Searched: {NVIDIA_SMI_PATHS}")
    return False


def run_benchmark() -> bool:
    """Run GPU benchmark with proper error handling"""

    # Use absolute path for nvidia-smi
    nvidia_smi_path = "/usr/bin/nvidia-smi"

    # Verify command exists
    if not os.path.exists(nvidia_smi_path):
        logger.error(f"nvidia-smi not found at {nvidia_smi_path}")

        # Try to find it
        if not check_gpu_available():
            return False

    try:
        # Run nvidia-smi with explicit path and timeout
        result = subprocess.run(
            [nvidia_smi_path], capture_output=True, timeout=10, check=False
        )

        if result.returncode != 0:
            logger.error(f"nvidia-smi failed with return code {result.returncode}")
            logger.error(f"stderr: {result.stderr.decode()}")
            return False

        # Use absolute path for output file
        output_dir = Path("/var/log")
        output_file = output_dir / "benchmark_results.txt"

        # Ensure directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write results
        with open(output_file, "w") as f:
            f.write(result.stdout.decode())

        logger.info(f"Benchmark complete, results written to {output_file}")
        return True

    except subprocess.TimeoutExpired:
        logger.error("nvidia-smi timed out after 10 seconds")
        return False
    except PermissionError as e:
        logger.error(f"Permission error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False


def main():
    """Main entry point"""
    logger.info("Starting GPU benchmark service")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"PATH environment: {os.environ.get('PATH', 'NOT SET')}")
    logger.info(f"USER: {os.environ.get('USER', 'NOT SET')}")

    success = run_benchmark()

    if success:
        logger.info("Benchmark completed successfully")
        sys.exit(0)
    else:
        logger.error("Benchmark failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
