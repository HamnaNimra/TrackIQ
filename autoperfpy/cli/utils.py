"""CLI utility functions for AutoPerfPy (automotive-specific).

Re-exports generic CLI utilities from trackiq_core and adds AutoPerfPy-specific wrappers.
"""

from typing import Optional, Tuple

from trackiq_core.cli.utils import output_path as _trackiq_output_path
from trackiq_core.cli.utils import write_result_to_csv as _trackiq_write_csv
from trackiq_core.cli.utils import run_default_benchmark as _trackiq_run_benchmark
from autoperfpy.device_config import resolve_device
from autoperfpy.auto_runner import run_single_benchmark


# Re-export with backward-compatible names
def _output_path(args, filename: str) -> str:
    """Return path for an output file inside the output directory (create dir if needed)."""
    return _trackiq_output_path(args, filename)


def _write_result_to_csv(result: dict, path: str) -> bool:
    """Write run result samples to a CSV file. Returns True if written."""
    return _trackiq_write_csv(result, path)


def _run_default_benchmark(
    device_id: Optional[str] = None,
    duration_seconds: int = 10,
) -> Tuple[dict, Optional[str], Optional[str]]:
    """Run a short benchmark for AutoPerfPy and return (data_dict, temp_csv_path, temp_json_path)."""
    return _trackiq_run_benchmark(
        device_resolver_fn=resolve_device,
        benchmark_runner_fn=run_single_benchmark,
        device_id=device_id,
        duration_seconds=duration_seconds,
    )
