"""CLI utility compatibility wrappers for AutoPerfPy.

This module delegates to the canonical helper implementations in
``autoperfpy.cli`` to avoid behavior drift across modules.
"""


# Re-export with backward-compatible names
def _output_path(args, filename: str) -> str:
    """Return path for an output file inside the output directory (create dir if needed)."""
    from autoperfpy.cli import _output_path as _cli_output_path

    return _cli_output_path(args, filename)


def _write_result_to_csv(result: dict, path: str) -> bool:
    """Write run result samples to a CSV file. Returns True if written."""
    from autoperfpy.cli import _write_result_to_csv as _cli_write_result_to_csv

    return _cli_write_result_to_csv(result, path)


def _run_default_benchmark(
    device_id: str | None = None,
    duration_seconds: int = 10,
) -> tuple[dict, str | None, str | None]:
    """Run a short benchmark for AutoPerfPy and return (data_dict, temp_csv_path, temp_json_path)."""
    from autoperfpy.cli import _run_default_benchmark as _cli_run_default_benchmark

    return _cli_run_default_benchmark(
        device_id=device_id,
        duration_seconds=duration_seconds,
    )
