"""Generic CLI module for trackiq_core.

Provides reusable CLI utilities and commands that can be used by any
application built on trackiq_core.
"""

from .utils import output_path, write_result_to_csv, run_default_benchmark
from .commands import run_devices_list, run_compare

__all__ = [
    "output_path",
    "write_result_to_csv",
    "run_default_benchmark",
    "run_devices_list",
    "run_compare",
]
