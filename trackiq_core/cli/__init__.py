"""Generic CLI module for trackiq_core.

Provides reusable CLI utilities and commands that can be used by any
application built on trackiq_core.
"""

from .commands import run_compare, run_devices_list
from .utils import output_path, run_default_benchmark, write_result_to_csv

__all__ = [
    "output_path",
    "write_result_to_csv",
    "run_default_benchmark",
    "run_devices_list",
    "run_compare",
]
