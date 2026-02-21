"""Generic CLI commands for trackiq_core."""

from .compare import run_compare
from .devices import run_devices_list

__all__ = [
    "run_devices_list",
    "run_compare",
]
