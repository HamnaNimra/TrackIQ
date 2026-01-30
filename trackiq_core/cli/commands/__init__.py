"""Generic CLI commands for trackiq_core."""

from .devices import run_devices_list
from .compare import run_compare

__all__ = [
    "run_devices_list",
    "run_compare",
]
