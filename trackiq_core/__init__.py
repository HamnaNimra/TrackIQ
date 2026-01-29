"""Shared abstractions for TrackIQ and AutoPerfPy.

Provides: dict_utils (safe_get), stats (percentile, stats_from_values),
env (command_available, nvidia_smi_available), config_io (load/save YAML/JSON).
"""

from trackiq_core.dict_utils import safe_get
from trackiq_core.stats import percentile, stats_from_values
from trackiq_core.env import command_available, nvidia_smi_available
from trackiq_core.config_io import (
    load_yaml_file,
    load_json_file,
    save_yaml_file,
    save_json_file,
    ensure_parent_dir,
)

__all__ = [
    "safe_get",
    "percentile",
    "stats_from_values",
    "command_available",
    "nvidia_smi_available",
    "load_yaml_file",
    "load_json_file",
    "save_yaml_file",
    "save_json_file",
    "ensure_parent_dir",
]
