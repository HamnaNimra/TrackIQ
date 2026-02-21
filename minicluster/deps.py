"""Centralized trackiq_core dependencies for minicluster.

This module centralizes all imports from trackiq_core so that when trackiq_core
becomes a standalone pip-installable package, only this file needs to change.
All other modules should import from minicluster.deps instead.
"""

# Regression detection and baseline management
from trackiq_core.utils.compare import (
    RegressionDetector,
    RegressionThreshold,
    MetricComparison,
)

# Schema definitions
from trackiq_core.schemas import AnalysisResult

# Distributed training configuration
from trackiq_core.distributed_validator import DistributedValidationConfig

# Configuration and utilities
from trackiq_core.utils.stats import percentile, stats_from_values
from trackiq_core.utils.dict_utils import safe_get
from trackiq_core.configs.config_io import (
    load_json_file,
    save_json_file,
    load_yaml_file,
    save_yaml_file,
    ensure_parent_dir,
)

__all__ = [
    "RegressionDetector",
    "RegressionThreshold",
    "MetricComparison",
    "AnalysisResult",
    "DistributedValidationConfig",
    "percentile",
    "stats_from_values",
    "safe_get",
    "load_json_file",
    "save_json_file",
    "load_yaml_file",
    "save_yaml_file",
    "ensure_parent_dir",
]
