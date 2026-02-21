"""Centralized trackiq_core dependencies for minicluster.

This module centralizes all imports from trackiq_core so that when trackiq_core
becomes a standalone pip-installable package, only this file needs to change.
All other modules should import from minicluster.deps instead.
"""

# Regression detection and baseline management
from minicluster.monitor.anomaly_detector import Anomaly, AnomalyDetector
from minicluster.monitor.health_reader import HealthReader
from minicluster.monitor.health_reporter import HealthReporter
from trackiq_core.configs.config_io import (
    ensure_parent_dir,
    load_json_file,
    load_yaml_file,
    save_json_file,
    save_yaml_file,
)

# Distributed training configuration
from trackiq_core.distributed_validator import DistributedValidationConfig

# Schema definitions
from trackiq_core.schemas import AnalysisResult
from trackiq_core.utils.compare import (
    MetricComparison,
    RegressionDetector,
    RegressionThreshold,
)
from trackiq_core.utils.dict_utils import safe_get

# Configuration and utilities
from trackiq_core.utils.stats import percentile, stats_from_values

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
    "HealthReader",
    "AnomalyDetector",
    "Anomaly",
    "HealthReporter",
]
