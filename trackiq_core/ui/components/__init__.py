"""Reusable UI components for TrackIQ dashboards."""

from trackiq_core.ui.components.metric_table import MetricTable
from trackiq_core.ui.components.loss_chart import LossChart
from trackiq_core.ui.components.regression_badge import RegressionBadge
from trackiq_core.ui.components.worker_grid import WorkerGrid
from trackiq_core.ui.components.power_gauge import PowerGauge
from trackiq_core.ui.components.comparison_table import ComparisonTable
from trackiq_core.ui.components.device_panel import DevicePanel
from trackiq_core.ui.components.result_browser import ResultBrowser

__all__ = [
    "MetricTable",
    "LossChart",
    "RegressionBadge",
    "WorkerGrid",
    "PowerGauge",
    "ComparisonTable",
    "DevicePanel",
    "ResultBrowser",
]
