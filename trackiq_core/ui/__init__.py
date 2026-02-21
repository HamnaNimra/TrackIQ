"""Public API for TrackIQ shared UI layer."""

from trackiq_core.ui.dashboard import TrackiqDashboard
from trackiq_core.ui.theme import TrackiqTheme, DARK_THEME, LIGHT_THEME
from trackiq_core.ui.launcher import run_dashboard
from trackiq_core.ui.components import (
    MetricTable,
    LossChart,
    RegressionBadge,
    WorkerGrid,
    PowerGauge,
    ComparisonTable,
    DevicePanel,
    ResultBrowser,
    RunHistoryLoader,
    TrendChart,
)

__all__ = [
    "TrackiqDashboard",
    "TrackiqTheme",
    "DARK_THEME",
    "LIGHT_THEME",
    "run_dashboard",
    "MetricTable",
    "LossChart",
    "RegressionBadge",
    "WorkerGrid",
    "PowerGauge",
    "ComparisonTable",
    "DevicePanel",
    "ResultBrowser",
    "RunHistoryLoader",
    "TrendChart",
]
