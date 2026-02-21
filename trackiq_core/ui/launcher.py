"""Dashboard launcher helpers."""

from typing import Optional, Type

from trackiq_core.schema import TrackiqResult
from trackiq_core.serializer import load_trackiq_result
from trackiq_core.ui.dashboard import TrackiqDashboard
from trackiq_core.ui.theme import DARK_THEME, TrackiqTheme


def run_dashboard(
    dashboard_class: Type[TrackiqDashboard],
    result_path: Optional[str] = None,
    result: Optional[TrackiqResult] = None,
    theme: TrackiqTheme = DARK_THEME,
) -> None:
    """Load result data, instantiate dashboard, and run it."""
    if result_path is not None:
        loaded = load_trackiq_result(result_path)
    elif result is not None:
        loaded = result
    else:
        raise ValueError("Either result_path or result must be provided.")

    dashboard = dashboard_class(result=loaded, theme=theme)
    dashboard.run()

