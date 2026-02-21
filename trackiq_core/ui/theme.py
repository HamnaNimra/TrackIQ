"""Theme primitives for TrackIQ dashboards."""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TrackiqTheme:
    """Visual theme used by TrackIQ dashboard components."""

    name: str
    background_color: str
    surface_color: str
    text_color: str
    accent_color: str
    pass_color: str
    fail_color: str
    warning_color: str
    chart_colors: List[str]


DARK_THEME = TrackiqTheme(
    name="dark",
    background_color="#0E1117",
    surface_color="#1A1F2B",
    text_color="#E6E9EF",
    accent_color="#E53935",
    pass_color="#2E7D32",
    fail_color="#C62828",
    warning_color="#F9A825",
    chart_colors=["#E53935", "#8D99AE", "#607D8B", "#90A4AE", "#B0BEC5"],
)

LIGHT_THEME = TrackiqTheme(
    name="light",
    background_color="#F7F9FC",
    surface_color="#FFFFFF",
    text_color="#1A2233",
    accent_color="#C62828",
    pass_color="#2E7D32",
    fail_color="#C62828",
    warning_color="#F9A825",
    chart_colors=["#C62828", "#5C6B7A", "#78909C", "#90A4AE", "#B0BEC5"],
)

