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
    font_family: str
    border_radius: str
    card_shadow: str
    header_gradient: str


DARK_THEME = TrackiqTheme(
    name="dark",
    background_color="#1a1a2e",
    surface_color="#16213e",
    text_color="#e8ecf1",
    accent_color="#E53935",
    pass_color="#43a047",
    fail_color="#e53935",
    warning_color="#ffb300",
    chart_colors=["#e53935", "#ef5350", "#7e57c2", "#5c6bc0", "#26c6da"],
    font_family="Inter, sans-serif",
    border_radius="8px",
    card_shadow="0 10px 24px rgba(0, 0, 0, 0.35)",
    header_gradient="linear-gradient(120deg, #e53935 0%, #4a148c 100%)",
)

LIGHT_THEME = TrackiqTheme(
    name="light",
    background_color="#ffffff",
    surface_color="#f5f5f5",
    text_color="#1f1f2d",
    accent_color="#e53935",
    pass_color="#43a047",
    fail_color="#e53935",
    warning_color="#ffb300",
    chart_colors=["#e53935", "#ef5350", "#5c6bc0", "#26a69a", "#546e7a"],
    font_family="Inter, sans-serif",
    border_radius="8px",
    card_shadow="0 8px 18px rgba(0, 0, 0, 0.12)",
    header_gradient="linear-gradient(120deg, #e53935 0%, #6a1b9a 100%)",
)
