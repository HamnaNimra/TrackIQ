"""Theme primitives for TrackIQ dashboards."""

from dataclasses import dataclass


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
    chart_colors: list[str]
    font_family: str
    border_radius: str
    card_shadow: str
    header_gradient: str


DARK_THEME = TrackiqTheme(
    name="dark",
    background_color="#0f172a",
    surface_color="#14213d",
    text_color="#e5e7eb",
    accent_color="#2563eb",
    pass_color="#16a34a",
    fail_color="#dc2626",
    warning_color="#d97706",
    chart_colors=["#2563eb", "#0ea5a4", "#f59e0b", "#a855f7", "#ef4444"],
    font_family='"Segoe UI", "Helvetica Neue", Arial, sans-serif',
    border_radius="10px",
    card_shadow="0 10px 24px rgba(2, 6, 23, 0.35)",
    header_gradient="linear-gradient(120deg, #0f6feb 0%, #0ea5a4 100%)",
)

LIGHT_THEME = TrackiqTheme(
    name="light",
    background_color="#f8fafc",
    surface_color="#ffffff",
    text_color="#1f2937",
    accent_color="#0f6feb",
    pass_color="#16a34a",
    fail_color="#dc2626",
    warning_color="#d97706",
    chart_colors=["#0f6feb", "#0ea5a4", "#f59e0b", "#7c3aed", "#e11d48"],
    font_family='"Segoe UI", "Helvetica Neue", Arial, sans-serif',
    border_radius="10px",
    card_shadow="0 8px 18px rgba(15, 23, 42, 0.10)",
    header_gradient="linear-gradient(120deg, #0f6feb 0%, #0ea5a4 100%)",
)
