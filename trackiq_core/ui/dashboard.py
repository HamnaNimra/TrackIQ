"""Base Streamlit dashboard class for TrackIQ tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
import json
from typing import List, Union

from trackiq_core.hardware.devices import get_all_devices
from trackiq_core.schema import TrackiqResult
from trackiq_core.ui.components import DevicePanel, ResultBrowser
from trackiq_core.ui.theme import DARK_THEME, LIGHT_THEME, TrackiqTheme


class TrackiqDashboard(ABC):
    """Abstract dashboard scaffold with common TrackIQ UI primitives."""

    def __init__(
        self,
        result: Union[TrackiqResult, List[TrackiqResult]],
        theme: TrackiqTheme = DARK_THEME,
        title: str = "TrackIQ Dashboard",
    ) -> None:
        self.result = result
        self.theme = theme
        self.title = title

    def _primary_result(self) -> TrackiqResult:
        if isinstance(self.result, list):
            return self.result[0]
        return self.result

    def configure_page(self) -> None:
        """Configure Streamlit page metadata."""
        import streamlit as st

        st.set_page_config(
            page_title=self.title,
            layout="wide",
            initial_sidebar_state="expanded",
        )

    def render_header(self) -> None:
        """Render title and theme toggle."""
        import streamlit as st

        if "theme" not in st.session_state:
            st.session_state["theme"] = self.theme.name if self.theme else DARK_THEME.name
        st.session_state["clock"] = datetime.now().strftime("%H:%M:%S")

        col_title, col_toggle = st.columns([6, 1])
        result = self._primary_result()
        live_badge = ""
        if st.session_state.get("trackiq_live_indicator"):
            live_badge = (
                f"<span style='margin-left:10px;padding:3px 8px;border-radius:999px;"
                f"background:{self.theme.pass_color};color:#fff;font-size:11px;font-weight:700;'>LIVE</span>"
            )
        with col_title:
            st.markdown(
                f"""
                <div class="trackiq-header-bar">
                    <div class="trackiq-header-title">{self.title}{live_badge}</div>
                    <div class="trackiq-header-subtitle">{result.tool_name} {result.tool_version}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_toggle:
            is_dark = st.session_state["theme"] == DARK_THEME.name
            if st.button("◐", use_container_width=True, key="trackiq_theme_toggle"):
                st.session_state["theme"] = (
                    LIGHT_THEME.name if is_dark else DARK_THEME.name
                )
            st.caption(st.session_state.get("clock", ""))

    def render_result_browser(self) -> None:
        """Render sidebar result browser expander."""
        import streamlit as st

        with st.expander("Load Result", expanded=False):
            ResultBrowser(theme=self.theme).render()

    def _build_html_report(self, result: TrackiqResult) -> str:
        """Build a lightweight self-contained HTML report for a single result."""
        metric_rows = "".join(
            f"<tr><td>{name}</td><td>{value}</td></tr>"
            for name, value in result.metrics.__dict__.items()
        )
        return (
            "<!doctype html><html><head><meta charset='utf-8'/>"
            f"<title>{result.tool_name} Report</title>"
            "<style>body{font-family:Arial,sans-serif;margin:20px;}table{border-collapse:collapse;width:100%;}"
            "th,td{border:1px solid #ddd;padding:8px;}th{background:#f5f5f5;}</style></head><body>"
            f"<h1>{result.tool_name} Report</h1>"
            f"<p><b>Tool Version:</b> {result.tool_version}<br/>"
            f"<b>Timestamp:</b> {result.timestamp.isoformat()}<br/>"
            f"<b>Hardware:</b> {result.platform.hardware_name}<br/>"
            f"<b>Workload:</b> {result.workload.name} ({result.workload.workload_type})</p>"
            "<h2>Metrics</h2><table><thead><tr><th>Metric</th><th>Value</th></tr></thead>"
            f"<tbody>{metric_rows}</tbody></table></body></html>"
        )

    def render_download_section(self) -> None:
        """Render common JSON/HTML download actions for the current result."""
        import streamlit as st

        result = self._primary_result()
        timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
        base_name = f"{result.tool_name}_{timestamp}"
        json_data = json.dumps(result.to_dict(), indent=2)
        html_data = self._build_html_report(result)

        st.markdown("---")
        st.markdown(
            f"<div style='font-weight:700;color:{self.theme.text_color};'>Downloads</div>",
            unsafe_allow_html=True,
        )
        col_json, col_html = st.columns(2)
        with col_json:
            st.download_button(
                "Download JSON",
                data=json_data,
                file_name=f"{base_name}.json",
                mime="application/json",
                key=f"download_json_{base_name}",
            )
        with col_html:
            st.download_button(
                "Download HTML Report",
                data=html_data,
                file_name=f"{base_name}.html",
                mime="text/html",
                key=f"download_html_{base_name}",
            )

    def render_device_panel(self) -> None:
        """Render cached device panel in the sidebar."""
        import streamlit as st

        if "devices" not in st.session_state:
            try:
                st.session_state["devices"] = get_all_devices()
            except Exception:
                st.session_state["devices"] = None

        devices = st.session_state.get("devices")
        if devices is None:
            st.info("Device detection unavailable")
            return
        DevicePanel(devices=devices, theme=self.theme).render()

    def render_sidebar(self) -> None:
        """Render common metadata in sidebar."""
        import streamlit as st

        loaded_result = st.session_state.get("loaded_result")
        if loaded_result is not None and not isinstance(self.result, list):
            self.result = loaded_result

        result = self._primary_result()
        with st.sidebar:
            self.render_result_browser()
            st.subheader("Run Metadata")
            st.markdown(f"**Tool:** {result.tool_name}")
            st.markdown(f"**Version:** {result.tool_version}")
            st.markdown(f"**Timestamp:** {result.timestamp.isoformat()}")
            st.markdown(f"**Hardware:** {result.platform.hardware_name}")
            st.markdown(f"**Workload:** {result.workload.name}")
            st.markdown(f"**Type:** {result.workload.workload_type}")
            self.render_device_panel()

    def render_footer(self) -> None:
        """Render a consistent footer."""
        import streamlit as st

        try:
            pkg_version = version("trackiq-core")
        except PackageNotFoundError:
            pkg_version = "dev"
        st.markdown("---")
        st.markdown(
            f"TrackIQ Core `{pkg_version}` | [Repository](https://github.com/trackiq/trackiq)"
        )

    def apply_theme(self, theme: TrackiqTheme) -> None:
        """Apply a CSS theme to the current Streamlit app."""
        import streamlit as st

        css = f"""
        <style>
        html, body, [class*="css"], .stApp {{
            font-family: {theme.font_family};
        }}
        .stApp {{
            background-color: {theme.background_color};
            color: {theme.text_color};
        }}
        ::-webkit-scrollbar {{
            width: 10px;
        }}
        ::-webkit-scrollbar-thumb {{
            background: {theme.accent_color};
            border-radius: 8px;
        }}
        ::-webkit-scrollbar-track {{
            background: {theme.surface_color};
        }}
        section[data-testid="stSidebar"] {{
            background-color: {theme.surface_color};
        }}
        [data-testid="stMetric"], [data-testid="stMetricValue"] {{
            color: {theme.text_color} !important;
        }}
        [data-testid="stMetric"] {{
            background: {theme.surface_color};
            border-radius: {theme.border_radius};
            box-shadow: {theme.card_shadow};
            padding: 10px;
        }}
        [data-testid="stDataFrame"], [data-testid="stTable"] {{
            background: {theme.surface_color};
            border-radius: {theme.border_radius};
        }}
        .stButton>button {{
            background: {theme.accent_color};
            color: #ffffff;
            border: 0;
            border-radius: {theme.border_radius};
        }}
        .stSuccess {{
            background-color: {theme.pass_color}22 !important;
            border-left: 4px solid {theme.pass_color};
        }}
        .stError {{
            background-color: {theme.fail_color}22 !important;
            border-left: 4px solid {theme.fail_color};
        }}
        .stWarning {{
            background-color: {theme.warning_color}22 !important;
            border-left: 4px solid {theme.warning_color};
        }}
        .trackiq-header-bar {{
            background: {theme.header_gradient};
            border-radius: {theme.border_radius};
            padding: 16px;
            box-shadow: {theme.card_shadow};
            margin-bottom: 10px;
        }}
        .trackiq-header-title {{
            color: #ffffff;
            font-size: 26px;
            font-weight: 700;
            margin: 0;
        }}
        .trackiq-header-subtitle {{
            color: #ffffff;
            font-size: 14px;
            opacity: 0.95;
            margin-top: 4px;
        }}
        .trackiq-card {{
            background-color: {theme.surface_color};
            border: 1px solid {theme.accent_color};
            border-radius: {theme.border_radius};
            box-shadow: {theme.card_shadow};
            padding: 12px;
            margin-bottom: 10px;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

    @abstractmethod
    def render_body(self) -> None:
        """Render tool-specific dashboard content."""

    def run(self) -> None:
        """Run full dashboard lifecycle."""
        import streamlit as st

        self.configure_page()
        active_theme = (
            LIGHT_THEME
            if st.session_state.get("theme", self.theme.name) == LIGHT_THEME.name
            else DARK_THEME
        )
        self.apply_theme(active_theme)
        self.render_header()
        self.render_sidebar()
        self.render_body()
        self.render_footer()
