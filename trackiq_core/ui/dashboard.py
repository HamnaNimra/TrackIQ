"""Base Streamlit dashboard class for TrackIQ tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib.metadata import PackageNotFoundError, version
from typing import List, Union

from trackiq_core.schema import TrackiqResult
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

        col_title, col_toggle = st.columns([4, 1])
        with col_title:
            st.title(self.title)
        with col_toggle:
            is_dark = st.session_state["theme"] == DARK_THEME.name
            if st.button("Toggle Theme", use_container_width=True):
                st.session_state["theme"] = (
                    LIGHT_THEME.name if is_dark else DARK_THEME.name
                )

    def render_sidebar(self) -> None:
        """Render common metadata in sidebar."""
        import streamlit as st

        result = self._primary_result()
        with st.sidebar:
            st.subheader("Run Metadata")
            st.markdown(f"**Tool:** {result.tool_name}")
            st.markdown(f"**Version:** {result.tool_version}")
            st.markdown(f"**Timestamp:** {result.timestamp.isoformat()}")
            st.markdown(f"**Hardware:** {result.platform.hardware_name}")
            st.markdown(f"**Workload:** {result.workload.name}")
            st.markdown(f"**Type:** {result.workload.workload_type}")

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
        .stApp {{
            background-color: {theme.background_color};
            color: {theme.text_color};
        }}
        section[data-testid="stSidebar"] {{
            background-color: {theme.surface_color};
        }}
        .trackiq-card {{
            background-color: {theme.surface_color};
            border: 1px solid {theme.accent_color};
            border-radius: 10px;
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

