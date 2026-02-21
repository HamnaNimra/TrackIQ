"""Interactive Streamlit app for trackiq-compare."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st

from trackiq_compare.ui.dashboard import CompareDashboard
from trackiq_core.serializer import load_trackiq_result


def _try_load(path: str):
    """Load a TrackiqResult from path, returning None on failure."""
    try:
        return load_trackiq_result(path)
    except Exception as exc:  # pragma: no cover - UI feedback path
        st.error(f"Failed to load result file '{path}': {exc}")
        return None


def main() -> None:
    """Render interactive compare app with file selectors and labels."""
    st.set_page_config(
        page_title="TrackIQ Compare Interactive Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("TrackIQ Compare Interactive Dashboard")
    st.caption("Load two canonical results and compare them with custom labels.")

    with st.sidebar:
        st.subheader("Compare Inputs")
        default_a = "output/autoperf_power.json"
        default_b = "minicluster_power.json"
        result_a_path = st.text_input("Result A Path", value=default_a)
        result_b_path = st.text_input("Result B Path", value=default_b)
        label_a = st.text_input("Label A", value="AMD MI300X")
        label_b = st.text_input("Label B", value="NVIDIA A100")
        load_clicked = st.button("Load Comparison", use_container_width=True)

    if load_clicked:
        if not Path(result_a_path).exists():
            st.error(f"Result A file not found: {result_a_path}")
        elif not Path(result_b_path).exists():
            st.error(f"Result B file not found: {result_b_path}")
        else:
            a = _try_load(result_a_path)
            b = _try_load(result_b_path)
            if a is not None and b is not None:
                st.session_state["compare_result_a"] = a
                st.session_state["compare_result_b"] = b
                st.session_state["compare_label_a"] = label_a
                st.session_state["compare_label_b"] = label_b

    a = st.session_state.get("compare_result_a")
    b = st.session_state.get("compare_result_b")
    if a is None or b is None:
        st.info("Set paths in the sidebar and click 'Load Comparison'.")
        return

    dash = CompareDashboard(
        result_a=a,
        result_b=b,
        label_a=st.session_state.get("compare_label_a"),
        label_b=st.session_state.get("compare_label_b"),
    )
    dash.apply_theme(dash.theme)
    dash.render_sidebar()
    dash.render_body()
    dash.render_footer()

