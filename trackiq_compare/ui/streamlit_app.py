"""Interactive Streamlit app for trackiq-compare."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from trackiq_compare.comparator import MetricComparator, SummaryGenerator
from trackiq_compare.ui.dashboard import CompareDashboard
from trackiq_core.serializer import load_trackiq_result
from trackiq_core.ui import ResultBrowser


def _try_load(path: str):
    """Load a TrackiqResult from path, returning None on failure."""
    try:
        return load_trackiq_result(path)
    except Exception as exc:  # pragma: no cover - UI feedback path
        st.error(f"Failed to load result file '{path}': {exc}")
        return None


def _discover_result_rows() -> list[dict]:
    """Return discovered TrackiqResult metadata rows from common output folders."""
    try:
        return ResultBrowser().to_dict()
    except Exception:
        return []


def main() -> None:
    """Render interactive compare app with file selectors and labels."""
    st.set_page_config(
        page_title="TrackIQ Compare Interactive Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("TrackIQ Compare Interactive Dashboard")
    st.caption("Configure comparison inputs and thresholds, then inspect graphs and winner logic.")

    with st.sidebar:
        st.subheader("Compare Inputs")
        input_mode = st.radio(
            "Input Mode",
            options=["Browse results", "Manual paths"],
            index=0,
            key="compare_input_mode",
        )
        rows = _discover_result_rows()
        result_a_path = ""
        result_b_path = ""
        if input_mode == "Browse results":
            if not rows:
                st.info("No results discovered. Switch to manual paths or generate runs first.")
            labels = [
                f"{idx + 1}. {row.get('tool_name', '?')} | {row.get('workload_name', '?')} | {row.get('timestamp', '?')}"
                for idx, row in enumerate(rows)
            ]
            if labels:
                idx_a = st.selectbox("Result A", options=list(range(len(labels))), format_func=lambda i: labels[i])
                remaining = [i for i in range(len(labels)) if i != idx_a] or [idx_a]
                idx_b = st.selectbox("Result B", options=remaining, format_func=lambda i: labels[i])
                result_a_path = str(rows[idx_a]["path"])
                result_b_path = str(rows[idx_b]["path"])
        else:
            default_a = "output/autoperf_power.json"
            default_b = "minicluster_power.json"
            result_a_path = st.text_input("Result A Path", value=default_a)
            result_b_path = st.text_input("Result B Path", value=default_b)

        label_a = st.text_input("Label A", value="Result A")
        label_b = st.text_input("Label B", value="Result B")
        regression_threshold = st.slider(
            "Regression Threshold (%)",
            min_value=0.5,
            max_value=50.0,
            value=5.0,
            step=0.5,
        )
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
                st.session_state["compare_regression_threshold"] = regression_threshold

    a = st.session_state.get("compare_result_a")
    b = st.session_state.get("compare_result_b")
    if a is None or b is None:
        st.info("Set inputs in the sidebar and click 'Load Comparison'.")
        return

    comp = MetricComparator(
        label_a=st.session_state.get("compare_label_a", label_a),
        label_b=st.session_state.get("compare_label_b", label_b),
    ).compare(a, b)
    summary = SummaryGenerator(
        regression_threshold_percent=float(st.session_state.get("compare_regression_threshold", regression_threshold))
    ).generate(comp)
    top = summary.largest_deltas[:3]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Winner", summary.overall_winner)
    with col2:
        st.metric("Comparable Metrics", len(comp.comparable_metrics))
    with col3:
        st.metric("Flagged Regressions", len(summary.flagged_regressions))
    if top:
        st.caption(
            "Top deltas: "
            + " | ".join(
                f"{item.metric_name} ({item.percent_delta:+.2f}%)" for item in top if item.percent_delta is not None
            )
        )

    dash = CompareDashboard(
        result_a=a,
        result_b=b,
        label_a=st.session_state.get("compare_label_a"),
        label_b=st.session_state.get("compare_label_b"),
        regression_threshold_percent=float(st.session_state.get("compare_regression_threshold", regression_threshold)),
    )
    dash.apply_theme(dash.theme)
    dash.render_sidebar()
    dash.render_body()
    dash.render_footer()


if __name__ == "__main__":
    main()
