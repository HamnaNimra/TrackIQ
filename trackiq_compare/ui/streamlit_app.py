"""Interactive Streamlit app for trackiq-compare."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure local repo packages are imported when launched via `streamlit run ...`.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trackiq_compare.comparator import MetricComparator, SummaryGenerator
from trackiq_compare.ui.dashboard import CompareDashboard
from trackiq_core.serializer import load_trackiq_result
from trackiq_core.ui import DARK_THEME, LIGHT_THEME, ResultBrowser

UI_THEME_OPTIONS = ["System", "Light", "Dark"]


def _resolve_trackiq_theme(theme: str):
    """Map UI theme selection to TrackIQ dashboard theme object."""
    if theme == "Dark":
        return DARK_THEME
    return LIGHT_THEME


def _apply_ui_style(theme: str = "System") -> None:
    """Apply visual polish for compare app."""
    prefers_dark = theme == "Dark"
    hero_text = "#d1d5db" if prefers_dark else "#4b5563"
    metric_border = "rgba(148,163,184,0.30)" if prefers_dark else "rgba(148,163,184,0.22)"
    metric_bg = "rgba(255,255,255,0.06)" if prefers_dark else "rgba(15,23,42,0.02)"
    st.markdown(
        """
        <style>
        .cmp-hero {
            border: 1px solid rgba(16,185,129,0.24);
            background: linear-gradient(135deg, rgba(59,130,246,0.12), rgba(16,185,129,0.10));
            border-radius: 14px;
            padding: 14px 16px;
            margin-bottom: 14px;
        }
        .cmp-hero h2 {
            margin: 0 0 4px 0;
            font-size: 1.24rem;
        }
        .cmp-hero p {
            margin: 0;
            color: %(hero_text)s;
            font-size: 0.95rem;
        }
        [data-testid="stMetric"] {
            border: 1px solid %(metric_border)s;
            border-radius: 12px;
            padding: 8px 10px;
            background: %(metric_bg)s;
        }
        button[kind="primary"] {
            border-radius: 10px !important;
        }
        </style>
        """
        % {
            "hero_text": hero_text,
            "metric_border": metric_border,
            "metric_bg": metric_bg,
        },
        unsafe_allow_html=True,
    )


def _render_page_intro() -> None:
    """Render top-level usage guidance."""
    st.markdown(
        """
        <div class="cmp-hero">
          <h2>TrackIQ Compare Studio</h2>
          <p>Load two results, tune thresholds, and inspect metric deltas and consistency regressions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Quick Start", expanded=False):
        st.markdown(
            "1. Pick two result files from `Browse results` or `Manual paths`.\n"
            "2. Click `Load Comparison`.\n"
            "3. Review winner summary, comparison graphs, and `Consistency Analysis`."
        )


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
    selected_theme = st.session_state.get("trackiq_compare_ui_theme", "Light")
    if selected_theme not in UI_THEME_OPTIONS:
        selected_theme = "Light"
    _apply_ui_style(selected_theme)
    st.title("TrackIQ Compare Interactive Dashboard")
    _render_page_intro()

    with st.sidebar:
        st.subheader("Compare Inputs")
        st.caption("Choose two outputs and threshold policy for regression detection.")
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
            else:
                st.caption(f"Discovered results: {len(rows)}")
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
        variance_threshold = st.slider(
            "Variance Threshold (%)",
            min_value=1.0,
            max_value=200.0,
            value=25.0,
            step=1.0,
        )
        with st.expander("Threshold guidance", expanded=False):
            st.markdown(
                "- `Regression Threshold`: flags large performance swings.\n"
                "- `Variance Threshold`: catches all-reduce consistency degradation even when averages look stable."
            )
        st.markdown("---")
        st.subheader("View Options")
        st.selectbox(
            "Theme",
            options=UI_THEME_OPTIONS,
            index=UI_THEME_OPTIONS.index(selected_theme),
            key="trackiq_compare_ui_theme",
        )
        load_clicked = st.button("Load Comparison", use_container_width=True, type="primary")

    # Auto-load first two discovered results to avoid blank first render.
    if "compare_result_a" not in st.session_state and "compare_result_b" not in st.session_state and len(rows) >= 2:
        auto_a = _try_load(str(rows[0]["path"]))
        auto_b = _try_load(str(rows[1]["path"]))
        if auto_a is not None and auto_b is not None:
            st.session_state["compare_result_a"] = auto_a
            st.session_state["compare_result_b"] = auto_b
            st.session_state["compare_label_a"] = "Result A"
            st.session_state["compare_label_b"] = "Result B"
            st.session_state["compare_regression_threshold"] = regression_threshold
            st.session_state["compare_variance_threshold"] = variance_threshold

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
                st.session_state["compare_variance_threshold"] = variance_threshold

    a = st.session_state.get("compare_result_a")
    b = st.session_state.get("compare_result_b")
    if a is None or b is None:
        st.info("Set inputs in the sidebar and click 'Load Comparison'.")
        return

    comp = MetricComparator(
        label_a=st.session_state.get("compare_label_a", label_a),
        label_b=st.session_state.get("compare_label_b", label_b),
        variance_threshold_percent=float(st.session_state.get("compare_variance_threshold", variance_threshold)),
    ).compare(a, b)
    summary = SummaryGenerator(
        regression_threshold_percent=float(st.session_state.get("compare_regression_threshold", regression_threshold))
    ).generate(comp)
    top = summary.largest_deltas[:3]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Winner", summary.overall_winner)
    with col2:
        st.metric("Comparable Metrics", len(comp.comparable_metrics))
    with col3:
        st.metric("Flagged Regressions", len(summary.flagged_regressions))
    with col4:
        st.metric("Consistency Findings", len(comp.consistency_findings))
    if top:
        finite_top = [
            item
            for item in top
            if item.percent_delta is not None and item.percent_delta not in (float("inf"), float("-inf"))
        ]
        if finite_top:
            st.caption(
                "Top deltas: " + " | ".join(f"{item.metric_name} ({item.percent_delta:+.2f}%)" for item in finite_top)
            )

    selected_theme = st.session_state.get("trackiq_compare_ui_theme", selected_theme)
    active_theme = _resolve_trackiq_theme(str(selected_theme))
    st.session_state["theme"] = active_theme.name
    dash = CompareDashboard(
        result_a=a,
        result_b=b,
        label_a=st.session_state.get("compare_label_a"),
        label_b=st.session_state.get("compare_label_b"),
        theme=active_theme,
        regression_threshold_percent=float(st.session_state.get("compare_regression_threshold", regression_threshold)),
        variance_threshold_percent=float(st.session_state.get("compare_variance_threshold", variance_threshold)),
    )
    dash.apply_theme(dash.theme)
    dash.render_sidebar()
    dash.render_body()
    dash.render_footer()


if __name__ == "__main__":
    main()
