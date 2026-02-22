"""Shared Streamlit chrome styling for TrackIQ family apps."""

from __future__ import annotations

import streamlit as st


def inject_app_shell_style(
    *,
    theme: str,
    hero_class: str,
    hero_border: str,
    hero_gradient: str,
    st_module=st,
) -> None:
    """Inject shared UI shell styling with per-app hero customization."""
    prefers_dark = theme == "Dark"
    card_bg = "rgba(255,255,255,0.06)" if prefers_dark else "rgba(15,23,42,0.02)"
    card_border = "rgba(148,163,184,0.35)" if prefers_dark else "rgba(148,163,184,0.24)"
    hero_text = "#d1d5db" if prefers_dark else "#4b5563"
    sidebar_border = "rgba(148,163,184,0.28)" if prefers_dark else "rgba(148,163,184,0.22)"
    muted_text = "#9ca3af" if prefers_dark else "#64748b"
    tab_active = "#0f6feb"
    tab_text = "#d1d5db" if prefers_dark else "#334155"
    app_bg = "#0f172a" if prefers_dark else "#f8fafc"

    css = """
        <style>
        .stApp {
            font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
            background: %(app_bg)s;
        }
        .main .block-container {
            max-width: 1200px;
            padding-top: 1.2rem;
            padding-bottom: 2.4rem;
        }
        section[data-testid="stSidebar"] {
            border-right: 1px solid %(sidebar_border)s;
        }
        section[data-testid="stSidebar"] [data-testid="stMarkdown"] p {
            line-height: 1.35;
            color: %(muted_text)s;
        }
        .%(hero_class)s {
            border: 1px solid %(hero_border)s;
            background: %(hero_gradient)s;
            border-radius: 14px;
            padding: 14px 16px;
            margin-bottom: 14px;
        }
        .%(hero_class)s h2 {
            margin: 0 0 4px 0;
            font-size: 1.24rem;
        }
        .%(hero_class)s p {
            margin: 0;
            color: %(hero_text)s;
            font-size: 0.95rem;
        }
        [data-testid="stMetric"] {
            border: 1px solid %(card_border)s;
            border-radius: 12px;
            padding: 8px 10px;
            background: %(card_bg)s;
            box-shadow: 0 1px 6px rgba(15, 23, 42, 0.05);
        }
        [data-testid="stMetricLabel"] p {
            margin-bottom: 2px;
            font-size: 0.82rem;
            line-height: 1.15;
        }
        [data-testid="stMetricValue"] {
            font-size: clamp(1.35rem, 1.1vw + 0.9rem, 2.05rem);
            line-height: 1.1;
        }
        [data-testid="stMetricValue"] > div {
            white-space: normal !important;
            overflow-wrap: anywhere;
        }
        [data-testid="stExpander"] {
            border: 1px solid %(card_border)s;
            border-radius: 10px;
        }
        [data-baseweb="tab-list"] {
            gap: 6px;
            border-bottom: 1px solid %(card_border)s;
            margin-bottom: 4px;
        }
        [data-baseweb="tab-list"] button {
            color: %(tab_text)s;
            border-radius: 9px 9px 0 0;
            padding-inline: 12px;
        }
        [data-baseweb="tab-list"] button[aria-selected="true"] {
            color: %(tab_active)s !important;
            box-shadow: inset 0 -2px 0 %(tab_active)s;
            font-weight: 700;
        }
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            border-radius: 10px;
            border: 1px solid %(card_border)s;
            overflow: hidden;
        }
        button[kind="primary"] {
            border-radius: 10px !important;
        }
        </style>
    """ % {
        "app_bg": app_bg,
        "sidebar_border": sidebar_border,
        "muted_text": muted_text,
        "hero_class": hero_class,
        "hero_border": hero_border,
        "hero_gradient": hero_gradient,
        "hero_text": hero_text,
        "card_border": card_border,
        "card_bg": card_bg,
        "tab_active": tab_active,
        "tab_text": tab_text,
    }
    st_module.markdown(css, unsafe_allow_html=True)
