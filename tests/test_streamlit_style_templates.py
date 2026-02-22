"""Regression tests for Streamlit CSS template rendering helpers."""

from __future__ import annotations

import autoperfpy.ui.streamlit_app as autoperf_streamlit
import minicluster.ui.streamlit_app as minicluster_streamlit
import trackiq_compare.ui.streamlit_app as compare_streamlit
import trackiq_core.ui.streamlit_app as core_streamlit


class _FakeStreamlit:
    """Minimal Streamlit shim for CSS injection tests."""

    css: str = ""

    @classmethod
    def markdown(cls, content, unsafe_allow_html=False):  # noqa: ANN001
        if unsafe_allow_html:
            cls.css = str(content)


def test_autoperfpy_apply_ui_style_renders_css(monkeypatch) -> None:
    """AutoPerfPy CSS helper should render without NameError placeholders."""
    monkeypatch.setattr(autoperf_streamlit, "st", _FakeStreamlit)
    autoperf_streamlit._apply_ui_style("Light")
    assert ".ap-hero" in _FakeStreamlit.css
    assert "font-family" in _FakeStreamlit.css


def test_minicluster_apply_ui_style_renders_css(monkeypatch) -> None:
    """MiniCluster CSS helper should render without NameError placeholders."""
    monkeypatch.setattr(minicluster_streamlit, "st", _FakeStreamlit)
    minicluster_streamlit._apply_ui_style("Light")
    assert ".mc-hero" in _FakeStreamlit.css
    assert "border-radius" in _FakeStreamlit.css


def test_trackiq_core_apply_ui_style_renders_css(monkeypatch) -> None:
    """TrackIQ Core CSS helper should render without NameError placeholders."""
    monkeypatch.setattr(core_streamlit, "st", _FakeStreamlit)
    core_streamlit._apply_ui_style("Dark")
    assert ".core-hero" in _FakeStreamlit.css
    assert "background" in _FakeStreamlit.css


def test_compare_apply_ui_style_renders_css(monkeypatch) -> None:
    """TrackIQ Compare CSS helper should render without NameError placeholders."""
    monkeypatch.setattr(compare_streamlit, "st", _FakeStreamlit)
    compare_streamlit._apply_ui_style("Dark")
    assert ".cmp-hero" in _FakeStreamlit.css
    assert 'button[kind="primary"]' in _FakeStreamlit.css
