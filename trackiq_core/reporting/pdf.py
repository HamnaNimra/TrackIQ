"""Standard PDF rendering backend for TrackIQ tools.

Primary backend is WeasyPrint for consistent HTML-to-PDF rendering.
Fallback backend is matplotlib text rendering for environments without
system HTML/PDF dependencies.
"""

from __future__ import annotations

import os
import re
import tempfile
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


PDF_BACKEND_AUTO = "auto"
PDF_BACKEND_WEASYPRINT = "weasyprint"
PDF_BACKEND_MATPLOTLIB = "matplotlib"
PDF_BACKENDS = (
    PDF_BACKEND_AUTO,
    PDF_BACKEND_WEASYPRINT,
    PDF_BACKEND_MATPLOTLIB,
)


@dataclass(frozen=True)
class PdfRenderOutcome:
    """Outcome for a PDF render attempt."""

    output_path: str
    backend_used: str
    used_fallback: bool


class PdfBackendError(RuntimeError):
    """Raised when PDF generation cannot be completed."""


def render_pdf_from_html(
    html_content: str,
    output_path: str,
    backend: str = PDF_BACKEND_AUTO,
    title: str = "TrackIQ Report",
    author: str = "TrackIQ",
    fallback_text: Optional[str] = None,
) -> PdfRenderOutcome:
    """Render PDF from HTML content using standardized backend strategy."""
    with tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, mode="w", encoding="utf-8"
    ) as handle:
        html_path = handle.name
        handle.write(html_content)
    try:
        return render_pdf_from_html_file(
            html_path=html_path,
            output_path=output_path,
            backend=backend,
            title=title,
            author=author,
            fallback_text=fallback_text,
        )
    finally:
        try:
            os.unlink(html_path)
        except OSError:
            pass


def render_pdf_from_html_file(
    html_path: str,
    output_path: str,
    backend: str = PDF_BACKEND_AUTO,
    title: str = "TrackIQ Report",
    author: str = "TrackIQ",
    fallback_text: Optional[str] = None,
) -> PdfRenderOutcome:
    """Render PDF from an HTML file path.

    Strategy (deterministic):
    1. WeasyPrint (primary)
    2. matplotlib text fallback
    """
    backend_name = _normalize_backend(backend)
    _ensure_parent_dir(output_path)
    weasy_error: Optional[Exception] = None
    mpl_error: Optional[Exception] = None

    if backend_name in (PDF_BACKEND_AUTO, PDF_BACKEND_WEASYPRINT):
        try:
            _render_with_weasyprint(html_path, output_path)
            return PdfRenderOutcome(
                output_path=str(output_path),
                backend_used=PDF_BACKEND_WEASYPRINT,
                used_fallback=False,
            )
        except Exception as exc:  # pragma: no cover - backend availability varies by env
            weasy_error = exc
            if backend_name == PDF_BACKEND_WEASYPRINT:
                raise PdfBackendError(_weasyprint_dependency_message(exc)) from exc

    if backend_name in (PDF_BACKEND_AUTO, PDF_BACKEND_MATPLOTLIB):
        text_payload = fallback_text
        if not text_payload:
            text_payload = _extract_text_from_html(Path(html_path).read_text(encoding="utf-8"))
        try:
            _render_with_matplotlib_fallback(
                output_path=output_path,
                title=title,
                author=author,
                body=text_payload,
            )
            return PdfRenderOutcome(
                output_path=str(output_path),
                backend_used=PDF_BACKEND_MATPLOTLIB,
                used_fallback=True,
            )
        except Exception as exc:  # pragma: no cover - backend availability varies by env
            mpl_error = exc
            if backend_name == PDF_BACKEND_MATPLOTLIB:
                raise PdfBackendError(_matplotlib_dependency_message(exc)) from exc

    msg = (
        "Unable to generate PDF with standardized backend chain "
        "(weasyprint -> matplotlib fallback). "
        f"WeasyPrint error: {weasy_error!s}. Matplotlib error: {mpl_error!s}"
    )
    raise PdfBackendError(msg)


def _normalize_backend(backend: str) -> str:
    value = str(backend or "").strip().lower()
    if value not in PDF_BACKENDS:
        allowed = ", ".join(PDF_BACKENDS)
        raise ValueError(f"Unsupported PDF backend '{backend}'. Supported: {allowed}")
    return value


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _render_with_weasyprint(html_path: str, output_path: str) -> None:
    from weasyprint import HTML as WeasyHTML

    WeasyHTML(filename=html_path).write_pdf(output_path)


def _render_with_matplotlib_fallback(
    output_path: str,
    title: str,
    author: str,
    body: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    lines = _wrap_lines(body, width=100)
    if not lines:
        lines = ["No report body content available."]

    lines_per_page = 44
    created_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")

    with PdfPages(output_path) as pdf:
        for start in range(0, len(lines), lines_per_page):
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.text(
                0.5,
                0.96,
                title,
                ha="center",
                va="top",
                fontsize=16,
                fontweight="bold",
                transform=ax.transAxes,
            )
            ax.text(
                0.5,
                0.93,
                f"Generated {created_at}",
                ha="center",
                va="top",
                fontsize=8,
                color="gray",
                transform=ax.transAxes,
            )
            y = 0.89
            for line in lines[start : start + lines_per_page]:
                ax.text(0.04, y, line, ha="left", va="top", fontsize=9, transform=ax.transAxes)
                y -= 0.02
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        info = pdf.infodict()
        info["Title"] = title
        info["Author"] = author
        info["Subject"] = "TrackIQ PDF Report"
        info["CreationDate"] = datetime.utcnow()


def _extract_text_from_html(html: str) -> str:
    no_script = re.sub(
        r"<(script|style)\b[^>]*>.*?</\1>",
        " ",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"<[^>]+>", " ", no_script)
    return re.sub(r"\s+", " ", text).strip()


def _wrap_lines(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for block in text.split("\n"):
        stripped = block.strip()
        if not stripped:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(stripped, width=width) or [""])
    return lines


def _weasyprint_dependency_message(exc: Exception) -> str:
    return (
        "PDF backend 'weasyprint' is not available or missing system libraries. "
        "Install Python package with `pip install weasyprint`, then install required "
        "OS dependencies (cairo/pango/gdk-pixbuf on Linux). "
        f"Original error: {exc!s}"
    )


def _matplotlib_dependency_message(exc: Exception) -> str:
    return (
        "PDF fallback backend 'matplotlib' is unavailable. "
        "Install with `pip install matplotlib`. "
        f"Original error: {exc!s}"
    )


__all__ = [
    "PDF_BACKEND_AUTO",
    "PDF_BACKEND_WEASYPRINT",
    "PDF_BACKEND_MATPLOTLIB",
    "PDF_BACKENDS",
    "PdfRenderOutcome",
    "PdfBackendError",
    "render_pdf_from_html",
    "render_pdf_from_html_file",
]
