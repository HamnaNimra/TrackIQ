"""Reporting helpers shared across TrackIQ tools."""

from .canonical import render_trackiq_result_html
from .pdf import (
    PDF_BACKEND_AUTO,
    PDF_BACKEND_MATPLOTLIB,
    PDF_BACKEND_WEASYPRINT,
    PDF_BACKENDS,
    PdfBackendError,
    PdfRenderOutcome,
    render_pdf_from_html,
    render_pdf_from_html_file,
)

__all__ = [
    "render_trackiq_result_html",
    "PDF_BACKEND_AUTO",
    "PDF_BACKEND_MATPLOTLIB",
    "PDF_BACKEND_WEASYPRINT",
    "PDF_BACKENDS",
    "PdfBackendError",
    "PdfRenderOutcome",
    "render_pdf_from_html",
    "render_pdf_from_html_file",
]
