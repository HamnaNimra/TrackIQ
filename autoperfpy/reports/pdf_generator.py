"""PDF report generation for AutoPerfPy.

This module delegates conversion to trackiq_core's standardized PDF backend:
1. WeasyPrint primary backend
2. matplotlib text fallback
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

from trackiq_core.reporting import (
    PDF_BACKEND_AUTO,
    PDF_BACKENDS,
    PdfRenderOutcome,
    render_pdf_from_html_file,
)


class PDFReportGenerator:
    """Generate PDF reports by rendering HTML then converting with shared backend."""

    def __init__(
        self,
        title: str = "Performance Analysis Report",
        author: str = "AutoPerfPy",
        theme: str = "light",
        pdf_backend: str = PDF_BACKEND_AUTO,
    ):
        self.title = title
        self.author = author
        self.theme = theme
        self.pdf_backend = str(pdf_backend or PDF_BACKEND_AUTO).strip().lower()
        self.metadata: dict[str, Any] = {}
        self._html_generator = None
        self.last_render_outcome: PdfRenderOutcome | None = None

    def _get_html_generator(self):
        if self._html_generator is None:
            from .html_generator import HTMLReportGenerator

            self._html_generator = HTMLReportGenerator(
                title=self.title,
                author=self.author,
                theme=self.theme,
            )
            for key, value in self.metadata.items():
                self._html_generator.add_metadata(key, value)
        return self._html_generator

    def add_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value
        if self._html_generator:
            self._html_generator.add_metadata(key, value)

    def add_summary_item(
        self,
        label: str,
        value: Any,
        unit: str = "",
        status: str = "neutral",
    ) -> None:
        self._get_html_generator().add_summary_item(label, value, unit, status)

    def add_section(self, name: str, description: str = "") -> None:
        self._get_html_generator().add_section(name, description)

    def add_html_figure(
        self,
        html_content: str,
        caption: str = "",
        section: str = "General",
        description: str = "",
    ) -> None:
        self._get_html_generator().add_html_figure(html_content, caption, section, description)

    def add_table(
        self,
        title: str,
        headers: list[str],
        rows: list[list[Any]],
        section: str = "General",
    ) -> None:
        self._get_html_generator().add_table(title, headers, rows, section)

    def add_charts_from_data(
        self,
        samples: list[dict],
        summary: dict[str, Any],
    ) -> None:
        try:
            from .charts import (
                add_charts_to_html_report,
                ensure_throughput_column,
                samples_to_dataframe,
            )

            df = samples_to_dataframe(samples)
            ensure_throughput_column(df)
            add_charts_to_html_report(self._get_html_generator(), df, summary)
        except ImportError:
            pass

    def generate_pdf(
        self,
        output_path: str,
        include_summary: bool = True,
        backend: str | None = None,
    ) -> str:
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )
        selected_backend = str(backend or self.pdf_backend or PDF_BACKEND_AUTO).lower()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as handle:
            html_path = handle.name
        try:
            self._get_html_generator().generate_html(
                html_path,
                include_summary=include_summary,
            )
            outcome = render_pdf_from_html_file(
                html_path=html_path,
                output_path=output_path,
                backend=selected_backend,
                title=self.title,
                author=self.author,
            )
            self.last_render_outcome = outcome
            return outcome.output_path
        finally:
            try:
                os.unlink(html_path)
            except OSError:
                pass

    def generate_html(
        self,
        output_path: str,
        include_summary: bool = True,
    ) -> str:
        return self._get_html_generator().generate_html(output_path, include_summary=include_summary)

    def clear(self) -> None:
        self.metadata = {}
        self.last_render_outcome = None
        if self._html_generator:
            self._html_generator.clear()
            self._html_generator = None


__all__ = ["PDFReportGenerator", "PDF_BACKENDS", "PDF_BACKEND_AUTO"]
