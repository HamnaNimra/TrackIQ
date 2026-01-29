"""PDF report generation for performance analysis."""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from typing import List, Any, Optional
import os


class PDFReportGenerator:
    """Generate consolidated PDF reports from performance analysis."""

    def __init__(self, title: str = "Performance Analysis Report", author: str = "AutoPerfPy"):
        """Initialize PDF report generator.

        Args:
            title: Report title
            author: Report author name
        """
        self.title = title
        self.author = author
        self.figures = []
        self.metadata = {}

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to report.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value

    def add_figure(self, fig: plt.Figure, caption: str = "") -> None:
        """Add a figure to the report.

        Args:
            fig: Matplotlib figure to add
            caption: Optional caption for the figure
        """
        self.figures.append({"figure": fig, "caption": caption})

    def add_figures_from_visualizer(self, visualizer, captions: Optional[List[str]] = None) -> None:
        """Add all figures from a PerformanceVisualizer.

        Args:
            visualizer: PerformanceVisualizer instance
            captions: Optional list of captions for figures
        """
        for i, fig in enumerate(visualizer.figures):
            caption = captions[i] if captions and i < len(captions) else f"Graph {i+1}"
            self.add_figure(fig, caption)

    def _create_title_page(self) -> plt.Figure:
        """Create a title page for the report.

        Returns:
            Matplotlib figure with title page
        """
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis("off")

        # Title
        ax.text(0.5, 0.85, self.title, ha="center", va="top", fontsize=28, fontweight="bold", transform=ax.transAxes)

        # Subtitle
        ax.text(
            0.5,
            0.78,
            "Performance Analysis Report",
            ha="center",
            va="top",
            fontsize=16,
            color="gray",
            transform=ax.transAxes,
        )

        # Metadata
        y_pos = 0.65
        ax.text(
            0.5, y_pos, "Report Details", ha="center", va="top", fontsize=14, fontweight="bold", transform=ax.transAxes
        )

        y_pos -= 0.08
        for key, value in self.metadata.items():
            ax.text(
                0.25, y_pos, f"{key}:", ha="right", va="top", fontsize=11, fontweight="bold", transform=ax.transAxes
            )
            ax.text(0.27, y_pos, str(value), ha="left", va="top", fontsize=11, transform=ax.transAxes)
            y_pos -= 0.05

        # Footer
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ax.text(
            0.5,
            0.05,
            f"Generated on {timestamp} by {self.author}",
            ha="center",
            va="bottom",
            fontsize=9,
            style="italic",
            color="gray",
            transform=ax.transAxes,
        )

        return fig

    def _create_summary_page(self) -> plt.Figure:
        """Create a summary page.

        Returns:
            Matplotlib figure with summary
        """
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis("off")

        ax.text(
            0.5, 0.95, "Report Summary", ha="center", va="top", fontsize=16, fontweight="bold", transform=ax.transAxes
        )

        y_pos = 0.85
        ax.text(
            0.05, y_pos, f"Total Graphs: {len(self.figures)}", ha="left", va="top", fontsize=12, transform=ax.transAxes
        )

        y_pos -= 0.08
        ax.text(0.05, y_pos, "Contents:", ha="left", va="top", fontsize=12, fontweight="bold", transform=ax.transAxes)

        y_pos -= 0.06
        for i, fig_data in enumerate(self.figures, 1):
            caption = fig_data.get("caption", f"Graph {i}")
            ax.text(0.1, y_pos, f"{i}. {caption}", ha="left", va="top", fontsize=10, transform=ax.transAxes)
            y_pos -= 0.05

        return fig

    def generate_pdf(self, output_path: str, include_summary: bool = True) -> str:
        """Generate consolidated PDF report.

        Args:
            output_path: Path to save PDF file
            include_summary: Whether to include a summary page

        Returns:
            Path to generated PDF
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        with PdfPages(output_path) as pdf:
            # Title page
            title_fig = self._create_title_page()
            pdf.savefig(title_fig, bbox_inches="tight")
            plt.close(title_fig)

            # Summary page
            if include_summary and self.figures:
                summary_fig = self._create_summary_page()
                pdf.savefig(summary_fig, bbox_inches="tight")
                plt.close(summary_fig)

            # Content pages
            for fig_data in self.figures:
                fig = fig_data["figure"]
                caption = fig_data.get("caption", "")

                # Add caption to figure if not already present
                if caption:
                    fig.text(0.5, 0.02, caption, ha="center", fontsize=10, style="italic", color="gray")

                pdf.savefig(fig, bbox_inches="tight")

            # Add metadata to PDF
            d = pdf.infodict()
            d["Title"] = self.title
            d["Author"] = self.author
            d["Subject"] = "Performance Analysis Report"
            d["Keywords"] = "Performance, Analysis, Benchmarking"
            d["CreationDate"] = datetime.now()

        return output_path

    def clear(self) -> None:
        """Clear all figures."""
        self.figures = []
        plt.close("all")


__all__ = ["PDFReportGenerator"]
