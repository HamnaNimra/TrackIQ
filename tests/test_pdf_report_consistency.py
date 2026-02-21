"""PDF reporting consistency and smoke tests across TrackIQ tools."""

from __future__ import annotations

import argparse
from pathlib import Path

import autoperfpy.cli as autoperf_cli
import pytest
import trackiq_core.reporting.pdf as pdf_backend
from minicluster.cli import cmd_report_pdf as minicluster_report_pdf
from trackiq_compare.cli import main as trackiq_compare_main


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "tool_outputs"
AUTOPERFPY_FIXTURE = FIXTURE_DIR / "autoperfpy_real_output.json"
MINICLUSTER_FIXTURE = FIXTURE_DIR / "minicluster_real_output.json"


def _assert_pdf(path: Path) -> None:
    assert path.exists()
    assert path.stat().st_size > 0
    assert path.read_bytes().startswith(b"%PDF")


def test_standard_pdf_backend_auto_falls_back_to_matplotlib(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Auto backend should deterministically fall back to matplotlib."""

    def _fail_weasyprint(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("weasyprint missing system libs")

    monkeypatch.setattr(pdf_backend, "_render_with_weasyprint", _fail_weasyprint)

    output = tmp_path / "fallback.pdf"
    outcome = pdf_backend.render_pdf_from_html(
        html_content="<html><body><h1>Fallback</h1></body></html>",
        output_path=str(output),
        backend=pdf_backend.PDF_BACKEND_AUTO,
        title="Fallback Test",
    )
    assert outcome.backend_used == pdf_backend.PDF_BACKEND_MATPLOTLIB
    assert outcome.used_fallback is True
    _assert_pdf(output)


def test_weasyprint_backend_error_message_is_clear(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Explicit weasyprint mode should raise clear dependency errors."""

    def _fail_weasyprint(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("cannot load libpango")

    monkeypatch.setattr(pdf_backend, "_render_with_weasyprint", _fail_weasyprint)
    with pytest.raises(pdf_backend.PdfBackendError, match="weasyprint"):
        pdf_backend.render_pdf_from_html(
            html_content="<html><body>x</body></html>",
            output_path=str(tmp_path / "weasyprint.pdf"),
            backend=pdf_backend.PDF_BACKEND_WEASYPRINT,
            title="Failure Test",
        )


def test_autoperfpy_report_pdf_from_canonical_fixture(tmp_path: Path) -> None:
    """autoperfpy report pdf should generate from canonical fixture input."""
    args = argparse.Namespace(
        csv=None,
        json=str(AUTOPERFPY_FIXTURE),
        device=None,
        duration=10,
        output="autoperfpy_fixture_report.pdf",
        export_json=None,
        export_csv=None,
        title="AutoPerfPy Fixture Report",
        author="tests",
        output_dir=str(tmp_path),
        pdf_backend=pdf_backend.PDF_BACKEND_MATPLOTLIB,
    )
    result = autoperf_cli.run_report_pdf(args, config=None)
    assert result is not None
    _assert_pdf(Path(result["output_path"]))


def test_minicluster_report_pdf_from_canonical_fixture(tmp_path: Path) -> None:
    """minicluster report pdf should generate from canonical fixture input."""
    output = tmp_path / "minicluster_fixture_report.pdf"
    args = argparse.Namespace(
        result=str(MINICLUSTER_FIXTURE),
        output=str(output),
        title="MiniCluster Fixture Report",
        pdf_backend=pdf_backend.PDF_BACKEND_MATPLOTLIB,
    )
    minicluster_report_pdf(args)
    _assert_pdf(output)


def test_trackiq_compare_report_pdf_from_canonical_fixtures(tmp_path: Path) -> None:
    """trackiq-compare report pdf should generate from two canonical fixtures."""
    output = tmp_path / "trackiq_compare_fixture_report.pdf"
    rc = trackiq_compare_main(
        [
            "report",
            "pdf",
            str(AUTOPERFPY_FIXTURE),
            str(MINICLUSTER_FIXTURE),
            "--output",
            str(output),
            "--label-a",
            "autoperfpy-fixture",
            "--label-b",
            "minicluster-fixture",
            "--pdf-backend",
            pdf_backend.PDF_BACKEND_MATPLOTLIB,
        ]
    )
    assert rc == 0
    _assert_pdf(output)
