"""End-to-end CLI integration tests across TrackIQ tools."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from trackiq_core.serializer import load_trackiq_result
from trackiq_core.validator import validate_trackiq_result_obj

ROOT = Path(__file__).resolve().parents[1]
AUTOPERFPY_CLI = ROOT / "autoperfpy_cli"


def _run_command(command: list[str], cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    """Run subprocess command and fail with detailed output on non-zero status."""
    process = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, (
        f"Command failed ({process.returncode}): {' '.join(command)}\n"
        f"STDOUT:\n{process.stdout}\n"
        f"STDERR:\n{process.stderr}"
    )
    return process


@pytest.fixture(scope="module")
def e2e_cli_artifacts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Generate canonical tool outputs via real CLI flows for cross-tool E2E checks."""
    work_dir = tmp_path_factory.mktemp("e2e-cli")
    autoperf_run_json = work_dir / "autoperf_run.json"
    minicluster_run_json = work_dir / "minicluster_run.json"
    health_checkpoint = work_dir / "health.json"

    _run_command(
        [
            sys.executable,
            str(AUTOPERFPY_CLI),
            "--output",
            autoperf_run_json.name,
            "--output-dir",
            str(work_dir),
            "run",
            "--manual",
            "--device",
            "cpu_0",
            "--collector",
            "synthetic",
            "--duration",
            "1",
            "--quiet",
        ]
    )
    _run_command(
        [
            sys.executable,
            "-m",
            "minicluster",
            "run",
            "--workers",
            "1",
            "--steps",
            "8",
            "--output",
            str(minicluster_run_json),
            "--health-checkpoint-path",
            str(health_checkpoint),
        ]
    )

    return {
        "work_dir": work_dir,
        "autoperf_run_json": autoperf_run_json,
        "minicluster_run_json": minicluster_run_json,
        "health_checkpoint": health_checkpoint,
    }


@pytest.mark.integration
def test_e2e_cli_producers_emit_canonical_trackiq_results(e2e_cli_artifacts: dict[str, Path]) -> None:
    """AutoPerfPy + MiniCluster CLI runs should emit canonical TrackiqResult payloads."""
    autoperf_result = load_trackiq_result(e2e_cli_artifacts["autoperf_run_json"])
    validate_trackiq_result_obj(autoperf_result)
    assert autoperf_result.tool_name == "autoperfpy"
    assert autoperf_result.workload.workload_type == "inference"

    minicluster_result = load_trackiq_result(e2e_cli_artifacts["minicluster_run_json"])
    validate_trackiq_result_obj(minicluster_result)
    assert minicluster_result.tool_name == "minicluster"
    assert minicluster_result.workload.workload_type == "training"

    assert e2e_cli_artifacts["health_checkpoint"].exists()
    monitor_status = _run_command(
        [
            sys.executable,
            "-m",
            "minicluster",
            "monitor",
            "status",
            "--checkpoint",
            str(e2e_cli_artifacts["health_checkpoint"]),
        ]
    )
    status_text = f"{monitor_status.stdout}\n{monitor_status.stderr}".upper()
    assert any(level in status_text for level in ("HEALTHY", "DEGRADED", "UNHEALTHY"))


@pytest.mark.integration
def test_e2e_cli_reports_and_cross_tool_compare(e2e_cli_artifacts: dict[str, Path], tmp_path: Path) -> None:
    """Reports and compare CLI commands should consume canonical outputs end-to-end."""
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    autoperf_report_name = "autoperf_e2e_report.html"
    autoperf_report_path = report_dir / autoperf_report_name
    autoperf_export_json = report_dir / "autoperf_e2e_report_data.json"
    autoperf_export_csv = report_dir / "autoperf_e2e_report_data.csv"

    _run_command(
        [
            sys.executable,
            str(AUTOPERFPY_CLI),
            "--output-dir",
            str(report_dir),
            "report",
            "html",
            "--json",
            str(e2e_cli_artifacts["autoperf_run_json"]),
            "--output",
            autoperf_report_name,
            "--title",
            "E2E AutoPerfPy Report",
        ]
    )
    assert autoperf_report_path.exists()
    assert autoperf_export_json.exists()
    assert autoperf_export_csv.exists()

    exported_autoperf_result = load_trackiq_result(autoperf_export_json)
    validate_trackiq_result_obj(exported_autoperf_result)
    assert exported_autoperf_result.tool_name == "autoperfpy"

    minicluster_report_path = report_dir / "minicluster_e2e_report.html"
    _run_command(
        [
            sys.executable,
            "-m",
            "minicluster",
            "report",
            "html",
            "--result",
            str(e2e_cli_artifacts["minicluster_run_json"]),
            "--output",
            str(minicluster_report_path),
            "--title",
            "E2E MiniCluster Report",
        ]
    )
    assert minicluster_report_path.exists()
    assert "MiniCluster" in minicluster_report_path.read_text(encoding="utf-8")

    compare_report_path = report_dir / "compare_e2e_report.html"
    _run_command(
        [
            sys.executable,
            "-m",
            "trackiq_compare",
            "run",
            str(autoperf_export_json),
            str(e2e_cli_artifacts["minicluster_run_json"]),
            "--html",
            str(compare_report_path),
            "--label-a",
            "autoperf-e2e",
            "--label-b",
            "minicluster-e2e",
        ]
    )
    assert compare_report_path.exists()
    compare_html = compare_report_path.read_text(encoding="utf-8")
    assert "TrackIQ Comparison Report" in compare_html
    assert "autoperf-e2e" in compare_html
    assert "minicluster-e2e" in compare_html
