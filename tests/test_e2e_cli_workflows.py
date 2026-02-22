"""End-to-end CLI integration tests across TrackIQ tools."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import dashboard as unified_dashboard
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


@pytest.mark.integration
def test_e2e_autoperf_bench_inference_and_plotly_report(tmp_path: Path) -> None:
    """bench-inference mock backend should produce JSON and feed into Plotly HTML report generation."""
    output_dir = tmp_path / "autoperf-bench"
    output_dir.mkdir(parents=True, exist_ok=True)
    bench_name = "bench_inference.json"
    bench_json = output_dir / bench_name

    _run_command(
        [
            sys.executable,
            str(AUTOPERFPY_CLI),
            "--output-dir",
            str(output_dir),
            "bench-inference",
            "--model",
            "mock-model",
            "--backend",
            "mock",
            "--num-prompts",
            "16",
            "--input-len",
            "64",
            "--output-len",
            "64",
            "--output",
            bench_name,
        ]
    )
    assert bench_json.exists()
    payload = json.loads(bench_json.read_text(encoding="utf-8"))
    assert payload["backend"] == "mock"
    assert payload["throughput_tokens_per_sec"] > 0

    report_name = "bench_inference_report.html"
    report_path = output_dir / report_name
    _run_command(
        [
            sys.executable,
            str(AUTOPERFPY_CLI),
            "--output-dir",
            str(output_dir),
            "report",
            "html",
            "--json",
            str(bench_json),
            "--output",
            report_name,
            "--title",
            "Bench Inference Report",
        ]
    )

    assert report_path.exists()
    html = report_path.read_text(encoding="utf-8")
    assert "AutoPerfPy Inference Benchmark Report" in html
    assert "Latency Percentiles (ms)" in html


@pytest.mark.integration
def test_e2e_minicluster_heatmap_and_fault_timeline_reports(tmp_path: Path) -> None:
    """minicluster report heatmap/fault-timeline commands should generate HTML outputs from JSON inputs."""
    workers_dir = tmp_path / "workers"
    workers_dir.mkdir(parents=True, exist_ok=True)
    (workers_dir / "worker_0.json").write_text(json.dumps({"worker_id": 0, "allreduce_time_ms": 4.0}), encoding="utf-8")
    (workers_dir / "worker_1.json").write_text(json.dumps({"worker_id": 1, "allreduce_time_ms": 9.5}), encoding="utf-8")

    heatmap_path = tmp_path / "heatmap.html"
    _run_command(
        [
            sys.executable,
            "-m",
            "minicluster",
            "report",
            "heatmap",
            "--results-dir",
            str(workers_dir),
            "--metric",
            "allreduce_time_ms",
            "--output",
            str(heatmap_path),
        ]
    )
    assert heatmap_path.exists()
    assert "Cluster Heatmap" in heatmap_path.read_text(encoding="utf-8")

    fault_report = {
        "loss_curve": [1.0, 0.9, 0.82, 0.8, 0.78],
        "results": [
            {
                "fault_type": "SLOW_WORKER",
                "injection_step": 2,
                "detected_step": 4,
                "was_detected": True,
            }
        ],
        "num_detected": 1,
        "num_faults": 1,
    }
    fault_json = tmp_path / "fault_report.json"
    fault_json.write_text(json.dumps(fault_report), encoding="utf-8")
    fault_timeline_path = tmp_path / "fault_timeline.html"

    _run_command(
        [
            sys.executable,
            "-m",
            "minicluster",
            "report",
            "fault-timeline",
            "--json",
            str(fault_json),
            "--output",
            str(fault_timeline_path),
        ]
    )
    assert fault_timeline_path.exists()
    assert "Fault Injection Validation Report" in fault_timeline_path.read_text(encoding="utf-8")


@pytest.mark.integration
def test_e2e_minicluster_bench_collective_smoke(tmp_path: Path) -> None:
    """minicluster bench-collective should run a tiny gloo benchmark and emit expected JSON fields."""
    pytest.importorskip("torch")

    output = tmp_path / "collective_bench.json"
    _run_command(
        [
            sys.executable,
            "-m",
            "minicluster",
            "bench-collective",
            "--workers",
            "2",
            "--size-mb",
            "1.0",
            "--iterations",
            "2",
            "--backend",
            "gloo",
            "--output",
            str(output),
        ]
    )
    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["workers"] == 2
    assert payload["backend"] == "gloo"
    assert payload["iterations"] == 2
    assert "p99_bandwidth_gbps" in payload


@pytest.mark.integration
def test_e2e_cli_pdf_reports_across_tools(e2e_cli_artifacts: dict[str, Path], tmp_path: Path) -> None:
    """PDF reporting should succeed end-to-end for producer and compare tools."""
    report_dir = tmp_path / "pdf-reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    autoperf_pdf = report_dir / "autoperf_e2e_report.pdf"
    minicluster_pdf = report_dir / "minicluster_e2e_report.pdf"
    compare_pdf = report_dir / "compare_e2e_report.pdf"

    _run_command(
        [
            sys.executable,
            str(AUTOPERFPY_CLI),
            "--output-dir",
            str(report_dir),
            "report",
            "pdf",
            "--json",
            str(e2e_cli_artifacts["autoperf_run_json"]),
            "--output",
            autoperf_pdf.name,
            "--pdf-backend",
            "matplotlib",
        ]
    )
    _run_command(
        [
            sys.executable,
            "-m",
            "minicluster",
            "report",
            "pdf",
            "--result",
            str(e2e_cli_artifacts["minicluster_run_json"]),
            "--output",
            str(minicluster_pdf),
            "--pdf-backend",
            "matplotlib",
        ]
    )
    _run_command(
        [
            sys.executable,
            "-m",
            "trackiq_compare",
            "report",
            "pdf",
            str(e2e_cli_artifacts["autoperf_run_json"]),
            str(e2e_cli_artifacts["minicluster_run_json"]),
            "--output",
            str(compare_pdf),
            "--pdf-backend",
            "matplotlib",
            "--label-a",
            "autoperf-e2e",
            "--label-b",
            "minicluster-e2e",
        ]
    )

    for pdf_path in (autoperf_pdf, minicluster_pdf, compare_pdf):
        assert pdf_path.exists()
        assert pdf_path.read_bytes().startswith(b"%PDF")
        assert pdf_path.stat().st_size > 0


@pytest.mark.integration
def test_e2e_unified_dashboard_cli_paths_for_minicluster_and_compare(
    e2e_cli_artifacts: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unified dashboard CLI should accept canonical result paths for non-interactive launch modes."""
    captured: dict[str, object] = {}

    def _fake_run_dashboard(dashboard_class, result_path=None, result=None, theme=None):  # noqa: ANN001
        captured["dashboard_class"] = dashboard_class
        captured["result_path"] = result_path
        captured["result"] = result
        captured["theme"] = theme

    monkeypatch.setattr(unified_dashboard, "run_dashboard", _fake_run_dashboard)

    rc_mini = unified_dashboard.main(
        [
            "--tool",
            "minicluster",
            "--result",
            str(e2e_cli_artifacts["minicluster_run_json"]),
        ]
    )
    assert rc_mini == 0
    assert captured["dashboard_class"].__name__ == "MiniClusterDashboard"
    assert str(captured["result_path"]) == str(e2e_cli_artifacts["minicluster_run_json"])

    captured.clear()
    rc_compare = unified_dashboard.main(
        [
            "--tool",
            "compare",
            "--result-a",
            str(e2e_cli_artifacts["autoperf_run_json"]),
            "--result-b",
            str(e2e_cli_artifacts["minicluster_run_json"]),
            "--label-a",
            "auto",
            "--label-b",
            "mini",
        ]
    )
    assert rc_compare == 0
    assert "CompareDashboardAdapter" in captured["dashboard_class"].__name__
    assert isinstance(captured["result"], list)
    assert len(captured["result"]) == 2


@pytest.mark.integration
def test_e2e_unified_dashboard_autoperf_entrypoint_uses_streamlit_and_renderer(
    e2e_cli_artifacts: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AutoPerfPy dashboard mode should initialize page config and route into interactive renderer."""
    captured: dict[str, object] = {}

    class _FakeStreamlit:
        @staticmethod
        def set_page_config(**kwargs):  # noqa: ANN003
            captured["page_config"] = kwargs

    def _fake_render_autoperf_interactive(*, args, key_prefix, title=None, caption=None):  # noqa: ANN001
        captured["args"] = args
        captured["key_prefix"] = key_prefix
        captured["title"] = title
        captured["caption"] = caption
        return 0

    monkeypatch.setitem(sys.modules, "streamlit", _FakeStreamlit)
    monkeypatch.setattr(unified_dashboard, "_render_autoperf_interactive", _fake_render_autoperf_interactive)

    rc = unified_dashboard.main(
        [
            "--tool",
            "autoperfpy",
            "--result",
            str(e2e_cli_artifacts["autoperf_run_json"]),
        ]
    )
    assert rc == 0
    assert captured["page_config"]["page_title"] == "AutoPerfPy Dashboard"
    assert captured["key_prefix"] == "trackiq_autoperf_only"
    assert "AutoPerfPy Interactive Dashboard" in str(captured["title"])
