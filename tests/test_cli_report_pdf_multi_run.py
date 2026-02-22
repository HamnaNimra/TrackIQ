"""Tests for multi-run PDF report handling in autoperfpy report command."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from autoperfpy.commands.report import run_report_pdf
from trackiq_core.reporting import PDF_BACKEND_MATPLOTLIB


def _normalize_report_input_data(data: object) -> dict[str, Any]:
    if isinstance(data, dict) and isinstance(data.get("tool_payload"), dict):
        payload = dict(data["tool_payload"])
        if "collector_name" not in payload and data.get("tool_name"):
            payload["collector_name"] = str(data.get("tool_name"))
        return payload
    if isinstance(data, dict):
        return data
    return {}


def _output_path(args: Any, filename: str) -> str:
    out_dir = Path(getattr(args, "output_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / Path(filename).name)


def test_run_report_pdf_supports_multi_run_json_and_exports_context(tmp_path: Path) -> None:
    """PDF report should handle list payloads and include report context in JSON export payload."""
    payload = [
        {
            "tool_name": "autoperfpy",
            "tool_payload": {
                "run_label": "cpu_0_fp32_bs1",
                "collector_name": "autoperfpy",
                "platform_metadata": {"device_name": "CPU"},
                "inference_config": {"precision": "fp32", "batch_size": 1},
                "summary": {"latency": {"p99_ms": 21.0}, "throughput": {"mean_fps": 44.0}, "power": {"mean_w": 55.0}},
                "samples": [],
            },
        },
        {
            "tool_name": "autoperfpy",
            "tool_payload": {
                "run_label": "nvidia_0_fp16_bs4",
                "collector_name": "autoperfpy",
                "platform_metadata": {"device_name": "NVIDIA"},
                "inference_config": {"precision": "fp16", "batch_size": 4},
                "summary": {"latency": {"p99_ms": 11.0}, "throughput": {"mean_fps": 122.0}, "power": {"mean_w": 130.0}},
                "samples": [],
            },
        },
    ]
    json_path = tmp_path / "runs.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    captured: dict[str, Any] = {}

    def _save_trackiq_wrapped_json(path: str, payload_obj: Any, workload_name: str, workload_type: str) -> None:
        captured["path"] = path
        captured["payload"] = payload_obj
        captured["workload_name"] = workload_name
        captured["workload_type"] = workload_type

    args = SimpleNamespace(
        csv=None,
        json=str(json_path),
        device=None,
        duration=10,
        output="multi_report.pdf",
        export_json=None,
        export_csv=None,
        title="Multi Run PDF Report",
        author="tests",
        output_dir=str(tmp_path),
        pdf_backend=PDF_BACKEND_MATPLOTLIB,
    )

    result = run_report_pdf(
        args,
        config=None,
        run_default_benchmark=lambda _device_id, _duration: (_ for _ in ()).throw(AssertionError("not expected")),
        normalize_report_input_data=_normalize_report_input_data,
        output_path=_output_path,
        save_trackiq_wrapped_json=_save_trackiq_wrapped_json,
        write_result_to_csv=lambda _result, _path: False,
    )

    assert isinstance(result, dict)
    pdf_path = Path(result["output_path"])
    assert pdf_path.exists()
    assert pdf_path.read_bytes().startswith(b"%PDF")
    assert captured["workload_name"] == "pdf_report_data"
    exported_payload = captured["payload"]
    assert isinstance(exported_payload, list)
    assert len(exported_payload) == 2
    assert isinstance(exported_payload[0], dict)
    assert exported_payload[0]["report_context"]["format"] == "pdf"
    assert "Human-readable performance artifact" in exported_payload[0]["report_context"]["purpose"]
