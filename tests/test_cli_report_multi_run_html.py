"""Tests for multi-run HTML report generation from JSON inputs."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from autoperfpy.commands.report import run_report_html


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


def test_run_report_html_includes_all_run_labels_for_multi_run_json(tmp_path: Path) -> None:
    """HTML report should include all run labels when JSON input contains multiple runs."""
    run_a = {
        "collector_name": "autoperfpy",
        "run_label": "cpu_0_fp32_bs1",
        "platform_metadata": {"device_name": "CPU"},
        "inference_config": {"precision": "fp32", "batch_size": 1, "accelerator": "cpu_0"},
        "summary": {
            "sample_count": 20,
            "duration_seconds": 2.0,
            "latency": {"p99_ms": 22.0, "p95_ms": 20.0, "p50_ms": 18.0, "mean_ms": 19.0},
            "throughput": {"mean_fps": 45.0},
            "power": {"mean_w": 50.0},
            "temperature": {"max_c": 62.0},
        },
        "samples": [],
    }
    run_b = {
        "collector_name": "autoperfpy",
        "run_label": "nvidia_0_fp16_bs4",
        "platform_metadata": {"device_name": "NVIDIA"},
        "inference_config": {"precision": "fp16", "batch_size": 4, "accelerator": "nvidia_0"},
        "summary": {
            "sample_count": 25,
            "duration_seconds": 2.0,
            "latency": {"p99_ms": 11.0, "p95_ms": 10.0, "p50_ms": 8.5, "mean_ms": 9.0},
            "throughput": {"mean_fps": 120.0},
            "power": {"mean_w": 135.0},
            "temperature": {"max_c": 71.0},
        },
        "samples": [],
    }
    payload = [
        {"tool_name": "autoperfpy", "tool_payload": run_a},
        {"tool_name": "autoperfpy", "tool_payload": run_b},
    ]
    json_path = tmp_path / "multi_runs.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    args = SimpleNamespace(
        csv=None,
        json=str(json_path),
        title="Performance Analysis Report",
        author="AutoPerfPy",
        theme="light",
        output="multi_report.html",
        export_json=None,
        export_csv=None,
        output_dir=str(tmp_path),
        device=None,
        duration=10,
    )

    result = run_report_html(
        args,
        config=None,
        run_default_benchmark=lambda _device_id, _duration: (_ for _ in ()).throw(AssertionError("not expected")),
        normalize_report_input_data=_normalize_report_input_data,
        output_path=_output_path,
        save_trackiq_wrapped_json=lambda *_args, **_kwargs: None,
        write_result_to_csv=lambda _result, _path: False,
    )
    assert isinstance(result, dict)
    html_path = Path(result["output_path"])
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")
    assert "cpu_0_fp32_bs1" in content
    assert "nvidia_0_fp16_bs4" in content
    assert "Run Overview" in content
    assert "Run Comparison" in content
