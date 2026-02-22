"""Tests for Plotly inference report path in autoperfpy report command."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from autoperfpy.commands.report import run_report_html

pytest.importorskip("plotly")


def _output_path(args: Any, filename: str) -> str:
    out_dir = Path(getattr(args, "output_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / Path(filename).name)


def test_run_report_html_uses_plotly_inference_path_for_bench_payload(tmp_path: Path) -> None:
    """Inference benchmark JSON should route to Plotly report output."""
    payload = {
        "backend": "mock",
        "model": "mock-model",
        "num_prompts": 32,
        "mean_ttft_ms": 120.0,
        "p99_ttft_ms": 180.0,
        "mean_tpot_ms": 40.0,
        "p99_tpot_ms": 70.0,
        "throughput_tokens_per_sec": 1500.0,
    }
    json_path = tmp_path / "bench.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    args = SimpleNamespace(
        csv=None,
        json=str(json_path),
        title="Performance Analysis Report",
        author="AutoPerfPy",
        theme="light",
        output="plotly_report.html",
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
        normalize_report_input_data=lambda data: data if isinstance(data, dict) else {},
        output_path=_output_path,
        save_trackiq_wrapped_json=lambda *_args, **_kwargs: None,
        write_result_to_csv=lambda _result, _path: False,
    )

    assert isinstance(result, dict)
    html_path = Path(result["output_path"])
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")
    assert "AutoPerfPy Inference Benchmark Report" in content
    assert "Latency Percentiles (ms)" in content
