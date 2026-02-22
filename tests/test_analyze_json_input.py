"""Tests for JSON-to-CSV analyze latency flow."""

from __future__ import annotations

import json
from types import SimpleNamespace

from autoperfpy.commands.analyze import run_analyze_latency


def test_run_analyze_latency_accepts_json_export(tmp_path) -> None:
    """Latency analyze command should consume JSON export without running a benchmark."""
    payload = {
        "collector_name": "synthetic",
        "run_label": "cpu_0_fp32_bs1",
        "inference_config": {"batch_size": 1},
        "samples": [
            {"timestamp": 1.0, "metrics": {"latency_ms": 10.0, "power_w": 30.0}},
            {"timestamp": 2.0, "metrics": {"latency_ms": 11.0, "power_w": 31.0}},
            {"timestamp": 3.0, "metrics": {"latency_ms": 12.0, "power_w": 32.0}},
        ],
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    args = SimpleNamespace(
        csv=None,
        json=str(path),
        device=None,
        duration=1,
    )

    def _unexpected_default(_device_id, _duration):
        raise AssertionError("run_default_benchmark should not be used when --json is provided")

    result = run_analyze_latency(args, config=None, run_default_benchmark=_unexpected_default)
    assert result is not None
    assert any("cpu_0_fp32_bs1" in key for key in result.metrics)


def test_run_analyze_latency_accepts_trackiq_wrapped_json_export(tmp_path) -> None:
    """Latency analyze command should read samples from canonical wrapper tool_payload."""
    payload = {
        "tool_name": "autoperfpy",
        "tool_payload": {
            "run_label": "wrapped_run",
            "inference_config": {"batch_size": 4},
            "samples": [
                {"timestamp": 1.0, "metrics": {"latency_ms": 20.0, "power_w": 60.0}},
                {"timestamp": 2.0, "metrics": {"latency_ms": 21.0, "power_w": 61.0}},
            ],
        },
    }
    path = tmp_path / "wrapped_run.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    args = SimpleNamespace(
        csv=None,
        json=str(path),
        device=None,
        duration=1,
    )

    result = run_analyze_latency(
        args,
        config=None,
        run_default_benchmark=lambda _device_id, _duration: (_ for _ in ()).throw(AssertionError("not expected")),
    )
    assert result is not None
    assert any("wrapped_run" in key for key in result.metrics)
