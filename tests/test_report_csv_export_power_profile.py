"""Tests for CSV export power fallback behavior."""

from __future__ import annotations

from autoperfpy.cli import _write_result_to_csv
from autoperfpy.commands.report import _write_multi_run_csv


def test_write_result_to_csv_uses_power_profile_when_sample_power_missing(tmp_path) -> None:
    """Single-run CSV export should backfill power_w from power_profile step_readings."""
    result = {
        "inference_config": {"batch_size": 2},
        "samples": [
            {"timestamp": 1.0, "metrics": {"latency_ms": 10.0}, "metadata": {"sample_index": 0}},
            {"timestamp": 2.0, "metrics": {"latency_ms": 20.0}, "metadata": {"sample_index": 1}},
        ],
        "power_profile": {
            "step_readings": [
                {"step": 0, "power_watts": 33.0},
                {"step": 1, "power_watts": 44.0},
            ]
        },
    }
    output = tmp_path / "single.csv"

    assert _write_result_to_csv(result, str(output))
    content = output.read_text(encoding="utf-8")
    assert ",33.0," in content
    assert ",44.0," in content


def test_write_multi_run_csv_uses_power_profile_when_sample_power_missing(tmp_path) -> None:
    """Multi-run CSV export should backfill power_w from power_profile step_readings."""
    runs = [
        {
            "run_label": "run_a",
            "inference_config": {"batch_size": 1},
            "samples": [
                {"timestamp": 1.0, "metrics": {"latency_ms": 5.0}, "metadata": {"sample_index": 0}},
                {"timestamp": 2.0, "metrics": {"latency_ms": 10.0}, "metadata": {"sample_index": 1}},
            ],
            "power_profile": {
                "step_readings": [
                    {"step": 0, "power_watts": 11.0},
                    {"step": 1, "power_watts": 22.0},
                ]
            },
        }
    ]
    output = tmp_path / "multi.csv"

    assert _write_multi_run_csv(runs, str(output))
    content = output.read_text(encoding="utf-8")
    assert ",11.0," in content
    assert ",22.0," in content
