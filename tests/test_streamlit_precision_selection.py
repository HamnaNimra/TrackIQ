"""Tests for Streamlit benchmark precision selection and run capping helpers."""

from dataclasses import dataclass

import pandas as pd

from autoperfpy.ui.streamlit_app import (
    _available_metric_sections,
    _build_run_summary_text,
    _cap_pairs_with_precision_coverage,
    _normalize_max_configs_per_device,
    _run_display_name,
    build_run_overview_row,
    build_sample_data_list,
)


@dataclass
class _FakeDevice:
    device_id: str


@dataclass
class _FakeConfig:
    precision: str
    batch_size: int


def test_normalize_max_configs_per_device_zero_means_unlimited() -> None:
    """UI value 0 should disable per-device max-config truncation."""
    assert _normalize_max_configs_per_device(0) is None
    assert _normalize_max_configs_per_device(-5) is None
    assert _normalize_max_configs_per_device(None) is None
    assert _normalize_max_configs_per_device(7) == 7


def test_cap_pairs_preserves_precision_coverage_before_fill() -> None:
    """When capped, selection should include every precision if capacity allows."""
    device = _FakeDevice(device_id="cpu_0")
    precisions = ["fp32", "fp16", "bf16", "int8", "int4", "mixed"]
    batches = [1, 4, 8]
    pairs: list[tuple[_FakeDevice, _FakeConfig]] = []
    for precision in precisions:
        for batch in batches:
            pairs.append((device, _FakeConfig(precision=precision, batch_size=batch)))

    selected = _cap_pairs_with_precision_coverage(pairs, max_total_runs=12)

    assert len(selected) == 12
    selected_precisions = {cfg.precision for _, cfg in selected}
    assert selected_precisions == set(precisions)


def test_build_sample_data_list_returns_single_demo_run() -> None:
    """Sample-data helper should return one valid demo run payload."""
    data_list = build_sample_data_list()
    assert len(data_list) == 1
    run = data_list[0]
    assert run.get("collector_name")
    assert isinstance(run.get("samples"), list)
    assert run.get("summary")


def test_build_run_overview_row_includes_config_fields() -> None:
    """Overview row should expose device/precision/batch and key summary values."""
    run = {
        "run_label": "nvidia_0_fp16_bs8",
        "platform_metadata": {"device_name": "NVIDIA A100"},
        "inference_config": {"precision": "fp16", "batch_size": 8, "accelerator": "nvidia_0"},
        "summary": {
            "sample_count": 120,
            "duration_seconds": 15.5,
            "latency": {"p99_ms": 42.1},
            "throughput": {"mean_fps": 812.0},
            "power": {"mean_w": 289.4},
        },
    }
    row = build_run_overview_row(run)
    assert row["Run"] == "nvidia_0_fp16_bs8"
    assert row["Device"] == "NVIDIA A100"
    assert row["Precision"] == "fp16"
    assert row["Batch Size"] == 8
    assert row["Samples"] == 120
    assert row["P99 Latency (ms)"] == 42.1


def test_run_display_name_prefers_label_and_filename_when_distinct() -> None:
    """Display name should preserve run label and include source filename context."""
    run = {"run_label": "mi300x_fp16_bs8", "source_filename": "results_gpu0.json"}
    assert _run_display_name(run, index=0) == "mi300x_fp16_bs8 (results_gpu0.json)"


def test_available_metric_sections_lists_known_groups() -> None:
    """Metric section helper should map dataframe columns to user-facing sections."""
    df = pd.DataFrame(
        [
            {
                "latency_ms": 1.0,
                "cpu_percent": 10.0,
                "power_w": 50.0,
                "memory_used_mb": 2048.0,
                "throughput_fps": 90.0,
            }
        ]
    )
    assert _available_metric_sections(df) == "Latency, Utilization, Power & Thermal, Memory, Throughput"


def test_build_run_summary_text_uses_p99_and_throughput() -> None:
    """Sidebar run summary should expose key latency and throughput snippets."""
    run = {"summary": {"latency": {"p99_ms": 27.32}, "throughput": {"mean_fps": 42.5}}}
    assert _build_run_summary_text(run) == "P99 27.32 ms, 42.5 FPS"
