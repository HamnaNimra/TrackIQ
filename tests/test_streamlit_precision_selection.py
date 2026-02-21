"""Tests for Streamlit benchmark precision selection and run capping helpers."""

from dataclasses import dataclass

from autoperfpy.ui.streamlit_app import (
    _cap_pairs_with_precision_coverage,
    _normalize_max_configs_per_device,
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
