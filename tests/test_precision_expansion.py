"""Tests for precision expansion and unsupported precision fallback."""

from __future__ import annotations

import argparse

import pytest

import autoperfpy.cli as autoperf_cli
from trackiq_core.configs.profiles import Profile, ProfileValidationError, validate_profile_precision
from trackiq_core.hardware.devices import (
    DEVICE_TYPE_CPU,
    DEVICE_TYPE_NVIDIA_GPU,
    DeviceProfile,
)
from trackiq_core.inference import (
    PRECISION_BF16,
    PRECISION_FP16,
    PRECISION_FP32,
    PRECISION_INT4,
    PRECISION_INT8,
    PRECISION_MIXED,
    PRECISIONS,
    enumerate_inference_configs,
    get_supported_precisions_for_device,
)


def test_precision_constants_include_new_modes() -> None:
    """TrackIQ precision constants should include BF16, INT4, and mixed."""
    assert PRECISION_BF16 in PRECISIONS
    assert PRECISION_INT4 in PRECISIONS
    assert PRECISION_MIXED in PRECISIONS


def test_device_capabilities_include_new_modes_where_supported() -> None:
    """Capability map should expose precision support by accelerator type."""
    cpu = DeviceProfile(
        device_id="cpu_0",
        device_type=DEVICE_TYPE_CPU,
        device_name="CPU",
    )
    nvidia = DeviceProfile(
        device_id="nvidia_0",
        device_type=DEVICE_TYPE_NVIDIA_GPU,
        device_name="NVIDIA",
    )
    cpu_supported = get_supported_precisions_for_device(cpu)
    nvidia_supported = get_supported_precisions_for_device(nvidia)

    assert PRECISION_INT4 not in cpu_supported
    assert PRECISION_INT4 in nvidia_supported
    assert PRECISION_BF16 in nvidia_supported
    assert PRECISION_MIXED in nvidia_supported


def test_enumerate_inference_configs_falls_back_for_unsupported_precision() -> None:
    """Unsupported precision requests should fall back to a supported mode per device."""
    cpu = DeviceProfile(
        device_id="cpu_0",
        device_type=DEVICE_TYPE_CPU,
        device_name="CPU",
    )
    pairs = enumerate_inference_configs(
        [cpu],
        precisions=[PRECISION_INT4],
        batch_sizes=[1],
        max_configs_per_device=1,
    )
    assert len(pairs) == 1
    _, cfg = pairs[0]
    assert cfg.precision == PRECISION_FP32


def test_profile_precision_validation_rejects_unsupported_precision() -> None:
    """Profile precision compatibility checks should reject unsupported precision."""
    profile = Profile(
        name="precision_profile",
        description="test profile",
        supported_precisions=[PRECISION_FP32, PRECISION_FP16, PRECISION_INT8],
    )
    validate_profile_precision(profile, PRECISION_FP16)
    with pytest.raises(ProfileValidationError, match="Supported precisions"):
        validate_profile_precision(profile, PRECISION_INT4)


def test_manual_run_precision_fallback_for_unsupported_device_precision(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Manual run should fallback to supported precision when request is unsupported."""
    cpu = DeviceProfile(
        device_id="cpu_0",
        device_type=DEVICE_TYPE_CPU,
        device_name="CPU",
    )
    observed: dict[str, str] = {}

    def _fake_run_single_benchmark(device, config, **kwargs):  # noqa: ANN001
        observed["precision"] = config.precision
        return {
            "run_label": f"{device.device_id}_{config.precision}_bs{config.batch_size}",
            "samples": [],
            "summary": {},
            "inference_config": config.to_dict(),
        }

    monkeypatch.setattr(autoperf_cli, "_resolve_device", lambda _: cpu)
    monkeypatch.setattr(autoperf_cli, "run_single_benchmark", _fake_run_single_benchmark)

    args = argparse.Namespace(
        device="cpu_0",
        precision=PRECISION_INT4,
        batch_size=1,
        warmup=None,
        iterations=None,
        duration=1,
        quiet=True,
        no_power=True,
        export=None,
        export_csv=None,
        output_dir="output",
    )
    result = autoperf_cli.run_manual_single(args)
    assert result is not None
    assert observed["precision"] == PRECISION_FP32
    assert "falling back" in capsys.readouterr().err.lower()


def test_cli_parser_accepts_new_precision_modes() -> None:
    """CLI parser should accept BF16/INT4/mixed in --precision."""
    parser = autoperf_cli.setup_parser()
    args_bf16 = parser.parse_args(
        ["run", "--manual", "--device", "cpu_0", "--precision", PRECISION_BF16]
    )
    args_int4 = parser.parse_args(
        ["run", "--manual", "--device", "cpu_0", "--precision", PRECISION_INT4]
    )
    args_mixed = parser.parse_args(
        ["run", "--manual", "--device", "cpu_0", "--precision", PRECISION_MIXED]
    )
    assert args_bf16.precision == PRECISION_BF16
    assert args_int4.precision == PRECISION_INT4
    assert args_mixed.precision == PRECISION_MIXED
