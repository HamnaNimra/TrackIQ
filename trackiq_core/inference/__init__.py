"""Inference configuration module for TrackIQ.

Provides InferenceConfig dataclass and utilities for enumerating inference
configurations across devices.
"""

from .config import (
    InferenceConfig,
    enumerate_inference_configs,
    PRECISION_FP32,
    PRECISION_FP16,
    PRECISION_BF16,
    PRECISION_INT8,
    PRECISION_INT4,
    PRECISION_MIXED,
    PRECISIONS,
    DEFAULT_BATCH_SIZES,
    DEFAULT_WARMUP_RUNS,
    DEFAULT_ITERATIONS,
    DEFAULT_STREAMS,
    get_supported_precisions_for_device,
    is_precision_supported,
    resolve_precision_for_device,
)

__all__ = [
    "InferenceConfig",
    "enumerate_inference_configs",
    "PRECISION_FP32",
    "PRECISION_FP16",
    "PRECISION_BF16",
    "PRECISION_INT8",
    "PRECISION_INT4",
    "PRECISION_MIXED",
    "PRECISIONS",
    "DEFAULT_BATCH_SIZES",
    "DEFAULT_WARMUP_RUNS",
    "DEFAULT_ITERATIONS",
    "DEFAULT_STREAMS",
    "get_supported_precisions_for_device",
    "is_precision_supported",
    "resolve_precision_for_device",
]
