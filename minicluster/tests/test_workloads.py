"""Tests for synthetic minicluster workload modules."""

from __future__ import annotations

import torch

from minicluster.runner.workloads import EmbeddingWorkload, TransformerWorkload


def test_transformer_workload_outputs_scalar_per_sample() -> None:
    """Transformer workload should produce batch x 1 output."""
    model = TransformerWorkload()
    x_batch = torch.randn(8, 32, 256)
    output = model(x_batch)
    assert output.shape == (8, 1)


def test_embedding_workload_outputs_scalar_per_sample() -> None:
    """Embedding workload should produce batch x 1 output."""
    model = EmbeddingWorkload()
    batch_size = 8
    bag_size = 16
    total = batch_size * bag_size
    indices = torch.randint(0, 10_000, (total,), dtype=torch.long)
    offsets = torch.arange(0, total, bag_size, dtype=torch.long)
    output = model(indices, offsets)
    assert output.shape == (batch_size, 1)
