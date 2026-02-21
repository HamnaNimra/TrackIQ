"""Synthetic workload modules for minicluster training simulations."""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover - dependency guard
    torch = None
    nn = None
    F = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


def _require_torch() -> None:
    """Raise actionable dependency error when torch extras are missing."""
    if _TORCH_IMPORT_ERROR is not None:
        raise ImportError(
            "PyTorch is required for minicluster workloads. " 'Install with: pip install -e ".[ml]"'
        ) from _TORCH_IMPORT_ERROR


if nn is not None:

    class TransformerWorkload(nn.Module):
        """Simulates attention-heavy transformer blocks used in LLM training."""

        def __init__(self, d_model: int = 256, heads: int = 4):
            super().__init__()
            self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, 1024),
                nn.ReLU(),
                nn.Linear(1024, d_model),
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, 1)

        def forward(self, x):
            attn_out, _ = self.attn(x, x, x, need_weights=False)
            x = self.norm1(x + attn_out)
            ff = self.ffn(x)
            x = self.norm2(x + ff)
            pooled = x.mean(dim=1)
            return self.head(pooled)


    class EmbeddingWorkload(nn.Module):
        """Simulates embedding-heavy recommendation models dominated by memory bandwidth."""

        def __init__(self, vocab_size: int = 10_000, embedding_dim: int = 128):
            super().__init__()
            self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim, mode="mean")
            self.fc1 = nn.Linear(embedding_dim, 256)
            self.fc2 = nn.Linear(256, 1)

        def forward(self, indices, offsets):
            x = self.embedding(indices, offsets)
            x = F.relu(self.fc1(x))
            return self.fc2(x)


else:

    class TransformerWorkload:  # pragma: no cover - exercised only when torch missing
        """Fallback type for optional torch dependency."""

        def __init__(self, *args, **kwargs):
            _require_torch()


    class EmbeddingWorkload:  # pragma: no cover - exercised only when torch missing
        """Fallback type for optional torch dependency."""

        def __init__(self, *args, **kwargs):
            _require_torch()

