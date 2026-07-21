"""Step 4: gated fusion of the two branches."""
import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """Gated fusion: learn a gating vector to combine mel and wav2vec branches."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.out_dim = dim
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, em1: torch.Tensor, em2: torch.Tensor):
        g = self.gate(torch.cat([em1, em2], dim=-1))  # [B, dim] in (0, 1)
        fused = g * em1 + (1.0 - g) * em2
        return self.norm(fused)
