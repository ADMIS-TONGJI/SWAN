"""Step 3: projection heads aligning the two branches to the same dimension."""
import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_dim = proj_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
