"""Step 1: wav2vec2 audio branch.

Input is a frozen wav2vec2 embedding [B, T, 768]. The branch performs
temporal mean pooling to obtain one vector per sample [B, 768].

Note: wav2vec2 itself is frozen and embeddings are extracted offline.
This module only processes the saved embedding tensors.
"""
import torch
import torch.nn as nn


class Wav2VecBranch(nn.Module):
    def __init__(self, in_dim: int = 768):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = in_dim

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb: [B, T, 768]
        Returns:
            [B, 768]
        """
        if emb.dim() != 3:
            raise ValueError(f"expected [B, T, D], got {tuple(emb.shape)}")
        return emb.mean(dim=1)
