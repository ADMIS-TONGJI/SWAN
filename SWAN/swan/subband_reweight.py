"""Step 2 core: energy-adaptive mel spectrogram subband reweighting.

The mel frequency axis is split into subbands. Per-subband energy is
computed by averaging over frequency bins and time frames. A small MLP
generates subband weights that rescale the spectrogram.

Numerical note: stored mel values are in dB (log scale, can be negative).
Direct softmax over dB energies would be distorted, so the module uses
instance-normalized energies and a tanh gate centered around 1.0.
"""
import torch
import torch.nn as nn


class EnergyAdaptiveSubbandReweight(nn.Module):
    def __init__(self, n_mels: int = 128, n_subbands: int = 8,
                 hidden: int = 32, weight_scale: float = 0.5):
        super().__init__()
        if n_mels % n_subbands != 0:
            raise ValueError(
                f"n_mels({n_mels}) must be divisible by n_subbands({n_subbands})")
        self.n_mels = n_mels
        self.n_subbands = n_subbands
        self.band_size = n_mels // n_subbands
        self.weight_scale = weight_scale

        self.gate = nn.Sequential(
            nn.Linear(n_subbands, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_subbands),
        )

    def subband_energy(self, mel: torch.Tensor) -> torch.Tensor:
        """Per-subband energy averaged over freq bins and time frames."""
        B, F, T = mel.shape
        bands = mel.view(B, self.n_subbands, self.band_size, T)
        return bands.mean(dim=(2, 3))

    def forward(self, mel: torch.Tensor):
        """
        Args:
            mel: [B, F=128, T=157] (dB)
        Returns:
            reweighted: [B, F, T]
            weights:    [B, n_subbands]
        """
        if mel.dim() != 3:
            raise ValueError(f"expected [B, F, T], got {tuple(mel.shape)}")
        B, F, T = mel.shape

        energy = self.subband_energy(mel)
        e_std = torch.clamp(energy.std(dim=1, keepdim=True), min=1e-5)
        e_norm = (energy - energy.mean(dim=1, keepdim=True)) / e_std

        gate = torch.tanh(self.gate(e_norm))
        weights = 1.0 + self.weight_scale * gate

        w_expand = weights.unsqueeze(-1).unsqueeze(-1)
        bands = mel.view(B, self.n_subbands, self.band_size, T)
        reweighted = (bands * w_expand).reshape(B, F, T)

        return reweighted, weights
