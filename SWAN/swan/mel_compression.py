"""Step 2 second half: CNN compression of the reweighted spectrogram."""
import torch
import torch.nn as nn


class ResidualConvBlock(nn.Module):
    """Two Conv2d + BN + ReLU layers with a 1x1 projection shortcut."""

    def __init__(self, in_ch: int, out_ch: int, time_stride: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1,
                               stride=(1, time_stride))
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(1, time_stride)),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + identity)
        return out


class MelCompression(nn.Module):
    """[B, F, T] -> [B, mel_out_dim]."""

    def __init__(self, n_mels: int = 128, channels=(16, 32, 64),
                 mel_out_dim: int = 256):
        super().__init__()
        self.n_mels = n_mels
        self.out_dim = mel_out_dim

        blocks = []
        in_ch = 1
        for ch in channels:
            blocks.append(ResidualConvBlock(in_ch, ch, time_stride=2))
            in_ch = ch
        self.blocks = nn.Sequential(*blocks)

        self.freq_pool = nn.AdaptiveAvgPool2d((8, 1))
        feat_dim = channels[-1] * 8
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, mel_out_dim),
            nn.LayerNorm(mel_out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [B, F, T]
        Returns:
            [B, mel_out_dim]
        """
        x = mel.unsqueeze(1)
        x = self.blocks(x)
        x = self.freq_pool(x)
        return self.proj(x)
