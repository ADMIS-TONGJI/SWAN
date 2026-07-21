"""Model configuration."""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Input
    w2v_dim: int = 768          # wav2vec2 embedding dimension
    n_mels: int = 128           # number of mel frequency bins

    # Step 2: subband reweighting + compression
    n_subbands: int = 8         # number of subbands on mel frequency axis
    subband_hidden: int = 32
    subband_weight_scale: float = 0.5
    mel_channels: tuple = (16, 32, 64)
    mel_out_dim: int = 256

    # Step 3: projection
    proj_dim: int = 256

    # Step 4: gated fusion
    fusion_dropout: float = 0.1

    # Step 5: classifier
    num_classes: int = 9
    cls_hidden: int = 256
    cls_dropout: float = 0.3

    proj_dropout: float = 0.1
