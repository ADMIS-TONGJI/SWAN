"""SWAN: multimodal underwater acoustic target classification.

Two branches: frozen wav2vec2 audio embeddings + mel spectrogram with
energy-adaptive subband reweighting, projected, fused via gated fusion,
and classified.
"""
from .config import ModelConfig
from .model import MultiModalNet

__all__ = ["ModelConfig", "MultiModalNet"]
