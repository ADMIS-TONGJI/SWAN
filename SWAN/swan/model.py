"""MultiModalNet: assemble the five-step pipeline.

Input:
    w2v_emb: [B, T, 768]  frozen wav2vec2 embeddings
    mel:     [B, 128, 157] mel spectrogram
Output:
    logits:  [B, num_classes]
"""
import torch
import torch.nn as nn

from .config import ModelConfig
from .wav2vec_branch import Wav2VecBranch
from .subband_reweight import EnergyAdaptiveSubbandReweight
from .mel_compression import MelCompression
from .projection import ProjectionHead
from .fusion import GatedFusion
from .classifier import Classifier


class MultiModalNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Step 1: wav2vec branch
        self.w2v_branch = Wav2VecBranch(in_dim=cfg.w2v_dim)

        # Step 2: mel subband reweighting + compression
        self.subband = EnergyAdaptiveSubbandReweight(
            n_mels=cfg.n_mels, n_subbands=cfg.n_subbands,
            hidden=cfg.subband_hidden, weight_scale=cfg.subband_weight_scale)
        self.mel_compress = MelCompression(
            n_mels=cfg.n_mels, channels=tuple(cfg.mel_channels),
            mel_out_dim=cfg.mel_out_dim)

        # Step 3: project both branches to the same dimension
        self.proj_w2v = ProjectionHead(
            cfg.w2v_dim, cfg.proj_dim, dropout=cfg.proj_dropout)
        self.proj_mel = ProjectionHead(
            cfg.mel_out_dim, cfg.proj_dim, dropout=cfg.proj_dropout)

        # Step 4: gated fusion
        self.fusion = GatedFusion(cfg.proj_dim, dropout=cfg.fusion_dropout)

        # Step 5: classifier
        self.classifier = Classifier(
            cfg.proj_dim, cfg.num_classes,
            hidden=cfg.cls_hidden, dropout=cfg.cls_dropout)

    def forward(self, w2v_emb: torch.Tensor, mel: torch.Tensor,
                return_aux: bool = False):
        # Step 1
        a = self.w2v_branch(w2v_emb)            # [B, 768]
        em2 = self.proj_w2v(a)                  # [B, proj_dim]

        # Step 2
        mel_rw, sub_w = self.subband(mel)       # [B, F, T], [B, n_subbands]
        m = self.mel_compress(mel_rw)           # [B, mel_out_dim]
        em1 = self.proj_mel(m)                  # [B, proj_dim]

        # Step 4
        fused = self.fusion(em1, em2)           # [B, proj_dim]

        # Step 5
        logits = self.classifier(fused)         # [B, num_classes]

        if return_aux:
            return logits, {"subband_weights": sub_w, "em1": em1, "em2": em2}
        return logits
