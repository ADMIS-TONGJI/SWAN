"""Smoke tests: forward/backward pass and optional real-data end-to-end run."""
import argparse
import glob
import time

import torch
import torch.nn as nn

from swan import ModelConfig, MultiModalNet


def count_params(m):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable


def smoke(device, num_classes=9, B=16, T=249):
    cfg = ModelConfig(num_classes=num_classes)
    model = MultiModalNet(cfg).to(device)

    w2v = torch.randn(B, T, 768, device=device)
    mel = torch.randn(B, 128, 157, device=device) * 15 - 20

    model.train()
    logits, aux = model(w2v, mel, return_aux=True)
    assert logits.shape == (B, num_classes), logits.shape
    assert not torch.isnan(logits).any(), "logits contain NaN"
    assert aux["subband_weights"].shape == (B, cfg.n_subbands)

    target = torch.randint(0, num_classes, (B,), device=device)
    loss = nn.functional.cross_entropy(logits, target)
    loss.backward()

    g = model.subband.gate[0].weight.grad
    assert g is not None and torch.isfinite(g).all(), "subband gate has no valid grad"

    total, trainable = count_params(model)

    model.zero_grad()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    n_iter = 20
    for _ in range(n_iter):
        model.zero_grad()
        out = model(w2v, mel)
        l = nn.functional.cross_entropy(out, target)
        l.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / n_iter * 1000

    print(f"[gated] device={device.type} "
          f"params={total/1e6:.2f}M (trainable {trainable/1e6:.2f}M) "
          f"loss={loss.item():.3f} "
          f"fwd+bwd={dt:.1f}ms/iter (B={B})")


def real_data_test(device, w2v_root, mel_root, num_classes=9):
    """Run one forward pass on a few real .pt samples."""
    if w2v_root is None or mel_root is None:
        print("[real] --w2v_root / --mel_root not provided, skipping")
        return

    w2v_files = sorted(glob.glob(f"{w2v_root}/train/*/*.pt"))[:8]
    mel_files = sorted(glob.glob(f"{mel_root}/train/*/*.pt"))[:8]
    if not w2v_files or not mel_files:
        print("[real] no .pt files found, skipping")
        return

    w2v = torch.stack([torch.load(f, map_location="cpu")["embedding"]
                       for f in w2v_files]).to(device)
    mel = torch.stack([torch.load(f, map_location="cpu")["mel"]
                       for f in mel_files]).to(device)
    print(f"[real] w2v={tuple(w2v.shape)} mel={tuple(mel.shape)}")

    cfg = ModelConfig(num_classes=num_classes)
    model = MultiModalNet(cfg).to(device).eval()
    with torch.no_grad():
        logits = model(w2v, mel)
    pred = logits.argmax(-1)
    print(f"[real] logits={tuple(logits.shape)} preds={pred.tolist()} OK")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--w2v_root", default=None)
    p.add_argument("--mel_root", default=None)
    p.add_argument("--num_classes", type=int, default=9)
    args = p.parse_args()

    torch.manual_seed(0)
    cpu = torch.device("cpu")
    print("=== CPU smoke ===")
    smoke(cpu, num_classes=args.num_classes)

    if torch.cuda.is_available():
        gpu = torch.device("cuda")
        print("\n=== GPU smoke ===")
        smoke(gpu, num_classes=args.num_classes)

    print("\n=== real data test ===")
    real_data_test(
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        args.w2v_root, args.mel_root, num_classes=args.num_classes)

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
