"""Evaluation script: evaluate the best checkpoint on the test set.

Example:
    python evaluate.py --ckpt ./ckpts/shipsear_gated_nocw_best.pt --device cuda
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from swan.model import MultiModalNet
from swan.dataset import PairedFeatureDataset, collate


@torch.no_grad()
def collect_preds(model, loader, device):
    model.eval()
    all_pred, all_true = [], []
    for w2v, mel, y in loader:
        w2v, mel = w2v.to(device), mel.to(device)
        logits = model(w2v, mel)
        all_pred.append(logits.argmax(-1).cpu())
        all_true.append(y)
    return torch.cat(all_pred).numpy(), torch.cat(all_true).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda",
                   help="cuda / cpu / cuda:N")
    p.add_argument("--save_result", default=None,
                   help="path to save test result json")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    cfg = ckpt["config"]
    saved_args = ckpt.get("args", {})
    w2v_root = saved_args.get("w2v_root")
    mel_root = saved_args.get("mel_root")
    if w2v_root is None or mel_root is None:
        raise ValueError("checkpoint does not contain w2v_root/mel_root; "
                         "please re-train with the updated script")

    model = MultiModalNet(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])

    test_set = PairedFeatureDataset(w2v_root, mel_root, "test")
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate)
    print(f"dataset={ckpt.get('dataset_name', 'unknown')} "
          f"fusion=gated test={len(test_set)}")

    pred, true = collect_preds(model, test_loader, device)

    try:
        from sklearn.metrics import (accuracy_score, f1_score,
                                     classification_report, confusion_matrix)
        class_names = [n for n, _ in sorted(
            ckpt["class_to_idx"].items(), key=lambda x: x[1])]
        acc = accuracy_score(true, pred)
        macro_f1 = f1_score(true, pred, average="macro")
        weighted_f1 = f1_score(true, pred, average="weighted")
        print(f"\n=== TEST RESULT ===")
        print(f"accuracy      = {acc:.4f}")
        print(f"macro-F1      = {macro_f1:.4f}")
        print(f"weighted-F1   = {weighted_f1:.4f}\n")
        print(classification_report(true, pred, target_names=class_names,
                                    digits=4, zero_division=0))
        print("confusion matrix (rows=true, cols=pred):")
        cm = confusion_matrix(true, pred)
        print(cm)

        if args.save_result:
            import json
            result = {
                "dataset": ckpt.get("dataset_name", "unknown"),
                "fusion": "gated",
                "ckpt": str(args.ckpt),
                "test_accuracy": float(acc),
                "macro_f1": float(macro_f1),
                "weighted_f1": float(weighted_f1),
                "per_class": classification_report(
                    true, pred, target_names=class_names,
                    digits=4, zero_division=0, output_dict=True),
                "confusion_matrix": cm.tolist(),
                "class_names": class_names,
            }
            Path(args.save_result).parent.mkdir(parents=True, exist_ok=True)
            with open(args.save_result, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nsaved test result -> {args.save_result}")
    except ImportError:
        acc = (pred == true).mean()
        print(f"\n=== TEST RESULT ===\naccuracy = {acc:.4f} "
              f"(install scikit-learn for full report)")


if __name__ == "__main__":
    main()
