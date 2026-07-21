"""Training script.

Example:
    python train.py \
        --w2v_root /path/to/w2v_embeddings \
        --mel_root /path/to/mel_spectrograms \
        --num_classes 9 \
        --dataset_name shipsear \
        --epochs 100 --device cuda
"""
import argparse
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from swan.config import ModelConfig
from swan.model import MultiModalNet
from swan.dataset import PairedFeatureDataset, collate, stratified_train_val_split


@torch.no_grad()
def run_eval(model, loader, device, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for w2v, mel, y in loader:
        w2v, mel, y = w2v.to(device), mel.to(device), y.to(device)
        logits = model(w2v, mel)
        loss_sum += criterion(logits, y).item() * y.size(0)
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--w2v_root", required=True,
                   help="root directory of pre-extracted wav2vec embeddings")
    p.add_argument("--mel_root", required=True,
                   help="root directory of pre-computed mel spectrograms")
    p.add_argument("--num_classes", type=int, required=True,
                   help="number of classes")
    p.add_argument("--dataset_name", default="custom",
                   help="dataset name used only for logging and checkpoint naming")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--step_size", type=int, default=20,
                   help="StepLR decay interval (epochs)")
    p.add_argument("--gamma", type=float, default=0.5,
                   help="StepLR decay factor")
    p.add_argument("--patience", type=int, default=15,
                   help="early stopping patience (epochs)")
    p.add_argument("--no_early_stop", action="store_true",
                   help="disable early stopping")
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda",
                   help="cuda / cpu / cuda:N")
    p.add_argument("--gpus", type=str, default=None,
                   help="GPU ids, e.g. '0' or '0,1,2'; enables DataParallel if multiple")
    p.add_argument("--class_weight", action="store_true",
                   help="inverse-frequency weighting for CrossEntropyLoss")
    p.add_argument("--cache", action="store_true",
                   help="preload dataset into RAM")
    p.add_argument("--out_dir", default="./ckpts")
    p.add_argument("--run_tag", type=str, default="",
                   help="suffix added to checkpoint filename")
    args = p.parse_args()

    torch.manual_seed(args.seed)

    multi_gpu = False
    gpu_ids = []
    if args.gpus is not None and torch.cuda.is_available():
        gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
        device = torch.device(f"cuda:{gpu_ids[0]}")
        multi_gpu = len(gpu_ids) > 1
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device} multi_gpu={multi_gpu} gpu_ids={gpu_ids or 'auto'}")

    full_train = PairedFeatureDataset(args.w2v_root, args.mel_root, "train")
    train_paths, val_paths = stratified_train_val_split(
        full_train, val_ratio=args.val_ratio, seed=args.seed)
    assert len(train_paths) > 0 and len(val_paths) > 0, "train/val split is empty"
    train_set = PairedFeatureDataset(
        args.w2v_root, args.mel_root, "train", rel_paths=train_paths,
        cache=args.cache)
    val_set = PairedFeatureDataset(
        args.w2v_root, args.mel_root, "train", rel_paths=val_paths,
        cache=args.cache)
    print(f"dataset={args.dataset_name} | train={len(train_set)} "
          f"val={len(val_set)} | classes={args.num_classes}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate, drop_last=False)
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate)

    cfg = ModelConfig(num_classes=args.num_classes)
    model = MultiModalNet(cfg).to(device)
    if multi_gpu:
        model = nn.DataParallel(model, device_ids=gpu_ids)

    if args.class_weight:
        import numpy as np
        counts = np.bincount(train_set.labels(),
                             minlength=args.num_classes).astype(float)
        counts[counts == 0] = 1.0
        w = counts.sum() / (len(counts) * counts)
        class_weights = torch.tensor(w, dtype=torch.float32, device=device)
        print(f"class_weight enabled: {w.round(3).tolist()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        print("class_weight disabled: plain CrossEntropyLoss")
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "cw" if args.class_weight else "nocw"
    suffix = f"_{args.run_tag}" if args.run_tag else ""
    ckpt_path = out_dir / f"{args.dataset_name}_gated_{tag}{suffix}_best.pt"

    best_val_acc, best_state, best_epoch = -1.0, None, -1
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for w2v, mel, y in train_loader:
            w2v, mel, y = w2v.to(device), mel.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(w2v, mel)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * y.size(0)
            tr_correct += (logits.argmax(-1) == y).sum().item()
            tr_total += y.size(0)
        scheduler.step()

        val_loss, val_acc = run_eval(model, val_loader, device, criterion)
        tr_acc = tr_correct / tr_total
        dt = time.time() - t0

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc, best_epoch = val_acc, epoch
            core = model.module if isinstance(model, nn.DataParallel) else model
            best_state = copy.deepcopy(core.state_dict())
            no_improve = 0
            torch.save({
                "state_dict": best_state,
                "config": cfg,
                "dataset_name": args.dataset_name,
                "num_classes": args.num_classes,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "class_to_idx": full_train.class_to_idx,
                "args": vars(args),
            }, ckpt_path)
        else:
            no_improve += 1

        print(f"epoch {epoch:3d} | lr {scheduler.get_last_lr()[0]:.2e} | "
              f"train loss {tr_loss/tr_total:.4f} acc {tr_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
              f"{dt:.1f}s{'  *best' if improved else ''}")

        if not args.no_early_stop and no_improve >= args.patience:
            print(f"early stop at epoch {epoch} "
                  f"(no val improvement for {args.patience} epochs)")
            break

    torch.save({
        "state_dict": best_state,
        "config": cfg,
        "dataset_name": args.dataset_name,
        "num_classes": args.num_classes,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "class_to_idx": full_train.class_to_idx,
        "args": vars(args),
    }, ckpt_path)
    print(f"\nbest val acc={best_val_acc:.4f} @epoch {best_epoch} "
          f"-> saved {ckpt_path}")


if __name__ == "__main__":
    main()
