"""Paired dataset loading pre-extracted wav2vec embeddings and mel spectrograms.

The two feature types share the same relative path layout:
    <w2v_root>/train/<class>/<name>.pt -> {"embedding": [T, 768], "label": int}
    <mel_root>/train/<class>/<name>.pt -> {"mel": [128, 157], "label": int}

Training uses a stratified 85/15 train/val split from the train set.
"""
from pathlib import Path

import torch
from torch.utils.data import Dataset


class PairedFeatureDataset(Dataset):
    def __init__(self, w2v_root, mel_root, split, rel_paths=None, cache=False):
        """
        Args:
            w2v_root: root directory of wav2vec embeddings.
            mel_root: root directory of mel spectrograms.
            split: "train" or "test".
            rel_paths: optional list of relative paths to use (for train/val subsets).
            cache: if True, preload all samples into memory.
        """
        self.w2v_root = Path(w2v_root)
        self.mel_root = Path(mel_root)
        self.split = split
        self.cache = cache
        self._cache = {}

        if rel_paths is None:
            rel_paths = self.discover(self.w2v_root, self.mel_root, split)
        self.rel_paths = list(rel_paths)
        if not self.rel_paths:
            raise RuntimeError(f"no paired samples for split={split}")

        self.class_to_idx = torch.load(
            self.w2v_root / "class_to_idx.pt", weights_only=False)

    @staticmethod
    def discover(w2v_root, mel_root, split):
        w2v_root, mel_root = Path(w2v_root), Path(mel_root)
        w_rel = {p.relative_to(w2v_root)
                 for p in w2v_root.glob(f"{split}/*/*.pt")}
        m_rel = {p.relative_to(mel_root)
                 for p in mel_root.glob(f"{split}/*/*.pt")}
        return sorted(w_rel & m_rel)

    def labels(self):
        """Return label for each sample, used for stratified splitting."""
        c2i = torch.load(self.w2v_root / "class_to_idx.pt", weights_only=False)
        return [c2i[rp.parent.name] for rp in self.rel_paths]

    def __len__(self):
        return len(self.rel_paths)

    def __getitem__(self, idx):
        if self.cache and idx in self._cache:
            return self._cache[idx]

        rp = self.rel_paths[idx]
        w_item = torch.load(self.w2v_root / rp, weights_only=False)
        m_item = torch.load(self.mel_root / rp, weights_only=False)

        w2v = w_item["embedding"].float()
        mel = m_item["mel"].float()
        label = int(w_item["label"])

        if int(m_item["label"]) != label:
            raise RuntimeError(f"label mismatch at {rp}")

        sample = (w2v, mel, label)
        if self.cache:
            self._cache[idx] = sample
        return sample


def collate(batch):
    w2v = torch.stack([b[0] for b in batch])
    mel = torch.stack([b[1] for b in batch])
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return w2v, mel, labels


def stratified_train_val_split(dataset, val_ratio=0.15, seed=42):
    """Stratified 85/15 train/val split from the train set.

    Returns:
        (train_rel_paths, val_rel_paths)
    """
    import random
    rng = random.Random(seed)

    labels = dataset.labels()
    by_class = {}
    for rp, y in zip(dataset.rel_paths, labels):
        by_class.setdefault(y, []).append(rp)

    train_paths, val_paths = [], []
    for y, paths in sorted(by_class.items()):
        paths = list(paths)
        rng.shuffle(paths)
        n_val = max(1, round(len(paths) * val_ratio))
        n_val = min(n_val, len(paths) - 1) if len(paths) > 1 else 0
        val_paths.extend(paths[:n_val])
        train_paths.extend(paths[n_val:])

    rng.shuffle(train_paths)
    rng.shuffle(val_paths)
    return train_paths, val_paths
