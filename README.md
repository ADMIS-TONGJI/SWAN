# SWAN

Heterogeneous Fusion of Spectrogram and Waveform with Sub-band Adaptation for Underwater Acoustic Target Recognition

## Overview

SWAN fuses two branches for underwater acoustic classification:

- **raw waveform branch**: pre-extracted, frozen wav2vec2 last-hidden-state embeddings, time-pooled.
- **spectrogram branch**: processed by an sub-band adaptive spectrogram learning module and a compression module.

The two branch embeddings are projected to the same dimension and fused with a **gated fusion** module, followed by an MLP classifier.

## Project structure

```
SWAN/
├── swan/                  # model package
│   ├── config.py          # ModelConfig
│   ├── model.py           # MultiModalNet
│   ├── wav2vec_branch.py
│   ├── subband_reweight.py
│   ├── mel_compression.py
│   ├── projection.py
│   ├── fusion.py          # GatedFusion
│   ├── classifier.py
│   └── dataset.py
├── train.py               # training script
├── evaluate.py            # evaluation script
├── README.md
└── requirements.txt
```

## Requirements

```bash
pip install -r requirements.txt
```

## Data preparation

The training scripts expect pre-extracted features with this layout:

```
<w2v_root>/
  ├── class_to_idx.pt
  ├── train/
  │   └── <class>/<name>.pt   # dict: {"embedding": [T, 768], "label": int}
  └── test/
      └── <class>/<name>.pt

<mel_root>/
  ├── train/
  │   └── <class>/<name>.pt   # dict: {"mel": [128, 157], "label": int}
  └── test/
      └── <class>/<name>.pt
```

Relative paths under `train/` and `test/` must match between `w2v_root` and `mel_root`.

## Training

```bash
python train.py \
    --w2v_root /path/to/w2v_embeddings \
    --mel_root /path/to/mel_spectrograms \
    --num_classes 9 \
    --dataset_name shipsear \
    --epochs 100 \
    --device cuda
```

Optional flags:

- `--cache`: preload all samples into RAM.
- `--out_dir ./ckpts`: checkpoint output directory.

Checkpoints are saved as `<dataset_name>_gated_<cw|nocw>[_<run_tag>]_best.pt`.

## Evaluation

```bash
python evaluate.py \
    --ckpt ./ckpts/shipsear_gated_nocw_best.pt \
    --device cuda
```

## Citation

If you use this code, please cite the corresponding work.
