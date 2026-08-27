# Trajectory Cache

This script creates normal-image DDIM trajectories from a trained InvAD model
for use by TrajLik-AD. It extracts EfficientNet features and caches `z_0`,
`z_seq`, `eps_seq`, and `delta_z_seq` for each training image.
Every sample is also marked with `split=train` and `is_normal=true`; the
validator uses these fields instead of dataset-specific path conventions.

## Inputs

- An InvAD YAML config containing the dataset path and model architecture.
- A matching checkpoint, selected with either `--save_dir` or
  `--checkpoint_path`.
- The normal training split of the dataset.

## Usage

Run from the project root.

Smoke test on eight images using the EMA checkpoint:

```bash
python -m scripts.cache_trajectories \
    --config baseline/InversionAD/configs/exp_dit_ad/all.yml \
    --checkpoint_path /path/to/checkpoint.pth \
    --cache_dir /kaggle/working/cache_smoke \
    --batch_size 1 \
    --max_images 8 \
    --force
```

For a full run, remove `--max_images` and change `--cache_dir`. To select an EMA
checkpoint from a training directory instead, replace `--checkpoint_path` with:

```bash
--save_dir /path/to/training/results --use_ema_model
```

The config architecture must match the checkpoint architecture. Cache metadata
records the preprocessing mode, a SHA-256 fingerprint of the complete config,
the exact InvAD checkpoint SHA-256 and size, and the ordered diffusion
`timestep_map`. Evaluation rejects a different checkpoint or schedule even when
the replacement has the same architecture and filename.

## Output

The cache directory groups `.pt` files by category and keeps `cache_meta.json`
and `cache_index.json` at its root. The index supports provenance checks and
category-balanced head training without loading all tensor files:

```text
cache/
├── bottle/
├── cable/
├── metal_nut/
├── cache_meta.json
└── cache_index.json
```

With EfficientNet-B4, three inversion steps, and no projection, each `.pt`
file contains:

```text
z_0:         (272, 16, 16)
z_seq:       (3, 272, 16, 16)
eps_seq:     (3, 272, 16, 16)
delta_z_seq: (3, 272, 16, 16)
a_end_coarse: (16, 16)
```

`a_end_coarse` is always computed from the original, unprojected final latent.
This preserves the official InvAD endpoint score when projection caching is on.

## Validation

```bash
python tests/cache_trajectories_check.py \
    --cache_dir /kaggle/working/cache_val
```

Evaluation rejects metadata that is missing the checkpoint identity or
`timestep_map`, as well as any recorded mismatch.

Use `--force` only when the existing cache may be overwritten.

## Organize an existing flat cache

Preview the moves first:

```bash
python -m scripts.organize_cache --cache_dir /path/to/cache
```

Then group the files by category and update `cache_index.json`:

```bash
python -m scripts.organize_cache --cache_dir /path/to/cache --apply
```
