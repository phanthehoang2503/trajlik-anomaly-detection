# TrajLik-AD

## Description

TrajLik-AD trains DCTE and ECTF on cached normal InvAD trajectories, fits
normal-only score calibration, and combines endpoint and path anomaly scores at
evaluation time.

## Input

- A trajectory cache created by `scripts.cache_trajectories`
- The matching InvAD config and checkpoint
- MVTec AD, VisA, or MPDD for evaluation

The main pipeline requires three inversion steps and `projection=none`.

Core pipeline utilities live in `trajlik/`; DCTE and ECTF remain separate
top-level packages.

## Output

Training reserves 5% of normal images for validation and another untouched 5%
for score calibration. It selects the best head and performs early stopping only
from full-trajectory normal validation NLL (`mask=False`). Masked validation is
run separately to monitor MSM. Evaluation produces image-level and pixel-level
metrics, with an optional JSON result file.

Training writes `head.pth` (the calibrated best head), `head_best.pth`,
`head_latest.pth`, and a JSON summary. The latest file is an uncalibrated
training snapshot; evaluation must use `head.pth` or `head_best.pth`. Logs include
train/validation NLL, NLL per dimension, MSM, base NLL, flow log-determinant,
learning rate, duration, and peak GPU memory. The complete epoch history and
split indices are stored in the final checkpoint.

The DCTE output is the terminal-LayerNorm trajectory code specified by the
proposal. ECTF receives that code directly. Training uses a randomly masked step
for the joint path-NLL and MSM objective; validation, calibration, and inference
use the complete trajectory.

## Requirements

- The packages in `requirements.txt`
- A validated normal-only cache containing `cache_meta.json` and
  `cache_index.json`
- A config matching both the InvAD and TrajLik checkpoints

Cache metadata binds the trained head to the exact InvAD checkpoint SHA-256 and
three-step `timestep_map`. Evaluation fails before loading data if either differs
or is missing.

## Usage

Run the following commands from the project root.

Train the TrajLik head:

```bash
python -m scripts.train_trajlik \
    --cache_dir /path/to/cache \
    --output_path results/trajlik/head.pth
```

Defaults are 50 maximum epochs, five-epoch patience, and a 90/5/5
train/validation/calibration split. `lambda_msm=1` is only an initial candidate;
main experiment values must be selected without anomaly labels.

Evaluate the complete pipeline:

```bash
python -m scripts.evaluate_trajlik \
    --config /path/to/config.yml \
    --invad_checkpoint /path/to/model.pth \
    --trajlik_checkpoint results/trajlik/head.pth \
    --output_json results/trajlik/metrics.json
```

Module details: [DCTE](dcte/README.md) and [ECTF](ectf/README.md).
