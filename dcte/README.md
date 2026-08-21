# DCTE

## Description

DCTE converts a three-step InvAD trajectory at each spatial position into one
trajectory code. During training it can mask one step for the MSM objective.

## Input

A cached or online trajectory containing:

- `z_0`: `[B, C, H, W]`
- `z_seq`, `eps_seq`, `delta_z_seq`: `[B, 3, C, H, W]`

The canonical `states`, `epsilons`, and `deltas` form is also accepted.

## Output

`DCTE.forward()` returns terminal-LayerNorm `trajectory_codes` with shape
`[B, H*W, trajectory_dim]` and the intermediate tensors required by MSM.

## Requirements

- Python 3.10+
- PyTorch
- Exactly three trajectory steps for the main TrajLik-AD protocol

## Usage

```python
from dcte import DCTE

model = DCTE(input_dim=272, num_steps=3)
output = model(trajectory_batch, mask=False)
codes = output["trajectory_codes"]
```
