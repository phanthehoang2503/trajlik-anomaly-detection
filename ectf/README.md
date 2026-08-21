# ECTF

## Description

ECTF is a conditional normalizing flow that assigns a likelihood directly to
each terminal-LayerNorm DCTE trajectory code using the InvAD endpoint energy and
initial feature map.

## Input

- `trajectory_codes`: `[B, H*W, trajectory_dim]`
- `endpoint_energy`: `[B, H, W]`
- `z0`: `[B, C, H, W]`

## Output

`EndpointConditionedTrajectoryFlow.forward()` returns per-position `path_nll`,
`log_prob`, flow latents, log determinants, and conditioning features.

## Requirements

- Python 3.10+
- PyTorch
- DCTE and ECTF must use the same `trajectory_dim`

## Usage

```python
from ectf import EndpointConditionedTrajectoryFlow

flow = EndpointConditionedTrajectoryFlow(
    trajectory_dim=64,
    z0_dim=272,
)
output = flow(trajectory_codes, endpoint_energy, z0)
path_nll = output["path_nll"]
```
