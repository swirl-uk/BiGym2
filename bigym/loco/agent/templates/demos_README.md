
## The demonstrations

{n_episodes} episodes recorded by a human teleoperating the same robot and controller
in VR, under `demos/`. Each `*.npz` holds one episode at 50 Hz. They contain only what a
policy could observe itself; object and fixture poses are **not** included.

| key | shape | meaning |
|---|---|---|
| `raw_outer_action` | (T, {action_dim}) | the physical action the operator sent each step, same layout as your `act` output{pitch_note} |
| `action` | (T, {action_dim}) | the same action normalised to [-1, 1] with `demos/metadata.json` `action_stats` |
| `low_dim_obs` | (T, {state_dim}) | proprioceptive vector (same as `obs["low_dim_obs"]`) |
| `rgb_obs` | (T, 3, 3, {cam_height}, {cam_width}) | head / right wrist / left wrist images |
| `reward`, `demo`, `is_expert`, `event_progress` | (T, 1) | bookkeeping |
| `seed` | (1,) | the seed of that episode (a development seed you can reset to) |

```python
import numpy as np
ep = np.load("demos/<file>.npz")
raw = ep["raw_outer_action"]          # (T, {action_dim})
print(raw[:, :4].round(2))            # base commands over time
```

They are for reading. Do not fit anything to them: the policy is hand-written
control logic, and the demonstrations are not available to `policy.py` at evaluation.
