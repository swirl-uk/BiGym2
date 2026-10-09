# Lower-body backend

A *backend* is a frozen RL locomotion policy behind the
{class}`~bigym.loco.controller.LowerBodyController` contract. Every control
step it reads a velocity-style command and returns position targets for the
joints it owns, the legs and usually the waist. The env routes the base slots
of the outer action to the backend, and its joints stay out of the agent's
action space.

## `groot_wbc_g1`

| backend | robot | policy |
|---|---|---|
| `groot_wbc_g1` | `g1_dex1` | NVIDIA GR00T-WBC (Decoupled WBC, Balance + Walk) |

BiGym 2.0 ships one backend, and the benchmark uses it (see
[Official configuration](official_configuration.md)). Commanded 0.35 m/s, it
walks 0.30 m/s.

`groot_wbc_g1` runs NVIDIA's frozen Decoupled WBC policy
{cite:p}`nvidia2025gr00tn1` from
[GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)
with the weights unmodified.

```python
from bigym.loco import make

make("move_plate")                                   # controller backend groot_wbc_g1
make("move_plate", controller={"backend": "groot_wbc_g1", "cmd_clip": 0.5})
```

## Command bounds

Each backend declares a typed {class}`~bigym.loco.command.CommandSpec`.
GR00T-WBC's training command ranges are not published, so its adapter
declares benchmark clips. A command inside them may still be outside the
training distribution.

The env derives the outer height and pitch bounds from this spec unless the
config overrides them. Both change action normalization, so the resolved
bounds must match the demonstration metadata.

## Configuration reference

{class}`bigym.loco.config.ControllerConfig` (`EnvConfig.controller`)
configures the controller. `controller=None` builds a floating base without
legs. Common fields, with their official defaults:

| field | default | meaning |
|---|---|---|
| `backend` | `"groot_wbc_g1"` | backend name |
| `base_action_mode` | `"lowerbody_cmd"` | base slots carry velocity and height commands |
| `pitch_command` | `True` | adds the torso-pitch slot (off for six tasks) |
| `reset_warmup_steps` | `200` | settle steps after reset before the agent engages |
| `deterministic_reset` | `True` | every reset restores the same controller state, so `reset(seed)` depends only on the seed |
| `passive_base_tilt` | `True` | unactuated pelvis roll and pitch joints keep the policy's trained balance dynamics. Adds no action slots and no floating-base proprioception |
| `cmd_clip` / `wz_clip` | 1.0 | symmetric clips on `vx`/`vy` and `wz` |
| `height_cmd_min/max`, `pitch_cmd_min/max` | `None` (from `command_spec`) | override the outer height and pitch slot bounds |
| `default_height_cmd` / `init_pelvis_z` | 0.74 / 0.74 | standing height command and reset pelvis height, in m |
| `model_path` | `None` (the packaged ONNX pair) | comma-separated Balance and Walk policy files |

## Adding a backend

Subclass {class}`~bigym.loco.base.LowerBodyBase`. It resolves joint
addresses, applies the reset pose, detects falls, clips commands and
snapshots replay state.

For replay state, declare `STATEFUL`, a map from snapshot key to attribute:
`STATEFUL = {"cmd": "command", "height_cmd": "height_command", ...}`. The
base class then provides bit-exact `get_state` and `set_state`, and
`set_state` raises `KeyError` when a snapshot misses a key. List every
attribute that shapes future targets.

You implement:

1. `__init__`: load the policy and set the attributes listed on
   `LowerBodyBase`: `controlled_joints`, `command_spec`, `control_dt`, the
   joint ranges (via `build_joint_ranges`), the command clips and the
   latched command.
2. `reset()`: put the joints in the reset pose (via `apply_pose`) and clear
   the mutable state.
3. `step()`: build the observation, run the policy, and return joint position
   targets in `controlled_joints` order.
4. `get_base_obs()`: base-frame linear velocity, angular velocity and
   projected gravity. Policies use different frame conventions, so the base
   class leaves this to you.

[`bigym/loco/adapters/groot_wbc.py`](https://github.com/swirl-uk/BiGym2/blob/main/bigym/loco/adapters/groot_wbc.py)
is a complete example. To put the controller in the env with a
`BackendBinding` and select it from your own package, see
[Building on BiGym 2.0](extending.md#add-a-lower-body-controller).
