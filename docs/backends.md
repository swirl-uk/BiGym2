# Lower-body backend

A *backend* is a frozen RL locomotion policy wrapped in the
{class}`~bigym.loco.controller.LowerBodyController` contract. It owns a set
of joints (legs, usually plus waist), consumes a velocity-style command every
control step, and produces joint position targets. The env routes the outer
action's base slots to the backend and keeps its joints out of the agent's
action space.

## `groot_wbc_g1`

| backend | robot | policy | weights | measured walk at cmd vx=0.35 |
|---|---|---|---|---|
| `groot_wbc_g1` | `g1_dex1` | NVIDIA GR00T-WBC (Decoupled WBC, Balance + Walk) | packaged official ONNX | 0.302 ± 0.002 m/s |

BiGym 2.0 ships one backend, `groot_wbc_g1`, and the benchmark uses it on
`g1_dex1` (see
[The official benchmark configuration](official_configuration.md)). The
walking speed above is a measurement of the checkpoint, not a target: the
policy tracks its velocity command closely.

`groot_wbc_g1` wraps NVIDIA's frozen Decoupled WBC, the lower-body
controller used by the GR00T N1.5/N1.6 models {cite:p}`nvidia2025gr00tn1`,
from the
[GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)
repository, pinned to a specific commit. The policy weights are unmodified;
the runtime is adapted to numpy (see `bigym/loco/adapters/_groot/PROVENANCE.md`). If you use it,
please cite that work.

```python
from bigym.loco import make

make("move_plate")                                   # controller backend groot_wbc_g1
make("move_plate", controller={"backend": "groot_wbc_g1", "cmd_clip": 0.5})
```

An unknown backend name, or a robot the backend does not support, raises an
error that says why (see
{func}`bigym.loco.adapters.resolve_backend_name`). To add a backend, see
[Adding a backend](#adding-a-backend).

## Command bounds

Each backend declares a typed {class}`~bigym.loco.command.CommandSpec`.
GR00T-WBC's upstream does not publish its training command ranges, so its
adapter declares benchmark clips instead. A command inside those bounds is
not guaranteed to be inside the training distribution.

The env derives the outer height/pitch bounds from this spec unless they are
overridden in config. Either source changes action normalization, so match
the resolved bounds to the demonstration metadata.

## Configuration reference

The controller is configured by {class}`bigym.loco.config.ControllerConfig`
(`EnvConfig.controller`; `None` is a floating base without legs). Common
fields, with their official defaults:

| field | default | meaning |
|---|---|---|
| `backend` | `"groot_wbc_g1"` | backend name |
| `base_action_mode` | `"lowerbody_cmd"` | routes the base slots to velocity/height commands |
| `pitch_command` | `True` | adds the torso-pitch slot (off for six tasks) |
| `reset_warmup_steps` | `200` | post-reset settle steps before the agent engages; `None` uses the adapter's recommendation (180) |
| `deterministic_reset` | `True` | every reset restores the controller state of the first reset |
| `cmd_clip` / `wz_clip` | 1.0 | symmetric clips on `vx`/`vy` and `wz` |
| `height_cmd_min/max`, `pitch_cmd_min/max` | `None` (from `command_spec`) | override the outer height and pitch slot bounds |
| `default_height_cmd` / `init_pelvis_z` | 0.74 / 0.74 | standing height command and reset pelvis height |
| `model_path` | `None` (the packaged ONNX pair) | comma-separated Balance and Walk policy files |

`passive_base_tilt` (default `True`) adds unactuated pelvis roll/pitch
joints so the free-base policy keeps its trained balance dynamics, without
adding action slots or floating-base proprioception. The benchmark
configuration uses it, and the substrate fingerprint records it.

## Adding a backend

Subclass {class}`~bigym.loco.base.LowerBodyBase`. The base class handles the
shared pipeline: joint address resolution, reset pose, failure detection,
command clipping, and **declarative replay state** (declare
`STATEFUL = {"cmd": "command", "height_cmd": "height_command", ...}`, snapshot
key to attribute, and you get bit-exact `get_state` /
`set_state`, with a `KeyError` if a snapshot misses a field). You implement:

1. `__init__`: load the policy and set the attributes listed on
   `LowerBodyBase`: `controlled_joints`, `command_spec`, `control_dt`, the
   joint ranges (via `build_joint_ranges`), the command clips and the
   latched command.
2. `reset()`: put the joints in the reset pose (via `apply_pose`) and clear
   the mutable state.
3. `step()`: build the observation, run the policy, and return joint position
   targets in `controlled_joints` order.
4. `get_base_obs()`: base-frame linear velocity, angular velocity and
   projected gravity. Frame conventions differ between policies, so the base
   class cannot provide this.

Then bind it to the env with a {class}`~bigym.loco.BackendBinding`: the
robot models it drives, the robot class it needs (which joints become
actuated, their PD gains) and how to build the controller.
[`bigym/loco/adapters/groot_wbc.py`](https://github.com/swirl-uk/BiGym2/blob/main/bigym/loco/adapters/groot_wbc.py)
is a complete example, including explicit state snapshots for the external
policy. To add a backend from your own package, see
[Building on BiGym](extending.md#4-add-a-lower-body-controller).

`CommandKind.VELOCITY` is the command kind every current backend uses;
`EE_POSE` (whole-body SE(3) targets) and `MOTION_REF` (reference-motion
trackers) are reserved for other kinds of controller.

Citation entries are in the [global bibliography](index.md#citing).
