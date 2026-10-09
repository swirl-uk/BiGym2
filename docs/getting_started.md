# Getting started

[Install](installation.md) BiGym 2.0 first.

## Make and reset

`make(task)` builds the task's official configuration: the Unitree G1 with
Dex1-1 two-finger grippers (`g1_dex1`), the GR00T-WBC lower body
(`groot_wbc_g1`), three 84×84 cameras, and the task's episode budget and
success hold.

```python
from bigym.loco import make

env = make("move_plate")
timestep = env.reset(seed=620000)
env.config                    # the resolved EnvConfig
```

`reset` runs a warmup before it returns: the lower-body controller steps
with the arms held while the robot settles into its stance. Your policy
takes over after the warmup. `reset_warmup_steps` defaults to 200 control
steps (4 s). Never set it to 0. The demonstrations start after the warmup,
so evaluation has to warm up too.

## One step

`make` returns a dm_env-style wrapper. `step(action)` advances one 50 Hz
control step and returns an `ExtendedTimeStep` with `rgb_obs`,
`low_dim_obs`, `reward` and `step_type`.

```python
import numpy as np

spec = env.action_spec()                        # shape (20,), float32
action = np.zeros(spec.shape, dtype=spec.dtype)
timestep = env.step(action)
timestep.rgb_obs.shape        # (3, 3, 84, 84): head, right_wrist, left_wrist
timestep.low_dim_obs.shape    # (50,)
timestep.step_type.name       # 'MID'
```

Actions are normalized to [-1, 1] per slot, and the env maps each slot onto
its range. A zero action commands zero base velocity and the middle of every
other range.

## What am I controlling?

```python
env.wholebody_action_layout()
# {'action_dim': 20, 'base_dim': 4, 'limb_names': (...14 arm joints...),
#  'gripper_count': 2, 'action_low': ..., 'action_high': ..., ...}

env.low_dim_component_slices()
# {'proprioception': (0, 44), 'proprioception_grippers': (44, 46),
#  'proprioception_floating_base': (46, 50)}
```

The four base slots are `[vx, vy, height, wz]`: forward and lateral velocity
in m/s, absolute pelvis height in m and yaw rate in rad/s. The 14 arm slots
are absolute joint position targets. The 2 gripper slots open (0) or close
(1) each hand.

Those are the `move_plate` values. The 14 tasks with the torso-pitch command
add a fifth base slot and have a 21-dim action and a 56-dim low-dimensional
observation. Read the layout from the env.
[Official configuration](official_configuration.md#action-layout) has the
full tables.

## Overrides

Keyword overrides or an `EnvConfig` change any setting. The env records what
departs from the official configuration, and evaluation reports a run as
unofficial when a departure affects results:

```python
env = make("move_plate", camera_keys=("head",), controller={"cmd_clip": 0.5})
env.config_overrides       # {'controller.cmd_clip': 0.5, 'camera_keys': ('head',)}

env = make("move_plate", env.config.override(camera_shape=(128, 128)))
```

## Demonstrations

```python
from bigym.loco import make_gym

env = make_gym("move_plate")
demos = env.get_demos(60)     # downloads move_plate's demos on first use
demo = demos[0]
demo["obs"]["rgb"].shape      # (T + 1, 3, 3, 84, 84)
demo["action"].shape          # (T, 20), in env.action_space
```

`make_gym` returns a gymnasium env with the same task.
`bigym-download --all` fetches every task ahead of time. See
[Demonstrations](demonstrations.md).

## Check the install

```bash
MUJOCO_GL=egl uv run python examples/loco_random_agent.py
```

It ends with one line per seed, such as
`seed 620000: total reward 0.000, last step MID`.
