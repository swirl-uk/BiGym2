# Getting started

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # uv, if you don't have it yet
git clone https://github.com/swirl-uk/BiGym2.git && cd BiGym2
uv sync --extra agent
```

`uv sync` makes the environment match exactly the extras you name, so
list every extra you want in one command: a second `uv sync --extra vr`
would remove what `--extra agent` installed.

Run the commands in these docs through `uv run` (`uv run bigym-download
--list`), or activate the environment once with `source .venv/bin/activate`
and drop the prefix. Every command writes its own shell completion script:
`bigym-agent --tyro-write-completion {bash,zsh,tcsh} PATH`.

| Extra | Adds | Needed for |
|---|---|---|
| *(none)* | environments, GR00T-WBC lower body, demonstration loader, evaluation, viewer | using the benchmark |
| `agent` | OpenCV, SciPy, mink | the coding-agent benchmark (`bigym-agent`) |
| `vr` | pyopenxr, mink | VR teleoperation (`bigym-collect`) |
| `lerobot` | LeRobot (and torch) | writing LeRobot datasets (`bigym-export-lerobot`, `bigym-rerender-lerobot`) |

`uv sync` also installs the `dev` dependency group (pytest, pre-commit, ruff, ty)
for working on BiGym itself; `--no-dev` leaves it out.

Reading the published demonstrations does not need `lerobot`: the loader
reads the LeRobot v3 files directly. `vr` and `lerobot` cannot share an
environment (pyopenxr needs `setuptools<70`, LeRobot needs `>=71`). A
collection machine uses `uv sync --extra vr --extra agent` and layers
LeRobot in for a single export:

```bash
uv run --with "lerobot[dataset]>=0.6" bigym-export-lerobot ...
```

The commands in these docs assume headless Linux and set
`MUJOCO_GL=egl`. On macOS, leave `MUJOCO_GL` unset. VR teleoperation needs
Linux (see [Collecting demonstrations](demo_collection.md)).

## First environment

`make(task)` builds the task's official configuration: robot, lower-body
controller, episode budget, warmup and success hold come from the
`EnvConfig` defaults and the task's entry in `bigym.loco.tasks`:

```python
from bigym.loco import make

env = make("move_plate")   # g1_dex1 + groot_wbc_g1, the official configuration
env.config                 # the resolved EnvConfig
```

For experiments of your own, override fields by keyword or pass an
`EnvConfig`. The env records what departs
from the official configuration, and evaluation reports such a run as
unofficial when a departure affects results:

```python
env = make("move_plate", camera_keys=("head",), controller={"cmd_clip": 0.5})
env.config_overrides       # {'controller.cmd_clip': 0.5, 'camera_keys': ('head',)}

env = make("move_plate", env.config.override(camera_shape=(128, 128)))
```

See [The official benchmark configuration](official_configuration.md) for
the full table of frozen values.

The runnable version of this page is
[`examples/loco_random_agent.py`](https://github.com/swirl-uk/BiGym2/blob/main/examples/loco_random_agent.py):

```bash
MUJOCO_GL=egl uv run python examples/loco_random_agent.py
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

`bigym-download --all` (or `--task NAME`) fetches ahead of time; see
[Demos, determinism & evaluation](demos_eval.md).

## What am I controlling?

Everything about the action/observation layout is introspectable — you never
have to count dimensions by hand:

```python
env.wholebody_action_layout()
# {'action_dim': 20, 'base_dim': 4, 'limb_names': (...14 arm joints...),
#  'gripper_count': 2, 'action_low': ..., 'action_high': ..., ...}

env.low_dim_component_slices()
# {'proprioception': (0, 44), 'proprioception_grippers': (44, 46),
#  'proprioception_floating_base': (46, 50)}

env.substrate_fingerprint()
# {'substrate_version': 'bigym2-mj381-v1', 'mujoco_version': '3.8.1', ...}
```

(Those are the `move_plate` values. Tasks with the torso-pitch command
(`ControllerConfig.pitch_command`, set per task by the
registry) have a 21-dim action and a 56-dim low-dimensional observation; see
[Tasks](tasks.md). Read the layout from the env rather than hard-coding
these dimensions.)

Under `base_action_mode="lowerbody_cmd"` the base slots carry locomotion
commands: `[vx, vy, wz]`, or `[vx, vy, height, wz]` when
`enable_all_floating_dof=True` (the official setting). The height slot's
bounds come from the backend's `command_spec` unless explicitly overridden;
the planar and yaw slots use the env's symmetric `cmd_clip` / `wz_clip`.
For GR00T-WBC these are benchmark clips, not published training ranges.
With pitch enabled, inspect `wholebody_action_layout()` for the extra slot's
position and bounds.

## The step loop

`make()` returns a dm_env-style wrapper: `reset(seed=...)` / `step(action)`
return an `ExtendedTimeStep` with `rgb_obs`, `low_dim_obs`, `reward`,
`step_type`. `env.action_spec()` gives shape/dtype/bounds.

## Reset settling

The lower-body policy needs a moment to settle after reset. The official
configuration runs **200** warmup steps (`ControllerConfig.reset_warmup_steps`),
a margin over the backend's measured settle time (180 for `groot_wbc_g1`);
`controller={"reset_warmup_steps": None}` uses the backend's value.

Demos are recorded from the post-settle engage moment, so evaluation must
settle too: never set this to 0.
