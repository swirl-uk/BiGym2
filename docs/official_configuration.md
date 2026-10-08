# The official benchmark configuration

Everything a leaderboard entry must match. Every value below is stored in
code — this page only assembles it, and each row names the module that owns
it. Anything not listed here (network sizes, optimizers, action-chunk
lengths) belongs to your method, not to the benchmark.

````{admonition} One call gets you all of it
:class: tip
{func}`bigym.loco.make` builds the official env for a task:

```python
from bigym.loco import make

env = make("move_plate")        # official embodiment, budget, criteria
```

The defaults of {class}`bigym.loco.EnvConfig` are the official values, and
each task's `TaskSpec` in {mod}`bigym.loco.tasks` adds only its episode
budget and its few differences (pitch command off, reach tolerance,
initialization profile). `env.config` is the resolved configuration.

`make(task, config=None, **overrides)` takes an `EnvConfig` or keyword
overrides; nested controller settings merge into the
task's own (`make("pick_box", controller={"reset_warmup_steps": 0})`).
`env.config_overrides` lists every field that departs from the task's
official configuration, and evaluation refuses a leaderboard record when any
of them affects results (see [Evaluation protocol](#evaluation-protocol)).
`make(task, controller=None)` builds a floating-base research env without the
lower-body controller; no released demonstrations were recorded in it.
````

## The configuration table

| Item | Official value | Where it lives in code |
|---|---|---|
| Robot | `g1_dex1` — Unitree G1 29-DoF with Dex1-1 parallel two-finger grippers | `EnvConfig.robot_model`; MJCF `bigym/envs/xmls/g1/g1_29dof_with_dex1_1.xml` |
| Lower-body backend | `groot_wbc_g1` — NVIDIA GR00T-WBC (frozen Balance + Walk ONNX pair) | `ControllerConfig.backend`; adapter `bigym/loco/adapters/groot_wbc.py` |
| Base action mode | `lowerbody_cmd` (base slots are velocity/height **commands**, not position deltas) | `ControllerConfig.base_action_mode` |
| Lower-body controller | on: the legs support the base | `EnvConfig.controller` (`None` is the floating-base research env) |
| Passive pelvis roll/pitch | `True` (unactuated tilt DoFs, seen only through the policy's pelvis IMU) | `ControllerConfig.passive_base_tilt`, `bigym/loco/adapters/groot_wbc.py` |
| Spawn stance | `keyframe` | `ControllerConfig.init_stance` |
| Control rate | **50 Hz** (`control_step_seconds = 0.02`; the G1 MJCF compiles at `timestep=0.001`, so one control step is 20 substeps) | substep count derived from the compiled timestep in `bigym/bigym_env.py`; stamped as `control_step_seconds` by `env.substrate_fingerprint()` |
| Cameras | 3 × RGB 84×84: `head`, `right_wrist`, `left_wrist` (`rgb_obs` shape `(3, 3, 84, 84)`, uint8) | `EnvConfig.camera_keys`, `EnvConfig.camera_shape` |
| Torso-pitch command | on, except for six tasks (the three reach tasks, `move_plate`, `drawer_top_open`, `drawer_top_close`) | `ControllerConfig.pitch_command`; the exceptions in `bigym.loco.tasks.TASKS` |
| Low-dim observation | 50 floats, or 56 with the pitch command | `env.low_dim_component_slices()` |
| Action | 20 floats, or 21 with the pitch command | `env.wholebody_action_layout()` |
| Outer floating DoFs | `enable_all_floating_dof=True` → 4 base slots (`vx, vy, height, wz`); the pitch command adds a fifth | `EnvConfig.enable_all_floating_dof` |
| Success hold | the task predicate holds for **1.0 s** (50 consecutive control steps) before the task reward is given; see [Evaluation protocol](#evaluation-protocol) | `EnvConfig.success_hold_seconds` |
| Success hold during collection | **3.0 s** hold (stricter; every demo tail carries ~3 s of stay-still supervision), trimmed to 1.0 s for training | `CollectConfig.collect_success_hold_seconds` (`bigym.vr.collect.config`) |
| Reach tolerance | **0.05 m** for `reach_target_single` / `_dual` / `_multi_modal` (the target-ball radius: the pinch centre must be inside the ball), instead of the class constant 0.1 | `EnvConfig.reach_tolerance`, set per task in `bigym.loco.tasks.TASKS` and applied in `bigym/envs/reach_target.py` |
| Demo downsample rate | 10 (one outer step per 10 env steps) | `EnvConfig.demo_down_sample_rate` |
| Reset warmup | **200** control steps before the agent engages | `ControllerConfig.reset_warmup_steps` |
| Initialization profile | `upstream`, except `g1_id_v1` for `drawer_top_open` / `drawer_top_close` | `EnvConfig.initialization_profile`; the exceptions in `bigym.loco.tasks.TASKS` |
| Episode budget | per task, see [Episode budgets](#episode-budgets) | `TaskSpec.episode_length` |
| Substrate | `bigym2-mj381-v1`, MuJoCo 3.8.1, Newton solver pin on G1 scenes | `bigym/loco/fingerprint.py::SUBSTRATE_VERSION`; stamped by `env.substrate_fingerprint()` |
| Demonstrations | 60 successful human VR demos per task | see [Demos & evaluation](demos_eval.md) |

Keep the official `reset_warmup_steps=200`. The adapter's own
`recommended_reset_warmup_steps` (180 for `groot_wbc_g1`) is the measured
settle time; the official value adds a margin. Do not set it to 0: demos start at
the post-settle engage moment, so evaluation has to settle too.

## Observation layout

The tables below describe a task without the pitch command, such as
`move_plate`. The differences with pitch are listed after each table.

```python
env.low_dim_component_slices()
# {'proprioception': (0, 44),
#  'proprioception_grippers': (44, 46),
#  'proprioception_floating_base': (46, 50)}
```

| Slice | Dims | Contents |
|---|---|---|
| `proprioception` | 0–44 | arm and finger joint positions and pelvis x, y, z, yaw (0–21), then their velocities (22–43). Leg and waist joints are **excluded**: the backend owns them |
| `proprioception_grippers` | 44–46 | left / right gripper state, 0 = open, 1 = closed |
| `proprioception_floating_base` | 46–50 | pelvis x, y, z, yaw as tracked by the base controller |

With the pitch command, the three waist joints (yaw, roll, pitch) are added
at the front of `proprioception` with their velocities, so the slices become
`(0, 50)`, `(50, 52)` and `(52, 56)`: a policy that commands the lean can
see it.

Visual observation: `rgb_obs` is `(3, 3, 84, 84)` uint8 — camera-major, then
CHW, in the order `head`, `right_wrist`, `left_wrist`.

## Action layout

20 floats, produced by `env.wholebody_action_layout()` (bounds below are the
resolved values for `g1_dex1 + groot_wbc_g1`):

| Slot(s) | Name | Range | Meaning |
|---|---|---|---|
| 0 | `vx` | [-1.0, 1.0] m/s | body-frame forward velocity command |
| 1 | `vy` | [-1.0, 1.0] m/s | body-frame lateral velocity command |
| 2 | `height` | [0.4, 1.0] m | absolute pelvis-height command (benchmark clip in `command_spec`) |
| 3 | `wz` | [-1.0, 1.0] rad/s | yaw rate command |
| 4–10 | left arm | joint limits | shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw |
| 11–17 | right arm | joint limits | same order, right side |
| 18 | left gripper | [0, 1] | 0 = open, 1 = closed |
| 19 | right gripper | [0, 1] | 0 = open, 1 = closed |

Base slots 0–3 are the outer floating DoFs `X, Y, Z, RZ` in that order — the
planar/height/yaw channels, not position deltas. The `vx`/`vy`/`wz` bounds
are the env's symmetric command clips (`cmd_clip` / `wz_clip`, default 1.0);
the height bounds come from the backend's `command_spec`. GR00T-WBC's
upstream does not publish training command ranges, so these are benchmark
clips, not evidence of the training distribution.
Arm slots are absolute joint position targets in radians.

With the pitch command, slot 4 is the torso pitch (radians, absolute,
+ = lean forward, [-0.2, 0.8]) and the arm and gripper slots move up by one
(left arm 5–11, right arm 12–18, grippers 19 and 20).

## Episode budgets

Budgets and their provenance live in `bigym.loco.tasks`, not in a
duplicate table here. Values are **env steps**; divide by
`demo_down_sample_rate` for outer control steps.

```python
from bigym.loco.tasks import TASKS, budget_provenance

for task, spec in sorted(TASKS.items()):
    print(task, spec.episode_length, budget_provenance(task))
```

The 20 benchmark tasks are marked `data_derived`: their budget is
`bigym.loco.tasks.BUDGET_RULE` (2x the longest successful demonstration,
rounded up to the next 500 env steps, minimum 2000) applied to the task's 60
demonstrations, and each dataset records the rule as
`recommended_episode_length_rule`. The other 20 registered tasks, which will
be released later, carry upstream floating-base placeholders
(`upstream_placeholder`). The collector's longer safety cap is not a
benchmark episode budget.

## Evaluation protocol

| Item | Value | Where |
|---|---|---|
| Episodes per evaluation | 100 | `bigym/loco/eval/protocol.py::EVAL_EPISODES` |
| Seeds | one contiguous block, episode *i* uses `620000 + i` | `EVAL_SEED_BASE`, `eval_seeds()` |
| Checkpoint spacing | every 5000 training steps, all evaluated | reporting convention |
| **Headline number** | **mean success rate over the last 5 checkpoints, ± standard error** | reporting convention |
| Appendix numbers | final checkpoint alone; per-checkpoint maximum (labelled as coming from the same seed block) | reporting convention |
| Record schema | `leaderboard_record()`, which embeds `substrate_fingerprint()` and refuses non-contiguous seed sets | `bigym/loco/eval/protocol.py` |
| Configuration check | `protocol_violations(env)` — lists every result-affecting field of `env.config` that departs from the task's official configuration; an empty list means the env is official | `bigym/loco/eval/protocol.py`, `bigym/loco/config.py` |

There is exactly **one** seed block. No separate validation block is used
for checkpoint selection: under a fixed training budget, "where the run
ended" has no selection bias, whereas early-stopping on a validation block
is free extra environment access and conflicts with the online methods'
interaction budget. Reporting a peak checkpoint as the headline number
overstates unstable runs — it is an appendix column, not the main table.

`peak_final_selection()` in `bigym/loco/eval/protocol.py` picks the peak and
final checkpoints for the appendix columns; it is not the headline
statistic.

The official configuration enables deterministic reset
(`ControllerConfig.deterministic_reset`): every reset restores the
controller state captured at the env's first reset, so `reset(seed)` depends
only on the seed and not on earlier episodes.

Overrides that only change what the policy is handed, not the episode it
plays, keep an env official: `frame_stack`, `normalize_low_dim_obs`,
`action_representation`, `upper_delta_scale_rad`, `event_progress_enabled`
and `render_mode`. Any other field, cameras included, is part of the
benchmark.

When reporting uncertainty, state the checkpoint, training-seed and
evaluation-episode aggregation levels; adjacent checkpoints are correlated.

An episode is a **success** when the task reported success
(`env.episode_succeeded()`: its predicate held for the success hold) and the
robot never fell during the episode (`bigym.loco.eval.is_success`). The robot has
fallen when its pelvis tilts more than about 53° from upright or drops below
the controller's height floor (the bottom of its height command range minus
0.10 m: 0.30 m for GR00T-WBC).
The controller's `is_failed()` checks this after every step and the env
latches it for the episode (`env.episode_fell()`). A fall does not end the
episode. `env.episode_termination()` labels each finished episode
`fell` / `success` / `physics_error` / `timeout` / `terminated`, and a fallen
episode is always `fell`. Report the `fell` share alongside the success rate.

## Baseline training recipe v1

Recipe v1 applies to the five baseline implementations: ACT, Diffusion
Policy, DEAS, CQN-AS and DrQ-v2+. Pretrained VLA comparisons such as
π0.5 have separately disclosed input processing and fine-tuning budgets.

**Layer 1 — interface (same task and configuration across the five baselines).**
Three 84×84 RGB cameras, the resolved low-dimensional/action layout,
60 demonstrations, the episode budget, and the evaluation protocol above.
π0.5 uses an explicitly derived 224×224 view; its pretrained inputs are
not silently changed to match the baseline regularizers.

**Layer 2 — training-time input regularization (recipe v1).**

| Knob | Value | Applies to |
|---|---|---|
| Image random-shift augmentation | pad **8** | all five baselines during updates |
| Proprioception masking | probability **0.2**, mode `sample_preserve_base` (per sample, zero non-base low-dimensional inputs; preserve base slots) | all five baselines during updates |
| Base-command exploration noise | Gaussian std **0.03** on the **first 3 normalized action coordinates**, mode `always` | online training only; no evaluation noise or perturbed demonstration labels |

The noise is applied by coordinate position, not channel name: in the
benchmark layout the first three coordinates are `vx, vy, height`, not
`vx, vy, wz`. Masking is per sample, not an independent 20% chance of
dropping each coordinate.

These are this benchmark's settings, not the upstream algorithms' defaults.

Successful-online-episode relabelling is enabled for the two online
baselines. Batch sizes are 256 offline, and 256 online + 256 demo samples
per online update.

**Layer 3 — the method itself.** Exploration schedules, BC weights and
margins, action-chunk length, open-loop horizon, EMA, KL weights and learning
rate follow each method's implementation. Disclose their resolved values;
do not assume they are all paper defaults.

### Training budget

The unit differs by algorithm class, so budgets are equal *within* a group
and are not compared across groups. Report the information budget alongside
the number.

| Group | Unit | Budget |
|---|---|---|
| Online (CQN-AS, DrQ-v2+) | environment interactions | 60 demos + **101k** interactions; update scheduling follows the trainer, including initial collection |
| Offline (ACT, Diffusion Policy, …) | dataset samples drawn | 60 demos + **256 × 101k ≈ 2.59e7** samples |
| VLA fine-tuning | fine-tuning steps | pre-training + 60 demos, disclosed per model |

The offline budget must clear each method's plateau; the
success-vs-epochs curve belongs in the appendix as evidence that it does.

## What makes two numbers comparable

`env.substrate_fingerprint()` is the identity of everything that can move
underneath a result: substrate version, MuJoCo version, task,
initialization profile, robot model, backend, `success_hold_seconds`, base
action mode, spawn stance, passive-tilt flag, control step, episode budget,
downsample rate, action dim, observation slices, solver, and a hash of every
compiled contact parameter. Two evaluations are comparable **iff** their
fingerprints match. Stamp it into every record.

The fingerprint does not cover the Python packages around MuJoCo. Official
results use Python 3.12 with the versions pinned in `uv.lock`; on Python 3.10
the lock resolves an older onnxruntime (1.23, the last release with 3.10
wheels) for the GR00T-WBC policy.
