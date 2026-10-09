# Official configuration

A leaderboard entry uses the configuration and the evaluation protocol on
this page. Anything not listed here, such as network sizes, optimizers or
action-chunk lengths, belongs to your method.

## Building the official env

{func}`bigym.loco.make` builds the official env for a task:

```python
from bigym.loco import make

env = make("move_plate")
env.config                    # the resolved configuration
```

The defaults of {class}`bigym.loco.config.EnvConfig` are the official values.
Each task's `TaskSpec` in `bigym.loco.tasks` adds its episode budget and its
few differences: pitch command off, reach tolerance, initialization profile.

`make(task, config=None, **overrides)` takes an `EnvConfig` or keyword
overrides. Nested controller settings merge into the task's own, as in
`make("pick_box", controller={"cmd_clip": 0.5})`. `env.config_overrides`
lists every field that departs from the task's official configuration.
Evaluation refuses a leaderboard record when any of them affects results.
`make(task, controller=None)` builds a floating-base research env without the
lower-body controller.

## The configuration table

| Item | Official value | Config field |
|---|---|---|
| Robot | `g1_dex1`: Unitree G1 29-DoF with Dex1-1 two-finger grippers | `robot_model` |
| Lower-body backend | `groot_wbc_g1`: NVIDIA GR00T-WBC, frozen Balance and Walk ONNX pair | `controller.backend` |
| Base action mode | `lowerbody_cmd`: the base slots are velocity and height commands | `controller.base_action_mode` |
| Passive pelvis roll and pitch | `True`: the pelvis tilt is unactuated | `controller.passive_base_tilt` |
| Spawn stance | `keyframe` | `controller.init_stance` |
| Control rate | 50 Hz: 0.02 s per control step, 10 env steps, 20 physics substeps of 1 ms | `demo_down_sample_rate` |
| Cameras | 3 × RGB 84×84: `head`, `right_wrist`, `left_wrist` | `camera_keys`, `camera_shape` |
| Torso-pitch command | on, except for six tasks: the three reach tasks, `move_plate`, `drawer_top_open`, `drawer_top_close` | `controller.pitch_command` |
| Success hold | the task predicate holds for 1.0 s (50 control steps) before the task reward is given | `success_hold_seconds` |
| Reach tolerance | 0.05 m for the three reach tasks: the pinch centre must be inside the target ball | `reach_tolerance` |
| Reset warmup | 200 control steps before the agent engages | `controller.reset_warmup_steps` |
| Initialization profile | `upstream`, except `g1_id_v1` for `drawer_top_open` and `drawer_top_close` | `initialization_profile` |
| Episode budget | per task, see [Episode budgets](#episode-budgets) | `episode_length` |

## Observation layout

`env.low_dim_component_slices()` returns the slices below for a task without
the pitch command, such as `move_plate`.

| Slice | Dims | Contents |
|---|---|---|
| `proprioception` | 0–44 | arm and finger joint positions and pelvis x, y, z, yaw (0–21), then their velocities (22–43). Leg and waist joints are excluded because the backend owns them |
| `proprioception_grippers` | 44–46 | left and right gripper state, 0 = open, 1 = closed |
| `proprioception_floating_base` | 46–50 | pelvis x, y, z, yaw as tracked by the base controller |

With the pitch command, the three waist joints (yaw, roll, pitch) and their
velocities are added at the front of `proprioception`. The slices become
`(0, 50)`, `(50, 52)` and `(52, 56)`.

`rgb_obs` is `(3, 3, 84, 84)` uint8: camera-major, then CHW, in the order
`head`, `right_wrist`, `left_wrist`.

## Action layout

The action has 20 floats, normalized to [-1, 1]. The ranges below are the raw
values each slot maps onto for `g1_dex1` with `groot_wbc_g1`, as reported by
`env.wholebody_action_layout()`:

| Slot(s) | Name | Range | Meaning |
|---|---|---|---|
| 0 | `vx` | [-1.0, 1.0] m/s | body-frame forward velocity command |
| 1 | `vy` | [-1.0, 1.0] m/s | body-frame lateral velocity command |
| 2 | `height` | [0.4, 1.0] m | absolute pelvis-height command |
| 3 | `wz` | [-1.0, 1.0] rad/s | yaw rate command |
| 4–10 | left arm | joint limits | shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw |
| 11–17 | right arm | joint limits | same order, right side |
| 18 | left gripper | [0, 1] | 0 = open, 1 = closed |
| 19 | right gripper | [0, 1] | 0 = open, 1 = closed |

The `vx`, `vy` and `wz` bounds are the env's command clips (`cmd_clip` and
`wz_clip`, default 1.0). The height bounds come from the backend. Arm slots
are absolute joint position targets in radians.

`get_demos()` returns demo actions normalized over these ranges. To use
other ranges, call `env.set_action_stats(min, max)` before `get_demos()` and
evaluate with the same ranges.

With the pitch command, slot 4 is the torso pitch (radians, absolute,
+ = lean forward, [-0.2, 0.8]) and the arm and gripper slots move up by one:
left arm 5–11, right arm 12–18, grippers 19 and 20.

## Episode budgets

`make(task)` sets the task's budget, `TASKS[task].episode_length`. Values are
env steps. Divide by `demo_down_sample_rate` (10) for control steps.

The 20 benchmark tasks get their budget from `BUDGET_RULE` applied to the
task's 60 published demonstrations: 2× the longest successful demonstration,
rounded up to the next 500 env steps, minimum 2000. The other 20 registered
tasks keep placeholder budgets from BiGym.

The `episode_length` inside a batch's `metadata.json` is the collector's cap,
at least 6000 control steps (2 min). It is not the benchmark budget. Use the
budget `make(task)` sets.

## Evaluation protocol

| Item | Value | Where |
|---|---|---|
| Episodes per evaluation | 100 | `EVAL_EPISODES` |
| Seeds | one contiguous block, episode *i* uses `620000 + i` | `eval_seeds()` |
| Checkpoint spacing | every 5000 training steps, all evaluated | reporting convention |
| Headline number | mean success rate over the last 5 checkpoints, ± standard error | reporting convention |
| Appendix numbers | final checkpoint alone, and per-checkpoint maximum (labelled as coming from the same seed block) | reporting convention |

There is one seed block. No separate validation block is used for checkpoint
selection. `peak_final_selection()` picks the peak and final checkpoints for
the appendix columns. `leaderboard_record()` embeds the substrate fingerprint
and refuses a seed set that is not one contiguous block.

When you report uncertainty, state the checkpoint, training-seed and
evaluation-episode aggregation levels. Adjacent checkpoints are correlated.

## Success and falls

An episode is a success when the task reported success
(`env.episode_succeeded()`) and the robot never fell (`env.episode_fell()`).
`bigym.loco.eval.is_success(env)` checks both.

The robot has fallen when its pelvis tilts more than 53° from upright or its
height drops below 0.30 m. The check runs after every control step, and a fall
is latched for the rest of the episode. A fall does not end the episode.

`env.episode_termination()` labels each finished episode `fell`, `success`,
`physics_error`, `timeout` or `terminated`. A fallen episode is always `fell`.
Report the `fell` share alongside the success rate.

## Official overrides

Overrides that only change what the policy is handed keep an env official:
`frame_stack`, `normalize_low_dim_obs`, `action_representation`,
`upper_delta_scale_rad`, `event_progress_enabled` and `render_mode`. Any other
field, cameras included, is part of the benchmark. `protocol_violations(env)`
lists every such field that departs from the task's official configuration.
The env is official when the list is empty.

## The reference runner

`evaluate` runs the protocol:

```python
from bigym.loco.eval import evaluate

result = evaluate(
    my_policy,
    task_name="move_plate",
    method="my-method",
    overrides={"frame_stack": 2},    # optional
)
result["record"]                     # leaderboard entry
result["protocol_violations"]        # [] for an official env
result["env_config"]                 # the resolved EnvConfig, as a dict
```

A policy is any callable from timestep to action. If it has a `reset()`
method, the runner calls it between episodes. A result with protocol
violations, or with fewer than 100 episodes (`episodes=N`), has no
leaderboard record.

The same runner from the shell, where `--policy` names a factory that returns
the policy:

```bash
uv run python -m bigym.loco.eval.runner --task move_plate --method my-method \
    --policy my_package.eval:make_policy --out record.json
```

## Baseline training recipe v1

The paper trained its five baselines (ACT, Diffusion Policy, DEAS, CQN-AS and
DrQ-v2+) with this recipe. It is not implemented in this repository.
`examples/train_act.py` is a minimal ACT example that does not follow it.

The five baselines use the official configuration above: three 84×84 RGB
cameras, the low-dimensional and action layouts, 60 demonstrations, the
episode budget and the evaluation protocol. π0.5 uses a derived 224×224 view
with its own pretrained input processing. Pretrained VLAs disclose their input
processing and fine-tuning budgets.

| Knob | Value | Applies to |
|---|---|---|
| Image random-shift augmentation | pad 8 | all five baselines during updates |
| Proprioception masking | probability 0.2, mode `sample_preserve_base`: per sample, zero the non-base low-dimensional inputs and keep the base slots | all five baselines during updates |
| Base-command exploration noise | Gaussian std 0.03 on the first 3 normalized action coordinates, mode `always` | online training only, with no evaluation noise and no perturbed demonstration labels |

The noise is applied by coordinate position. In the benchmark layout the
first three coordinates are `vx`, `vy` and `height`. Masking is decided once
per sample for all non-base inputs together. The original algorithms use
other defaults.

The two online baselines relabel successful online episodes. Batch sizes are
256 offline, and 256 online plus 256 demo samples per online update.

Exploration schedules, BC weights and margins, action-chunk length, open-loop
horizon, EMA, KL weights and learning rate follow each method's
implementation. Disclose their resolved values.

### Training budget

The budget unit differs by algorithm group. Budgets are equal within a group
and are not compared across groups. Report the information budget with the
number.

| Group | Unit | Budget |
|---|---|---|
| Online (CQN-AS, DrQ-v2+) | environment interactions | 60 demos + 101k interactions, including initial collection |
| Offline (ACT, Diffusion Policy, …) | dataset samples drawn | 60 demos + 256 × 101k ≈ 2.59e7 samples |
| VLA fine-tuning | fine-tuning steps | pre-training + 60 demos, disclosed per model |

Update scheduling for the online group follows each trainer. Show in the
appendix, with a success-versus-epochs curve, that each offline method
plateaus within its budget.
