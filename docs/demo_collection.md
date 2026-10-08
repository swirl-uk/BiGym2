# Collecting demonstrations (VR)

```{note}
Collection needs an OpenXR runtime and has been run on Linux with WiVRn.
macOS has no OpenXR runtime, so the collector does not run there. Install with `uv sync --extra vr --extra agent`; see
[Install](getting_started.md#install). WebXR support is on the roadmap.
```

Demos are collected by **VR teleoperation**: a human wears an OpenXR headset
(we use a Quest 3 streamed over Wi-Fi via [WiVRn](https://github.com/WiVRn/WiVRn));
the frozen lower-body backend keeps the robot walking from stick commands
while the operator's hand poses drive the arms through differential IK.

```{image} _static/vr_pipeline.svg
:alt: VR teleoperation pipeline — sticks to locomotion commands, hand poses through mink IK to arm targets, triggers to grippers; merged into the outer action, stepped through the env with the frozen lower body, saved as a schema-v1 npz episode with an engage snapshot.
:width: 100%
```

## Division of labor

- **This repo** owns everything that defines the demo *format and physics*:
  the mink-based upper-body IK (`bigym/vr/ik/` — 7-DoF×2 on G1,
  zero-jump engage, ~0.3 ms/solve {cite:p}`zakka2024mink`), the engage
  snapshot definition, the npz schema v1 + validator
  (`bigym.loco.demos`), and `bigym/tools/loco/validate_demos.py`.
- The interactive collector is `bigym-collect` (`bigym/vr/collect/`). It
  records in the task's official environment, `bigym.loco.make(task)`, on
  the real G1 legs under `groot_wbc_g1`; every released demonstration was
  recorded this way, none on a floating base. Only three things differ
  from the official env: the longer success hold below, an episode cap
  that never binds (at least 2 min), and per-step event progress.

  ```bash
  uv run --no-sync bigym-collect --task move_plate
  uv run --no-sync bigym-collect --task pick_box --episodes 10 --spectator viser
  ```

  The session settings (`bigym.vr.collect.config.CollectConfig`: stick
  scales, output directory, headset view, spectators) are command-line
  flags; `bigym-collect --help` lists them all. The environment itself is
  not a collector setting: the batch's `metadata.json` records the resolved
  configuration (`env_config`).

  Batches go to `./bigym_demos/<task>/<timestamp>` unless `--out-dir` says
  otherwise. Playback: `uv run bigym-view --demo-dir bigym_demos`
  (`--viewer mujoco` for a native window). Export and re-render are
  `bigym-export-lerobot` and `bigym-rerender-lerobot`.

## The engage moment

Recording does **not** start at `reset()`. The operator first starts an
attempt (robot stands, arms held), moves their hands to a comfortable pose,
then *engages* hand tracking. Engagement is zero-jump — the IK tracks pose
deltas from the engage pose, so hands never snap. At that instant the
collector snapshots the **full restorable state**:

- MuJoCo `qpos / qvel / ctrl / qacc_warmstart`, and
- the controller state (`get_lowerbody_state()`: obs histories, slew
  anchors, last actions, yaw targets).

The episode's actions are recorded from the engage moment on. Replay
restores the snapshot and replays the actions — bit-exact (see
[Demos & evaluation](demos_eval.md)). This is also why evaluation must let
the robot settle after reset (`reset_warmup_steps`): demos start at the
post-settle engage moment.

## Command mapping

Default stick scales are |vx| ≤ 0.35 m/s, |vy| ≤ 0.25 m/s and
|wz| ≤ 0.5 rad/s. Consult the selected backend's `command_spec` and the
collector's resolved limits: GR00T-WBC exposes benchmark clips because its
upstream training command ranges are not published.

| input | command | note |
|---|---|---|
| left stick Y / X | `vx` / `vy` | body-frame translation in walk mode |
| right stick X | `wz` | base yaw |
| right stick Y (G1) | height (integrated) | always height; right-stick axis dominance prevents simultaneous turn/height commands |
| hold left grip | pitch mode | requires a pitch slot; while held, left Y leans the torso and walking is suspended; releasing keeps the lean |
| triggers | grippers | the Dex1-1 gripper is one open/close scalar per hand |

## Success-only, 60 per task

The collector saves **successful** episodes only — the operator reviews each
pending episode and keeps or discards it. Every benchmark task has **60
successful human demonstrations**, the same count for every task, so demo
count is never a confound between tasks or between methods.

An attempt succeeds by the evaluation's definition
(`bigym.loco.eval.is_success`): the task reported success and the robot never
fell. Only the hold is stricter during collection: the collector holds the
task's success predicate for **3.0 s** (`collect_success_hold_seconds`) where
training and evaluation require **1.0 s** (`success_hold_seconds`). Every
demo tail therefore carries roughly three seconds of stay-still supervision in
the raw batch.
Before training, cut the batch to the training hold: the collector prints
the command when it exits, and runs it itself with `--export-lerobot`.

```bash
python -m bigym.loco.demos.success_hold --demo-dir <raw batch>
```

The cut writes `<raw batch>_hold1s` and never modifies the raw batch. It
replays every episode from its engage snapshot on the task's official env,
ends it on the control step where that env latches success, moves the single
terminal reward to the new last frame and sets `success_hold_seconds` in the
metadata (both the flat `task` block and `env_config`) to the training hold.

Read the actual attempts and keeps from `metadata.json.collection`,
including `sessions` for resumed batches and `unaccounted_episodes` for
files on disk that no session accounts for. The number of released
episodes says nothing about the keep rate.

## Validating a batch

```bash
uv run python bigym/tools/loco/validate_demos.py <demo_dir>
```

runs the schema-v1 validator over every episode (required keys, engage
snapshot completeness, metadata consistency) and prints actionable
violations. Always read the batch's `metadata.json` before training on it —
`episode_length` and `demo_down_sample_rate` must match.

Citation entries live in the [global bibliography](index.md#citing).
