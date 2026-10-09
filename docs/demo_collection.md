# Collecting demonstrations (VR)

```{note}
Collection needs an OpenXR runtime, so it runs on Linux and not on macOS.
Install with `uv sync --extra vr --extra agent` (see
[Installation](installation.md)).
```

An operator wears an OpenXR headset. We use a Quest 3 streamed over Wi-Fi
with [WiVRn](https://github.com/WiVRn/WiVRn). The frozen lower body walks
the robot from stick commands, and the operator's hand poses drive the arms
through mink differential IK {cite:p}`zakka2024mink`.

## Running the collector

`bigym-collect` records in the task's official environment,
`bigym.loco.make(task)`, with the G1 legs under `groot_wbc_g1`. Every
released demonstration was recorded this way. The collector's env differs
from the official one in two settings: a longer success hold (below) and an
episode cap of at least 2 min that never binds.

```bash
uv run --no-sync bigym-collect --task move_plate
uv run --no-sync bigym-collect --task pick_box --episodes 10 --spectator viser
```

`bigym-collect --help` lists the session settings. Batches go to
`./bigym_demos/<task>/<timestamp>` unless `--out-dir` says otherwise. Play
them back with `uv run bigym-view --demo-dir bigym_demos`.

## The engage moment

After `reset()`, the robot stands and settles with its arms held. The
operator moves their hands to a comfortable pose and engages hand tracking.
The IK tracks pose changes from that pose, so the hands never jump.

Recording starts at the engage. The collector saves the full simulator and
controller state at that instant (see
[Determinism and versioning](determinism.md#bit-exact-replay)), and replay
restores it and replays the actions bit-exactly. The demos start after the
settle, so evaluation must let the robot settle after reset too
(`reset_warmup_steps`).

## Command mapping

| input | command | range |
|---|---|---|
| left stick Y / X | `vx` / `vy` | ±0.35 m/s / ±0.25 m/s |
| right stick X | `wz` | ±0.5 rad/s |
| right stick Y | height, integrated at 4 mm per step | 0.4–0.8 m |
| hold left grip, then left stick Y | torso pitch, integrated at 0.8 rad/s (pitch tasks only) | −0.2 to 0.8 rad |
| triggers | grippers | open or closed |

The collector caps the height command at 0.80 m. Above that, the GR00T-WBC
policy locks the knees and stops stepping. Height and pitch keep their value
when the stick is released. Holding the left grip suspends walking.

## Success and the training view

The collector saves successful episodes only, and the operator keeps or
discards each one. Every benchmark task has 60 human demonstrations.

An attempt succeeds by the evaluation's definition: the task reported
success and the robot never fell. Only the hold differs. The collector
requires the success predicate to hold for 3.0 s, where training and
evaluation require 1.0 s (`success_hold_seconds`).

Cut the raw batch before training. The collector prints this command when it
exits and runs it itself with `--export-lerobot`:

```bash
python -m bigym.loco.demos.success_hold --demo-dir <raw batch>
```

The cut replays each episode from its engage snapshot on the official env
and ends it on the step where that env latches success. It writes
`<raw batch>_hold1s` and leaves the raw batch untouched.

## Validating a batch

```bash
uv run python bigym/tools/loco/validate_demos.py <demo_dir>
```

It checks every episode against the npz schema and the batch's
`metadata.json`, and exits non-zero on any violation. Before training, check
that the batch's `demo_down_sample_rate` matches your env.
