# Demonstrations

Each benchmark task has 60 successful human VR demonstrations. They are
published as the Hugging Face dataset
[`SWIRL-Lab/bigym-g1-native60`](https://huggingface.co/datasets/SWIRL-Lab/bigym-g1-native60),
one LeRobot v3 folder per task, with frames stored as lossless PNG.

## Getting the demonstrations

`env.get_demos(n)` downloads the task's folder on first use.
`bigym-download` fetches ahead of time:

```bash
uv run bigym-download --list                  # tasks the dataset provides
uv run bigym-download --all                   # every task
uv run bigym-download --task move_plate pick_box
uv run bigym-download --all --local-dir bigym-data   # into a folder instead of the cache
```

Both share the Hugging Face cache and its variables (`HF_HOME`, `HF_TOKEN`,
`HF_HUB_OFFLINE=1`). `BIGYM_DATASET_REPO` and `BIGYM_DATASET_REVISION` select
another repository or pin a revision.

## Watching them

`uv run bigym-view` opens a browser viewer for the dataset. `--task move_plate`
opens that task first, and `--demo-dir bigym-data` opens a downloaded folder.

## Loading them

`env.get_demos()` checks the dataset against the env and decodes it:

- On a `make` env it returns lists of `ExtendedTimeStep` with the env's own
  frame stacking.
- On a `make_gym` env it returns gymnasium-shaped trajectories:
  `obs[i] --action[i]--> obs[i + 1]`.

It raises an error when the task has no published demonstrations or the
dataset does not match the env. The [FAQ](faq.md) covers both.

Decoded, the 60 demonstrations of a long task take a lot of memory: about
15 GB for `stack_blocks`. On a `make_gym` env, `get_demos(60,
decode_images=False)` keeps the frames as PNG bytes instead (about 5 GB for
`stack_blocks`), and `demo["obs"]["rgb"][i]` decodes observation `i` when you
index it. [`examples/train_act.py`](https://github.com/swirl-uk/BiGym2/blob/main/examples/train_act.py)
trains this way, decoding in its data loader workers.

`bigym.loco.demos.dataset.load_episodes(task_dir)` is the reader underneath.
It reads the LeRobot folder with pyarrow and Pillow, without installing
`lerobot`. Row `t` holds the observation at step `t` with the action, reward
and discount of the transition that produced it, and row 0 is the reset row.

The stored `action` column is normalized over the ranges in the task's
`metadata.json` (`action_stats`), and `get_demos()` converts it to the env's
ranges. To evaluate a policy trained on the stored column directly, give the
env those ranges first:

```python
stats = metadata["action_stats"]
env.set_action_stats(np.array(stats["min"]), np.array(stats["max"]))
```

## How an episode ends

Each episode ends on the step where the official env latches success, cut
from a recording made with the collector's longer 3.0 s hold.

## Formats

The collector writes one npz file per episode. Loading one needs only numpy:

```python
from bigym.loco.demos import load_episode, validate_episode

episode = load_episode(path)
violations = validate_episode(episode)   # empty when valid
```

- `bigym-view --demo-dir` reads both formats. `get_demos()` and
  `load_episodes` read LeRobot folders only.
- Video-mode LeRobot exports are for viewing. The training loader rejects
  them because the codec is lossy.
- Every step stores the full simulator state (`full_qpos`, `full_qvel`), so
  a batch can be re-rendered at another resolution, such as 224×224 for a VLA.
- Every episode stores its
  [engage snapshot](determinism.md#bit-exact-replay) for exact replay.
- In the parquet files, frame *k* holds observation *k* and the action
  executed from it.

## Converting and re-rendering

Both commands need the `lerobot` extra (see
[Installation](installation.md#extras)). `bigym-export-lerobot` converts an
npz batch to LeRobot v3:

```bash
uv run bigym-export-lerobot --demo-dir <npz batch> \
    --repo-id <hf-user>/<dataset> --root <output dir>
```

`bigym-rerender-lerobot` takes the same arguments plus `--size`, re-renders
the cameras (224×224 by default) and writes a LeRobot dataset:

```bash
uv run bigym-rerender-lerobot --demo-dir <npz batch> \
    --repo-id <hf-user>/<dataset>-224 --root <output dir>
```

## Batch metadata

Each batch's `metadata.json` stores the configuration it was collected in
under `env_config`, and `EnvConfig.from_metadata(metadata)` reads it back.
That configuration holds the collector's episode cap, which is longer than
the benchmark budget (see [Episode budgets](official_configuration.md#episode-budgets)).

## Every task at once

`scripts/demo_grid.py` plays one demonstration of every published task in a
single looping viser scene, for screenshots and recordings:

```bash
MUJOCO_GL=egl uv run python scripts/demo_grid.py
MUJOCO_GL=egl uv run python scripts/demo_grid.py --tasks move_plate,pick_box --columns 2
MUJOCO_GL=egl uv run python scripts/demo_grid.py --record grid.mp4 --record-size 1920x1080
MUJOCO_GL=egl uv run python scripts/demo_grid.py --screenshot grid.png --frame 400
```

`--help` lists the other options. `--export-states FILE.npz` saves the chosen
episodes' states, and `--states FILE.npz` builds the grid from that file
without the dataset.
