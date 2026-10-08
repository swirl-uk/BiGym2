# Demos, determinism & evaluation

## Getting the demonstrations

The public demonstrations are one Hugging Face dataset repository with a
folder per task (LeRobot v3, lossless PNG frames). `env.get_demos(n)` fetches
the current task's folder on first use through `bigym.loco.demos.hub`, and
`bigym-download` fetches ahead of time:

```bash
uv run bigym-download --list                  # tasks the dataset provides
uv run bigym-download --all                   # mirror everything
uv run bigym-download --task move_plate pick_box
uv run bigym-download --all --local-dir bigym-data   # into a folder, not the cache
```

`uv run bigym-view` browses the demonstrations in the browser: pick a task
and an episode in the sidebar (`--task move_plate` opens that one first). A
downloaded folder opens with `--demo-dir bigym-data`, or from the viewer's
*Demo directory* panel, which browses the folders of the machine running
`bigym-view` and marks the ones holding demonstrations with ●.

Everything goes through `huggingface_hub.snapshot_download`, so the Hub
cache (`$HF_HOME` / `$HF_HUB_CACHE`) is shared between the two paths and the
usual knobs apply: `HF_TOKEN` for a private repository, `HF_HUB_OFFLINE=1` to
stay on the cache. `BIGYM_DATASET_REPO` / `BIGYM_DATASET_REVISION` point the
loader at another repository or pin a revision.

Each published episode ends with a 1 s success hold, the same hold
evaluation scores: the demonstrations were recorded with a 3 s hold and cut
before release on the step where the official env latches success
(`metadata.json` → `success_hold_trim`), so replaying one ends on its last
row and they are ready to train on as downloaded.

The dataset covers the 20 benchmark tasks. `bigym-download --list` shows
them along with the registered tasks whose demonstrations come in a later
release; `get_demos()` on one of those raises
`bigym.loco.demos.hub.DemosUnavailableError`, which lists both groups.

`bigym.loco.demos.dataset.load_episodes(task_dir)` is the reader (pyarrow +
Pillow; the `lerobot` package is never imported). It returns replay-format
episodes: row `t` holds the observation at step `t` and the action, reward
and discount of the transition that produced it. `env.get_demos()` on a
`make` env returns those rows as `ExtendedTimeStep` lists with the
env's own frame stacking; on a `make_gym` env it returns gymnasium-shaped
trajectories (`obs[i] --action[i]--> obs[i + 1]`). Either way the env adopts
the dataset's `action_stats`, so learned `[-1, 1]` actions de-normalize
exactly as they were recorded, and the loader refuses a dataset whose
embodiment, backend, down-sample rate, cameras or observation/action layout
differ from the env.

## Demo schema v1

Native demos are npz episodes with a versioned, validated schema
(`bigym.loco.demos`). Loading needs **numpy + json only** — no VR stack, no
torch:

```python
from bigym.loco.demos import load_episode, validate_episode

episode = load_episode(path)
violations = validate_episode(episode)   # actionable strings, empty = valid
```

Each episode carries the engage snapshot (`init_qpos/qvel/ctrl/
qacc_warmstart` + the controller's full mutable state), so a demo restores
into the exact simulator state it was recorded from.

Always read the demo `metadata.json` before training/eval and match
`episode_length` and `demo_down_sample_rate` to it — a wrong time limit
looks exactly like an algorithm failure (constant short episodes, 0%
success).

## Bit-exact replay

Mid-episode save/restore is bit-exact (`max |qpos_A - qpos_B| = 0.0` over 50
steps). Three design decisions make this true:

1. **Newton solver pin.** PGS warmstarts its dual variables from
   variable-size internals tied to the previous step's contact structure —
   state that cannot be reseeded from an npz. The G1 scenes therefore pin
   the Newton solver.
2. **Declarative controller state.** Every mutable field that shapes future
   targets (obs histories, slew anchors, last actions, gait clocks) is
   declared in `STATEFUL` and snapshotted; a missing field is a loud
   `KeyError`, not a silent divergence.
3. **Derived state recomputed every step.** `mj_step` leaves kinematics,
   contacts and sensors at the state before its last substep, so the env
   runs `mj_forward` at the end of every control step; a restored state
   (which runs `mj_forward` too) and an uninterrupted run read the same
   values.

## Evaluation protocol

`bigym.loco.eval` freezes the protocol: **100 episodes** per evaluation,
episode *i* seeded `620000 + i` (one contiguous block, no separate
validation block). An episode is a success when the task reported success
(`env.episode_succeeded()`: its predicate held for the success hold) and the
robot never fell during the episode (`bigym.loco.eval.is_success`).
The robot has fallen when its pelvis tilts more than about 53° from upright
or drops below the controller's height floor (the bottom of its height
command range minus 0.10 m: 0.30 m for GR00T-WBC). `leaderboard_record()`
refuses non-contiguous seed sets, and every record embeds the env's `substrate_fingerprint()` — results from
different substrates cannot be silently mixed.

Checkpoints are evaluated every 5000 training steps, and the reported
headline number is the **mean success rate over the last five checkpoints,
± standard error**. Final-checkpoint-only and per-checkpoint-maximum columns
belong in an appendix. Report checkpoint, training-seed and episode
aggregation separately. Deterministic reset is described in
[The official benchmark configuration](official_configuration.md#evaluation-protocol).

### The reference runner

Don't hand-roll an eval loop — run the protocol:

```python
from bigym.loco.eval import evaluate

result = evaluate(
    my_policy,                       # callable: timestep -> action
    task_name="move_plate",
    method="my-cool-method",
    overrides={"frame_stack": 2},    # optional; the official env otherwise
)
result["record"]                     # leaderboard entry
result["env_config"]                 # the resolved EnvConfig, as a dict
```

The runner builds `bigym.loco.make(task_name, config, **overrides)`. A
policy is any callable (plus an optional `reset()` called between episodes
for stateful policies). Overrides of presentation-only fields
(`frame_stack`, `normalize_low_dim_obs`, `action_representation`,
`upper_delta_scale_rad`, `event_progress_enabled`, `render_mode`) keep the
run official; any other override is listed in
`result["protocol_violations"]` and the result has no leaderboard record.
Running fewer episodes (`episodes=N`) likewise works for smoke-testing but
returns a summary **without** a record, so off-protocol numbers can never
masquerade as benchmark numbers. CLI:
`uv run python -m bigym.loco.eval.runner --task ... --method ... --policy module:factory [--overrides JSON]`.

## Distribution format: LeRobot v3 (lossless)

The npz schema above is the *collector output* and the conversion source.
The canonical format is **LeRobot v3 in lossless PNG mode**: recorded
84×84 images plus full simulator state. A 224×224 re-render is a separate
derived view, not a replacement for the canonical observations.

- **Lossless is the point.** Video-mode exports exist for visualization
  only and are rejected by the training loader: a benchmark whose
  observations went through a lossy codec is not the benchmark that was
  evaluated.
- **Nothing is dropped in conversion.** Cameras, low-dim observations, the
  outer action and the per-step full-body extras (`full_qpos`, `full_qvel`,
  raw/expanded actions, lower-body command and targets, reward) become
  first-class LeRobot features. Because `full_qpos` / `full_qvel` are
  carried per step, a dataset can be **re-rendered at any resolution** from
  this single copy — 84×84 for the benchmark, 224×224 for a VLA, without
  re-collecting anything.
- **Engage snapshots survive as a sidecar.** The per-episode
  `init_qpos / init_qvel / init_ctrl / init_qacc_warmstart`, the controller
  state (`lb_state.*`), the seed and the pre-engage step count live in
  `meta/episode_init_states.json` with explicit dtypes, so deterministic
  replay still works from the LeRobot copy.
- **Frame alignment is versioned.** Alignment v2 (`meta/alignment.json`)
  stores frame *k* with the action executed *from* observation *k*, which is
  what an external consumer reading the raw parquet expects; the source npz
  convention (index *t* holds the state *after* action *t*) is shifted
  during export.
- **Loading is transparent.** A demo directory containing `meta/info.json`
  is read through the LeRobot path, an npz directory as npz. The reader uses
  pyarrow only, so the training environment never imports `lerobot`.

Conversion lives in `bigym.loco.demos.lerobot_export`. It needs LeRobot
(Python >= 3.12): the `lerobot` extra, or `uv run --with "lerobot[dataset]>=0.6"`
as in [Install](getting_started.md#install).

```bash
uv run bigym-export-lerobot \
    --demo-dir  <npz batch> \
    --repo-id   <hf-user>/<dataset> \
    --root      <output dir> \
    --task-text "Move the plate from one dish rack to the other."
```

A conversion should reconstruct every episode bit-identically: compare the
raw bytes of both demo paths (NaN payloads and dtypes included).
`bigym-rerender-lerobot` renders a derived resolution; all non-image columns
of the derived dataset stay byte-identical to the source.

## Substrate versioning

`SUBSTRATE_VERSION` names the tuple (physics version, backend weights,
obs/action layout, reward/success criteria, demo set). Numbers are
comparable **iff** their fingerprints match; changing any component bumps
the version. This is what "final results" means here: not "the code didn't
move", but "the substrate didn't".

## Task registry

`bigym.loco.tasks.TASKS` holds one `TaskSpec` per task: its env class,
episode budget (with per-task provenance: `data_derived` or
`upstream_placeholder`, available through `budget_provenance(name)`) and
the few settings where it departs from the `EnvConfig` defaults.
`task_config(name)` returns the resolved official configuration and
`bigym.loco.make(name)` builds the env from it; to replicate the official
evaluation settings, read them from here rather than from training configs.
The assembled table is
[The official benchmark configuration](official_configuration.md).

Every batch's `metadata.json` records the configuration it was collected
in, as `env_config` and as the flat `task` and `lowerbody_policy` blocks;
`EnvConfig.from_metadata(metadata)` reads `env_config` back into an
`EnvConfig` for `make`.

## Showing every task at once

`python scripts/demo_grid.py` puts one human demonstration of **every published task**
into a single looping viser scene — twenty environments side by side, for a
screenshot of the whole benchmark or a screen recording:

```bash
MUJOCO_GL=egl uv run python scripts/demo_grid.py                 # every task, 5 columns
MUJOCO_GL=egl uv run python scripts/demo_grid.py --tasks move_plate,pick_box --columns 2
MUJOCO_GL=egl uv run python scripts/demo_grid.py --camera 7 --tint --sky grey
MUJOCO_GL=egl uv run python scripts/demo_grid.py --exit-after-seconds 60   # headless smoke run
```

Each scene is a `make` env built **without cameras**
(`camera_keys=()`; the grid never reads a pixel), reset on the seed its
demonstration was collected on, and posed frame by frame from the stored
`full_qpos` with `mj_forward` plus the task's own `_on_step` hook — the
kinematic replay of `bigym.loco.demos.kinematic` that `bigym-view` uses, so the props, the target
placement and the reach-target highlight are the demonstration's own. Only
the `full_qpos` column and `meta/episode_init_states.json` are read, the way
`bigym.loco.agent.demo_video.TaskDemos` reads them, so no camera PNG is ever
decoded.

Layout: every environment is built and reset first, its footprint measured
from the geoms it actually draws, and only then are the scenes packed — a
column is as wide as its widest scene, a row as deep as its deepest, and
neighbouring footprints clear each other by `--clearance` (0.6 m). Each
scene is centred in its cell on its footprint rather than its origin, so an
asymmetric kitchen lines up with a bare reach-target instead of sitting a
metre off. `--gap G` replaces all of that with a uniform pitch. One
**column** steps along +y and one **row** back along -x, `--columns 5` by
default, so twenty tasks read as 4 rows of 5; the log prints the resulting
column widths, row depths, pitches and the grid's extent. The camera starts
where the whole grid is in frame, on a lens computed from that extent —
viser's own ~80° field of view would leave the grid in a band across the
middle of the picture.
`--episode median` (the default) shows the demonstration of median length of
each task, `--episode N` a fixed index. `--loop all` holds each scene's last
frame until the longest demo ends and restarts the grid together; `--loop
each` lets every scene wrap on its own length. `--labels` (on) draws the task
name on a plate above the robot's pelvis (`move_plate` reads `Move plate`),
`--tint` gives each robot a palette colour (display only), `--sky
black|grey|transparent` sets the backdrop, and `--hide-gui` (on) minimises
viser's panel so only a small *Playback* folder — play, frame, fps and a
*focus* dropdown that flies to one scene — is one click away.

For twenty tasks, expect about 90 s to build the environments and bake
their meshes, about 7.5 GB of memory, and roughly 20 grid updates per
second; the limit is viser's scene updates, not MuJoCo. `--fps` is demo
time, not a render rate: at the default 50 the demonstrations play in real
time, frames that cannot be drawn in time are skipped, and the measured
render rate is logged as `[grid] playback N fps measured`.

### Recording a video

Live playback of twenty scenes is too choppy to film off the screen. `--record` captures the same scene offline instead: the
playhead is stepped one frame at a time with no wall clock anywhere, every
scene is posed, the browser is asked for a picture of that frame, and the
picture goes straight into an ffmpeg pipe. The file is smooth at
`--record-fps` however long each capture took.

```bash
# 250 frames (5 s at 50 fps) of the whole grid, 1080p
MUJOCO_GL=egl uv run python scripts/demo_grid.py --record grid.mp4 \
    --record-size 1920x1080 --record-fps 50 --record-frames 250
# one frame, as a PNG
MUJOCO_GL=egl uv run python scripts/demo_grid.py --screenshot grid.png --frame 400
```

`--path fly` flies the camera instead of fixing it: a close shot of the
first scene, a dolly down the aisle between the first two rows, a turn at
the far end, the way back down the aisle between the last two, then a rise
onto the overview which it holds. The aisles are the real gaps between the
measured footprints, so the camera never clips a worktop, and position,
look-at and field of view follow one Catmull-Rom spline (with time-aware
tangents, eased at the start, the turn and the end) so nothing jerks.
`--duration` sets its length and `--record-frames` follows from it;
`--path-dump FILE.json` writes the keyframes out to edit and
`--path-keyframes FILE.json` flies an edited set (`{"t", "position",
"look_at", "fov"}` in world metres and degrees). `--path-preview` renders
the same path at 480×270 and 5 fps, to check the motion in a minute rather
than an hour. The live viewer flies the path too: `--path fly` without
`--record` moves every connected browser's camera on the wall clock,
looping.

viser has no server-side renderer — `ClientHandle.get_render` asks a
connected browser's canvas — so a capture needs a client. Any client will
do: open the URL in your own (GPU) browser and the recording starts as soon
as it connects, and the log names the client it is rendering from.
`--browser headless` (the default) waits 3 s for one of yours and then starts its own
headless Chrome/Chromium on the page with SwiftShader (no display, no GPU, no
sandbox) and closes it when the file is written; `--browser none` waits for
you to open the printed URL instead, and the recording starts by itself the
moment the page connects. `--record-size` is **not** limited by the browser
window: `get_render` renders offscreen, and the window only sets the aspect
ratio. The remaining knobs: `--record-frames` (default: one pass over the longest
demonstration), `--record-camera overview|<index>`, `--record-then-stay` to
keep serving the live scene afterwards, and `--frame` for the screenshot's
playhead position.

A capture also waits for the page before it starts: `get_render` does not
await anything the browser is still loading, and the first render of a
twenty-scene page comes back with a white sky and most of the grid missing,
so the recorder renders small throwaway frames until two in a row are
identical (`[grid] page settled after N renders`).

Headless capture is software rendering, so it is slow: about 10 s per
1920×1080 frame of the twenty-scene grid (2–3 s for a two-scene page), so
the 250-frame example above takes about 40 minutes for 5 s of 50 fps H.264.
With a GPU and a desktop, `--browser none` plus your own browser window is
much faster; a smaller `--record-size` also helps.

### Rendering without the dataset

The demonstrations are 28 GB; what the grid needs to draw them is the
per-frame `full_qpos` and the seed of one episode per task, which is a few
MB. `--export-states` writes exactly that, and `--states` builds the grid
from it with no Hub access at all:

```bash
# where the dataset lives (20 tasks -> 3.4 MB)
uv run python scripts/demo_grid.py --export-states grid_states.npz
# anywhere else, including a laptop with no dataset and no EGL
uv run python scripts/demo_grid.py --states grid_states.npz --path fly --record grid.mp4
```

The npz holds one float32 `full_qpos` array per task (posing writes qpos and
calls `mj_forward`, so float32 is identical to the double for anything a
picture shows) plus a JSON header with each task's seed, episode index and
source file, the dataset repo and revision, and the `--episode` rule they
were chosen by. `--tasks` then selects and orders tasks from the file.
Nothing on this path is Linux-only: the environments are built with no
cameras, so no EGL or `MUJOCO_GL` is needed, and ffmpeg is looked up on
`PATH` (or taken from the `imageio-ffmpeg` wheel).
