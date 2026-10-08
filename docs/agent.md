# The coding-agent benchmark

`bigym.loco.agent` runs the other kind of policy this benchmark measures. A
learned baseline reads the three onboard cameras and the proprioceptive
vector and emits an action; a **coding agent** never sees a training loop —
it is given one sentence of task description, a video of a human
demonstration, and a budget of environment steps, and it writes
`policy.py` by hand. The submitted file is then scored on the same hidden
seed block as everything else, through the same frozen lower-body
controller.

The whole thing ships with the package: `bigym-agent` builds the sandbox,
serves the environment, runs the harness, snapshots every policy version the
agent writes, and scores the submission.

## What is measured

A **cell** is one task × one session. Inside it:

- **The task statement is one sentence** — a plain instruction naming what to
  do and to which object, with nothing about positions, sizes or the
  tolerance the success test uses
  (`bigym.loco.agent.sandbox.TASK_SENTENCES`). No task card, no hints.
  `docs/api.md` in the sandbox describes the action, the observation and the
  client, and nothing about the scene.
- **The demonstration is a video** of one human demo, rendered from the
  robot's own three cameras (`head`, `left_wrist`, `right_wrist`) at the
  policy-facing resolution (84×84), 25 fps, one frame per two control steps.
  There is no outside camera, for the agent as much as for the policy.
- **The interaction budget is 101,000 environment steps**, and a reset costs
  200 steps (the controller warm-up) plus the steps of the episode. The
  server keeps the ledger and refuses the step that would exceed the cap.
- **Development seeds are the seeds the demonstrations were collected on**
  (60 for a published task), listed in `seeds.json`. The server refuses any
  other seed, including the evaluation block.
- **Scoring is the success rate over 100 hidden seeds**, `620000 + i` — the
  frozen evaluation block from
  [Demos, determinism & evaluation](demos_eval.md). Nothing the agent reports
  about its own performance is used.
- **A fall is a failure.** The lower-body controller reports it, the episode
  runs out the clock, and the episode is scored 0.
- **The evaluation is re-run and must agree.** `--spot-check N` repeats the
  first N episodes after the block and requires identical per-episode
  success, so a policy that reads the wall clock or unseeded randomness is
  caught rather than averaged.
- **The policy never shares a process with the simulator it is scored on.**
  `evaluate`, `replay` and `record` start the policy in a fresh interpreter
  and exchange observations, actions and tool calls with it over a pipe; the
  reward, the fall check and the success test are computed on the other side.
  A policy that goes looking for the environment (through `gc`, say) finds
  none.
- **A policy that imports the simulator scores 0.** `policy.py` and its
  helpers must not import `bigym`, `mujoco`, `dm_control` or `mink`.
- **The policy is hand-written control.** No learned component — no network,
  regression or lookup table fitted to data. Tuning constants by running
  episodes is the intended way to work.
- **The libraries are numpy, scipy, OpenCV and pillow**, plus the standard
  library. Nothing else is installed and nothing can be installed.

### `strict` and `tools`

`--interface strict` is the default and the benchmark row: the policy signs
**the learned baselines' I/O contract** — the three 84×84 cameras and the raw
`low_dim_obs` vector in, the physical action out. No named state fields, no
wrist positions, no camera calibration, no IK. A number produced this way is
comparable to a learned policy's number, because the two saw the same thing.

`--interface tools` is the documented ablation: the same environment with
named state, wrist positions in world frame, `camera_info` / `pixel_to_ray`
and an IK solver on top. It answers "how much of the difficulty is the
interface?", and it is a separate, labelled row — never mixed into the
`strict` column. `--no-ik` and `--no-hand-pos` withhold one tool at a time.

## Install

```bash
uv sync --extra agent     # in the clone: the server, the evaluator, the sandbox templates
```

The commands below are run as `uv run bigym-agent ...` (or plain
`bigym-agent` once `.venv` is activated).

The extra adds `opencv-python-headless` and `scipy` — what a submitted
`policy.py` may import, installed on the evaluator's side so it runs the same
code the agent wrote — and `mink`, which backs the `tools` interface's IK and
is imported lazily, so a `strict` session never touches it.

Two things are not Python dependencies:

- **ffmpeg** on `PATH`, for the demonstration videos and the evaluation
  recordings.
- **docker**, if you want the container isolation the benchmark rows were
  produced with. Without it, `--isolation soft` runs the harness on the host
  under its own tool rules and `bigym-agent audit` checks afterwards what it
  reached for.

Rendering is EGL: `export MUJOCO_GL=egl`.

## The cell

Everything one session produces lives in one directory, `<root>/<task>/`:

```text
<root>/<task>/
  run.json              configuration and result: harness, model, effort, interface,
                        image, auth mode, budget, start/end, verdict, token usage
                        and cost
  transcript.jsonl      the unified transcript, one JSON object per event
  transcript.md         its human-readable rendering
  raw/                  the harness's own event stream, its stderr, its home directory
  ledger.jsonl          every reset and episode: seed, outcome, budget charged
  budget.json           the interaction budget as the server left it
  dev/vNNN_<stamp>/     every development episode, as replay batches per policy version
  server.log            the environment server's own log
  sandbox/              exactly what the agent saw and wrote
    PROMPT.md  README.md  docs/api.md  harness/  run_episodes.py  python
    policy.py  seeds.json  demo_head.mp4  demo_left_wrist.mp4  demo_right_wrist.mp4
  sandbox_config.json   the resolved sandbox configuration, the environment settings
                        (env_tools) included
  allowed_seeds.json    the development seeds the server will accept
  initial_policy.py     the template the agent started from
  policies/             index.json + v000/policy.py (the template), v001/policy.py, ...
  eval/vNNN/            hidden-seed evaluation of one version
    episodes.csv  smoothness.csv  summary.json  videos/  batch/
  replays/vNNN/         on-demand rollouts: batch/ + videos/
```

`policies/index.json` is the session's history of policy versions. The
environment server watches `sandbox/policy.py` while the session runs and
records every distinct content hash, with why it was recorded:

```json
{"version": 1, "dir": "v001", "sha256": "6a34987c...",
 "time": "2026-09-21T00:01:27+01:00", "trigger": "write",
 "helpers": [], "train_run": null, "train_success": null, "train_episodes": null}
```

`trigger` is `write` when the watcher saw new content, `run` when the agent
executed that content through `run_episodes.py` (then `train_run`,
`train_success` and `train_episodes` are filled from the run), and
`submission` for the content at session end. The last entry is what gets
scored.

`eval/vNNN/` and `replays/vNNN/` each hold a `batch/` that is an ordinary
replay-format batch (`metadata.json` + one `.npz` per episode), so the demo
viewer opens them with no special case.

## Running a session

### 1. Docker: nothing to set up by hand

An agent session gets exactly one network capability: its own model
endpoint. Everything else — package indexes, code hosts, search engines, the
demonstration dataset on the Hub — is refused, because this environment is
open source and a fetch or a search is a ground-truth channel into a
benchmark that is supposed to measure control design.

That is two docker networks and one allowlist proxy:
`bigym-agent-internal` is `--internal` (no route off the host) and the agent
container joins only it; `bigym-agent-egress` is an ordinary bridge; a
[tinyproxy](https://tinyproxy.github.io/) sits on both and forwards only the
hosts in its filter file (`FilterDefaultDeny Yes`): the model endpoints of
the shipped harnesses, nothing else. The harness images carry python with
the libraries a submitted policy may use, ffmpeg and ripgrep, and no pip,
curl or git. The Codex and Claude Code images install the latest CLI when
they are built, and `run.json` records the CLI version each session ran
with. To pin a version, build the image yourself with
`--build-arg CODEX_VERSION=<version>` (or `CLAUDE_CODE_VERSION`) under the
default image name; `bigym/loco/agent/docker/proxy/README.md` has the build
commands.

`bigym-agent run` creates the networks, starts the proxy and builds the
harness image the first time each is missing, so on a fresh host the first
`run` takes a few minutes longer and nothing needs preparing. Several
sessions, of any harness, share the one proxy.

```bash
uv run bigym-agent proxy status                 # what is there
uv run bigym-agent proxy up --harness codex     # prepare ahead of time (optional)
uv run bigym-agent proxy down                   # remove the proxy container
```

`--no-docker-setup` creates nothing, and a non-default `--docker-network`,
`--proxy` or `--container` is never touched.
`bigym/loco/agent/docker/proxy/README.md` has the manual equivalents and the
one-liner that checks from inside the network that the model endpoint
answers and nothing else does.

### 2. Credentials

Use an API key. Set it in the environment, or write it to a mode-600 file
under `~/.bigym-agent/` (`$BIGYM_AGENT_HOME`):

| harness | API key |
|---|---|
| `codex` | `$OPENAI_API_KEY` or `~/.bigym-agent/openai_api_key` |
| `claude` | `$ANTHROPIC_API_KEY` or `~/.bigym-agent/anthropic_api_key` |

```bash
export OPENAI_API_KEY=<your-key>
uv run bigym-agent run --task move_plate --harness codex --model <model-id>
```

Without an API key, the harness falls back to a subscription login:

| harness | subscription login |
|---|---|
| `codex` | the ChatGPT login in `auth.json` under `--codex-home` (`$BIGYM_AGENT_CODEX_HOME`, default `~/.bigym-agent/codex_home`), bind-mounted writable so the CLI can refresh the token; log in once with `CODEX_HOME=~/.bigym-agent/codex_home codex login` |
| `claude` | a token from `claude setup-token`, in `$CLAUDE_CODE_OAUTH_TOKEN` or a mode-600 `~/.bigym-agent/claude_oauth_token` |

Each session logs which mode it uses and what to set for the other, and
`run.json` records it under `auth` (`api_key` or `subscription`).

A credential is never on a command line: the host process table is
readable, and agents do run `ps`. In a container, Codex reads its API key
from an `auth.json` and Claude Code reads its key or token from a mode-600
env file, both in a session-private directory under
`~/.bigym-agent/sessions/` that is removed when the session ends, so neither
lands in the cell. With `--isolation soft` the key reaches the host CLI
through its environment.

```{note}
For a Claude Code subscription the token must come from `claude setup-token`.
Copying a host `.credentials.json` does not work: host and container then
refresh the same short-lived token, and the later refresh revokes the
earlier one mid-session.
```

### 3. Run one cell

```bash
export MUJOCO_GL=egl
uv run bigym-agent run --task move_plate --harness codex --model <model-id> --effort high
```

That creates `bigym-agent-runs/move_plate/` in the current directory (`--root`
puts the runs elsewhere; bigym's own `.gitignore` already lists
`bigym-agent-runs/`, add the same line to yours — a cell is hundreds of
megabytes of episodes and videos), builds the sandbox, starts the environment
server and waits for its workers, launches the harness in its container, waits
for it with a wall-clock limit, stops the server, writes `run.json` and the
transcript, and scores the submission. `--harness codex` runs OpenAI models and
`--harness claude` Claude models; a `--model` from the other vendor (`claude-*`
with codex, `gpt-*` or `o*` with claude) is refused before anything starts.
Add `--dry-run` to print every command it would run and write nothing — the
fastest way to check images, mounts, credentials and GPU choice before
spending a session:

```bash
uv run bigym-agent run --task move_plate --harness codex --model <model-id> --effort high --dry-run
```

Several tasks in one command: `--task move_plate pick_box --parallel 2`. The
environment renders with EGL and the session takes the GPU with the most free
memory, refusing to start when the best one is under `--min-free-mib` — a full
GPU makes EGL workers crash rather than queue. EGL numbers GPUs in its own
order, which on multi-GPU hosts is not `nvidia-smi`'s; the launcher reads
the driver's `EGL_CUDA_DEVICE_NV` map at start and logs it (`EGL map
(nvidia-smi index -> EGL device): ...`). `--gpu` picks the GPU by its
`nvidia-smi` index and renders on that GPU's EGL device;
`BIGYM_AGENT_EGL_MAP` (`cuda:egl` pairs) overrides the detected map.

Sessions are registered in `<root>/registry.sqlite`, which refuses a second
live run of the same cell and a duplicate container name.

### 4. Watch it

```bash
uv run bigym-agent status              # live sessions: budget, evaluation progress
uv run bigym-agent status --all        # finished ones too
uv run bigym-agent kill move_plate     # stop a session, its container and its server
uv run bigym-agent gc                  # close dead rows, list orphan containers
```

Every episode the agent runs during development is recorded by the server
as a replay batch under `<cell>/dev/vNNN_<stamp>/batch/` (one directory per
policy version that was current; `serve --no-dev-recording` switches it off),
so a running session can be watched from any machine that sees the directory:

```bash
uv run bigym-view --demo-dir bigym-agent-runs/move_plate --follow
```

`--follow` rescans the cell every few seconds: new episodes, new policy
versions and the growing token count appear in place; nothing is started at
launch time and nothing needs to be. Tick *auto-jump to newest* to follow
the latest batch.

### 5. What happens at the end

When the harness exits, `run` decides what the session is worth before
scoring it. A submission identical to the template the builder handed over is
`void`, not a 0/100 — a crashed session always leaves a valid-looking
`policy.py`, and scoring it is indistinguishable from a real failure. An
interrupted session is scored and recorded as interrupted.

Then, in order: the harness's raw event stream, stderr and home directory move
under `raw/`; `transcript.jsonl` and `transcript.md` are written; `run.json`
gets the verdict, the budget actually spent, token usage and cost; and the
last policy version is evaluated on the hidden seeds
(`bigym-agent evaluate <cell> --version <last>`) unless `--no-eval`.
`--eval-detached` starts that evaluation in the background instead of waiting
for it, and `--eval-episodes N` shortens it — the benchmark's own figure is
100 episodes, so a shorter block is a smoke test, not a result.

`run.json` records the evaluation's command together with the environment it
ran under (`MUJOCO_GL`, the EGL device, `BIGYM_AGENT_EVAL_JOBS`), its exit
status and where its `summary.json` is. An evaluation that dies leaves the
session's verdict untouched but marks `evaluation.status` as `failed`;
`bigym-agent status` then shows the cell as `EVAL FAILED (unscored)`, and
`bigym-agent evaluate <cell> --version <n>` scores it by hand. A cell is
scored only when its `summary.json` exists, never because the session ended.

`bigym-agent report <cell>` rebuilds the transcript from `raw/` afterwards —
useful when a price table or an adapter changed. `bigym-agent usage <root>`
totals tokens and API-equivalent cost per session, and `bigym-agent audit
<root>` flags every tool call in a transcript that named something outside
the sandbox (which is how you check a `--isolation soft` session after the
fact).

## Looking at the results

```bash
uv run bigym-agent policies bigym-agent-runs/move_plate            # versions with train/eval scores and provenance
uv run bigym-agent policies bigym-agent-runs/move_plate --rebuild  # recompute the derived fields of an older cell
```

Training scores come from the server ledger (every episode the agent ran,
attributed to the version that was current), not from what the agent
reports, and each version records the command index, budget, elapsed time
and cumulative tokens at which it was written.

```text
bigym-agent-runs/move_plate  task=move_plate
version  time                       trigger     train            eval
v001     2026-09-21T00:01:27+01:00  write       -                -
```

The viewer opens one cell, one session of cells, or every session at once.
`--demo-dir` takes several paths, and each one may be a cell
(`<root>/<task>/`), a session root (`<root>`) or a directory holding such
roots:

```bash
MUJOCO_GL=egl uv run bigym-view --demo-dir bigym-agent-runs/move_plate --port 8080
MUJOCO_GL=egl uv run bigym-view --demo-dir bigym-agent-runs   # one session
MUJOCO_GL=egl uv run bigym-view --demo-dir runs_s1 runs_s2 runs_s3
```

The startup log says what it found, which rollout it opened and why:

```text
[viewer] agent runs under /runs: 3 sessions (s1, s2, s3) x 9 tasks, 27 cells
[viewer] task move_plate · session s2 · cell /runs/s2/move_plate
[viewer] session: codex <model-id> effort=high interface=strict verdict=ok
[viewer] rollouts: evaluation 1 · development 6 · replays 0 · 7 policy versions
[viewer] opening Evaluation v007 · 100 episodes · 2% · submission — the evaluation of the submission
```

Every `eval/vNNN/batch`, `replays/vNNN/batch` and `dev/vNNN_<stamp>/batch`
is an ordinary replay-format batch, so playback, the camera panels and the
figure capture all work unchanged. The sidebar of a cell holds, in order:

- **Session**: a *task* dropdown, a *session* dropdown (one run root is one
  session) and the card from `run.json` — harness, model, effort, interface,
  budget, verdict, wall clock, tokens and cost. Moving either dropdown opens
  that cell's default rollout.
- **Rollouts**: a *kind* dropdown (Evaluation / Development / Replays) and a
  *rollout* dropdown within it. Items read `v007 · 100 episodes · 2%`,
  `while policy.py was v001 · 66 episodes · 3%` or
  `policy_ab12cd34 · 2 episodes`; the submission's evaluation is marked
  `· submission` and is what opens by default. Development batches are
  grouped by the `policy.py` version that was current while they ran, which
  is why they are labelled "while policy.py was v001" and not "v001's
  episodes" — agents often develop in other files while `policy.py` still
  holds the template.
- **Policy versions**: the table with train and eval scores, the command
  index, budget and tokens each version was written at; Replay and Evaluate
  buttons that run `bigym-agent replay` / `bigym-agent evaluate` as
  subprocesses and pick up the batches they write (logs in
  `<cell>/viewer_jobs/`); a *policy path* field to roll out any policy file
  (`replays/policy_<sha8>/`).
- **Policy code**: the selected version's `policy.py` with highlighting, its
  path and a `vscode://` link, and a unified diff against any other version.
- **Transcript**: what the agent said and ran between the selected version
  and the next one, with prev/next buttons to walk the session.
- **Compare**: up to six slots, each a *session* x *version* of the task on
  screen (or a loose policy file), one *seed* and a Compare button.
- **Episode**: every episode of the open rollout with seed, outcome, length
  and termination, selectable; a reward strip under the playhead.

### Comparing policies side by side

Compare mode rebuilds the scene with one environment per slot, each robot in
its own colour from a fixed six-colour palette and carrying a plate of text
above its pelvis:

```text
s2 · v007 · submission · train 40% · eval 2%
✗ · timeout
```

Slots are placed on a grid: one **column** steps `--compare-gap` metres
(2.5 by default) along +y, one **row** steps the same distance back along
-x, so a fourth policy stands behind the first rather than running off the
side of the screen. `--layout` (and the panel's *layout* dropdown) picks the
grid — `auto` keeps a single row up to three slots and folds into two
columns from the fourth, and `row`, `2 columns` and `3 columns` say it
outright. The camera starts where the whole grid is in view, on the -x side
and raised over the near row once there is more than one:

```text
[viewer] compare layout auto: 4 slots as 2 columns x 2 rows, gap 2.5 m
[viewer] compare slot 3: slot3 at row 1 column 1 (x -2.50, y +2.50), built hidden, …
[viewer] compare camera: at (-7.05, 1.25, 3.60) looking at (-1.25, 1.25, 0.80) — every scene in view
```

The **Compare** folder's *focus* dropdown flies every connected client's
camera to a pose in front of one slot's robot, looking at its pelvis; the
free view is the grid overview above. Its *camera panels* checkbox is off by
default — compare mode is about the scene, and the panels render four MuJoCo
views a tick for one slot out of several — and when it is on the *camera
slot* dropdown still says whose view they show.

Names are shortened where the shortening loses nothing. When every slot's
session starts the same way (`codex_high_s1`, `codex_high_s2`, …) the shared
prefix is dropped from the labels and the legend and named once in the
folder's header (`sessions codex_high_*`). A `v001` whose `policy.py` is
still byte-identical to the cell's `initial_policy.py` is marked
`(template)`, which is why three of them behave identically. And a score
that was never measured is named rather than dashed out: `no dev episodes`
where the agent ran none, `not evaluated` where no hidden-seed evaluation
exists — the same two words the **Policy versions** table's `train` and
`eval` columns use (a plain `—` still means "simply unknown").

The plate above each robot is an image that turns to face the camera. It
shows the outcome; the live step counter is in the sidebar legend, next to
the slot's colour swatch. The tint is display only: the camera panels still
render the policy's own, untinted view.

Every slot is built with its meshes hidden (the legend counts `building
2/3 …`), posed to frame 0 of its own replay — the environment's reset pose
is *not* the first recorded frame — and only then are all of them revealed
together.

The **Compare** folder in the sidebar opens with one slot per session that
holds the task, each on that session's submission, and says so:

```text
3 policies: one submission per session · Add adds another (up to 6)
```

*Add policy* appends a slot (session, version, an optional policy file and
its own *Remove*), up to six. One playhead drives every slot, a dropdown
says whose onboard cameras the camera panels show, and *Leave compare*
returns to the single-rollout scene. The same thing can be started from the
shell, which is also how a headless smoke test reaches it:

```bash
MUJOCO_GL=egl uv run bigym-view \
    --demo-dir runs_s1/move_plate runs_s2/move_plate runs_s3/move_plate \
    --compare "s1:v008,s2:v007,s3:last" --seed 620003 --layout auto
```

`--compare` is a comma-separated list of up to six `<session>:<version>`
slots. The version is `vNNN`, `NNN`, `last` or `submission`; the session is
matched by name or by any substring that names exactly one session. There is
no task in the spec — compare is always one task, the one the viewer would
open, so `--demo-dir` (or `--batch`) picks it. `--seed` is the seed every
slot is posed on.

Slots read their frames from what is already stored, preferring
`eval/vNNN/batch`, then `replays/vNNN/batch`, then `dev/`. **An evaluation
batch holds every hidden seed (620000-620099), so comparing submissions on a
hidden seed needs no new rollout at all**:

```text
[viewer] compare slot 0: s1 v008 seed 620003 <- eval/v008
[viewer] compare slot 1: s2 v007 seed 620003 <- eval/v007
[viewer] compare slot 2: s3 v017 seed 620003 <- eval/v017
[viewer] compare: every slot already stores seed 620003 (evaluation batches hold every hidden seed) — no replay needed
[viewer] compare slot 0: built hidden, 1701 frames from eval/v008/seed620003.npz, 132 geoms tinted
[viewer] compare slot 0: geom_rgba restored after the mesh bake — model untouched: True
[viewer] compare slot 0 label: s1 · v008 · submission · train 100% · eval 36%
[viewer] compare slot 0: posed to frame 0 of the replay (not the reset pose), pelvis (0.02, 0.00, 0.76)
[viewer] compare slot 0: billboard (image) at (0.02, 0.00, 1.91) = pelvis + offset (0.00, 0.00) + 1.15 m, facing (-1, 0, 0)
[viewer] compare: 3 scenes revealed together
[viewer] compare: task move_plate · seed 620003 · 3 scenes built · 3 slots as 3 columns x 1 row, gap 2.5 m
```

Any slot that has no episode on that seed is produced first with
`bigym-agent replay <cell> --version N --seeds <seed> --no-video`, and the
scene is built once every file exists. That works on a version that already
has a replay: `replays/vNNN` is **added to**, so a second compare on a new
seed records only what is missing and the rest of the batch is left where it
is.

```text
replay: adding to …/replays/v001/batch, which already holds 1 episode(s); seeds it has are kept (pass --force to re-record them)
seed 500: already recorded -> …/replays/v001/batch/seed500.npz
seed 501: success=0 length=700 end=timeout -> …/replays/v001/batch/seed501.npz
```

### Recording a comparison

`--record OUT.mp4` writes the scene from the overview camera and exits.
viser has no server-side renderer, so a headless Chrome/Chromium is started
on the page and asked for one picture per video frame while the playhead is
stepped by hand — the file is smooth at its frame rate however slowly the
software rasteriser draws (`--browser none` waits for you to open the URL
in your own browser instead, which is much faster on a machine with a GPU):

```bash
MUJOCO_GL=egl uv run bigym-view --demo-dir <runs>/s1/move_plate <runs>/s2/move_plate \
    --compare "s1:v008,s2:v007" --seed 620005 \
    --record compare.mp4 --record-seconds 10
```

The episodes play at their recorded speed and wrap around; `--record-size`
(default 1920x1080), `--record-fps` (default: the episodes' own rate),
`--record-speed` (2 = twice real time) and `--record-then-stay` (keep
serving afterwards) complete the set. Missing rollouts are produced first,
exactly as for a live compare.

The camera is `--record-view overview` (the live compare's camera: behind
the robots, high, every scene in view) or `oblique` — in front of them and
to one side, lower and closer, the three-quarter view that shows the hands
and what is on the tables rather than four backs. Either way the lens is
opened just enough to hold the slots' own box; `--record-zoom 1.35` narrows
it further. The billboards are drawn `--record-label-scale` times bigger
than live (default 1.5): a video is watched from further away than a tab.
A whole 1701-step episode as a 17-second clip, watchable labels included:

```bash
MUJOCO_GL=egl uv run bigym-view --demo-dir <runs>/s1/move_plate <runs>/s2/move_plate <runs>/s3/move_plate \
    --compare "s1:v008,s2:v003,s2:v007,s3:last" --seed 620005 \
    --record move_plate.mp4 --record-seconds 17 --record-fps 25 --record-speed 2 \
    --record-view oblique --record-zoom 1.35 --record-label-scale 1.8
```

The frames come from a browser, so the machine's GPU is what sets the pace:
a headless Chrome on a server without one draws about a frame every five
seconds through SwiftShader, while `--browser none` with the page open in a
laptop's Chrome captures at tens of frames a second. The runs are portable:
a cell's `eval/vNNN/batch/*.npz`, `metadata.json`, `policies/`, `run.json`
and the transcript are all the viewer needs (no `sandbox/`, `raw/` or
Docker), and the lower-body controller runs on CPU through onnxruntime.

Both commands are also useful on their own. `replay` rolls a version out on
seeds you choose and writes a viewable batch plus mp4s:

```bash
uv run bigym-agent replay bigym-agent-runs/move_plate --version 1 --seeds 620000-620004
uv run bigym-agent replay bigym-agent-runs/move_plate --version 1 --seeds 620000,620042 --no-video
```

`evaluate` scores a version on the hidden block, writing `episodes.csv`
incrementally and `summary.json` at the end:

```bash
uv run bigym-agent evaluate bigym-agent-runs/move_plate --version 1
```

`summary.json` carries the success rate, the substrate fingerprint, the
interface and image cap the evaluation ran with, the policy's sha256, and
`protocol_violations` (empty when the policy asked for nothing it was not
given). An evaluation is one block of seeds, so `evaluate` still refuses to
overwrite a finished one unless you pass `--force`; a replay is a set of
seeds, so it adds the ones its directory does not have and `--force` there
means "re-record every seed I asked for". Both resolve a relative cell
path.

Outside a cell, `evaluate` and `record` take a bare policy file:

```bash
uv run bigym-agent evaluate --task move_plate --policy policy.py --out eval_out
uv run bigym-agent record   --task move_plate --policy policy.py --seeds 620000,620042 --out clips
```

And the demonstration videos can be rendered on their own, at any resolution
and including the outside view the sandbox never hands to an agent:

```bash
uv run bigym-agent demo-video --task move_plate --out demo_clips --views head,third --size 224x224
```

## Bring your own agent

Any harness can run the benchmark. The simplest way is `--harness custom`:

```bash
uv run bigym-agent run --task move_plate \
    --harness custom --command 'my-harness --prompt PROMPT.md'
```

The command runs with the **sandbox as its working directory** and
`AGENT_SANDBOX`, `BIGYM_AGENT_PROMPT` and `BIGYM_AGENT_CELL` in its
environment; the task statement is `PROMPT.md` in that directory.
Everything else is identical — the same server, the same budget, the same
policy-version snapshots, the same evaluation. Its stdout and stderr are
captured under `raw/`, and the stdout becomes the cell's transcript, so
`bigym-agent report` works for a custom session too. A custom command is an
arbitrary host program, so it runs under `--isolation soft`: nothing enforces
the network allowlist, and `bigym-agent audit` afterwards is how you check
what it reached for.

If you want to drive the pieces yourself, they are three commands:

```bash
uv run bigym-agent sandbox --task move_plate --cell runs/move_plate

MUJOCO_GL=egl uv run bigym-agent serve --task move_plate \
    --sandbox runs/move_plate/sandbox --ledger-dir runs/move_plate \
    --workers 1 --budget-steps 101000 \
    --allowed-seeds runs/move_plate/allowed_seeds.json
# ... your agent works in runs/move_plate/sandbox ...
# SIGTERM the server: it records the final policy as `submission` and exits.

uv run bigym-agent evaluate runs/move_plate --version 1
```

`serve` prints one `ready` line per worker; wait for all of them before
starting the agent. The snapshot watcher runs whether or not you used `run`,
so `policies/index.json` is there either way.

## Reproducibility knobs

Three flags change what a number means, so they belong in the label of any
result you report. They are environment settings, which `sandbox`, `serve`,
`evaluate`, `record` and `run` all take with the same names: `run` hands its
settings to the sandbox builder, the server and the evaluation alike, the
sandbox builder records them in `sandbox_config.json` (`env_tools`), and the
evaluation and the replays of a cell read them back from there, so a cell is
always scored in the environment its session ran against.

- **`--interface strict|tools`** — the I/O contract, as above. The evaluation
  stamps it into `summary.json`.
- **`--image-cap WxH`** — the policy-facing resolution ceiling, `84x84` by
  default, exactly what the learned baselines are rendered at. The server
  refuses a larger request, and the demonstration is rendered at the same cap
  so an agent cannot measure the scene offline at a precision its own runtime
  images will never reach.
- **`--slew`** — the command rate limiter (6 rad/s arm joints, 0.7/s base,
  0.03 rad/step² arm acceleration; `--joint-vmax`, `--accel` (`None` for
  velocity caps only) and `--lowpass` tune it). **Off by default and
  not part of the benchmark**: the learned baselines' commands are not
  rate-limited either, so a rate-limited agent policy is not comparable to
  them. It exists to replay a submission that was developed against it.

`--tier privileged` (object positions in the observation) and
`--allow-external-cameras` (the free third-person views) are debugging and
replay aids on the same footing: useful for understanding a failure, never
part of a reported number.

## What the agent sees

`bigym/loco/agent/templates/PROMPT.md` is the whole prompt. It is filled in
per session — the demonstration line, the seed count, the budget and the
task sentence — and copied into the sandbox as `PROMPT.md`:

```text
You are writing the control policy for a simulated Unitree G1 humanoid robot.
Your submission is `policy.py` in this directory. `docs/api.md` describes the
observation, action and tool interface and how to run episodes.

What you have
- {video_line}
- Training seeds: the {n_seeds} seeds listed in `seeds.json`, the seeds the demonstration
  set was collected on. Object placements are drawn per seed; the hidden evaluation
  seeds draw from the same distribution.
- A budget of {budget} environment steps for everything you run. Each reset
  costs 200 steps plus the steps of the episode.

How you are scored
- After you stop, `policy.py` is run once on 100 hidden seeds. Your score is the
  success rate over those 100 episodes; nothing you report is used. An episode in
  which the robot falls counts as failed.
- Episodes are re-run and must end the same way: no wall-clock time,
  no unseeded randomness.

Rules
- The policy is hand-written control logic. It may not contain learned
  components (no training a network, regression or lookup table from data).
  Take whatever you can from the demonstration video; tuning constants by
  running episodes is fine.
- Libraries: numpy, scipy, OpenCV (cv2), pillow and the Python standard library;
  nothing else is installed and nothing can be installed. No network, no changes
  to the harness. The simulator is reachable only through the client.
- The robot's own cameras (`head`, `left_wrist`, `right_wrist`) and its body
  state are the only view of the scene, for you while developing as much as
  for the policy. There is no outside camera.
- Write code to files and run them with `./python file.py`; inline `python -c '...'`
  is refused by the sandbox and only wastes a turn.

Task: {task_sentence}
```

Everything else the agent has is in the sandbox: `docs/api.md` (the action
table, the `low_dim_obs` layout for its action dimension, the client and the
tools its interface grants), `harness/` (the client, the episode contract and
the batch runner), `run_episodes.py`, `seeds.json`, the demonstration files,
and a `policy.py` template to fill in.
