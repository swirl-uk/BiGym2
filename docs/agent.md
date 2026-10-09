# The coding-agent benchmark

A coding agent gets a one-sentence task, a video of a human demonstration and
a budget of environment steps, and writes `policy.py`. `bigym-agent` scores
that file on the same evaluation seeds as the learned policies, through the
same frozen lower-body controller.

## Rules and scoring

A cell is one task × one session. Inside it:

- The task statement is one sentence naming what to do and to which object.
  It says nothing about positions, sizes or the success tolerance.
- The demonstration is one human demo rendered from the robot's own cameras
  (`head`, `left_wrist`, `right_wrist`) at 84×84 and 25 fps. There is no
  outside camera, for the agent or for the policy.
- The budget is 101,000 environment steps. A reset costs 200 steps (the
  controller warm-up) plus the steps of the episode. The server refuses the
  step that would exceed the budget.
- The development seeds are the seeds the demonstrations were collected on
  (60 for a published task), listed in `seeds.json`. The server refuses any
  other seed.
- The score is the success rate over the 100 hidden seeds of the
  [evaluation protocol](official_configuration.md#evaluation-protocol).
  Nothing the agent reports about its own performance is used.
- A fall is a failure.
- After the 100 evaluation episodes, the first 10 are rerun
  (`--spot-check N`) and must give identical per-episode success. A policy
  that reads the wall clock or unseeded randomness fails this check.
- The policy runs in its own interpreter and exchanges observations and
  actions with the simulator over a pipe. A policy whose code imports
  `bigym`, `mujoco`, `dm_control` or `mink` is rejected and scores 0.
- The policy is hand-written control with no learned component: no network,
  regression or lookup table fitted to data. Tuning constants by running
  episodes is allowed.
- The policy may use numpy, scipy, OpenCV, pillow and the standard library.
  Nothing else is installed and nothing can be installed.

The agent's full prompt is
[`bigym/loco/agent/templates/PROMPT.md`](https://github.com/swirl-uk/BiGym2/blob/main/bigym/loco/agent/templates/PROMPT.md).
Each session fills in the task sentence, the seed count, the budget and the
demonstration line. The sandbox also holds `docs/api.md` (the action, the
observation, the client and the tools the interface grants), the client under
`harness/`, `run_episodes.py`, `seeds.json`, the demonstration videos and a
`policy.py` template.

## Interfaces

`--interface strict` is the default and the benchmark row. The policy gets the
learned baselines' inputs and outputs: the three 84×84 cameras and the raw
`low_dim_obs` vector in, the physical action out. It has no named state, wrist
positions, camera calibration or IK.

`--interface tools` is an ablation, reported as a separate row. It adds named
state, wrist positions in the world frame, `camera_info`, `pixel_to_ray` and
an IK solver. `--no-ik` and `--no-hand-pos` withhold one tool each.

## Install

```bash
uv sync --extra agent
export MUJOCO_GL=egl
```

The `agent` extra adds OpenCV and scipy, which a submitted policy may import,
and `mink` for the IK of the `tools` interface. You also need `ffmpeg` on
`PATH` for the videos, and Docker for the container isolation the benchmark
rows use. The commands below run as `uv run bigym-agent ...`.

## Run a session

The agent container can reach only its model API. `bigym-agent run` creates
the Docker networks and the allowlist proxy and builds the harness image the
first time each is missing.
[`docker/proxy/README.md`](https://github.com/swirl-uk/BiGym2/blob/main/bigym/loco/agent/docker/proxy/README.md)
covers the setup, the manual commands and how to pin a CLI version.

### Credentials

Use an API key, in the environment or in a mode-600 file under
`~/.bigym-agent/` (`$BIGYM_AGENT_HOME`):

| harness | API key |
|---|---|
| `codex` | `$OPENAI_API_KEY` or `~/.bigym-agent/openai_api_key` |
| `claude` | `$ANTHROPIC_API_KEY` or `~/.bigym-agent/anthropic_api_key` |

Without a key, `codex` uses the ChatGPT login in
`~/.bigym-agent/codex_home` (`CODEX_HOME=~/.bigym-agent/codex_home codex login`)
and `claude` uses a token from `claude setup-token`, in
`$CLAUDE_CODE_OAUTH_TOKEN` or `~/.bigym-agent/claude_oauth_token`.
`run.json` records which mode a session used. A copied host
`.credentials.json` fails in a container, because host and container refresh
the same token and revoke each other mid-session.

### Run

```bash
uv run bigym-agent run --task move_plate --harness codex --model <model-id> --effort high
```

This writes the cell `bigym-agent-runs/move_plate/` in the current directory
(`--root` puts it elsewhere). A cell holds hundreds of megabytes, so add
`bigym-agent-runs/` to your `.gitignore`.

Each harness runs only its own vendor's models (`codex` for OpenAI, `claude`
for Anthropic), and `run` refuses a mismatch before anything starts.
`--dry-run` prints every command the run would execute and writes nothing.
`--task move_plate pick_box --parallel 2` runs two sessions at once.

Each session takes the GPU with the most free memory and refuses to start if
it has less than 6000 MiB free (`--min-free-mib`). `--gpu` picks a GPU by its
`nvidia-smi` index.

### Watch

```bash
uv run bigym-agent status              # live sessions: budget, evaluation progress
uv run bigym-agent status --all        # finished ones too
uv run bigym-agent kill move_plate     # stop a session, its container and its server
uv run bigym-view --demo-dir bigym-agent-runs/move_plate --follow
```

With `--follow` the viewer rescans the cell every few seconds, so new
development episodes and policy versions appear while the agent works.

## After the run

A session whose `policy.py` is still the untouched template is `void` and is
not scored. Otherwise `run` scores the last policy version on the hidden
seeds. An interrupted session is scored and marked `interrupted`.

`--no-eval` skips the evaluation, `--eval-detached` runs it in the background,
and `--eval-episodes N` shortens it for a smoke test. Reported numbers use
100 episodes.

If the evaluation dies, `bigym-agent status` shows the cell as
`EVAL FAILED (unscored)`. Score it by hand:

```bash
uv run bigym-agent evaluate bigym-agent-runs/move_plate --version <n>
```

A cell is scored only once `eval/vNNN/summary.json` exists. `evaluate` will
not overwrite a finished evaluation without `--force`.

`replay` rolls a version out on seeds you choose and writes a viewable batch
and mp4s to `replays/vNNN/`. It runs only the seeds the directory does not
already hold. `--no-video` skips the mp4s.

```bash
uv run bigym-agent replay bigym-agent-runs/move_plate --version 1 --seeds 620000-620004
```

`summary.json` holds the success rate, the interface and image cap the
evaluation ran with, and the substrate fingerprint. `protocol_violations`
lists the config fields that depart from the task's official configuration,
and is empty for an official run. A policy that imported the simulator is
named in `rejected`.

## Reading results

```text
bigym-agent-runs/move_plate/
  run.json                configuration, verdict, budget spent, tokens and cost
  transcript.md           what the agent said and ran
  policies/               every version of policy.py the agent wrote
  eval/vNNN/summary.json  the hidden-seed score of version NNN
  dev/                    the agent's own episodes, per policy version
  sandbox/                what the agent saw and wrote
```

The server snapshots `sandbox/policy.py` each time its content changes. The
last version is the submission. `policies` lists every version with its
training and evaluation scores:

```bash
uv run bigym-agent policies bigym-agent-runs/move_plate
```

Training scores come from the server ledger. Each episode counts toward the
version that was current when it ran.

The viewer opens a cell, a runs root, or several roots at once:

```bash
uv run bigym-view --demo-dir bigym-agent-runs/move_plate
```

### Comparing policies

`--compare` plays up to six policies of one task side by side on one seed.
Each slot is `<session>:<version>`, where the version is `vNNN`, `NNN`, `last`
or `submission` and the session is matched by name or by a substring that
names exactly one session.

```bash
uv run bigym-view --demo-dir runs_s1/move_plate runs_s2/move_plate \
    --compare "s1:v008,s2:last" --seed 620003
```

Adding `--record compare.mp4 --record-seconds 10` writes the comparison to an
mp4 and exits. The recording renders in a headless Chrome. `--browser none`
waits for you to open the page in your own browser, which is much faster on a
machine with a GPU. `bigym-view --help` lists the remaining flags.

## Bring your own agent

```bash
uv run bigym-agent run --task move_plate \
    --harness custom --command 'my-harness --prompt PROMPT.md'
```

The command runs in the sandbox directory, where `PROMPT.md` holds the task,
with `AGENT_SANDBOX`, `BIGYM_AGENT_PROMPT` and `BIGYM_AGENT_CELL` in its
environment. The server, budget, version snapshots and evaluation are the
same as for the shipped harnesses. Its stdout becomes the cell's transcript.

A custom command runs on the host under `--isolation soft`, so nothing
enforces the network allowlist. Check afterwards which tool calls named
anything outside the sandbox:

```bash
uv run bigym-agent audit bigym-agent-runs
```

To drive the pieces yourself:

```bash
uv run bigym-agent sandbox --task move_plate --cell runs/move_plate
uv run bigym-agent serve --task move_plate --sandbox runs/move_plate/sandbox \
    --ledger-dir runs/move_plate --allowed-seeds runs/move_plate/allowed_seeds.json
# Wait for one "ready" line per worker, run your agent in runs/move_plate/sandbox,
# then send SIGTERM to the server. It records the final policy as the submission.
uv run bigym-agent evaluate runs/move_plate --version <n>
```

## Flags that change a result

Report these with any result. `sandbox`, `serve`, `evaluate`, `record` and
`run` take them under the same names, and a cell's evaluation reads them back
from its `sandbox_config.json`.

- `--interface strict|tools`, described above.
- `--image-cap WxH`, the policy-facing resolution ceiling. The default is
  `84x84`, the resolution of the learned baselines. The demonstration is
  rendered at the same cap.
- `--slew`, a rate limiter on the arm and base commands. It is off by default
  and not part of the benchmark.

`--tier privileged` (object positions in the observation) and
`--allow-external-cameras` (outside views) are debugging aids. A number
produced with either is not a benchmark result.
