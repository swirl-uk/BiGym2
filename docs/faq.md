# FAQ

## Installation

### `No module named 'cv2'` (or `scipy`, `mink`)

The environment was synced without the extra that provides the module. Name
every extra in one `uv sync`, as in [Installation](installation.md#extras).

## Rendering

### Rendering fails on a headless Linux machine

MuJoCo defaults to a windowed OpenGL context, which needs a display. Render
through EGL instead:

```bash
export MUJOCO_GL=egl
```

### Rendering fails on macOS with `MUJOCO_GL=egl`

macOS has no EGL. Leave `MUJOCO_GL` unset and drop it from the commands in
these pages.

## Environment and demonstrations

### Can I set `reset_warmup_steps` to 0 to save time?

No. The demonstrations start after the 200-step warmup, so a run without it
does not match them and is not official.

### Which episode length should I use?

Use the length `make(task)` sets (`TASKS[task].episode_length`). The
`episode_length` in a batch's `metadata.json` is the collector's cap
([Episode budgets](official_configuration.md#episode-budgets)).

### `get_demos()` raises `DemosUnavailableError`

The task has no published demonstrations yet. `bigym-download --list` shows
which tasks do.

### `get_demos()` says the demonstrations do not match this env

The loader compares the dataset's robot, backend, downsample rate, cameras and
observation and action dimensions with the env. A camera override such as
`camera_keys=("head",)` is the usual cause. Build the env with
`make(task)` or `make_gym(task)`, or re-render the dataset for another camera
setup with `bigym-rerender-lerobot`.

### How do I work offline?

Download once, then tell the Hub client to stay on its cache:

```bash
uv run bigym-download --all
export HF_HUB_OFFLINE=1
```

## Evaluation

### My result has no leaderboard record

The runner writes a record only for an official run with 100 episodes.
`result["protocol_violations"]` lists every result-affecting override.
[Official overrides](official_configuration.md#official-overrides) lists the
ones that keep a run official.

### Are my numbers comparable with someone else's?

Official results are comparable when their `substrate_version` matches. See
[Determinism and versioning](determinism.md#substrate-version-and-fingerprint).

## VR collection

### Can I collect demonstrations on macOS?

No. The collector needs an OpenXR runtime, and macOS has none. Use Linux with
WiVRn and a Quest 3, as in [Collecting demonstrations](demo_collection.md).

## Coding-agent benchmark

### `bigym-agent run` refuses my model

Each harness runs only its own vendor's models (`codex` for OpenAI, `claude`
for Anthropic). Switch the harness to match the model.

### My Claude Code subscription session stops authenticating mid-run

A copied host `.credentials.json` shares one short-lived token between host
and container, and one refresh revokes the other. Create a dedicated token for
container sessions:

```bash
claude setup-token        # then set CLAUDE_CODE_OAUTH_TOKEN
```
