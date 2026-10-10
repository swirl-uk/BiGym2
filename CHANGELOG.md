# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `make_gym(...).get_demos(n, decode_images=False)` keeps the demo frames as
  PNG bytes and decodes an observation when it is indexed: about 7 GB at peak
  instead of 22.5 GB for `stack_blocks`.

### Changed

- A fresh env now maps `[-1, 1]` actions onto the raw outer bounds, so a zero
  action commands the middle of each range (a 0.7 m pelvis height). Previously
  it used `[-1, 1]` per slot, which commanded a 0 m pelvis height and capped
  arm targets at ±1 rad. To evaluate a policy trained under the old default,
  restore its ranges with `env.set_action_stats(min, max)`.
- `get_demos()` now normalizes demo actions over `env.action_stats` and leaves
  them unchanged, so the actions mean the same in a fresh env. Previously it
  switched the env to the dataset's ranges, which a fresh evaluation env did
  not have.
- `get_demos()` with `frame_stack=1` no longer copies demo frames.
- Loading demonstrations needs less memory: 16.5 GB instead of 22.5 GB at
  peak for the 60 demos of `stack_blocks`.
- `bigym-agent` puts the published demonstration videos in a sandbox and
  downloads only them and the task's metadata, a few MB per task. Previously
  it downloaded all of a task's demonstrations (5.6 GB for `stack_blocks`) to
  render one. Other demonstration settings still render locally, and download
  only the data file that holds the demonstration they show.
- `examples/train_act.py` trains offline on the demonstrations, then runs the
  evaluation protocol. Previously it took one gradient step per env step,
  which ran the simulator for nothing, and kept every decoded frame in memory.

### Fixed

- `make_gym(..., normalize_low_dim_obs=True).get_demos()` now sets the low-dim
  mean/std from the demos. Previously the gym path skipped it.
- `examples/train_act.py` now resets action chunking between evaluation
  episodes. Previously the next episode started with the last one's chunks.

## [1.0.0]

First release of BiGym 2.0. It builds on [BiGym](https://github.com/NeuracoreAI/bigym)
(whose releases, up to 4.2.0, are listed in
[its changelog](https://github.com/NeuracoreAI/bigym/blob/master/CHANGELOG.md))
and replaces its floating-base robots with a walking humanoid.

### Added

- **Unitree G1 with Dex1-1 grippers**, the benchmark embodiment.
- **Controller in the loop** (`bigym.loco`): NVIDIA's GR00T-WBC lower-body
  policy runs inside every environment step while the agent commands the
  upper body, the grippers and a base velocity. The runtime ships in the
  package and runs on onnxruntime, so there is no torch dependency.
- **20 benchmark tasks** with G1 scene presets, across reaching, table-top,
  dishwasher and kitchen-counter scenes. The other 20 upstream tasks are
  registered and will be released later.
- **One configuration object** (`bigym.loco.EnvConfig`): a frozen dataclass
  whose defaults are the official configuration. `make(task, config=None,
  **overrides)` and `make_gym` build any task from it or from keyword
  overrides; each task's `TaskSpec` (`bigym.loco.tasks`) holds its
  episode budget and its few departures. The env records `env.config` and
  `env.config_overrides`, and `EnvConfig.from_metadata` rebuilds the
  configuration a demonstration batch was recorded in.
- **Evaluation protocol** (`bigym.loco.eval`): 100 fixed evaluation seeds;
  an episode is a success when the env reports the task done
  (`env.episode_succeeded()`: the task predicate held for
  `success_hold_seconds`) and the robot never fell during the episode; a run whose configuration departs from the official one in a
  result-affecting field gets no leaderboard record; every result carries a
  substrate fingerprint of the physics it was produced with.
- **Building on BiGym in your own package**: `register_task` and
  `register_backend` (with a `BackendBinding`) add tasks and lower-body
  controllers from another package, and `"pkg.module:ATTR"` names reach them
  without registration; see `docs/extending.md`.
- **Demonstrations on the Hugging Face Hub**
  ([`SWIRL-Lab/bigym-g1-native60`](https://huggingface.co/datasets/SWIRL-Lab/bigym-g1-native60)):
  60 VR demonstrations per task with full simulator state. `env.get_demos()`
  downloads a task on first use and `bigym-download` fetches ahead of time.
  `GymnasiumEnv.get_demos` returns gymnasium-shaped trajectories.
- **Coding-agent benchmark** (`bigym-agent`, extra `agent`): an agent gets
  the task sentence, one demonstration video and an environment-step budget,
  writes `policy.py`, and is scored on the same hidden seeds as learned
  policies. Codex and Claude Code harnesses run in Docker behind an
  allowlist proxy.
- **VR collection and viewing** (`bigym.vr`): `bigym-collect` (extra `vr`)
  records in the official env and judges an attempt with the evaluation's
  `is_success`, with session settings in `CollectConfig` (command-line
  flags); `bigym.loco.demos.success_hold` cuts a raw batch on the step the
  official env latches success. `bigym-view` plays the published
  demonstrations, your own batches and agent sessions, and compares policies
  side by side.
- **LeRobot export** (`bigym-export-lerobot`, `bigym-rerender-lerobot`,
  extra `lerobot`).
- **`examples/train_act.py`**: a minimal ACT example, ending in the
  evaluation protocol.
- **Command lines built with tyro**: each command's flags come from a
  settings dataclass, booleans take `--x` / `--no-x`, and
  `--tyro-write-completion {bash,zsh,tcsh} PATH` writes a shell completion
  script.
- **Licensing records**: `NOTICE`, `THIRD_PARTY_NOTICES.md`, `CITATION.cff`
  and a dataset card template.
- **Type checking** with ty, with generated MuJoCo stubs in `typings/`.

### Changed

- **One top-level package**: the wheel installs only `bigym`; the VR code and
  development scripts are `bigym.vr` and `bigym.tools`.
- **Native MuJoCo scene**: an environment builds its scene as one
  `mujoco.MjSpec`, compiles it once and exposes `env.spec`, `env.model`,
  `env.data` and `env.simulation` in place of `env.mojo`; robot and prop
  elements are MuJoCo's own `MjsBody`, `MjsGeom` and so on.
- **Dependencies**: onnxruntime and tyro are core dependencies; the VR
  runtime moved to the `vr` extra; `dm_control`, `mojo`, `mujoco_utils`,
  `dearpygui`, `safetensors` and `wget` are gone.

### Removed

- The H1, Hello Robot Stretch and Google Robot embodiments and
  `TorqueActionMode`.
- The upstream floating-base demonstration store (`demonstrations/`); the
  G1 dataset replaces it.
- The dearpygui demo recorder and player, and the floating-base examples.
