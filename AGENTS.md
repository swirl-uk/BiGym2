# AGENTS.md

BiGym 2.0 is a MuJoCo benchmark for humanoid whole-body manipulation: a
Unitree G1 whose legs are driven by a frozen GR00T-WBC controller, kitchen
tasks, VR demonstrations on the Hugging Face Hub, and a coding-agent benchmark
(`bigym-agent`). The user docs are in `docs/`; read the relevant page before
guessing how something works.

## Environment

- Use `uv` and `uv run`, never `pip` or bare `python`.
- `uv sync` makes the environment match exactly the extras named in that one
  command, so name them all at once: `uv sync --extra vr --extra agent`. After
  a hand-picked sync, run commands with `uv run --no-sync`.
- On headless Linux, set `MUJOCO_GL=egl`.

## Checks

Run these before calling a change done:

- Lint and format: `uv run pre-commit run --all-files`
- Types: `uv run ty check` (needs the `vr` and `agent` extras). After a
  mujoco version change, run `bash typings/generate_mujoco_stubs.sh` and
  commit the stubs.
- Tests: `MUJOCO_GL=egl uv run pytest tests/`. `--run-slow` adds whole
  GR00T-in-the-loop episodes and rendering; it needs a GPU and CI skips it.
- After changing how a harness is launched, also run
  `BIGYM_AGENT_CLI_TESTS=1 uv run pytest tests/test_agent_host_isolation.py`
  (starts the real Claude Code and Codex CLIs; both must be logged in).
- Docs: `uv sync --group docs && uv run --no-sync sphinx-build -b html -W docs docs/_build/html`
  (warnings fail the build).

## Invariants

- IMPORTANT: the substrate (physics options, robot model, controller loop,
  obs/action layout, success semantics) is frozen under `SUBSTRATE_VERSION` in
  `bigym/loco/fingerprint.py`. A failing `tests/test_substrate_golden.py`,
  `tests/test_bit_exact_replay.py` or `tests/test_task_success_from_demos.py`
  without an intended substrate change is a regression: fix the code, never
  regenerate the goldens or `tests/fixtures/demo_endpoints.npz` to make it
  pass. For an intended change, follow the regeneration steps in those files'
  docstrings and review the reported diff.
- Bit-exact replay depends on the Newton solver and the full controller
  snapshot (`docs/determinism.md`). Env and evaluation code must not read the
  wall clock or unseeded randomness.
- `bigym/loco/adapters/_groot/` is vendored upstream code. Record every edit
  in its `PROVENANCE.md`. No torch and no git dependencies.
- The README snippet (`make_gym(task)`, `env.get_demos(n)`, the gymnasium
  loop) is the reference usage and must keep running as written.
- `bigym/loco/agent/templates/` is everything a benchmark agent is told.
  Editing it changes the benchmark: update `prompt_version` in
  `bigym/loco/agent/sandbox.py` and never add task-specific hints. Nothing
  outside the sandbox may reach an agent session.
- Reported benchmark results come from `--isolation container` (the default).
  `--isolation soft` is for development only; never switch to it to get past
  a Docker error.

## Git

- Commit messages follow Conventional Commits, `<type>(<scope>): <summary>`,
  with the types and scopes listed in `CONTRIBUTING.md`. Pull requests are
  squash-merged, so the PR title is the commit message.
