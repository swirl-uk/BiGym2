---
name: running-agent-sweeps
description: Runs a sweep of BiGym coding-agent benchmark sessions (one harness and model, a task list, several sessions per task), watches it, checks that every cell was scored, and reports per-task results. Use when launching, monitoring or summarizing a bigym-agent experiment wave.
disable-model-invocation: true
---

# Running a bigym-agent sweep

A sweep is one configuration (harness, model, effort and the flags that
change a result) run on a task list, with N sessions per task. `bigym-agent
run --root <name> --sessions N` writes session k to the runs root
`<name>_s<k>`; a cell is `<root>/<task>/`. Every flag and file named here
is described in `docs/agent.md`.

Copy this checklist and keep it updated:

```
- [ ] 1. Protocol written down
- [ ] 2. Preflight passed
- [ ] 3. Sweep launched
- [ ] 4. Every cell finished
- [ ] 5. Every cell scored or excluded
- [ ] 6. Report written
```

## 1. Protocol

Write these into `<name>_notes.md` next to the roots before launching:

- `--harness`, `--model`, `--effort`, `--interface`, `--image-cap`,
  `--service-tier` and the harness image. Sweeps that will be compared differ
  in one of these only.
- The task list and N. Use N >= 3: scores vary widely between sessions of
  the same task, and two sessions are not enough to compare models.
- Which session ends are scored. A cell with `verdict.state` `ok` always is;
  decide now whether one that hit the wall-clock limit (`interrupted`,
  reason `exit=-1`) is too. Every other `interrupted` cell (stopped by hand,
  or by a failure outside the agent) is excluded and rerun.
- A new name for the sweep. Never run a second sweep into existing roots.

## 2. Preflight

- `docker info` works without sudo. A new docker group membership needs a
  new login shell.
- Credentials are set up as in `docs/agent.md` ("Credentials").
- `nvidia-smi`: each running session needs 6000 MiB free on its GPU. On a
  shared machine, pin the sweep with `--gpu <index>`.
- `--parallel p` is the number of sessions running at once over the whole
  sweep. Choose it from the account's usage window and from
  `uv run bigym-agent usage <roots of an earlier sweep>` (tokens and cost per
  cell).
- Dry run and read the commands:

```bash
uv run bigym-agent run --task <tasks> --root ~/bigym-agent-runs/<name> \
    --sessions <N> --harness <harness> --model <model> --effort <effort> --dry-run
```

## 3. Launch

From the bigym checkout:

```bash
mkdir -p ~/bigym-agent-runs
nohup uv run bigym-agent run --task <tasks> --root ~/bigym-agent-runs/<name> \
    --sessions <N> --parallel <p> \
    --harness <harness> --model <model> --effort <effort> \
    > ~/bigym-agent-runs/<name>.log 2>&1 &
```

Keep the default `--isolation container`. Record the launch time and the
bigym commit in the notes.

## 4. Monitor

```bash
uv run bigym-agent status --root ~/bigym-agent-runs/<name>_s1
uv run bigym-view --demo-dir ~/bigym-agent-runs/<name>_s1/<task> --follow
```

- `uv run bigym-agent kill --root <root> <task>` stops one cell and leaves
  the rest of the sweep running. The cell is recorded as `interrupted`
  (reason `killed`) and is not scored.
- If the `ledger.jsonl` of every running cell stops at the same second, the
  sweep was killed. A rate limit slows sessions down instead.

## 5. Close out

Check every cell under every root:

- `run.json`: `verdict.state` and `verdict.reason`. Apply the rule from
  step 1; list every excluded cell in the notes with its reason, and rerun
  it in its root with `--force`. `void` means no submission and no score.
- `eval/vNNN/summary.json` exists. `status --all` shows the cell as
  `EVAL FAILED (unscored)` otherwise; score it with
  `uv run bigym-agent evaluate <cell> --version <n>`.
- `summary.json`: `episodes` is 100, `protocol_violations` is empty,
  `spotcheck_identical` is true.
- `uv run bigym-agent usage <roots> --csv <name>_usage.csv` for tokens and
  cost.

## 6. Report

Per task, every session's `success_rate`, their mean and their range:

```
| task | s1 | s2 | s3 | mean | range |
```

Take the suite mean only over tasks that have all N sessions. Under the
table, give the protocol from step 1, `harness_version` from `run.json`,
the substrate fingerprint from `summary.json`, the excluded cells and the
total cost.
