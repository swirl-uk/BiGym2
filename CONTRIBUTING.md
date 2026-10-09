# Contributing

## Setup

```bash
uv sync --extra agent
uv run pre-commit install
```

The hooks run ruff, the YAML and XML checks and the commit-message check on
every commit. CI runs the same hooks and the test suite.

## Commit messages

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):

```text
<type>(<scope>): <summary>
```

- `type` is one of `feat`, `fix`, `docs`, `test`, `refactor`, `perf`,
  `build`, `ci`, `chore`, `style` or `revert`.
- `scope` is optional and names the area: `envs`, `tasks`, `robots`,
  `lowerbody`, `demos`, `eval`, `agent`, `collect`, `viewer`.
- The summary is lowercase, in the imperative, with no trailing period.

```text
feat(agent): add the tools interface
fix(demos): cut each episode on the official success latch
docs(tasks): state every success condition in plain English
```

## Pull requests

Pull requests are squash-merged, so the pull request title becomes the commit
message on `main`. Write the title in the same format.
