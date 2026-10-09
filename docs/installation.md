# Installation

BiGym 2.0 needs Python 3.10 or newer and runs on Linux and macOS. It installs
from source with [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # uv, if you don't have it yet
git clone https://github.com/swirl-uk/BiGym2.git && cd BiGym2
uv sync
```

## Extras

The core install has the environments, the GR00T-WBC lower body, the
demonstration loader, the evaluation runner and the viewer.

| Extra | Adds | Needed for |
|---|---|---|
| `agent` | OpenCV, SciPy, mink | the coding-agent benchmark (`bigym-agent`) |
| `vr` | pyopenxr, mink | VR teleoperation (`bigym-collect`) |
| `lerobot` | LeRobot and torch | writing LeRobot datasets (`bigym-export-lerobot`, `bigym-rerender-lerobot`) |

The `lerobot` extra needs Python 3.12 or newer. Below 3.12 it installs
nothing. Reading the published demonstrations does not need it.

`uv sync` makes the environment match the extras you name. List every extra
you want in one command. A later `uv sync --extra vr` removes what
`--extra agent` installed.

`vr` and `lerobot` cannot share an environment. pyopenxr needs
`setuptools<70` and LeRobot needs `setuptools>=71`. A collection machine
installs `vr` and `agent` and adds LeRobot for a single export:

```bash
uv sync --extra vr --extra agent
uv run --with "lerobot[dataset]>=0.6" bigym-export-lerobot ...
```

`uv sync` also installs the `dev` group. `--no-dev` skips it. `--group docs`
adds Sphinx.

## Running commands

Run commands through `uv run`, for example `uv run bigym-download --list`.
To drop the prefix, activate the environment once with
`source .venv/bin/activate`.

On headless Linux, set `MUJOCO_GL=egl`. On macOS, leave it unset.

VR collection needs Linux with an OpenXR runtime. See
[Collecting demonstrations](demo_collection.md).
