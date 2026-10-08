# Building on BiGym

Your own tasks and lower-body controllers can live in a separate package
that depends on `bigym`; nothing in BiGym has to change. This page sets up
such a package, adds a task and a controller to it, and runs them with
BiGym's tools. The code samples come from
[`tests/fixtures/external_extension.py`](https://github.com/swirl-uk/BiGym2/blob/main/tests/fixtures/external_extension.py),
which the test suite builds and steps.

## Structure

```text
my_lab/
  pyproject.toml          # depends on bigym
  src/my_lab/
    __init__.py
    tasks.py              # TaskSpecs and their env classes
    wbc.py                # lower-body controller and its BackendBinding
```

## 1. Depend on bigym

```bash
uv init --package my_lab
cd my_lab
```

::::{tab-set}

:::{tab-item} Source
```bash
uv add "bigym @ git+https://github.com/swirl-uk/BiGym2"
```
:::

:::{tab-item} Local
```bash
git clone https://github.com/swirl-uk/BiGym2.git
uv add --editable /path/to/BiGym2
```
:::

::::

Extras go in the requirement as usual, e.g.
`"bigym[vr] @ git+https://github.com/swirl-uk/BiGym2"` for `bigym-collect`
(see [Install](getting_started.md#install) for what each extra adds). Keep
the `[build-system]` table that `uv init --package` writes: without it your
package is not installed into the environment, and BiGym's command-line
tools cannot import `my_lab`.

## 2. Configure an existing task

Keyword overrides or an {class}`~bigym.loco.config.EnvConfig` change any
setting of a task; `env.config_overrides` lists what differs from the
task's own configuration:

```python
from bigym.loco import make

env = make("move_plate", camera_keys=("head",), controller={"cmd_clip": 0.5})
```

[First environment](getting_started.md#first-environment) and the
[API reference](api.md#configuration) list the fields.

## 3. Add a task

A task is a {class}`~bigym.loco.tasks.TaskSpec`: a `BiGymEnv` subclass,
an episode budget in env steps, and the `EnvConfig` fields where the task
departs from the defaults (in `make`'s override form). The shortest route is to subclass a task you
start from and change its class attributes or its `_initialize_env`,
`_on_reset`, `_success` and `_fail` hooks
([`bigym/envs/move_plates.py`](https://github.com/swirl-uk/BiGym2/blob/main/bigym/envs/move_plates.py)
shows all four):

```{literalinclude} ../tests/fixtures/external_extension.py
:language: python
:start-at: class PreciseMovePlateG1
```

`_initialize_env` adds props to the scene's `mujoco.MjSpec` (`self.spec`),
which is compiled once afterwards; the other hooks read and write the
compiled arrays through `self.model.bind(element)` and
`self.data.bind(element)`. Element names are full scene names, such as
`"dishwasher/door"`. After writing state, call `self.simulation.forward()`
before reading derived quantities such as `xpos`.

Build it by a registered name or by its `"pkg.module:ATTR"` reference:

```python
from bigym.loco import make, register_task
from my_lab.tasks import TASK

register_task("precise_move_plate", TASK)
env = make("precise_move_plate")

env = make("my_lab.tasks:TASK")  # imported on first use, no registration
```

A registered name exists only in processes that ran `register_task`. The
reference works in any process with `my_lab` installed, including the ones
BiGym's tools start, and it is the name the env records in its metadata,
so a recorded batch rebuilds the same task. Use the reference on the
command line.

A `TaskSpec` can also pin the task's controller:
`overrides={"controller": {"backend": "my_lab.wbc:HOLD_POSE"}}`.

## 4. Add a lower-body controller

A controller subclasses {class}`~bigym.loco.base.LowerBodyBase` (the
contract is in [Adding a backend](backends.md#adding-a-backend)). This one
holds the legs and waist at a standing pose:

```{literalinclude} ../tests/fixtures/external_extension.py
:language: python
:start-after: BINDING = ReusedGrootBinding()
:end-before: class HoldPoseBinding
```

`STATEFUL` maps snapshot keys to the attributes that shape future targets, so
`get_state` / `set_state` restore the controller bit-exactly on replay. A
learned controller loads its policy in `__init__`, runs it in `step()`,
and returns its weight files from `weight_files()`, which puts their
SHA-256 in the substrate fingerprint;
[`bigym/loco/adapters/groot_wbc.py`](https://github.com/swirl-uk/BiGym2/blob/main/bigym/loco/adapters/groot_wbc.py)
does all three.

A {class}`~bigym.loco.BackendBinding` puts the controller in the env: the
robot models it drives, the robot class it needs (which joints are
actuated, with which PD gains) and how to build it:

```{literalinclude} ../tests/fixtures/external_extension.py
:language: python
:start-at: class HoldPoseBinding
:end-at: HOLD_POSE = HoldPoseBinding()
```

Select it like a task, by reference or by a name given to
{func}`~bigym.loco.adapters.register_backend`:

```python
from bigym.loco import make, register_backend
from my_lab.wbc import HOLD_POSE

env = make("move_plate", controller={"backend": "my_lab.wbc:HOLD_POSE"})

register_backend("hold_pose", HOLD_POSE)
env = make("move_plate", controller={"backend": "hold_pose"})
```

## 5. Use them with BiGym's tools

| Tool | Your task | Your controller |
|---|---|---|
| `make`, `make_gym` | name or reference | name or reference |
| `python -m bigym.loco.eval.runner` | `--task my_lab.tasks:TASK` | `--overrides '{"controller": {"backend": "my_lab.wbc:HOLD_POSE"}}'`, or pinned by the `TaskSpec` |
| `bigym-collect` | `--task my_lab.tasks:TASK` | `groot_wbc_g1` only |
| `python -m bigym.loco.demos.success_hold trim`, `bigym-view`, `bigym-export-lerobot`, `bigym-rerender-lerobot` | read from the batch metadata | read from the batch metadata |
| `env.get_demos()` | from a dataset repo you publish (below) | — |
| `bigym-agent` | BiGym's built-in tasks only | `groot_wbc_g1` only |

`bigym-agent` needs a one-sentence brief and published demonstrations for
each task it runs, and both exist for the built-in tasks only.

The evaluation runner loads the policy from a `module:factory` import spec
whose factory returns a callable from timestep to action:

```bash
uv run python -m bigym.loco.eval.runner --task my_lab.tasks:TASK \
    --method my-method --policy my_lab.policies:make_policy --out result.json
```

## Collecting demonstrations for your task

On a machine set up for [VR collection](demo_collection.md), with
`bigym[vr]` in your project:

```bash
uv run bigym-collect --task my_lab.tasks:TASK --export-lerobot \
    --task-text "Move the plate into the other rack."
```

The batch goes to `./bigym_demos/my_lab.tasks-TASK/<timestamp>`: file and
directory names replace the colon with a dash, and the batch metadata keeps
the exact task name. After the session the collector trims a training view
to the task's success hold (`<timestamp>_hold1s` for a 1 s hold) and
exports it to LeRobot in `<timestamp>_hold1s_lerobot`. `--task-text` is the
language instruction written to the LeRobot dataset; BiGym's instruction
table covers only its own tasks.

`env.get_demos()` reads demonstrations from a Hugging Face dataset repo with
one folder per task. Upload the export as the folder `my_lab.tasks-TASK/`
of a dataset repo of yours and set `BIGYM_DATASET_REPO` to it:

```bash
BIGYM_DATASET_REPO=my-org/my-lab-demos uv run python train.py
```
