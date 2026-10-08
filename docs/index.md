# BiGym 2.0

**A whole-body humanoid manipulation benchmark with locomotion in the loop.**

BiGym 2.0 extends [BiGym](https://arxiv.org/abs/2407.07788) with a second,
humanoid-locomotion line: instead of a floating base, a **frozen RL
lower-body controller runs inside the environment step** and keeps the robot
balanced and walking from velocity commands, while your agent owns the rest
of the action:

```text
outer action = [ base command (vx, vy, height, wz[, pitch]) | arm joint targets | grippers ]
```

```python
from bigym.loco import make

env = make("move_plate")   # the official G1 configuration
timestep = env.reset(seed=620000)
```

`make` builds the task's official configuration from the task registry:
Unitree G1 with Dex1-1 grippers (`g1_dex1`), the NVIDIA GR00T-WBC lower
body (`groot_wbc_g1`) and three 84×84 cameras. `move_plate` has a 20-dim
action and a 50-dim low-dimensional observation; tasks with the torso-pitch
command have 21 and 56 (see [Tasks](tasks.md)).
Overrides (`make("move_plate", camera_keys=("head",))`) give research
variants; the env records them, and evaluation reports a run
as unofficial when they affect results.

```{image} _static/architecture.svg
:alt: BiGym 2.0 architecture — the agent sends an outer action (base velocity command + arm targets + grippers) to the env, which routes the base command to the frozen GR00T-WBC lower-body controller and the arm targets to PD actuators inside the MuJoCo scene; VR teleoperation produces demos through the same outer action, and the evaluation protocol stamps results with a substrate fingerprint.
:width: 100%
```

## Why trust the numbers

- **Bit-exact replay.** Mid-episode save/restore reproduces trajectories to
  the last bit (Newton solver pin + declarative controller-state snapshots).
- **Substrate versioning.** Every env stamps a
  `substrate_fingerprint()` (MuJoCo version, backend, action/observation
  layout, solver), so results from different substrates are never silently
  compared.
- **Frozen protocols.** Demo schema v1 and the evaluation protocol are
  versioned and validated in code, including the fixed 100-episode seed
  block and the task registry that pins each task's budget and
  configuration.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 🚀 Getting started
:link: getting_started
:link-type: doc
Install, first env in 5 lines, and how to introspect every action/obs dim.
:::

:::{grid-item-card} 📋 Official configuration
:link: official_configuration
:link-type: doc
The frozen benchmark row: embodiment, observation/action layout, budgets,
eval protocol, baseline training recipe.
:::

:::{grid-item-card} 🗂️ Tasks
:link: tasks
:link-type: doc
The 20 benchmark tasks: success predicates, reset randomisation, budgets, previews.
:::

:::{grid-item-card} 🦿 Lower-body backend
:link: backends
:link-type: doc
GR00T-WBC, the command-bounds contract, and how to add a backend.
:::

:::{grid-item-card} 🔬 Demos & evaluation
:link: demos_eval
:link-type: doc
Demo schema v1, bit-exact replay, and the frozen eval protocol.
:::

:::{grid-item-card} 🥽 Collecting demos
:link: demo_collection
:link-type: doc
VR teleoperation: mink IK arms, stick-commanded locomotion, engage snapshots.
:::

:::{grid-item-card} 🤖 Coding-agent benchmark
:link: agent
:link-type: doc
An agent writes `policy.py` by hand against a sandboxed env, scored on the
same hidden seeds.
:::

:::{grid-item-card} 🧩 Building on BiGym
:link: extending
:link-type: doc
Your own tasks and lower-body controllers in a package that depends on
`bigym`.
:::
::::

```{toctree}
:hidden:
:caption: Guides

getting_started
official_configuration
tasks
backends
demos_eval
demo_collection
agent
extending
```

```{toctree}
:hidden:
:caption: Reference

api
dataset_card_template
```

## Projects built on BiGym

To add yours, open a pull request adding a line here.

## Citing

If you use BiGym 2.0, please cite:

```bibtex
@article{zhang2026bigym2,
  title   = {BiGym 2.0: Benchmarking Learned and Agent-Developed Policies for Humanoid Household Manipulation},
  author  = {Zhang, Zexi and Zhu, Zecheng and Chen, Zidong and Tuya, Zulkhuu and James, Stephen},
  journal = {arXiv preprint arXiv:2610.07594},
  year    = {2026}
}
```

BiGym 2.0 builds on BiGym {cite:p}`chernyadev2024bigym` and MuJoCo
{cite:p}`todorov2012mujoco`; the lower-body backend and the VR collection
stack build on GR00T N1's Decoupled WBC {cite:p}`nvidia2025gr00tn1` and
mink {cite:p}`zakka2024mink` — please cite them alongside this benchmark.

```{bibliography}
```
