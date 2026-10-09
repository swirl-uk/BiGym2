# BiGym 2.0

BiGym 2.0 is a benchmark for whole-body humanoid manipulation. A Unitree G1
with two-finger grippers performs 20 household tasks in MuJoCo. A frozen
lower-body controller, NVIDIA's GR00T-WBC, runs inside every environment step
and keeps the robot balanced and walking. Your policy commands the base, the
14 arm joints and the two grippers. Each task has 60 human VR demonstrations.
Learned policies and coding agents are scored on the same 100 evaluation
seeds.

```{image} https://bigym2.github.io/readme/fig2_bigym_to_bigym2.gif
:alt: Left, BiGym: the H1 pelvis is commanded directly and the legs play back an animation. Right, BiGym 2.0: GR00T-WBC walks and balances the G1.
:width: 100%
```

Build a task, reset it and take one step:

```python
import numpy as np
from bigym.loco import make

env = make("move_plate")                # the task's official configuration
timestep = env.reset(seed=620000)       # the first evaluation seed
action = np.zeros(env.action_spec().shape, dtype=np.float32)
timestep = env.step(action)             # one 50 Hz control step
print(timestep.rgb_obs.shape)           # (3, 3, 84, 84)
print(timestep.low_dim_obs.shape)       # (50,)
```

```{toctree}
:maxdepth: 1
:caption: Start here

installation
getting_started
```

```{toctree}
:maxdepth: 1
:caption: The benchmark

official_configuration
tasks
backends
demonstrations
determinism
```

```{toctree}
:maxdepth: 1
:caption: Tools

demo_collection
agent
extending
```

```{toctree}
:maxdepth: 1
:caption: Reference

api
dataset_card_template
```

```{toctree}
:maxdepth: 1
:caption: Further reading

faq
research
```

## License and citation

The code is released under the Apache 2.0 License. The GR00T-WBC weights are
NVIDIA's, under the NVIDIA Open Model License.

Cite BiGym 2.0 with the BibTeX entry on the [Research](research.md) page.
