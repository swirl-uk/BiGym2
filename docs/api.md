# API reference

## Environment

```{eval-rst}
.. autofunction:: bigym.loco.make

.. autofunction:: bigym.loco.make_gym
```

## Configuration

```{eval-rst}
.. autoclass:: bigym.loco.config.EnvConfig
   :members: override, differences, from_metadata

.. autoclass:: bigym.loco.config.ControllerConfig

.. autoclass:: bigym.loco.config.WholeBodyConfig

.. autofunction:: bigym.loco.config.resolve_config

.. autofunction:: bigym.loco.config.conformance_violations
```

## Task registry

```{eval-rst}
.. autoclass:: bigym.loco.tasks.TaskSpec
   :members: config

.. autofunction:: bigym.loco.tasks.task_config

.. autofunction:: bigym.loco.tasks.budget_provenance

.. autofunction:: bigym.loco.tasks.all_task_names

.. autofunction:: bigym.loco.register_task
```

## Contract layer

```{eval-rst}
.. automodule:: bigym.loco.command
   :members: CommandKind, CommandField, CommandSpec, velocity_spec

.. automodule:: bigym.loco.controller
   :members: LowerBodyController, OutputSpec

.. autoclass:: bigym.loco.base.LowerBodyBase
   :members: STATEFUL, controlled_joints, command_spec, control_dt, output_spec,
             build_joint_addresses, build_joint_ranges, find_sensor, apply_pose,
             set_command, is_failed, get_state, set_state,
             get_base_obs, recommended_reset_warmup_steps
```

## Backends

```{eval-rst}
.. automodule:: bigym.loco.adapters
   :members: register_backend, resolve_backend_name

.. autoclass:: bigym.loco.BackendBinding
   :members: robot_models, supports_passive_base_tilt, robot_cls,
             build_controller, configure_model
```

## Demos

```{eval-rst}
.. automodule:: bigym.loco.demos.schema
   :members:

.. automodule:: bigym.loco.demos.io
   :members:

.. automodule:: bigym.loco.demos.success_hold
   :members: latch_batch

.. automodule:: bigym.loco.demos.hub
   :members: DemosUnavailableError

.. automodule:: bigym.loco.demos.dataset
   :members: load_episodes
```

## Demo collection

```{eval-rst}
.. autoclass:: bigym.vr.collect.config.CollectConfig
```

## Evaluation

```{eval-rst}
.. autofunction:: bigym.loco.eval.runner.evaluate

.. automodule:: bigym.loco.eval.protocol
   :members:
```

## Coding-agent benchmark

```{eval-rst}
.. automodule:: bigym.loco.agent

.. autofunction:: bigym.loco.agent.envtools.make_env

.. autoclass:: bigym.loco.agent.episode.Tools

.. autofunction:: bigym.loco.agent.episode.load_policy

.. autofunction:: bigym.loco.agent.episode.run_episode
```
