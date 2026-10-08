"""The coding-agent benchmark: an LLM agent writes the control policy.

An agent works in a sandbox directory containing a task brief, a policy
template and a harness that talks to an environment server over a unix socket.
The server (:mod:`bigym.loco.agent.server`) owns the simulator, counts the
agent's interaction budget, keeps a ledger and refuses the hidden evaluation
seeds. When the session ends, :mod:`bigym.loco.agent.evaluate` scores the
submitted ``policy.py`` on those hidden seeds with the same episode loop the
agent developed against (:mod:`bigym.loco.agent.episode`), the policy running
in an interpreter of its own that holds no simulator
(:mod:`bigym.loco.agent.policy_process`).

Public surface::

    from bigym.loco.agent import EnvTools, EnvToolsConfig, make_env, Tools
    from bigym.loco.agent import load_policy, run_episode

``EnvTools(task, EnvToolsConfig(...))`` builds the environment a policy runs
against, with the settings the command line takes.

Names are resolved lazily, so importing this package pulls in neither the
simulator nor the optional ``bigym[agent]`` dependencies.
"""

from __future__ import annotations

from typing import Any

_LAZY = {
    "EnvTools": ("bigym.loco.agent.envtools", "EnvTools"),
    "EnvToolsConfig": ("bigym.loco.agent.cli", "EnvToolsConfig"),
    "make_env": ("bigym.loco.agent.envtools", "make_env"),
    "task_pitch_enabled": ("bigym.loco.agent.envtools", "task_pitch_enabled"),
    "Tools": ("bigym.loco.agent.episode", "Tools"),
    "load_policy": ("bigym.loco.agent.episode", "load_policy"),
    "run_episode": ("bigym.loco.agent.episode", "run_episode"),
}

__all__ = [
    "EnvTools",
    "EnvToolsConfig",
    "make_env",
    "task_pitch_enabled",
    "Tools",
    "load_policy",
    "run_episode",
]


def __getattr__(name: str) -> Any:
    """Import a public name on first use (PEP 562).

    Args:
        name: Attribute name.

    Returns:
        The requested object.

    Raises:
        AttributeError: The package has no such attribute.
    """
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(target[0]), target[1])


def __dir__() -> list[str]:
    """List the public names, lazy ones included."""
    return sorted(set(__all__) | set(globals()))
