"""The frozen BiGym 2.0 evaluation protocol.

One definition shared by every leaderboard entry:

- 100 episodes per (task, checkpoint) evaluation;
- fixed seed block starting at 620000, reserved for evaluation by convention
  (episode i uses ``seed_base + i``; training-seed overlap is not checked here);
- checkpoints every 5k training steps, each evaluated on the same block;
  the reported number is the MEAN OF THE LAST 5 CHECKPOINTS (80k..100k of a
  101k budget) +- standard error.
  ``peak_final_selection`` serves the appendix tables. Selecting
  the maximum on the same test block introduces selection bias; do not use
  it as the main-table result or treat correlated checkpoints as independent
  training seeds. This module records individual checkpoints, not that
  aggregate; disclose checkpoint/seed/episode aggregation when reporting it;
- an episode is a success when the task reported success and the robot
  never fell (:func:`is_success`). Evaluation must keep reward shaping off.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from bigym.loco.config import conformance_violations

EVAL_PROTOCOL_VERSION = 1
EVAL_EPISODES = 100
# The one definition of the evaluation seed block; everything that draws,
# refuses or defaults to evaluation seeds imports it from here.
EVAL_SEED_BASE = 620000
# Seeds reserved for evaluation, ``[lo, hi)``: the protocol block plus room for
# offset blocks after it (``bigym-agent evaluate --seed-offset``). The coding
# agent's environment server refuses every one of them.
EVAL_SEED_RESERVED = (EVAL_SEED_BASE, EVAL_SEED_BASE + 10000)


def is_success(env: Any) -> bool:
    """Whether the env's episode so far counts as a success.

    Args:
        env: An env from :func:`bigym.loco.make` or ``make_gym``, or any
            wrapper that forwards ``episode_succeeded`` and ``episode_fell``.

    Returns:
        True when the task reported success (``env.episode_succeeded()``)
        and the robot never fell (``env.episode_fell()``).
    """
    return env.episode_succeeded() and not env.episode_fell()


def eval_seeds(
    episodes: int = EVAL_EPISODES, *, seed_base: int = EVAL_SEED_BASE
) -> list[int]:
    """The per-episode reset seeds of one evaluation block."""
    return [int(seed_base) + i for i in range(int(episodes))]


def peak_final_selection(
    step_to_score: Mapping[int, float],
) -> dict[str, int]:
    """Pick the peak-score and final checkpoints from step->score curves."""
    if not step_to_score:
        raise ValueError("No evaluated checkpoints to select from.")
    steps = sorted(int(s) for s in step_to_score)
    peak = max(steps, key=lambda s: (float(step_to_score[s]), -s))
    return {"peak_step": peak, "final_step": steps[-1]}


def protocol_violations(env: Any) -> list[str]:
    """Why an env's results are not official; empty when they are.

    Compares the configuration the env was built from (``env.config``)
    with its task's official configuration (``env.official_config``). Every
    result-affecting field that differs is listed; fields that only change
    rendering or the policy-side view of observations and actions are
    exempt (see :func:`bigym.loco.config.affects_results`).

    Args:
        env: An env from :func:`bigym.loco.make` or ``make_gym``, or any
            wrapper that forwards ``config`` and ``official_config``.

    Returns:
        One ``"field: official X, env Y"`` line per departing field.
    """
    config = getattr(env, "config", None)
    official = getattr(env, "official_config", None)
    if config is None or official is None:
        return ["env does not record its configuration (build it with make)"]
    return conformance_violations(config, official)


def leaderboard_record(
    *,
    task: str,
    method: str,
    backend: str,
    success_rate: float,
    episodes: int,
    checkpoint: str,
    substrate_fingerprint: Mapping[str, Any],
    seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """One leaderboard JSON entry (schema v1)."""
    seeds = [int(s) for s in seeds] if seeds is not None else eval_seeds(episodes)
    if len(seeds) != int(episodes):
        raise ValueError(f"{len(seeds)} seeds for {episodes} episodes")
    # The protocol is a contiguous fixed block; a record storing only
    # seed_base must not silently misdescribe shuffled/ad-hoc seed sets.
    if seeds != list(range(seeds[0], seeds[0] + len(seeds))):
        raise ValueError(
            "Eval seeds must be the contiguous block seed_base..seed_base+N-1 "
            "required by the frozen protocol."
        )
    return {
        "eval_protocol_version": EVAL_PROTOCOL_VERSION,
        "task": str(task),
        "method": str(method),
        "backend": str(backend),
        "success_rate": float(success_rate),
        "episodes": int(episodes),
        "checkpoint": str(checkpoint),
        "seed_base": int(seeds[0]),
        "substrate": dict(substrate_fingerprint),
    }
