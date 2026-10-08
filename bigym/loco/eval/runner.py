"""Reference evaluation runner — the executable form of protocol v1.

The protocol module defines the evaluation block (100 episodes, contiguous
seeds from 620000); this runner builds the task's official environment and
emits a leaderboard record only when ``protocol_violations`` finds no
result-affecting departure from the official configuration.

Library use::

    from bigym.loco.eval import evaluate

    result = evaluate(
        my_policy,                      # callable: ExtendedTimeStep -> action
        task_name="move_plate",
        method="my-cool-method",
        # The env is bigym.loco.make(task_name, config, **overrides): the
        # official configuration unless told otherwise. Overrides of
        # result-affecting fields suppress the record.
        overrides={"frame_stack": 2},
    )
    print(result["record"])             # leaderboard entry, fingerprint inside
    print(result["protocol_violations"])  # [] for an official env

CLI use (the policy is loaded from an import spec ``module:factory`` where
``factory()`` returns the policy callable)::

    python -m bigym.loco.eval.runner \
        --task move_plate --method my-cool-method \
        --policy my_package.eval:make_policy \
        [--overrides '{"camera_keys": ["head"]}'] \
        --out record.json

Policy contract: a callable ``policy(timestep) -> np.ndarray`` (action for
``env.action_spec()``); if it has a ``reset()`` method it is called between
episodes (stateful policies: action chunking, RNNs).

Protocol parameters are deliberately NOT configurable. Running fewer
episodes (``episodes=N``) is supported for smoke-testing the plumbing, but
such runs return a summary WITHOUT a leaderboard record — a number produced
off-protocol should never look like a benchmark number. The same holds for
an env built with result-affecting overrides
(``bigym.loco.eval.protocol_violations``): the summary is returned, the
departing fields are listed, and no record is produced.

Per-episode records (seed, success, length, reward, termination reason,
fell) are returned under ``episode_records`` and written next to
``--out`` as ``<out>.episodes.csv``.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import tyro

from bigym.loco import make
from bigym.loco.config import EnvConfig
from bigym.loco.eval.protocol import (
    EVAL_EPISODES,
    eval_seeds,
    is_success,
    leaderboard_record,
    protocol_violations,
)


def evaluate(
    policy: Callable[..., np.ndarray],
    *,
    task_name: str,
    method: str,
    checkpoint: str = "n/a",
    episodes: int = EVAL_EPISODES,
    config: EnvConfig | None = None,
    overrides: Mapping[str, Any] | None = None,
    env: Any = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run one evaluation block; return summary (+ leaderboard record).

    Pass either ``config`` / ``overrides`` (forwarded to
    :func:`bigym.loco.make` with ``task_name``; the env is created and
    closed here) or a ready ``env`` (kept open).

    Returns ``{"task", "method", "episodes", "success_rate",
    "episode_rewards", "substrate", "env_config", "protocol_violations",
    ["record"]}``; ``record`` (the protocol-v1 leaderboard entry) only for
    a full block on an official env.
    """
    episodes = int(episodes)
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    owns_env = env is None
    if owns_env:
        env = make(task_name, config, **dict(overrides or {}))
    try:
        seeds = eval_seeds(episodes)
        episode_rewards: list[float] = []
        episode_records: list[dict[str, Any]] = []
        successes = 0
        for index, seed in enumerate(seeds):
            if hasattr(policy, "reset"):
                policy.reset()  # ty: ignore[call-non-callable]
            timestep = env.reset(seed=seed)
            total_reward = 0.0
            length = 0
            while not timestep.last():
                action = policy(timestep)
                timestep = env.step(action)
                total_reward += float(timestep.reward or 0.0)
                length += 1
            fell = bool(env.episode_fell())
            success = is_success(env)
            successes += int(success)
            episode_rewards.append(total_reward)
            termination = env.episode_termination()
            episode_records.append(
                {
                    "episode": index,
                    "seed": int(seed),
                    "success": int(success),
                    "length": int(length),
                    "reward": float(total_reward),
                    "termination": termination,
                    "fell": int(fell),
                }
            )
            if verbose:
                print(
                    f"[eval] ep {index + 1}/{episodes} seed={seed} "
                    f"reward={total_reward:.3f} success={int(success)} "
                    f"end={termination} running_sr={successes / (index + 1):.2%}"
                )
        success_rate = successes / episodes
        fingerprint = env.substrate_fingerprint()
        violations = protocol_violations(env)
        env_config = getattr(env, "config", None)
        result: dict[str, Any] = {
            "task": str(task_name),
            "method": str(method),
            "episodes": episodes,
            "success_rate": success_rate,
            "episode_rewards": episode_rewards,
            "episode_records": episode_records,
            "substrate": dict(fingerprint),
            "env_config": env_config.to_dict() if env_config is not None else None,
            "protocol_violations": violations,
        }
        if violations and verbose:
            print(
                "[eval] env departs from the official configuration, NO "
                "leaderboard record: " + "; ".join(violations)
            )
        if episodes == EVAL_EPISODES and not violations:
            result["record"] = leaderboard_record(
                task=task_name,
                method=method,
                backend=str(fingerprint.get("lowerbody_backend") or "none"),
                success_rate=success_rate,
                episodes=episodes,
                checkpoint=checkpoint,
                substrate_fingerprint=fingerprint,
                seeds=seeds,
            )
        elif verbose:
            print(
                f"[eval] {episodes} episodes < protocol {EVAL_EPISODES}: "
                "smoke run, NO leaderboard record produced."
            )
        return result
    finally:
        if owns_env:
            env.close()


def _load_policy(spec: str) -> Callable[..., np.ndarray]:
    module_name, sep, attr = spec.partition(":")
    if not sep:
        raise SystemExit(
            f"--policy must be 'module:factory' (a zero-arg factory returning "
            f"the policy callable), got {spec!r}"
        )
    factory = getattr(importlib.import_module(module_name), attr)
    return factory()


@dataclass
class EvalConfig:
    """Settings of the protocol evaluation run."""

    task: str
    """Task name (see TASKS)."""
    method: str
    """Method name for the record."""
    policy: str
    """Import spec 'module:factory'; factory() returns the policy callable."""
    overrides: str = "{}"
    """JSON dict of EnvConfig fields applied last, e.g. '{"frame_stack": 2}'."""
    checkpoint: str = "n/a"
    """Checkpoint name for the record."""
    episodes: int = EVAL_EPISODES
    """Episodes to run; fewer than the protocol's 100 is a smoke run with no
    record."""
    out: Path | None = None
    """Write result JSON here."""


def main() -> None:
    """Command-line entry point: run the protocol eval and report results."""
    args = tyro.cli(EvalConfig, description=__doc__)

    result = evaluate(
        _load_policy(args.policy),
        task_name=args.task,
        method=args.method,
        checkpoint=args.checkpoint,
        episodes=args.episodes,
        overrides=json.loads(args.overrides),
    )
    print(
        f"\n{args.method} on {args.task}: success_rate="
        f"{result['success_rate']:.2%} over {result['episodes']} episodes"
    )
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}")
        episodes_csv = args.out.with_suffix(args.out.suffix + ".episodes.csv")
        with episodes_csv.open("w") as f:
            f.write("episode,seed,success,length,reward,termination,fell\n")
            for rec in result["episode_records"]:
                f.write(
                    f"{rec['episode']},{rec['seed']},{rec['success']},{rec['length']},"
                    f"{rec['reward']},{rec['termination']},{rec['fell']}\n"
                )
        print(f"wrote {episodes_csv}")


if __name__ == "__main__":
    main()
