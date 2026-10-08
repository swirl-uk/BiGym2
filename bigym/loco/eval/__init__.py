"""Frozen evaluation protocol for BiGym 2.0 leaderboard numbers."""

from bigym.loco.eval.protocol import (
    EVAL_EPISODES,
    EVAL_PROTOCOL_VERSION,
    EVAL_SEED_BASE,
    EVAL_SEED_RESERVED,
    eval_seeds,
    is_success,
    leaderboard_record,
    peak_final_selection,
    protocol_violations,
)


def evaluate(*args, **kwargs):
    """Reference runner (lazy import: pulls in the full env stack)."""
    from bigym.loco.eval.runner import evaluate as _evaluate

    return _evaluate(*args, **kwargs)


__all__ = [
    "EVAL_EPISODES",
    "EVAL_PROTOCOL_VERSION",
    "EVAL_SEED_BASE",
    "EVAL_SEED_RESERVED",
    "eval_seeds",
    "evaluate",
    "is_success",
    "leaderboard_record",
    "peak_final_selection",
    "protocol_violations",
]
