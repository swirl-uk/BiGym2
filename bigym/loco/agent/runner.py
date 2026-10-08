"""Run the agent's policy on development seeds and print per-episode results.

    python run_episodes.py --seeds 0-9            # seeds 0..9
    python run_episodes.py --seeds 3,7,11 --jobs 4

Writes ``runs/<timestamp>.json`` (per-episode records) and a snapshot of
``policy.py`` next to it.

This module is copied into an agent sandbox as ``harness/runner.py``, so it
must keep working when imported as a top-level module.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent

try:  # inside the package
    from .client import connect
    from .episode import load_policy, run_episode
except ImportError:  # copied into a sandbox next to harness/client.py
    sys.path.insert(0, str(HERE))
    from client import connect  # ty: ignore[unresolved-import]
    from episode import load_policy, run_episode  # ty: ignore[unresolved-import]

SANDBOX = Path(os.environ.get("AGENT_SANDBOX", HERE.parent))


def parse_seeds(spec: str) -> list[int]:
    """Expand a seed specification such as ``3,7,11`` or ``20-40``.

    Args:
        spec: Comma-separated seeds and inclusive ``a-b`` ranges.

    Returns:
        The seeds in the order they were written.
    """
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _one(args):
    policy_path, seed, save_frames = args
    try:
        policy = load_policy(policy_path)
        with connect(SANDBOX) as env:
            rec = run_episode(env, policy, seed)
            if save_frames and not rec["success"]:
                try:
                    # The robot's own head camera at the final state: the only
                    # view of the scene that exists.
                    rec["last_frame"] = env.render(
                        "head", f"frames/seed{seed}_last.png"
                    )
                except Exception as exc:
                    # A frame problem must never replace the episode record.
                    rec["last_frame"] = None
                    rec["last_frame_error"] = f"{type(exc).__name__}: {exc}"
            rec["budget"] = env.budget()
        return rec
    except Exception as exc:
        return {
            "seed": seed,
            "success": 0,
            "length": 0,
            "reward": 0.0,
            "termination": "error",
            "fell": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def main(argv=None):
    """Run the policy on a set of seeds and write a records file.

    Args:
        argv: Command line arguments, or None for ``sys.argv[1:]``.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--seeds",
        default=None,
        help="e.g. 3,7,11 or 20-40; default: every seed in seeds.json",
    )
    ap.add_argument("--policy", default=str(SANDBOX / "policy.py"))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument(
        "--no-frames",
        action="store_true",
        help="do not save the last frame of failed episodes",
    )
    args = ap.parse_args(argv)
    if args.seeds is None:
        sj = SANDBOX / "seeds.json"
        seeds = json.loads(sj.read_text()) if sj.exists() else parse_seeds("0-9")
    else:
        seeds = parse_seeds(args.seeds)
    policy_path = Path(args.policy).resolve()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    runs = SANDBOX / "runs"
    runs.mkdir(exist_ok=True)
    shutil.copy(policy_path, runs / f"{stamp}_policy.py")

    jobs = [(str(policy_path), s, not args.no_frames) for s in seeds]
    if args.jobs > 1:
        with mp.get_context("fork").Pool(args.jobs) as pool:
            recs = pool.map(_one, jobs)
    else:
        recs = [_one(j) for j in jobs]

    print(f"{'seed':>6} {'success':>7} {'steps':>6} {'fell':>5}  note")
    for r in recs:
        note = r.get("error") or r.get("termination", "")
        if r.get("last_frame"):
            note += f"  frame={Path(r['last_frame']).relative_to(SANDBOX)}"
        yes_no = "yes" if r["success"] else "no"
        fell = "yes" if r["fell"] else "no"
        print(f"{r['seed']:>6} {yes_no:>7} {r['length']:>6} {fell:>5}  {note}")
    n_ok = sum(r["success"] for r in recs)
    budget = next((r["budget"] for r in reversed(recs) if r.get("budget")), None)
    print(f"success {n_ok}/{len(recs)}", end="")
    if budget:
        print(
            f"   budget used {budget['used']:,} / {budget['cap']:,} steps "
            f"({budget['remaining']:,} left)"
        )
    else:
        print()
    for r in recs:
        if r.get("traceback"):
            print(
                f"--- seed {r['seed']} traceback ---\n{r['traceback']}", file=sys.stderr
            )
    out = runs / f"{stamp}.json"
    out.write_text(
        json.dumps(
            {"policy_snapshot": f"{stamp}_policy.py", "seeds": seeds, "episodes": recs},
            indent=1,
        )
    )
    print(f"records: {out.relative_to(SANDBOX)}")


if __name__ == "__main__":
    main()
