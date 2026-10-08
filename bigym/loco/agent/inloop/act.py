#!/usr/bin/env python
"""Turn-based robot commands for the in-the-loop evaluation (the model calls these).

  ./act status                      current observation, steps used, episode state
  ./act look                        save the robot's own camera images and print their paths
  ./act move_hand left X Y Z [--steps N]   move a wrist site to a world point, hold N steps
  ./act walk VX VY WZ --steps N     base velocity command for N steps (m/s, m/s, rad/s)
  ./act hold --steps N              stand still for N steps
  ./act gripper left VALUE [--steps N]     gripper command 0 (open) .. 1 (closed)

Every command advances the simulation and prints the new observation. The
episode ends by itself on success, on failure or at the time limit.

This file is copied into an episode directory next to ``episode.json`` (which
says which worker socket to use and what the per-episode command cap is) and
``harness/wire.py``, so it must keep working as a top-level script.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

try:  # inside the package
    from .. import wire
except ImportError:  # copied into an episode directory next to harness/wire.py
    sys.path.insert(0, str(HERE / "harness"))
    import wire  # ty: ignore[unresolved-import]

JOINT_STEP = 6.0 * 0.02  # rad per control step (6 rad/s, as in demo collection)
BASE_STEP = 0.7 * 0.02  # command units per step (0.7 /s slew, as in demo collection)
COUNTER_NAME = "commands_used.txt"

# Episode configuration, filled by load_config() from episode.json.
CFG: dict = {
    "dir": str(HERE),
    "socket": None,
    "episode_rel": "",
    "max_commands": 30,
    "tier": "privileged",  # "images": no object positions are printed
    "smooth": False,  # ramp targets like the teleoperation conditioning
}


def load_config(episode_dir=None) -> dict:
    """Read ``episode.json`` from the episode directory.

    Args:
        episode_dir: The episode directory, or None for this file's directory.

    Returns:
        The configuration dictionary (also stored in :data:`CFG`).
    """
    d = Path(episode_dir or HERE)
    CFG.update(json.loads((d / "episode.json").read_text()))
    CFG["dir"] = str(d)
    return CFG


def counter_path() -> Path:
    """Return the path of the local command counter file."""
    return Path(CFG["dir"]) / COUNTER_NAME


def call(sock, msg):
    """Send one request and return the reply, exiting on a server error.

    Args:
        sock: Socket connected to the episode's worker.
        msg: The request.

    Returns:
        The reply dictionary.
    """
    wire.send(sock, msg)
    r = wire.recv(sock)
    if not r.get("ok"):
        sys.exit(f"ERROR: {r.get('error')}")
    return r


def fmt(v, nd=3):
    """Format a vector for the model to read.

    Args:
        v: A sequence of numbers, or None.
        nd: Decimal places.

    Returns:
        The formatted string.
    """
    if v is None:
        return "n/a"
    return "[" + ", ".join(f"{float(x):.{nd}f}" for x in v) + "]"


def print_obs(obs, extra=None):
    """Print the observation in the form the model reads it.

    Args:
        obs: The observation dictionary, or None when no episode is active.
        extra: An extra line to print at the end.
    """
    if obs is None:
        print("episode not active")
        return
    tier = CFG.get("tier", "privileged")
    tp = obs.get("target_pos")
    # withheld unless the server runs the tools interface with wrist positions
    lh, rh = obs.get("left_hand_pos"), obs.get("right_hand_pos")
    if tier == "images":
        tp = None
    print(
        f"step {obs['t']}/{obs['time_limit']}  base_pos {fmt(obs['base_pos'])}  "
        f"base_yaw {float(obs['base_yaw']):+.3f} rad"
    )
    print(
        f"left_hand {fmt(lh)}  right_hand {fmt(rh)}  "
        f"grippers L {float(obs['left_gripper']):.2f} R {float(obs['right_gripper']):.2f}  "
        f"fell {obs['fell']}"
    )
    if tp is not None and lh is not None:
        d = float(np.linalg.norm(np.asarray(tp) - np.asarray(lh)))
        print(
            f"target_pos {fmt(tp)}  distance(left_hand, target) {d:.3f} m  "
            f"(success needs <= 0.050 held 1 s)"
        )
    elif tp is not None:
        print(
            f"target_pos {fmt(tp)}  (success needs the left wrist within 0.050 m, held 1 s)"
        )
    if tier == "images":
        print(
            "object positions are not provided in this tier: use `./act look` and judge "
            "from the images"
        )
    elif "plate_pos" in obs:
        pp = np.asarray(obs["plate_pos"])
        up = np.asarray(obs["plate_up_axis"])
        sites = np.asarray(obs["rack_target_sites"])
        dists = np.linalg.norm(sites - pp, axis=1)
        j = int(np.argmin(dists))
        ang = float(
            np.degrees(
                np.arccos(
                    np.clip(np.dot(up / (np.linalg.norm(up) + 1e-9), [0, -1, 0]), -1, 1)
                )
            )
        )
        print(
            f"plate_pos {fmt(pp)}  plate_up_axis {fmt(up, 2)}  angle(up, -y) {ang:.0f} deg "
            f"(success needs <= 20)"
        )
        print(f"target rack slots (x y z): {'; '.join(fmt(s_) for s_ in sites)}")
        print(
            f"nearest target slot #{j} at {dists[j]:.3f} m (success needs <= 0.050, plate "
            f"touching rack, not table, not held)"
        )
        print(
            f"start rack slots: {'; '.join(fmt(s_) for s_ in np.asarray(obs['rack_start_sites']))}"
        )
    if extra:
        print(extra)


def bump_counter(st=None):
    """Count one command locally and stop the episode past the cap.

    The server enforces the cap as well; this mirror is what the model reads.

    Args:
        st: The server's status reply, when one was just read.

    Returns:
        The new command count.
    """
    path = counter_path()
    n = int(path.read_text()) + 1 if path.exists() else 1
    if st is not None and st.get("commands_used") is not None:
        n = int(st["commands_used"]) + 1
    path.write_text(str(n))
    max_commands = int(CFG.get("max_commands", 30))
    if n > max_commands:
        sys.exit(
            f"command budget exhausted ({max_commands} commands). The episode is over."
        )
    return n


def run_steps(sock, raw, steps, start=None):
    """Apply one command for a number of steps, stopping early when the episode ends.

    With ``smooth``, the commanded arm joints and base velocities ramp from
    ``start`` toward ``raw`` at the demo-collection slew limits instead of
    jumping.

    Args:
        sock: Socket connected to the episode's worker.
        raw: The physical action to apply.
        steps: How many control steps to apply it for.
        start: The command to ramp from, when smoothing.

    Returns:
        ``(obs, done, termination, reward)``.
    """
    smooth = bool(CFG.get("smooth", False))
    obs, done, term, total = None, False, None, 0.0
    raw = np.asarray(raw, dtype=np.float64)
    cur = (
        np.asarray(start, dtype=np.float64).copy()
        if (smooth and start is not None)
        else raw.copy()
    )
    for _ in range(int(steps)):
        if smooth:
            d = raw - cur
            lim = np.full_like(cur, JOINT_STEP)
            lim[[0, 1, 3]] = BASE_STEP
            lim[2] = 0.01  # height: 0.5 m/s
            lim[18:20] = 1.0  # grippers are slew-limited by the env itself
            cur = cur + np.clip(d, -lim, lim)
        r = call(sock, {"op": "step", "action": [float(x) for x in cur]})
        obs, total = r["obs"], total + float(r["reward"])
        if r["done"]:
            done, term = True, r.get("termination")
            break
    return obs, done, term, total


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(add_help=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("look")
    m = sub.add_parser("move_hand")
    m.add_argument("side", choices=("left", "right"))
    m.add_argument("x", type=float)
    m.add_argument("y", type=float)
    m.add_argument("z", type=float)
    m.add_argument("--steps", type=int, default=60)
    w = sub.add_parser("walk")
    w.add_argument("vx", type=float)
    w.add_argument("vy", type=float)
    w.add_argument("wz", type=float)
    w.add_argument("--steps", type=int, required=True)
    h = sub.add_parser("hold")
    h.add_argument("--steps", type=int, required=True)
    g = sub.add_parser("gripper")
    g.add_argument("side", choices=("left", "right"))
    g.add_argument("value", type=float)
    g.add_argument("--steps", type=int, default=30)
    return ap


def main(argv=None):
    """Run one robot command and print the resulting observation.

    Args:
        argv: Command line arguments, or None for ``sys.argv[1:]``.
    """
    a = _parser().parse_args(argv)
    load_config()
    max_commands = int(CFG.get("max_commands", 30))

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(CFG["socket"])
    st = call(s, {"op": "status"})
    if a.cmd == "status":
        print_obs(st["obs"])
        used = int(counter_path().read_text()) if counter_path().exists() else 0
        print(f"commands used {used}/{max_commands}")
        if not st["episode_active"]:
            print(f"EPISODE OVER: {json.dumps(st.get('last_result'))}")
        return
    if not st["episode_active"]:
        sys.exit(
            f"EPISODE OVER: {json.dumps(st.get('last_result'))}. "
            f"No further commands are possible."
        )
    n = bump_counter(st)
    if a.cmd == "look":
        paths = []
        # onboard cameras only: there is no outside view of the scene
        for cam in ("head", "left_wrist", "right_wrist"):
            rel = f"{CFG['episode_rel']}/frames/cmd{n:02d}_{cam}.png"
            paths.append(call(s, {"op": "render", "camera": cam, "path": rel})["path"])
        print_obs(st["obs"])
        print("images: " + "  ".join(paths))
        print(f"commands used {n}/{max_commands}")
        return
    raw = np.asarray(call(s, {"op": "hold_action"})["action"], dtype=np.float64)
    start = raw.copy()  # current targets: where a ramp starts from
    steps = a.steps
    if a.cmd == "move_hand":
        kw = {f"{a.side}_pos": [a.x, a.y, a.z]}
        sol = call(s, {"op": "ik", **kw})["arm_qpos"]
        raw[4:18] = sol
        if abs(a.steps) > 400:
            sys.exit("--steps must be <= 400")
    elif a.cmd == "walk":
        if max(abs(a.vx), abs(a.vy)) > 0.5 or abs(a.wz) > 1.0 or a.steps > 400:
            sys.exit("limits: |vx|,|vy| <= 0.5 m/s, |wz| <= 1.0 rad/s, --steps <= 400")
        raw[0], raw[1], raw[3] = a.vx, a.vy, a.wz
    elif a.cmd == "hold":
        if a.steps > 400:
            sys.exit("--steps must be <= 400")
    elif a.cmd == "gripper":
        if a.steps > 400:
            sys.exit("--steps must be <= 400")
        raw[18 if a.side == "left" else 19] = float(np.clip(a.value, 0, 1))
    obs, done, _term, _rew = run_steps(s, raw, steps, start=start)
    if a.cmd == "walk" and not done:
        # stop the base after a walk command so the next observation is settled
        moving = raw.copy()
        raw[0], raw[1], raw[3] = 0.0, 0.0, 0.0
        obs2, done, _term, _rew2 = run_steps(
            s, raw, 25 if not CFG.get("smooth", False) else 40, start=moving
        )
        obs = obs2 or obs
    print_obs(obs)
    if done:
        st = call(s, {"op": "status"})
        print(f"EPISODE OVER: {json.dumps(st.get('last_result'))}")
    else:
        print(f"commands used {n}/{max_commands}")


if __name__ == "__main__":
    main()
