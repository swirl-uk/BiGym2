"""Environment server: owns the simulator, counts the agent's interaction budget.

Runs OUTSIDE the agent sandbox. Forks ``--workers`` processes, each holding one
environment and listening on ``<sandbox>/sock/w<k>.sock``. The step counter is
shared across workers; the append-only ledger lives in ``--ledger-dir`` (not
visible to the agent). Wire protocol: length-prefixed JSON (see wire.py).

The supervisor process also watches the sandbox's ``policy.py`` and records
every distinct version under ``<ledger-dir>/policies/`` (see snapshots.py),
labelling the content it finds at shutdown as the session's submission. This
is the session's edit history and needs nothing from the agent.

Every episode a worker runs is recorded as a replay batch under
``<ledger-dir>/dev/vNNN_<stamp>/batch/`` -- one directory per policy version,
the format the demo viewer reads -- so what the agent tried can be watched
while the session runs. ``--no-dev-recording`` switches that off.

    bigym-agent serve --task reach_target_single --sandbox DIR \
        --ledger-dir DIR2 --workers 4 --budget-steps 101000
"""

from __future__ import annotations

import base64
import json
import multiprocessing as mp
import os
import shutil
import signal
import socket
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from bigym.loco.eval.protocol import is_success

from . import wire
from .batch import Recorder, batch_metadata, write_episode_npz
from .cli import ServeConfig, limit_blas_threads, parse_command
from .envtools import (
    EVAL_SEED_HI,
    EVAL_SEED_LO,
    EnvTools,
)
from .snapshots import (
    DEV_DIR,
    POLICIES_DIR,
    PolicyWatcher,
    compact_stamp,
)

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import SynchronizedString

# Development seeds are [0, MAX_TRAIN_SEED): the 60 seeds the demonstrations were
# collected on, i.e. exactly the placements the learned baselines train on.
MAX_TRAIN_SEED = 60
ONBOARD_CAMERAS = ("head", "right_wrist", "left_wrist")
# Outside views; only reachable under --allow-external-cameras.
EXTERNAL_CAMERAS = ("third_person", "front")
CAMERAS = ONBOARD_CAMERAS + EXTERNAL_CAMERAS


class Budget:
    """Shared interaction-step counter and append-only ledger."""

    def __init__(self, cap: int, ledger_path: Path):
        """Create the counter.

        Args:
            cap: Total environment steps the agent may spend.
            ledger_path: File every event is appended to.
        """
        self.cap = int(cap)
        self.used = mp.Value("q", 0)
        self.lock = mp.Lock()
        self.ledger_path = ledger_path

    def charge(self, n: int = 1) -> int:
        """Charge n steps.

        Args:
            n: Steps to charge.

        Returns:
            The new total.

        Raises:
            RuntimeError: The budget is exhausted.
        """
        with self.lock:
            if self.used.value + n > self.cap:
                raise RuntimeError(
                    f"interaction budget exhausted: {self.used.value}/{self.cap} steps used"
                )
            self.used.value += n
            return int(self.used.value)

    def state(self) -> dict:
        """Return ``{'used', 'cap', 'remaining'}``."""
        with self.lock:
            return {
                "used": int(self.used.value),
                "cap": self.cap,
                "remaining": self.cap - int(self.used.value),
            }

    def log(self, worker: int, event: dict) -> None:
        """Append one event to the ledger.

        Args:
            worker: Worker index, or -1 for the supervisor.
            event: Event fields.
        """
        entry = {"ts": time.time(), "worker": worker, **event}
        with self.lock:
            with open(self.ledger_path, "a") as f:
                f.write(json.dumps(entry) + "\n")


class CurrentVersion:
    """The policy version the sandbox holds, shared with the workers.

    The watcher runs in the supervisor process and the episodes run in the
    workers, so the version an episode belongs to travels through shared
    memory: a counter and the directory name (``v006_20260921_010637``)
    the recordings of that version go into. Version 0 means "before the
    first snapshot".
    """

    def __init__(self, stamp: str):
        """Start at version 0, stamped with the server's start time.

        Args:
            stamp: Compact time (``YYYYMMDD_HHMMSS``) of the server start.
        """
        self.number = mp.Value("i", 0)
        self.label = cast("SynchronizedString", mp.Array("c", 32))
        self.set(0, stamp)

    def set(self, number: int, stamp: str) -> None:
        """Publish the version the sandbox now holds.

        Args:
            number: Version number.
            stamp: Compact time of that version's snapshot.
        """
        with self.number.get_lock():
            self.number.value = int(number)
            self.label.value = f"v{int(number):03d}_{stamp}".encode()[:31]

    def name(self) -> str:
        """Return the directory name of the current version."""
        with self.number.get_lock():
            return self.label.value.decode(errors="replace")


class DevRecorder:
    """Record every episode a worker runs, as a replay batch under ``dev/``.

    One batch directory per policy version (``dev/vNNN_<stamp>/batch/``), so
    the demo viewer can watch what the agent is trying while the session
    runs. This is bookkeeping only: the arrays are read out of the
    simulator after the step the agent asked for, nothing the policy sees
    changes, and any failure is reported once and then ignored -- a
    recording problem must never cost the agent its session.

    Episodes are written when they end, when they are abandoned (a reset
    while one was running, an ``abandon`` call, or the server shutting the
    worker down), and never while they are only half done: the file is
    built under ``dev/.staging/`` and hard-linked into the batch directory,
    so a reader polling ``batch/*.npz`` sees whole episodes only.

    An episode is ``seed<S>_<n>.npz``, where ``n`` counts the episodes of
    the directory: an agent runs the same seed over and over, so the seed
    alone does not name a file, and the number says which attempt it was.
    """

    def __init__(self, dev_dir: Path, current: CurrentVersion, worker: int, tools):
        """Bind the recorder to one worker's environment.

        Args:
            dev_dir: ``<ledger-dir>/dev``.
            current: The shared current policy version.
            worker: Worker index (names the staging file).
            tools: The wrapper holding the environment.
        """
        self.dev_dir = Path(dev_dir)
        self.current = current
        self.worker = int(worker)
        self.tools = tools
        self.recorder = None
        self.batch: Path | None = None
        self.seed: int | None = None
        self.counters: dict[Path, int] = {}
        self.described: set[Path] = set()
        self.warned: set[str] = set()
        self.writing = False

    # ------------------------------------------------------------ episodes
    def start(self, seed: int) -> None:
        """Begin recording the episode that just reset.

        Args:
            seed: The episode's seed.
        """
        try:
            self.abandon()
            if self.recorder is None:
                self.recorder = Recorder(self.tools)
            self.recorder.reset()
            self.recorder.capture_state()
            self.batch = self.dev_dir / self.current.name() / "batch"
            self.seed = int(seed)
            self.describe(self.batch)
        except Exception as exc:
            self.failed("start", exc)
            self.batch, self.seed = None, None

    def step(self, action, reward: float) -> None:
        """Record the state one step left the simulator in.

        Args:
            action: The raw physical action that was applied.
            reward: The reward it earned.
        """
        if self.batch is None or self.recorder is None:
            return
        try:
            self.recorder.capture_state()
            self.recorder.capture_step(action, reward)
        except Exception as exc:
            self.failed("step", exc)
            self.batch = None

    def finish(self, *, success: int, fell: bool, termination: str) -> Path | None:
        """Write the finished episode and forget it.

        Args:
            success: 1 when the episode succeeded.
            fell: Whether the robot fell.
            termination: The episode's termination label.

        Returns:
            The path written, or None when there was nothing to write.
        """
        batch, self.batch = self.batch, None
        if batch is None or self.recorder is None or self.writing:
            return None
        self.writing = True
        try:
            arrays = self.recorder.arrays()
            if len(arrays["action"]) == 0:
                return None
            return self.write(
                batch, arrays, success=success, fell=fell, termination=termination
            )
        except Exception as exc:
            self.failed("write", exc)
            return None
        finally:
            self.writing = False
            self.recorder.reset()

    def abandon(self) -> Path | None:
        """Write a running episode that stopped without an outcome.

        Returns:
            The path written, or None when no episode was running.
        """
        if self.batch is None:
            return None
        fell = False
        try:
            fell = bool(self.tools.outer.episode_fell())
        except Exception:  # the outcome is unknown; the episode still is not
            pass
        return self.finish(success=0, fell=fell, termination="abandoned")

    # -------------------------------------------------------------- files
    def describe(self, batch: Path) -> None:
        """Write a batch directory's ``metadata.json`` once.

        Args:
            batch: The batch directory.
        """
        if batch in self.described:
            return
        self.described.add(batch)
        if (batch / "metadata.json").exists():
            return
        source = {
            "kind": "agent_dev",
            "cell": str(self.dev_dir.parent),
            "version": int(self.current.number.value),
        }
        batch.mkdir(parents=True, exist_ok=True)
        payload = batch_metadata(self.tools, source=source)
        tmp = batch / f".metadata.w{self.worker}.tmp"
        tmp.write_text(json.dumps(payload, indent=1, default=str))
        os.replace(tmp, batch / "metadata.json")

    def write(self, batch: Path, arrays, *, success, fell, termination) -> Path:
        """Write one episode into a batch directory, atomically.

        The episode is written into a staging directory outside the batch
        and hard-linked in under the first free ``seed<S>_<n>.npz``: the
        link fails rather than overwriting, so two workers finishing at the
        same moment cannot lose an episode, and no reader ever opens a file
        that is still being written.

        Args:
            batch: The batch directory.
            arrays: The recorded arrays.
            success: 1 when the episode succeeded.
            fell: Whether the robot fell.
            termination: The episode's termination label.

        Returns:
            The path written.
        """
        staging = self.dev_dir / ".staging"
        staging.mkdir(parents=True, exist_ok=True)
        batch.mkdir(parents=True, exist_ok=True)
        tmp = write_episode_npz(
            staging / f"w{self.worker}_{os.getpid()}.npz",
            full_qpos=arrays["full_qpos"],
            full_qvel=arrays["full_qvel"],
            action=arrays["action"],
            reward=arrays["reward"],
            seed=int(self.seed or 0),
            success=int(success),
            fell=bool(fell),
            length=int(len(arrays["action"])),
            termination=str(termination or ""),
        )
        number = self.counters.get(batch, 0)
        while True:
            path = batch / f"seed{int(self.seed or 0)}_{number}.npz"
            try:
                os.link(tmp, path)
            except FileExistsError:
                number += 1
                continue
            break
        self.counters[batch] = number + 1
        os.unlink(tmp)
        return path

    def failed(self, what: str, exc: Exception) -> None:
        """Report a recording failure once per kind.

        Args:
            what: Which step failed.
            exc: The exception.
        """
        if what in self.warned:
            return
        self.warned.add(what)
        print(
            f"[dev] {what} failed, recording is off for this worker: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )


def check_reset_seed(
    seed: int,
    allowed_seeds: frozenset | None = None,
    max_train_seed: int = MAX_TRAIN_SEED,
    allow_eval_seeds: bool = False,
) -> int:
    """Check that the agent may reset on this seed.

    The hidden evaluation block ``[EVAL_SEED_LO, EVAL_SEED_HI)`` is refused
    unless the server runs the in-the-loop evaluation, where the model itself
    is the policy and the episodes are the protocol episodes.

    Args:
        seed: Requested seed.
        allowed_seeds: Explicit development seed set, or None for the range.
        max_train_seed: Development seeds are ``[0, max_train_seed)`` when no
            explicit set is given.
        allow_eval_seeds: Permit the hidden block (in-the-loop evaluation).

    Returns:
        The seed, as an int.

    Raises:
        ValueError: The seed is outside the development seeds.
    """
    seed = int(seed)
    if allow_eval_seeds:
        return seed
    ok = (
        (seed in allowed_seeds)
        if allowed_seeds is not None
        else (0 <= seed < max_train_seed)
    )
    if not ok or EVAL_SEED_LO <= seed < EVAL_SEED_HI:
        raise ValueError(
            f"seed {seed} is not allowed: use the development seeds listed in seeds.json"
            if allowed_seeds is not None
            else f"seed {seed} is not allowed: use training seeds in [0, {max_train_seed})"
        )
    return seed


# sun_path is 108 bytes on Linux; leave room for ``/w<k>.lock`` names.
SOCK_PATH_LIMIT = 96


def prepare_sock_dir(sandbox: Path) -> Path:
    """Return the directory the worker sockets bind in, creating it.

    Unix socket paths are limited to 108 bytes, so a sandbox deep in the
    filesystem cannot hold its own sockets. In that case a short directory is
    created under the system temp dir and ``<sandbox>/sock`` becomes a
    symlink to it; clients resolve the link before connecting (the launcher
    bind-mounts the resolved directory at ``<client-root>/sock`` in a
    container).

    Args:
        sandbox: The resolved sandbox directory.

    Returns:
        The real directory holding ``w<k>.sock`` files.
    """
    link = sandbox / "sock"
    if link.is_symlink():
        target = Path(os.readlink(link))
        if target.is_dir():
            return target
        link.unlink()
    if len(str(link / "w99.sock").encode()) <= SOCK_PATH_LIMIT:
        link.mkdir(parents=True, exist_ok=True)
        return link
    if link.is_dir():
        if any(link.iterdir()):
            raise RuntimeError(
                f"{link} is too long for unix sockets and not empty; "
                "move the sandbox to a shorter path or empty sock/"
            )
        link.rmdir()
    short = Path(tempfile.mkdtemp(prefix="bigym-agent-sock-"))
    os.chmod(short, 0o755)
    link.symlink_to(short, target_is_directory=True)
    return short


def _client_rel(path: str, client_root: str | None) -> str:
    """Path as the client wrote it -> path relative to the sandbox.

    Inside the agent container the sandbox is mounted at ``client_root`` (e.g.
    /work), so absolute container paths are mapped back onto the host sandbox.

    Args:
        path: The path the client sent.
        client_root: Where the client sees the sandbox, or None.

    Returns:
        A sandbox-relative path.
    """
    if client_root and path.startswith(client_root.rstrip("/") + "/"):
        return path[len(client_root.rstrip("/")) + 1 :]
    return path


def _client_path(out: Path, sandbox: Path, client_root: str | None) -> str:
    """Host path of a file under the sandbox -> the path the client should use.

    Args:
        out: Host path of the file.
        sandbox: Host sandbox directory.
        client_root: Where the client sees the sandbox, or None.

    Returns:
        The path to report to the client.
    """
    if client_root:
        return str(Path(client_root) / out.resolve().relative_to(sandbox.resolve()))
    return str(out)


# In-loop mode counts commands: a connection that uses one of these is one
# command of the per-episode cap.
ACTING_OPS = ("step", "render", "image", "ik", "hold_action")


class EpisodeService:
    """One worker's environment and its episode, served one request at a time.

    The episode outlives a connection: the agent's runner connects once per
    command in in-loop mode, and once per run of the development episodes
    otherwise.
    """

    def __init__(
        self,
        worker: int,
        args: ServeConfig,
        tools: EnvTools,
        budget: Budget,
        allowed_seeds: frozenset | None,
        dev: DevRecorder | None,
    ):
        """Bind the service to a worker's environment.

        Args:
            worker: Worker index.
            args: The server's settings.
            tools: The worker's environment.
            budget: The shared budget.
            allowed_seeds: Explicit development seed set, or None for the
                range ``[0, args.train_seeds)``.
            dev: The development-episode recorder, or None.
        """
        self.worker = worker
        self.args = args
        self.sandbox = args.sandbox.resolve()
        self.tools = tools
        self.budget = budget
        self.allowed_seeds = allowed_seeds
        self.dev = dev
        # the controller warm-up of a reset really runs, except in the
        # in-the-loop evaluation, whose episodes are the protocol's
        self.reset_cost = 0 if args.allow_eval_seeds else args.reset_cost
        self.seed: int | None = None
        self.reward = 0.0
        self.length = 0
        self.max_commands: int | None = None
        self.commands_used: int | None = None
        self.last_result: dict | None = None
        self.replies = {
            "info": self.reply_info,
            "budget": self.reply_budget,
            "reset": self.reply_reset,
            "step": self.reply_step,
            "abandon": self.reply_abandon,
            "status": self.reply_status,
            "hold_action": self.reply_hold_action,
            "ik": self.reply_ik,
            "view": self.reply_view,
            "image": self.reply_image,
            "camera_info": self.reply_camera_info,
            "render": self.reply_render,
        }

    def serve(self, conn: socket.socket) -> None:
        """Answer the requests of one connection until the client hangs up.

        A request that fails is answered with the error and the connection
        stays open.

        Args:
            conn: The accepted connection.
        """
        counted = False
        while True:
            try:
                msg = wire.recv(conn)
            except (ConnectionError, OSError):
                return
            try:
                op = msg.get("op")
                if op in ACTING_OPS and self.seed is not None and not counted:
                    self.count_command()
                    counted = True
                if op not in self.replies:
                    raise ValueError(f"unknown op {op!r}")
                reply = self.replies[op](msg)
            except Exception as exc:  # report to the agent, keep serving
                reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                if not isinstance(exc, (ValueError, RuntimeError)):
                    reply["traceback"] = traceback.format_exc()
            try:
                wire.send(conn, reply)
            except OSError:
                return

    def count_command(self) -> None:
        """Charge one command of the episode's cap, when it has one.

        Raises:
            RuntimeError: The episode's commands are used up.
        """
        if self.max_commands is None:
            return
        used = self.commands_used or 0
        if used >= int(self.max_commands):
            raise RuntimeError(
                f"command budget exhausted ({self.max_commands} commands). "
                "The episode is over."
            )
        self.commands_used = used + 1

    def camera(self, msg: dict) -> str:
        """Return the camera a request names, refusing one the policy may not use.

        Args:
            msg: The request.

        Returns:
            The camera name.

        Raises:
            ValueError: The camera is not one of the allowed ones.
        """
        camera = str(msg.get("camera", "head"))
        # EnvTools refuses these too; keep both gates.
        allowed = ONBOARD_CAMERAS if self.tools.onboard_only else CAMERAS
        if camera not in allowed:
            raise ValueError(f"camera must be one of {allowed}")
        return camera

    def reply_info(self, msg: dict) -> dict:
        """Describe the environment."""
        return {"ok": True, "info": self.tools.info()}

    def reply_budget(self, msg: dict) -> dict:
        """Report the interaction budget."""
        return {"ok": True, "budget": self.budget.state()}

    def reply_reset(self, msg: dict) -> dict:
        """Start an episode on a development seed, abandoning a running one."""
        seed = check_reset_seed(
            msg["seed"],
            self.allowed_seeds,
            self.args.train_seeds,
            self.args.allow_eval_seeds,
        )
        if self.reset_cost > 0:
            self.budget.charge(self.reset_cost)
        if self.seed is not None and self.length > 0:
            self.budget.log(
                self.worker,
                {
                    "event": "episode_abandoned",
                    "seed": self.seed,
                    "length": self.length,
                    "reward": self.reward,
                },
            )
        if self.dev is not None:
            # the episode the agent walked away from is still worth
            # seeing; it is read out before the state is replaced
            self.dev.abandon()
        self.tools.reset(seed)
        self.max_commands = msg.get("max_commands")
        self.commands_used = 0
        if self.args.record_dir is not None:
            self.tools.start_recording(
                self.args.record_dir
                / f"w{self.worker}_seed{seed}_{int(time.time())}.mp4"
            )
        if self.dev is not None:
            self.dev.start(seed)
        self.seed, self.reward, self.length = seed, 0.0, 0
        self.budget.log(
            self.worker, {"event": "reset", "seed": seed, "budget": self.budget.state()}
        )
        return {"ok": True, "obs": self.tools.observation()}

    def reply_step(self, msg: dict) -> dict:
        """Apply one action of the running episode."""
        if self.seed is None:
            raise RuntimeError("call reset(seed) before step")
        tools, dev = self.tools, self.dev
        raw = np.asarray(msg["action"], dtype=np.float32)
        used = self.budget.charge(1)
        ts = tools.step(raw)
        r = float(ts.reward or 0.0)
        if dev is not None:
            dev.step(raw, r)
        self.reward += r
        self.length += 1
        done = bool(ts.last())
        term, fell = None, False
        if done:
            fell = bool(tools.outer.episode_fell())
            success = is_success(tools.outer)
            term = tools.outer.episode_termination()
            if dev is not None:
                dev.finish(success=int(success), fell=fell, termination=str(term))
            self.budget.log(
                self.worker,
                {
                    "event": "episode_end",
                    "seed": self.seed,
                    "success": int(success),
                    "length": self.length,
                    "reward": self.reward,
                    "termination": term,
                    "fell": fell,
                    "budget_used": used,
                },
            )
        reply = {
            "ok": True,
            "obs": tools.observation(),
            "reward": r,
            "done": done,
            "termination": term,
            "budget_used": used,
        }
        if done:
            tools.stop_recording()
            self.last_result = {
                "seed": self.seed,
                "success": int(success),
                "length": self.length,
                "reward": self.reward,
                "termination": term,
                "fell": fell,
            }
            self.seed = None
        return reply

    def reply_abandon(self, msg: dict) -> dict:
        """End the running episode without an outcome and free the worker."""
        self.tools.stop_recording()
        if self.dev is not None:
            self.dev.abandon()
        if self.seed is not None:
            fell = bool(self.tools.outer.episode_fell())
            self.budget.log(
                self.worker,
                {
                    "event": "episode_abandoned",
                    "seed": self.seed,
                    "length": self.length,
                    "reward": self.reward,
                    "fell": fell,
                },
            )
            self.last_result = {
                "seed": self.seed,
                "success": 0,
                "length": self.length,
                "reward": self.reward,
                "termination": "agent_stopped",
                "fell": fell,
            }
        self.seed = None
        return {"ok": True, "last_result": self.last_result}

    def reply_status(self, msg: dict) -> dict:
        """Report the episode, the budget and the command count."""
        active = self.seed is not None
        return {
            "ok": True,
            "episode_active": active,
            "seed": self.seed,
            "length": self.length,
            "reward": self.reward,
            "last_result": self.last_result,
            "budget": self.budget.state(),
            "commands_used": self.commands_used,
            "max_commands": self.max_commands,
            "obs": self.tools.observation() if active else None,
        }

    def reply_hold_action(self, msg: dict) -> dict:
        """Return the action that holds the current pose."""
        return {"ok": True, "action": self.tools.hold_action()}

    def reply_ik(self, msg: dict) -> dict:
        """Solve the arm joints reaching the requested wrist poses."""
        arm_qpos = self.tools.ik(
            left_pos=msg.get("left_pos"),
            left_quat=msg.get("left_quat"),
            right_pos=msg.get("right_pos"),
            right_quat=msg.get("right_quat"),
        )
        return {"ok": True, "arm_qpos": arm_qpos}

    def reply_view(self, msg: dict) -> dict:
        """Refuse an outside view of the scene, and log the attempt.

        There is no outside view for the agent: it would be scene information
        the learned baselines never get. The op is an explicit refusal (and a
        ledger entry) rather than an unknown op.
        """
        self.budget.log(
            self.worker, {"event": "view_refused", "camera": str(msg.get("camera"))}
        )
        raise ValueError(
            "there is no outside view of the scene; the robot's own cameras "
            f"{ONBOARD_CAMERAS} are the only cameras (op 'render' / 'image')"
        )

    def reply_image(self, msg: dict) -> dict:
        """Return one camera image as base64 PNG."""
        camera = self.camera(msg)
        w, h = int(msg.get("width", 84)), int(msg.get("height", 84))
        png = self.tools.image_png(camera, w, h)
        return {"ok": True, "png_b64": base64.b64encode(png).decode()}

    def reply_camera_info(self, msg: dict) -> dict:
        """Return the calibration of every camera."""
        return {"ok": True, "cameras": self.tools.camera_info()}

    def reply_render(self, msg: dict) -> dict:
        """Save a PNG from one camera to a path inside the sandbox."""
        camera = self.camera(msg)
        client_root = self.args.client_root
        rel = _client_rel(
            str(msg.get("path", f"frames/{int(time.time() * 1000)}_{camera}.png")),
            client_root,
        )
        out = (self.sandbox / rel).resolve()
        if self.sandbox not in out.parents:
            raise ValueError("render path must be inside the sandbox directory")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(self.tools.render_png(camera))
        return {"ok": True, "path": _client_path(out, self.sandbox, client_root)}


def serve_environment_worker(
    worker: int,
    args: ServeConfig,
    budget: Budget,
    allowed_seeds: frozenset | None,
    dev_dir: Path | None,
    current: CurrentVersion | None,
):
    """Hold one environment and serve one connection at a time.

    Args:
        worker: Worker index (names the socket).
        args: The server's settings.
        budget: The shared budget.
        allowed_seeds: Explicit development seed set, or None for the range.
        dev_dir: Record every episode as a replay batch under this
            directory (``<ledger-dir>/dev``), or None for no recording.
        current: The shared current policy version, which names the batch
            directory an episode belongs to.
    """
    # ``sandbox/sock`` may be a symlink to a short directory (unix socket
    # paths are limited to 108 bytes); bind on the resolved path.
    sock_path = (args.sandbox.resolve() / "sock").resolve() / f"w{worker}.sock"
    if sock_path.exists():
        sock_path.unlink()
    tools = EnvTools(args.task, args.env_tools)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_path))
    os.chmod(sock_path, 0o666)
    srv.listen(1)
    dev = None
    if dev_dir is not None and current is not None:
        dev = DevRecorder(dev_dir, current, worker, tools)

        def flush(_signum, _frame):
            """Write the episode in flight before the supervisor kills us."""
            try:
                dev.abandon()
            finally:
                os._exit(0)

        # The supervisor terminates its workers at shutdown; an episode the
        # agent was still running is as interesting as any other.
        signal.signal(signal.SIGTERM, flush)
    service = EpisodeService(worker, args, tools, budget, allowed_seeds, dev)
    print(f"[server] worker {worker} ready on {sock_path}", flush=True)
    while True:
        conn, _ = srv.accept()
        try:
            service.serve(conn)
        finally:
            conn.close()


def main(argv: list[str] | ServeConfig | None = None):
    """Start the workers and supervise them until they are stopped.

    Args:
        argv: The parsed settings, or command line arguments (None for
            ``sys.argv[1:]``).
    """
    limit_blas_threads()
    args = parse_command(ServeConfig, argv)
    started = time.time()
    allowed = (
        frozenset(int(s) for s in json.loads(args.allowed_seeds.read_text()))
        if args.allowed_seeds
        else None
    )
    sandbox = args.sandbox.resolve()
    sock_dir = prepare_sock_dir(sandbox)
    (args.ledger_dir).mkdir(parents=True, exist_ok=True)
    (args.ledger_dir / "sock_dir").write_text(str(sock_dir) + "\n")
    if sock_dir != sandbox / "sock":
        print(f"[server] sockets in {sock_dir} (sandbox/sock links there)", flush=True)
    (sandbox / "frames").mkdir(exist_ok=True)
    args.ledger_dir.mkdir(parents=True, exist_ok=True)
    ledger = args.ledger_dir / "ledger.jsonl"
    budget = Budget(args.budget_steps, ledger)
    prev = args.ledger_dir / "budget.json"
    if prev.exists():  # a restarted server continues the same budget
        budget.used.value = int(json.loads(prev.read_text()).get("used", 0))
    env_tools = args.env_tools
    budget.log(
        -1,
        {
            "event": "server_start",
            "task": args.task,
            "budget_cap": args.budget_steps,
            "workers": args.workers,
            "tier": env_tools.tier,
            "reset_cost": args.reset_cost,
            "train_seeds": args.train_seeds,
            "allowed_seeds": sorted(allowed) if allowed is not None else None,
            "slew": env_tools.slew,
            "accel": env_tools.accel if env_tools.slew else None,
            "lowpass": env_tools.lowpass if env_tools.slew else None,
            "onboard_only": not env_tools.allow_external_cameras,
            "client_root": args.client_root,
            "interface": env_tools.interface,
            "hand_pos": env_tools.interface == "tools" and env_tools.hand_pos,
            "ik": env_tools.interface == "tools" and env_tools.ik,
            "calibration": env_tools.interface == "tools",
            "image_cap": env_tools.image_cap,
            "external_view": False,
        },
    )
    ledger_dir = args.ledger_dir.resolve()
    dev_dir = ledger_dir / DEV_DIR if args.dev_recording else None
    # Shared with the workers: the version an episode belongs to, and the
    # directory name its recording goes into.
    current = CurrentVersion(compact_stamp(started)) if dev_dir else None
    watcher = None
    if args.snapshots:
        watcher = PolicyWatcher(
            sandbox,
            args.policies_dir or (ledger_dir / POLICIES_DIR),
            task=args.task,
            ledger_dir=ledger_dir,
            started=started,
            on_version=(
                None
                if current is None
                else lambda entry: current.set(
                    int(entry.get("version") or 0),
                    compact_stamp(entry.get("ts") or entry.get("time")),
                )
            ),
        )
        # Polled before the workers exist so the sandbox's starting policy.py
        # is version 1 and the first episode is already attributed to it.
        watcher.poll()

    procs = []
    for k in range(args.workers):
        p = mp.Process(
            target=serve_environment_worker,
            args=(k, args, budget, allowed, dev_dir, current),
            daemon=True,
        )
        p.start()
        procs.append(p)

    if watcher is not None:
        # The polling thread starts after the workers so a failing worker is
        # reported first.
        watcher.start()

    def finalize_policies():
        if watcher is None:
            return
        try:
            entry = watcher.finalize()
        except Exception as exc:  # shutdown must not depend on the snapshots
            print(f"[policies] final snapshot failed: {exc}", flush=True)
            return
        if entry is not None:
            print(
                f"[policies] submission = v{int(entry['version']):03d} "
                f"({len(watcher.versions())} versions)",
                flush=True,
            )

    def stop(*_):
        finalize_policies()
        budget.log(-1, {"event": "server_stop", "budget": budget.state()})
        for p in procs:
            p.terminate()
        for p in procs:
            p.join(timeout=5)
        for p in procs:
            if p.is_alive():
                # A worker writes its unfinished episode when it is asked to
                # stop, which means it handles SIGTERM itself -- and a handler
                # only runs once the call the worker is inside returns. A
                # worker busy in a long library call would otherwise outlive
                # the server, so the wait is bounded and then it is killed.
                p.kill()
                p.join(timeout=5)
        if sock_dir != sandbox / "sock":
            # the short temp directory is ours; the sandbox keeps a dangling
            # link that the next server run replaces
            shutil.rmtree(sock_dir, ignore_errors=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while True:
        time.sleep(5)
        (args.ledger_dir / "budget.json").write_text(json.dumps(budget.state()))
        if not any(p.is_alive() for p in procs):
            print("[server] all workers died", flush=True)
            stop()


if __name__ == "__main__":
    mp.set_start_method("fork")
    main()
