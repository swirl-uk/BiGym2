"""A policy file run in its own interpreter, driven over a pair of pipes.

The evaluator owns the simulator. A policy imported into the evaluator's
process could reach that simulator through the garbage collector (or
``sys.modules``, or a frame walk) and rewrite the task's success test, read
the episode seed, or plan on a copy of the physics. A :class:`PolicyProcess`
runs the policy in a fresh interpreter instead: started with ``-I`` from
``sys.executable`` (exec'd, never forked, so no copy of the simulator exists
there), it imports only this file, ``episode.py`` and the policy. One control
step is one round trip::

    parent -> child   ("act", obs)                       pickle
    child  -> parent  {"op": "action", "value": <f32>}   JSON

and every tool call the policy makes in between is forwarded to the parent,
which runs it on the real environment and sends the result back. The episode
loop, the reward, the termination and the success test stay in the parent:
:func:`episode.run_episode` is unchanged, a :class:`PolicyProcess` is simply
the ``policy`` it is handed.

Nothing the child sends is unpickled. Its messages are JSON whose only
extensions are tuples and plain numeric arrays (dtype, shape, raw bytes), so a
policy cannot smuggle code into the parent through a reply.

A policy may not import the simulator (:data:`FORBIDDEN_MODULES`) to build its
own copy and plan on it. The evaluator scans the policy's directory for such
imports before it runs an episode (:func:`forbidden_imports`), and the child
refuses them at run time as a backstop for the imports a static scan cannot
see; either way the policy is rejected and scores 0.

This module is also the child's entry script, so it imports nothing from
``bigym`` at module level.
"""

from __future__ import annotations

import ast
import base64
import builtins
import importlib.util
import json
import math
import os
import subprocess
import sys
import traceback
from multiprocessing.connection import Connection
from pathlib import Path, PurePath
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
# The key that marks a JSON object as an encoded tuple or array.
WIRE_TAG = "__bigym_wire__"
# Array kinds a message may carry: bool, signed, unsigned, float, complex.
NUMERIC_KINDS = "biufc"
# The largest message the parent reads from the child: far above an image at
# the resolution cap, low enough that a runaway policy cannot exhaust memory.
MAX_MESSAGE = 64 << 20
# How long a closing child may take to exit before it is killed.
CLOSE_TIMEOUT_S = 5.0
# Top-level packages a policy may not import: the simulator and the packages
# that build or drive one. A policy that imports any of them scores 0.
FORBIDDEN_MODULES = frozenset({"bigym", "mujoco", "dm_control", "mink"})


class PolicyError(RuntimeError):
    """The policy failed in its process: it raised, or the process died."""


class ForbiddenImport(PolicyError):
    """The policy imported a module of :data:`FORBIDDEN_MODULES`; it is rejected."""


def _episode_module():
    """Return the episode contract, ``episode.py`` next to this file."""
    if __package__:
        from . import episode

        return episode
    spec = importlib.util.spec_from_file_location(
        "bigym_agent_episode", HERE / "episode.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tool_names() -> frozenset[str]:
    """Return the tool methods a policy may call: the public ``episode.Tools`` surface.

    Each ``Tools`` method forwards to the environment method of the same
    name with its own arguments, which is what lets the child's ``Tools`` ask
    the parent for an environment call by that name.
    """
    tools = _episode_module().Tools
    return frozenset(
        name
        for name, value in vars(tools).items()
        if callable(value) and not name.startswith("_")
    )


# --------------------------------------------------------------------------
# Forbidden imports
# --------------------------------------------------------------------------


def _forbidden(name: object) -> bool:
    return isinstance(name, str) and name.split(".")[0] in FORBIDDEN_MODULES


def _imported_names(tree: ast.AST):
    """Yield ``(line, module)`` for every absolute import in a parsed source.

    Covers the import statements and ``__import__`` / ``import_module`` calls
    whose module is a string constant.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.lineno, node.module
        elif isinstance(node, ast.Call) and node.args:
            func = node.func
            name = func.id if isinstance(func, ast.Name) else None
            if isinstance(func, ast.Attribute):
                name = func.attr
            arg = node.args[0]
            if (
                name in ("__import__", "import_module")
                and isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
            ):
                yield node.lineno, arg.value


def forbidden_imports(policy_path) -> list[str]:
    """Find the imports of :data:`FORBIDDEN_MODULES` in a policy's directory.

    Every ``.py`` file next to the policy is scanned, because the policy's
    directory is on its import path and any of them can be imported as a
    helper. A file that does not parse is skipped: importing it fails the
    run anyway.

    Args:
        policy_path: The policy file.

    Returns:
        One ``"<file>:<line> imports <module>"`` per hit, in file order;
        empty when the policy imports none of them.
    """
    hits = []
    for path in sorted(Path(policy_path).resolve().parent.glob("*.py")):
        try:
            tree = ast.parse(path.read_bytes(), filename=str(path))
        except (OSError, SyntaxError, ValueError):
            continue
        hits += [
            f"{path.name}:{line} imports {module}"
            for line, module in sorted(_imported_names(tree))
            if _forbidden(module)
        ]
    return hits


class _ForbiddenFinder:
    """A ``sys.meta_path`` finder that refuses :data:`FORBIDDEN_MODULES`.

    It remembers the first refusal, so a policy that catches the ImportError
    is still reported.
    """

    def __init__(self):
        self.hit: str | None = None

    def find_spec(self, name, path=None, target=None):
        """Refuse a forbidden module; leave every other one to the next finder."""
        if not _forbidden(name):
            return None
        if self.hit is None:
            self.hit = f"{_importer()} imports {name}"
        raise ImportError(
            f"{name}: a policy may not import "
            f"{', '.join(sorted(FORBIDDEN_MODULES))}; it is rejected and scores 0",
            name=name,
        )


def _importer() -> str:
    """Return ``<file>:<line>`` of the code that asked for the import being refused."""
    skip = {
        str(Path(__file__).resolve()),
        str(Path(importlib.__file__ or "").resolve()),
    }
    for frame in reversed(traceback.extract_stack()):
        if (
            frame.filename.startswith("<")
            or str(Path(frame.filename).resolve()) in skip
        ):
            continue
        return f"{Path(frame.filename).name}:{frame.lineno}"
    return "?"


# --------------------------------------------------------------------------
# The child -> parent encoding
# --------------------------------------------------------------------------


def _to_json(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)):
        arr = np.asarray(obj)
        if arr.dtype.kind not in NUMERIC_KINDS:
            raise TypeError(f"cannot send an array of dtype {arr.dtype}")
        return {
            WIRE_TAG: "ndarray" if isinstance(obj, np.ndarray) else "scalar",
            "dtype": arr.dtype.str,
            "shape": list(arr.shape),
            "data": base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode(),
        }
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, tuple):
        return {WIRE_TAG: "tuple", "items": [_to_json(v) for v in obj]}
    if isinstance(obj, list):
        return [_to_json(v) for v in obj]
    if isinstance(obj, dict):
        if not all(isinstance(k, str) for k in obj) or WIRE_TAG in obj:
            raise TypeError("cannot send a dict whose keys are not plain strings")
        return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, PurePath):
        return str(obj)
    raise TypeError(f"cannot send a {type(obj).__name__} to the evaluator")


def encode(obj: Any) -> bytes:
    """Encode a message from the child.

    Args:
        obj: None, bools, numbers, strings, lists, tuples, dicts with string
            keys, paths and numeric numpy arrays or scalars, nested freely.

    Returns:
        The UTF-8 JSON bytes.

    Raises:
        TypeError: Something else is in the message.
    """
    return json.dumps(_to_json(obj), separators=(",", ":")).encode()


def _array(item: dict) -> Any:
    dtype = np.dtype(str(item["dtype"]))
    if dtype.kind not in NUMERIC_KINDS or dtype.hasobject:
        raise ValueError(f"array of dtype {dtype} refused")
    shape = item["shape"]
    if not isinstance(shape, list) or not all(
        isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in shape
    ):
        raise ValueError(f"bad array shape {shape!r}")
    raw = base64.b64decode(str(item["data"]), validate=True)
    if len(raw) != math.prod(shape) * dtype.itemsize:
        raise ValueError("array data does not match its shape")
    arr = np.frombuffer(raw, dtype=dtype).reshape(shape)
    return arr[()] if item[WIRE_TAG] == "scalar" else arr.copy()


def _from_json(obj: Any) -> Any:
    if isinstance(obj, list):
        return [_from_json(v) for v in obj]
    if not isinstance(obj, dict):
        return obj
    kind = obj.get(WIRE_TAG)
    if kind is None:
        return {k: _from_json(v) for k, v in obj.items()}
    if kind == "tuple":
        return tuple(_from_json(v) for v in obj["items"])
    if kind in ("ndarray", "scalar"):
        return _array(obj)
    raise ValueError(f"unknown wire tag {kind!r}")


def decode(data: bytes) -> Any:
    """Decode a message from the child: the inverse of :func:`encode`.

    Args:
        data: The bytes the child sent.

    Returns:
        The message.

    Raises:
        ValueError: The bytes are not a well-formed message.
    """
    try:
        return _from_json(json.loads(data))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"malformed message: {type(exc).__name__}: {exc}") from exc


# --------------------------------------------------------------------------
# Parent side
# --------------------------------------------------------------------------


class PolicyProcess:
    """A policy file in its own interpreter, with the policy contract on top.

    ``load`` imports a policy file in the child (a fresh ``Policy`` instance,
    exactly as :func:`episode.load_policy` makes one) and returns this object,
    whose ``reset(obs, tools)`` and ``act(obs, tools)`` run that instance and
    serve its tool calls from ``tools``, the parent's real
    :class:`episode.Tools`. One process serves any number of episodes::

        with PolicyProcess() as proc:
            for seed in seeds:
                rec = run_episode(env, proc.load(policy_path), seed)

    Anything the policy raises, in ``load``, ``reset`` or ``act``, comes back
    as a :class:`PolicyError` carrying the policy's traceback, so a failing
    policy still ends the run the way it would in-process. A policy that tried
    to import the simulator raises :class:`ForbiddenImport` instead, whether or
    not it caught the ImportError.
    """

    def __init__(self, python: str | None = None):
        """Start the child interpreter.

        Args:
            python: The interpreter to run the policy with; defaults to
                ``sys.executable``.
        """
        self.python = python or sys.executable
        child_read, parent_write = os.pipe()
        parent_read, child_write = os.pipe()
        # -I: no script directory, user site or PYTHON* variables, so the
        # package's own modules cannot shadow helpers next to the policy.
        cmd = [self.python, "-I"]
        if sys.dont_write_bytecode:
            cmd.append("-B")
        cmd += [str(HERE / "policy_process.py"), str(child_read), str(child_write)]
        try:
            self._proc = subprocess.Popen(
                cmd, pass_fds=(child_read, child_write), stdin=subprocess.DEVNULL
            )
        except BaseException:
            os.close(parent_read)
            os.close(parent_write)
            raise
        finally:
            os.close(child_read)
            os.close(child_write)
        self._outbox = Connection(parent_write, readable=False)
        self._inbox = Connection(parent_read, writable=False)
        self._tools = tool_names()

    def __enter__(self) -> PolicyProcess:
        """Return the running process."""
        return self

    def __exit__(self, *exc) -> None:
        """Stop the child."""
        self.close()

    def load(self, path) -> PolicyProcess:
        """Instantiate the ``Policy`` of a file in the child.

        Args:
            path: Path to the policy file.

        Returns:
            This object, now standing for the fresh instance.

        Raises:
            PolicyError: The file failed to import or defines no ``Policy``.
        """
        self._send(("load", str(Path(path).resolve())))
        self._serve(None, "loaded")
        return self

    def reset(self, obs, tools) -> None:
        """Run the policy's ``reset`` in the child, serving its tool calls.

        Args:
            obs: The first observation.
            tools: The parent's :class:`episode.Tools`, which runs the
                policy's tool calls.
        """
        self._send(("reset", obs, tools.info))
        self._serve(tools, "done")

    def act(self, obs, tools) -> np.ndarray:
        """Run the policy's ``act`` in the child, serving its tool calls.

        Args:
            obs: The current observation.
            tools: The parent's :class:`episode.Tools`.

        Returns:
            The action, as the flat float32 array the episode loop makes of it.
        """
        self._send(("act", obs))
        action = self._serve(tools, "action")
        if not (
            isinstance(action, np.ndarray)
            and action.dtype == np.float32
            and action.ndim == 1
        ):
            raise PolicyError("the policy process sent a malformed action")
        return action

    def close(self) -> None:
        """Ask the child to exit, and kill it if it does not."""
        if self._proc.poll() is None:
            try:
                self._outbox.send(("close",))
            except OSError:
                pass
        self._outbox.close()
        self._inbox.close()
        try:
            self._proc.wait(timeout=CLOSE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()

    def _send(self, msg) -> None:
        try:
            self._outbox.send(msg)
        except OSError as exc:
            raise PolicyError(self._died()) from exc

    def _died(self) -> str:
        try:
            code = self._proc.wait(timeout=CLOSE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            code = None
        return f"the policy process exited unexpectedly (exit status {code})"

    def _serve(self, tools, until: str):
        """Answer tool calls until the child sends ``until``; return its value."""
        while True:
            try:
                data = self._inbox.recv_bytes(MAX_MESSAGE)
            except (EOFError, OSError) as exc:
                raise PolicyError(self._died()) from exc
            try:
                msg = decode(data)
            except ValueError as exc:
                raise PolicyError(f"the policy process sent {exc}") from None
            op = msg.get("op") if isinstance(msg, dict) else None
            if op == until:
                return msg.get("value")
            if op == "forbidden_import":
                raise ForbiddenImport(str(msg.get("reason")))
            if op == "error":
                raise PolicyError(
                    f"the policy raised {msg.get('error')}\n"
                    f"--- policy traceback ---\n{msg.get('traceback')}"
                )
            if op == "tool" and tools is not None:
                self._run_tool(tools, msg)
                continue
            raise PolicyError(f"unexpected message from the policy process: {op!r}")

    def _run_tool(self, tools, msg: dict) -> None:
        name, args, kwargs = msg.get("name"), msg.get("args"), msg.get("kwargs")
        try:
            if not isinstance(name, str) or name not in self._tools:
                raise AttributeError(f"tools has no method {name!r}")
            if not isinstance(args, list) or not isinstance(kwargs, dict):
                raise TypeError("malformed tool call")
            reply: tuple = ("result", getattr(tools, name)(*args, **kwargs))
        except Exception as exc:
            # The type name and message only: an exception object may hold
            # references into the environment, which must not reach the child.
            reply = ("raise", type(exc).__name__, str(exc))
        self._send(reply)


# --------------------------------------------------------------------------
# Child side
# --------------------------------------------------------------------------


def _tool_error(name: str, message: str) -> Exception:
    """Rebuild a tool's exception: a built-in type keeps its type, others are RuntimeError."""
    cls = getattr(builtins, name, None)
    if isinstance(cls, type) and issubclass(cls, Exception):
        return cls(message)
    return RuntimeError(f"{name}: {message}")


class _PipeEnv:
    """The child's environment: every call is forwarded to the parent.

    ``episode.Tools`` wraps it exactly as it wraps a real environment, so the
    policy sees the same ``tools`` surface, signatures and defaults as in
    development.
    """

    def __init__(self, inbox: Connection, outbox: Connection, info):
        self._inbox = inbox
        self._outbox = outbox
        self.info = info

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)

        def call(*args, **kwargs):
            self._outbox.send_bytes(
                encode(
                    {"op": "tool", "name": name, "args": list(args), "kwargs": kwargs}
                )
            )
            reply = self._inbox.recv()
            if reply[0] == "raise":
                raise _tool_error(reply[1], reply[2])
            return reply[1]

        return call


def serve_policy_requests(read_fd: int, write_fd: int) -> None:
    """Serve the parent's requests until it closes the pipe.

    Args:
        read_fd: The pipe the parent's requests arrive on.
        write_fd: The pipe the replies go out on.
    """
    episode = _episode_module()
    inbox = Connection(read_fd, writable=False)
    outbox = Connection(write_fd, readable=False)
    # Installed after the child's own imports, before any policy code runs.
    finder = _ForbiddenFinder()
    sys.meta_path.insert(0, finder)
    policy = tools = None
    while True:
        try:
            msg = inbox.recv()
        except EOFError:
            return
        op = msg[0]
        if op == "close":
            return
        try:
            if op == "load":
                policy = None
                policy = episode.load_policy(msg[1])
                reply: dict = {"op": "loaded"}
            elif policy is None:
                raise RuntimeError(f"{op} before a policy was loaded")
            elif op == "reset":
                tools = episode.Tools(_PipeEnv(inbox, outbox, msg[2]))
                policy.reset(msg[1], tools)
                reply = {"op": "done"}
            elif op == "act":
                raw = policy.act(msg[1], tools)
                # The conversion run_episode applies, done here so that only a
                # flat float32 array crosses the pipe.
                value = np.asarray(raw, dtype=np.float32).reshape(-1)
                reply = {"op": "action", "value": value}
            else:
                raise ValueError(f"unknown request {op!r}")
        except BaseException as exc:  # SystemExit from policy code ends the episode too
            reply = {
                "op": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        if finder.hit is not None:
            reply = {"op": "forbidden_import", "reason": finder.hit}
        outbox.send_bytes(encode(reply))


if __name__ == "__main__":
    serve_policy_requests(int(sys.argv[1]), int(sys.argv[2]))
