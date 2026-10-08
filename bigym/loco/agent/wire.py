"""Length-prefixed JSON messages over a unix socket.

No pickle: the agent side of the socket is untrusted, so the only thing the
server ever parses is JSON.

This module is copied into an agent sandbox as ``harness/wire.py``, so it must
keep working when imported as a top-level module.
"""

from __future__ import annotations

import json
import socket
import struct

import numpy as np

_HDR = struct.Struct("!I")
MAX_MSG = 64 * 1024 * 1024


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def send(sock: socket.socket, obj) -> None:
    """Send one JSON message (numpy arrays and scalars are converted).

    Args:
        sock: Connected stream socket.
        obj: Any JSON-serialisable object, numpy values included.
    """
    data = json.dumps(obj, default=_default).encode()
    sock.sendall(_HDR.pack(len(data)) + data)


def _recvall(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf.extend(chunk)
    return bytes(buf)


def recv(sock: socket.socket):
    """Receive one JSON message.

    Args:
        sock: Connected stream socket.

    Returns:
        The decoded message.

    Raises:
        ValueError: The announced length exceeds :data:`MAX_MSG`.
        ConnectionError: The peer closed the socket mid-message.
    """
    (n,) = _HDR.unpack(_recvall(sock, _HDR.size))
    if n > MAX_MSG:
        raise ValueError("message too large")
    return json.loads(_recvall(sock, n).decode())
