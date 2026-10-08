"""Shared command-line helpers.

Every BiGym command parses a dataclass of its settings with :func:`tyro.cli`;
the field docstrings are the help text.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NoReturn


def usage_error(message: str, prog: str | None = None) -> NoReturn:
    """Reject flags that parse but do not go together; exits with status 2."""
    name = prog or Path(sys.argv[0]).name
    print(f"{name}: error: {message}", file=sys.stderr)
    raise SystemExit(2)
