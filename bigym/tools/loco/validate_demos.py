#!/usr/bin/env python3
"""Validate native demo directories against the frozen schema.

Usage:
    python bigym/tools/loco/validate_demos.py <demo_dir> [<demo_dir> ...]

Checks every episode .npz against bigym.loco.demos.schema v1 and the
directory metadata.json; exits non-zero when any file violates the schema.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import tyro

from bigym.loco.demos import load_episode, load_metadata, validate_episode
from bigym.loco.demos.schema import validate_metadata


@dataclass
class ValidateConfig:
    """Demo directories to validate."""

    demo_dirs: tyro.conf.Positional[list[Path]]
    """Demo directories (each with metadata.json and episode .npz files)."""
    max_episodes: int = -1
    """Cap per dir (-1 = all)."""


def main() -> int:
    """Validate every given demo directory; return 1 if any check failed."""
    args = tyro.cli(ValidateConfig, description=__doc__)

    failures = 0
    for demo_dir in args.demo_dirs:
        metadata = load_metadata(demo_dir)
        if metadata is None:
            failures += 1
            print(f"[FAIL] {demo_dir}: no metadata.json (required by schema v1)")
        else:
            for issue in validate_metadata(metadata):
                failures += 1
                print(f"[FAIL] {demo_dir}/metadata.json: {issue}")

        episodes = sorted(Path(demo_dir).rglob("*.npz"))
        if args.max_episodes > 0:
            episodes = episodes[: args.max_episodes]
        if not episodes:
            failures += 1
            print(f"[FAIL] {demo_dir}: no .npz episodes found")
            continue
        ok = 0
        for path in episodes:
            issues = validate_episode(load_episode(path))
            if issues:
                failures += 1
                print(f"[FAIL] {path}: {issues}")
            else:
                ok += 1
        print(f"[ok] {demo_dir}: {ok}/{len(episodes)} episodes valid")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
