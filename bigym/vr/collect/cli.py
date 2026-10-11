"""``bigym-collect``: run one VR collection session, then build the training view."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from bigym.vr.collect.config import CollectConfig


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the collector CLI, run one session, then post-process the batch."""
    args = list(sys.argv[1:] if argv is None else argv)
    # MuJoCo picks its GL backend when it is first imported, and the config
    # module imports it, so --mujoco-gl is applied before the config is read.
    os.environ["MUJOCO_GL"] = _flag_value(args, "--mujoco-gl") or "glfw"
    from bigym.vr.collect.config import default_out_dir, parse_cli

    config = parse_cli(args)
    os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
    from bigym.vr.collect.runtime import (
        abort_on_foreign_vr_server,
        configure_openxr_logging,
        configure_openxr_runtime,
    )

    configure_openxr_logging(config.openxr_log_level)
    abort_on_foreign_vr_server()
    configure_openxr_runtime()

    from bigym.vr.collect.session import CollectorSession

    out_dir = (
        default_out_dir(config.task)
        if config.out_dir is None
        else config.out_dir.expanduser().resolve()
    )
    collector = CollectorSession(config, out_dir)
    try:
        collector.run()
    finally:
        if not config.keep_empty_session and not any(out_dir.glob("*.npz")):
            shutil.rmtree(out_dir, ignore_errors=True)
            print(f"[cleanup] no episodes saved, removed session dir: {out_dir}")
        elif (out_dir / "metadata.json").exists():
            post_process(
                config,
                out_dir,
                collection_hold=float(collector.env_config.success_hold_seconds),
                training_hold=float(collector.training_hold_seconds),
            )


def _flag_value(args: list[str], flag: str) -> str | None:
    """The value of ``flag`` (``--flag X`` or ``--flag=X``) in ``args``, if given."""
    for i, arg in enumerate(args):
        if arg == flag and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def post_process(
    config: CollectConfig,
    out_dir: Path,
    *,
    collection_hold: float,
    training_hold: float,
) -> None:
    """Cut the raw batch to the training hold and optionally export it.

    The raw batch is never modified; a failure here leaves it intact and
    prints the command to finish by hand.
    """
    from bigym.loco.demos import success_hold
    from bigym.loco.objref import path_name

    cut_cmd = [
        "python",
        "-m",
        "bigym.loco.demos.success_hold",
        "--demo-dir",
        str(out_dir),
    ]
    training_dir: Path | None = out_dir
    if collection_hold > training_hold + 1e-9:
        print(
            f"[training-view] raw reward is at {collection_hold:g}s; "
            f"train/eval terminate at {training_hold:g}s"
        )
        print(f"[training-view] {shlex.join(cut_cmd)}")
        training_dir = None
        if config.export_lerobot:
            try:
                training_dir = success_hold.latch_batch(out_dir)
                print(f"[training-view] wrote {training_dir}")
            except Exception as exc:
                print(
                    f"[training-view] FAILED ({exc}); raw batch is intact, "
                    "run the command above by hand"
                )
    elif collection_hold < training_hold - 1e-9:
        training_dir = None
        print(
            f"[training-view] raw hold {collection_hold:g}s is shorter than "
            f"train/eval {training_hold:g}s; recollect"
        )

    if not config.export_lerobot:
        return
    if training_dir is None:
        print("[export-lerobot] skipped: no reward-aligned training view")
        return
    # LeRobot cannot share an environment with the vr extra; uv runs the
    # exporter in its own environment from the script's inline metadata.
    from bigym.loco.demos import lerobot_export

    export_root = training_dir.parent / f"{training_dir.name}_lerobot"
    cmd = [
        "uv",
        "run",
        "--python",
        "3.12",
        str(Path(lerobot_export.__file__)),
        "--demo-dir",
        str(training_dir),
        "--repo-id",
        f"local/bigym-{path_name(config.task)}",
        "--root",
        str(export_root),
    ]
    if config.task_text is not None:
        cmd += ["--task-text", config.task_text]
    print(f"[export-lerobot] {shlex.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print(f"[export-lerobot] wrote {export_root}")
    except Exception as exc:  # never lose a session over the export
        print(
            f"[export-lerobot] FAILED ({exc}); npz batch is intact, re-run the "
            "command above by hand"
        )


if __name__ == "__main__":
    main()
