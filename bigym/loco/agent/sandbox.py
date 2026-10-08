"""Build the sandbox an agent works in: one task, one session.

    bigym-agent sandbox --task move_plate --cell RUNS/move_plate

A *cell* is one task x one session. This writes ``<cell>/sandbox/``, the only
directory the agent can see, plus the two files the launcher and the
environment server read next to it (``sandbox_config.json`` with the fully
resolved configuration, and ``allowed_seeds.json`` with the development
seeds).

What the agent is given:

- ``PROMPT.md`` (also copied to ``README.md``): the task in one sentence, the
  seeds, the step budget and the rules.
- ``docs/api.md``: the observation, the action and the tools, in the rendering
  that matches the interface (``strict`` or ``tools``) and the action layout
  (20 floats, or 21 with the torso-pitch command).
- ``harness/``: a copy of the client, the episode runner and the wire protocol.
  It is a copy, not an import: the sandbox has no access to ``bigym`` and the
  simulator is reachable only through a Unix socket in ``sock/``.
- ``policy.py``: the template to fill in, and ``run_episodes.py`` to run it.
- ``python``: the interpreter wrapper the agent is told to use.
- the demonstrations, in the form ``--demo`` asks for.

Demonstrations come from the public dataset (:mod:`bigym.loco.demos.hub`), not
from any local recording. The development seeds are the seeds those
demonstrations were collected on, so the agent develops on exactly the object
placements the learned baselines train on; every other seed is refused by the
server, the hidden evaluation block included.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import stat
import sys
from pathlib import Path

import numpy as np

from bigym.loco.demos import hub
from bigym.loco.demos.dataset import load_episodes

from . import demo_video
from .cli import SandboxConfig, parse_command
from .envtools import task_pitch_enabled
from .settings import claude_settings, claude_settings_soft

TEMPLATES = Path(__file__).resolve().parent / "templates"

# Directories every sandbox has: docs, the harness copy, the runner's outputs,
# the frames it saves for failed episodes, the server's sockets and the
# harness settings.
SANDBOX_DIRS = ("docs", "harness", "runs", "frames", "sock", ".claude")

# Keys of a demonstration a policy could also observe itself. Everything else
# a recording holds (full simulator state, object and fixture poses,
# controller internals) is dropped: it would be ground truth the policy has no
# way to obtain at evaluation time.
DEMO_KEEP = (
    "rgb_obs",
    "low_dim_obs",
    "action",
    "raw_outer_action",
    "reward",
    "discount",
    "demo",
    "is_expert",
    "event_progress",
    "seed",
)

RUN_EPISODES = '''#!/usr/bin/env python
"""Run policy.py on the development seeds. See docs/api.md."""
import os
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
os.environ.setdefault("AGENT_SANDBOX", str(here))
sys.path.insert(0, str(here / "harness"))

import runner  # noqa: E402

runner.main()
'''

# One sentence per benchmark task, as you would say it to a person: what to do
# and to which object, and nothing about positions, sizes or the tolerance the
# success test uses. "Gently" is the word for the placement tasks, where
# dropping the object from a height fails the task.
TASK_SENTENCES = {
    "reach_target_single": "Touch the red sphere with your left hand.",
    "reach_target_dual": (
        "Touch the red sphere with your left hand and the green sphere with "
        "your right hand at the same time."
    ),
    "reach_target_multi_modal": "Touch the red sphere with either hand.",
    "stack_blocks": "Move the blocks to the other table and stack them on the green pad.",
    "move_plate": (
        "Take the plate out of the left dish rack and gently stand it in the "
        "right dish rack."
    ),
    "move_two_plates": (
        "Take both plates out of the left dish rack and gently stand them in "
        "the right dish rack."
    ),
    "flip_cup": "Turn the upside-down cup the right way up and gently set it back down.",
    "flip_cutlery": "Take the spoon from the mug, flip it, and place it back.",
    "dishwasher_open": "Open the dishwasher door and pull both racks out.",
    "dishwasher_close": "Push the dishwasher racks in and close the door.",
    "dishwasher_open_trays": (
        "The dishwasher door is already open; pull both racks out."
    ),
    "dishwasher_close_trays": ("The dishwasher door is open; push both racks back in."),
    "dishwasher_load_cups": (
        "Gently place the two mugs from the counter into the upper rack of the "
        "dishwasher."
    ),
    "dishwasher_unload_cups": (
        "Take the two mugs out of the dishwasher's upper rack and gently set "
        "them on the counter."
    ),
    "dishwasher_unload_cups_long": (
        "Take the mug out of the dishwasher and gently put it away in the wall cabinet."
    ),
    "dishwasher_load_cutlery": (
        "Move the knife and fork from the mug on the counter into the "
        "dishwasher's cutlery basket."
    ),
    "dishwasher_unload_cutlery": (
        "Take the knife and fork out of the dishwasher's cutlery basket and "
        "gently put them in the mug on the counter."
    ),
    "dishwasher_unload_cutlery_long": (
        "Take the cutlery out of the dishwasher and gently put it away in the "
        "cutlery drawer."
    ),
    "dishwasher_load_plates": (
        "Take the plates from the dish rack and gently slot them into the "
        "dishwasher's lower rack."
    ),
    "dishwasher_unload_plates": (
        "Take the plates out of the dishwasher's lower rack and gently stand "
        "them in the dish rack."
    ),
    "dishwasher_unload_plates_long": (
        "Take the plate out of the dishwasher and gently put it away in the "
        "wall cabinet."
    ),
    "drawer_top_open": "Open the top drawer of the kitchen cabinet.",
    "drawer_top_close": "Close the open top drawer of the kitchen cabinet.",
    "drawers_open_all": "Open all the drawers of the kitchen cabinet.",
    "drawers_close_all": "Close all the open drawers of the kitchen cabinet.",
    "wall_cupboard_open": "Open both doors of the wall cabinet.",
    "wall_cupboard_close": "Close both open doors of the wall cabinet.",
    "cupboards_open_all": "Open every door and drawer of the kitchen units.",
    "cupboards_close_all": "Close every open door and drawer of the kitchen units.",
    "take_cups": (
        "Take the cups out of the wall cabinet and gently set them on the counter."
    ),
    "put_cups": (
        "Take the cups from the counter and gently put them away in the wall cabinet."
    ),
    "pick_box": (
        "Pick the cardboard box up from the side table and gently set it down "
        "on the kitchen counter."
    ),
    "store_box": "Pick the cardboard box up and gently put it away in the cupboard.",
    "saucepan_to_hob": (
        "Take the saucepan out of the cabinet and gently set it on the hob."
    ),
    "store_kitchenware": "Gently put all the kitchenware away in the cupboard.",
    "sandwich_toast": (
        "Use the spatula to move the sandwich from the chopping board into the "
        "frying pan."
    ),
    "sandwich_flip": "Use the spatula to flip the sandwich over in the frying pan.",
    "sandwich_remove": (
        "Use the spatula to move the sandwich from the frying pan onto the "
        "chopping board."
    ),
    "store_groceries_lower": (
        "Gently put the groceries from the counter away on the lower cabinet shelf."
    ),
    "store_groceries_upper": (
        "Gently put the groceries from the counter away on the upper cabinet shelf."
    ),
}


def task_sentence(task: str) -> str:
    """Return the one-sentence description of a task.

    Args:
        task: Task name.

    Returns:
        The sentence the prompt states the task with.

    Raises:
        KeyError: No sentence is registered for the task.
    """
    if task not in TASK_SENTENCES:
        raise KeyError(
            f"no task sentence for {task!r}; add one to "
            "bigym.loco.agent.sandbox.TASK_SENTENCES"
        )
    return TASK_SENTENCES[task]


def resolve_layout(text: str, pitch: bool) -> str:
    """Keep the action-layout section that matches the task.

    ``api_strict.md`` documents the 20-dim and the 21-dim proprioception
    layouts in blocks fenced by ``<!-- layout: 20 -->`` / ``<!-- layout: 21 -->``;
    a sandbox gets exactly one of them.

    Args:
        text: The document.
        pitch: Whether the task is on the 21-dim (torso-pitch) layout.

    Returns:
        The document with one layout block kept and the fences removed.

    Raises:
        ValueError: A fence survived (the document changed shape).
    """
    keep, drop = ("21", "20") if pitch else ("20", "21")
    start, end = f"<!-- layout: {drop} -->\n", f"<!-- /layout: {drop} -->\n"
    first = text.find(start)
    if first >= 0:
        last = text.find(end, first)
        text = text[:first] + text[last + len(end) :]
    text = text.replace(f"<!-- layout: {keep} -->\n", "")
    text = text.replace(f"<!-- /layout: {keep} -->\n", "")
    if "<!-- layout:" in text or "<!-- /layout:" in text:
        raise ValueError("api.md still carries a layout fence")
    return text


def pitch_layout(api: str) -> str:
    """Rewrite the 20-float action table into the 21-float torso-pitch layout.

    Args:
        api: The API document.

    Returns:
        The rewritten document.

    Raises:
        ValueError: The action table did not match.
    """
    api = api.replace(
        "## Action: 20 floats, physical units", "## Action: 21 floats, physical units"
    )
    api = api.replace(
        "| 3 | `wz` yaw rate command, rad/s (+ = counter-clockwise) | -1..1 |\n"
        "| 4..10 | left arm joint targets, rad: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw | joint limits |\n"
        "| 11..17 | right arm joint targets, same order | joint limits |\n"
        "| 18 | left gripper command | 0..1 (0 = fully open, 1 = fully closed) |\n"
        "| 19 | right gripper command | 0..1 |",
        "| 3 | `wz` yaw rate command, rad/s (+ = counter-clockwise) | -1..1 |\n"
        "| 4 | `pitch` torso pitch command, rad, absolute (+ = lean forward, 0 = upright); the controller tilts the torso, useful for reaching low or far | -0.2..0.8 |\n"
        "| 5..11 | left arm joint targets, rad: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw | joint limits |\n"
        "| 12..18 | right arm joint targets, same order | joint limits |\n"
        "| 19 | left gripper command | 0..1 (0 = fully open, 1 = fully closed) |\n"
        "| 20 | right gripper command | 0..1 |",
    )
    api = api.replace(
        "`tools.hold_action()` -> 20 floats: zero base velocity, current height, current arm and gripper targets.",
        "`tools.hold_action()` -> 21 floats: zero base velocity, current height and torso pitch, current arm and gripper targets.",
    )
    # Only the tools interface documents writing an IK solution into the action.
    api = api.replace(
        "Write the result into `raw[4:18]`.", "Write the result into `raw[5:19]`."
    )
    api = api.replace("| `low_dim_obs` | (50,) |", "| `low_dim_obs` | (56,) |")
    if "21 floats" not in api or "| 4 | `pitch`" not in api or "(56,)" not in api:
        raise ValueError("api.md action table did not match the 20-float layout")
    return api


def image_cap_doc(api: str, cap: tuple[int, int]) -> str:
    """State the policy-facing resolution ceiling in the API document.

    The agent is told the same number the server, the evaluation and the
    demonstration videos enforce. The shipped document states the 84x84 case,
    which is the benchmark setting.

    Args:
        api: The API document.
        cap: ``(width, height)`` ceiling.

    Returns:
        The document with the ceiling stated.

    Raises:
        ValueError: The camera-resolution paragraph did not match.
    """
    anchor = (
        "(the 84x84 views a learned policy gets; this is also the largest size you\n"
        "  may request, larger requests are refused)"
    )
    if anchor not in api:
        raise ValueError("api.md camera-resolution text did not match")
    if tuple(cap) == (84, 84):
        return api
    width, height = int(cap[0]), int(cap[1])
    replacement = (
        "(the 84x84 views a learned policy gets; you may request up to\n"
        f"  {width}x{height} from the same cameras, larger requests are refused)"
    )
    api = api.replace(anchor, replacement)
    return api.replace(
        "84x84, the observation itself",
        f"up to {width}x{height}, the observation itself",
    )


def seeds_doc(api: str, seeds: list[int]) -> str:
    """Replace the seed paragraph with the run's development seeds.

    Args:
        api: The API document.
        seeds: The development seeds.

    Returns:
        The document naming the seeds and ``seeds.json``.

    Raises:
        ValueError: The seed paragraph did not match.
    """
    anchor = (
        "Seeds: any integer in `[0, 200)`. Each seed fixes the initial placement of\n"
        "the objects."
    )
    if anchor not in api:
        raise ValueError("api.md seeds paragraph did not match")
    listed = ", ".join(str(seed) for seed in seeds)
    return api.replace(
        anchor,
        f"Seeds: the {len(seeds)} seeds the demonstration set was collected on "
        f"(also in `seeds.json`):\n{listed}.\nAny other seed is refused. Each seed "
        "fixes the initial placement of the objects.",
    )


def task_keys_doc(api: str, tier: str) -> str:
    """Say what the observation holds beyond the robot's own body.

    Args:
        api: The API document.
        tier: ``images`` (the benchmark setting) or ``privileged``.

    Returns:
        The document with the task-key sentence resolved.

    Raises:
        ValueError: The sentence did not match.
    """
    anchor = "Task-specific keys are listed in `docs/task.md`."
    if anchor not in api:
        raise ValueError("api.md task-key sentence did not match")
    if tier == "privileged":
        return api.replace(
            anchor,
            "This run uses the privileged tier: the observation also carries the "
            "task's object positions, under task-specific keys.",
        )
    return api.replace(
        anchor,
        "There are no task-specific keys: the observation contains no object positions.",
    )


def strip_ik_harness(source: str, name: str) -> str:
    """Remove the IK method from a harness file copied into the sandbox.

    The server refuses the operation under the ``strict`` interface either
    way; dropping it here keeps the code the agent can read from advertising
    a tool it cannot use.

    Args:
        source: The file's text.
        name: The file name.

    Returns:
        The text without the IK method.

    Raises:
        ValueError: The method was not found between the expected anchors.
    """
    marks = {
        "client.py": ("    def ik(self, left_pos=None", "    def render(self, camera:"),
        "episode.py": ("    def ik(self, **kw):", "    def render(self, camera="),
    }
    if name not in marks:
        return source
    start, end = marks[name]
    first, last = source.find(start), source.find(end)
    if not 0 <= first < last:
        raise ValueError(f"{name}: ik method not found between the expected anchors")
    return source[:first] + source[last:]


def strip_calibration_harness(source: str, name: str) -> str:
    """Remove camera calibration from a harness file copied into the sandbox.

    Args:
        source: The file's text.
        name: The file name.

    Returns:
        The text without ``camera_info`` and ``pixel_to_ray``.

    Raises:
        ValueError: The methods were not found, or one survived.
    """
    marks = {
        "client.py": [
            ("    def pixel_to_ray(self, camera: str", "    def budget(self) -> dict:")
        ],
        "episode.py": [("    def camera_info(self):", "\n\ndef load_policy(")],
    }
    if name not in marks:
        return source
    for start, end in marks[name]:
        first, last = source.find(start), source.find(end)
        if not 0 <= first < last:
            raise ValueError(f"{name}: {start.strip()} not found between the anchors")
        source = source[:first].rstrip("\n") + "\n" + source[last:]
    if "def camera_info" in source or "def pixel_to_ray" in source:
        raise ValueError(f"{name}: calibration methods left in the sandbox copy")
    return source


def policy_template(pitch: bool, interface: str) -> str:
    """Render the starting ``policy.py`` for one interface and action layout.

    Args:
        pitch: Whether the task is on the 21-float action layout.
        interface: ``strict`` or ``tools``.

    Returns:
        The file's text.

    Raises:
        ValueError: A withheld tool survived the rewrite.
    """
    text = (TEMPLATES / "policy_template.py").read_text()
    if pitch:
        text = text.replace("must return 20 floats", "must return 21 floats").replace(
            "raw = [vx, vy, height, wz, left arm (7), right arm (7), left gripper, right gripper]",
            "raw = [vx, vy, height, wz, torso_pitch, left arm (7), right arm (7), "
            "left gripper, right gripper]",
        )
    if interface == "strict":
        text = "\n".join(line for line in text.split("\n") if "tools.ik(" not in line)
        text = text.replace(
            "  tools.image(camera, w, h) -> RGB array; tools.camera_info(); "
            "tools.pixel_to_ray(camera, u, v, w, h)",
            "  tools.image(camera, w, h) -> RGB array from head / left_wrist / "
            "right_wrist (84x84)",
        )
        for withheld in ("tools.ik", "camera_info", "pixel_to_ray"):
            if withheld in text:
                raise ValueError(f"policy.py template still mentions {withheld}")
    return text


def harness_sources(interface: str, with_demo_reader: bool) -> dict[str, str]:
    """Render the harness files copied into ``<sandbox>/harness/``.

    The sandbox cannot import ``bigym``: it gets a copy of the client, the
    episode contract, the runner and the wire protocol, which are written to
    keep working as top-level modules.

    Args:
        interface: ``strict`` or ``tools``.
        with_demo_reader: Also copy the demonstration-video reader.

    Returns:
        File name to text.
    """
    package = Path(__file__).resolve().parent
    names = ["client.py", "episode.py", "runner.py", "wire.py"]
    if interface == "tools":
        names.append("geometry.py")
    out = {}
    for name in names:
        text = (package / name).read_text()
        if interface == "strict":
            text = strip_ik_harness(text, name)
            text = strip_calibration_harness(text, name)
        out[name] = text
    if with_demo_reader:
        out["demo.py"] = (TEMPLATES / "demo.py").read_text()
    return out


def video_demo_sentence(record: dict) -> str:
    """The prompt's "What you have" line for a rendered demonstration.

    Args:
        record: The render record from :func:`bigym.loco.agent.demo_video.render_demo`.

    Returns:
        The sentence, already indented for the prompt's bullet list.
    """
    files = ", ".join(f"`{name}`" for name in record["files"])
    cameras = ", ".join(view.replace("_", " ") for view in record["views"])
    count = len(record["episodes"])
    what = (
        "one human demonstration of the task"
        if count == 1
        else f"{count} human demonstrations of the task, one after another"
    )
    return (
        f"{files}: {what}, recorded at the same time from the\n"
        f"  robot's {cameras} cameras ({record['size']}, {record['fps']} fps, frame k "
        "of each file is the same instant).\n"
        "  `harness/demo.py` reads them as RGB arrays in the same layout as the live "
        "cameras and maps\n  frames to control steps (see docs/api.md)."
    )


def video_demo_doc(record: dict) -> str:
    """The API document's demonstration section for rendered videos.

    Args:
        record: The render record.

    Returns:
        The section text.
    """
    width, height = record["size"].split("x")
    count = len(record["episodes"])
    what = (
        "One human teleoperation episode"
        if count == 1
        else f"{count} human teleoperation episodes, played one after another,"
    )
    return (
        (TEMPLATES / "api_demo.md")
        .read_text()
        .format(
            n_episodes_text=what,
            files=", ".join(f"`{name}`" for name in record["files"]),
            width=width,
            height=height,
            fps=record["fps"],
            every=record["every"],
        )
    )


def files_demo_doc(episodes: int, action_dim: int, state_dim: int, camera_shape) -> str:
    """The API document's demonstration section for the npz demonstration set.

    Args:
        episodes: How many episodes were written.
        action_dim: Width of the physical action.
        state_dim: Width of the proprioceptive vector.
        camera_shape: ``(height, width)`` of the recorded images.

    Returns:
        The section text.
    """
    return (
        (TEMPLATES / "demos_README.md")
        .read_text()
        .format(
            n_episodes=episodes,
            action_dim=action_dim,
            state_dim=state_dim,
            cam_height=int(camera_shape[0]),
            cam_width=int(camera_shape[1]),
            pitch_note=" (index 4 = torso pitch)" if action_dim == 21 else "",
        )
    )


def prompt_document(
    task: str, seeds: list[int], budget_steps: int, demo: str, demo_line: str
) -> str:
    """Render ``PROMPT.md``.

    Args:
        task: Task name.
        seeds: The development seeds.
        budget_steps: The environment-step budget.
        demo: ``video``, ``files`` or ``none``.
        demo_line: The "What you have" line describing the demonstrations.

    Returns:
        The prompt.

    Raises:
        ValueError: A placeholder was left unfilled.
    """
    text = (
        (TEMPLATES / "PROMPT.md")
        .read_text()
        .format(
            video_line=demo_line,
            n_seeds=len(seeds),
            budget=f"{budget_steps:,}",
            task_sentence=task_sentence(task),
        )
    )
    if demo == "files":
        text = text.replace(
            "Take whatever you can from the demonstration video;",
            "Take whatever you can from the demonstrations;",
        )
    elif demo == "none":
        text = text.replace(
            "  Take whatever you can from the demonstration video; tuning constants by\n"
            "  running episodes is fine.",
            "  Tuning constants by running episodes is fine.",
        )
    if "{" in text or "}" in text:
        raise ValueError("PROMPT.md still carries an unfilled placeholder")
    return text


def write_demo_files(demos, out_dir: Path, max_episodes: int = -1) -> dict:
    """Write the demonstration set as stripped npz files.

    Args:
        demos: A :class:`bigym.loco.agent.demo_video.TaskDemos`.
        out_dir: Destination directory (``<sandbox>/demos``).
        max_episodes: Episode cap, or -1 for all of them.

    Returns:
        A record of what was written: the files, the episode count and the
        array widths the document quotes.

    Raises:
        ValueError: A written episode kept a stripped key.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    action_dim = state_dim = 0
    for name, episode in load_episodes(demos.dir, max_episodes):
        arrays = {key: value for key, value in episode.items() if key in DEMO_KEEP}
        leaked = [key for key in arrays if key.startswith("full_")]
        if leaked:
            raise ValueError(f"{name}: stripped demo still carries {leaked}")
        action_dim = int(np.asarray(arrays["raw_outer_action"]).shape[-1])
        state_dim = int(np.asarray(arrays["low_dim_obs"]).shape[-1])
        path = out_dir / Path(name).with_suffix(".npz").name
        np.savez_compressed(path, **arrays)  # ty: ignore[invalid-argument-type]
        written.append(path.name)
    metadata = demos.metadata
    # Only what the reader needs to make sense of the arrays: the action
    # normalisation and the camera order. The recorded controller settings and
    # the scene description stay out of the sandbox.
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "control_step_seconds": metadata.get("control_step_seconds"),
                "action_stats": metadata.get("action_stats"),
                "task": {
                    "camera_keys": metadata["task"].get("camera_keys"),
                    "camera_shape": metadata["task"].get("camera_shape"),
                },
            },
            indent=1,
        )
    )
    return {
        "files": written,
        "episodes": len(written),
        "action_dim": action_dim,
        "state_dim": state_dim,
        "camera_shape": tuple(metadata["task"].get("camera_shape", (84, 84))),
    }


def python_wrapper(isolation: str) -> str:
    """The ``python`` wrapper script the agent is told to run its files with.

    A symlink would not do: a virtual environment is found relative to
    ``argv[0]``, so a link into the interpreter loses the environment.

    Args:
        isolation: ``soft`` (this interpreter) or ``container`` (the image's).

    Returns:
        The shell script.
    """
    if isolation == "container":
        return '#!/bin/sh\nexec /usr/local/bin/python3 "$@"\n'
    return f'#!/bin/sh\nexport MUJOCO_GL=egl\nexec {sys.executable} "$@"\n'


def _make_executable(path: Path) -> None:
    """Give a file the execute bit for everyone who can read it."""
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def build(args: SandboxConfig) -> dict:
    """Write the sandbox and the cell files for one session.

    Args:
        args: The sandbox settings.

    Returns:
        The resolved configuration, as written to ``sandbox_config.json``.

    Raises:
        SystemExit: The sandbox exists and ``--force`` was not given.
    """
    task_sentence(args.task)  # fail before any download when the task is unknown
    env_tools = args.env_tools
    cap = env_tools.image_size
    isolation = args.isolation
    cell = Path(args.cell).resolve()
    sandbox = cell / "sandbox"
    if sandbox.exists() and any(sandbox.iterdir()) and not args.force:
        raise SystemExit(f"{sandbox} exists and is not empty (use --force)")

    demos = demo_video.TaskDemos(args.task)
    seeds = demos.seeds
    pitch = env_tools.pitch and task_pitch_enabled(args.task)

    for name in SANDBOX_DIRS:
        (sandbox / name).mkdir(parents=True, exist_ok=True)

    demo_record: dict = {"demo": args.demo}
    demo_section = ""
    demo_line = "No demonstrations in this run."
    if args.demo == "video":
        record = demo_video.render_demo(
            args.task,
            sandbox,
            size=cap,
            views=demo_video.DEFAULT_VIEWS,
            which=args.demo_episode,
            episodes=args.demo_episodes,
            demos=demos,
        )
        (sandbox / "demo.json").write_text(
            json.dumps(
                {
                    "every": record["every"],
                    "fps": record["fps"],
                    "frames": record["frames"],
                    "cameras": record["views"],
                },
                indent=1,
            )
        )
        demo_section = video_demo_doc(record)
        demo_line = video_demo_sentence(record)
        demo_record.update(record)
    elif args.demo == "files":
        written = write_demo_files(
            demos, sandbox / "demos", max_episodes=args.demo_max_files
        )
        demo_section = files_demo_doc(
            written["episodes"],
            written["action_dim"],
            written["state_dim"],
            written["camera_shape"],
        )
        demo_line = (
            f"`demos/` - {written['episodes']} human demonstrations of the task as "
            "npz files,\n  for reading only (see `docs/api_demo.md`)."
        )
        demo_record.update(written)

    api = (TEMPLATES / f"api_{env_tools.interface}.md").read_text()
    api = resolve_layout(api, pitch)
    if pitch:
        api = pitch_layout(api)
    api = image_cap_doc(api, cap)
    api = task_keys_doc(api, env_tools.tier)
    api = seeds_doc(api, seeds)
    if demo_section:
        (sandbox / "docs" / "api_demo.md").write_text(demo_section.lstrip("\n"))
        api = api.rstrip("\n") + "\n" + demo_section
    (sandbox / "docs" / "api.md").write_text(api)

    for name, text in harness_sources(
        env_tools.interface, with_demo_reader=args.demo == "video"
    ).items():
        (sandbox / "harness" / name).write_text(text)

    template = policy_template(pitch, env_tools.interface)
    (sandbox / "policy.py").write_text(template)
    # The exact bytes handed to the agent, archived outside the sandbox: the
    # submission is compared against this file to tell whether the agent wrote
    # anything at all, and reconstructing the template from the flags is what
    # makes that check wrong.
    cell.mkdir(parents=True, exist_ok=True)
    (cell / "initial_policy.py").write_text(template)

    prompt = prompt_document(args.task, seeds, args.budget_steps, args.demo, demo_line)
    (sandbox / "PROMPT.md").write_text(prompt)
    (sandbox / "README.md").write_text(prompt)
    (sandbox / "seeds.json").write_text(json.dumps(seeds))
    (sandbox / "run_episodes.py").write_text(RUN_EPISODES)
    _make_executable(sandbox / "run_episodes.py")
    wrapper = sandbox / "python"
    if wrapper.is_symlink() or wrapper.exists():
        wrapper.unlink()
    wrapper.write_text(python_wrapper(isolation))
    _make_executable(wrapper)

    client_root = args.client_root if isolation == "container" else str(sandbox)
    settings = (
        claude_settings(Path(client_root), effort=args.effort)
        if isolation == "container"
        else claude_settings_soft(sandbox, args.effort)
    )
    (sandbox / ".claude" / "settings.json").write_text(json.dumps(settings, indent=1))
    (sandbox / "claude_settings.json").write_text(json.dumps(settings, indent=1))

    (cell / "allowed_seeds.json").write_text(json.dumps(seeds))
    config = {
        "task": args.task,
        "task_sentence": task_sentence(args.task),
        "cell": str(cell),
        "sandbox": str(sandbox),
        "budget_steps": args.budget_steps,
        "workers": args.workers,
        "env_tools": dataclasses.asdict(env_tools),
        "harness": args.harness,
        "isolation": isolation,
        "client_root": client_root if isolation == "container" else None,
        "effort": args.effort,
        "action_dim": 21 if pitch else 20,
        "low_dim_obs_dim": 56 if pitch else 50,
        "pitch": pitch,
        "seed_rule": "the seeds the task's demonstrations were collected on",
        "allowed_seeds": seeds,
        "train_seeds": len(seeds),
        "dataset_repo": hub.dataset_repo(),
        "dataset_revision": hub.dataset_revision(),
        "dataset_dir": str(demos.dir),
        "demo": demo_record,
        "prompt_version": "v1 one-sentence task, rules inline",
    }
    (cell / "sandbox_config.json").write_text(json.dumps(config, indent=1))

    # Claude Code keeps per-project auto-memory keyed by the working directory.
    # A sandbox path can be reused across sessions, and carrying one session's
    # notes into the next would break the benchmark's independence.
    slug = "".join(ch if ch.isalnum() else "-" for ch in str(sandbox))
    memory = Path.home() / ".claude" / "projects" / slug / "memory"
    if memory.exists():
        shutil.rmtree(memory, ignore_errors=True)
    return config


def main(argv: list[str] | SandboxConfig | None = None) -> int:
    """Build a sandbox from the command line.

    Args:
        argv: The parsed settings, or command line arguments (None for
            ``sys.argv[1:]``).

    Returns:
        The process exit status.
    """
    args = parse_command(SandboxConfig, argv)
    try:
        config = build(args)
    except SystemExit as error:
        print(f"bigym-agent sandbox: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"bigym-agent sandbox: {error}", file=sys.stderr)
        return 1
    print(f"sandbox: {config['sandbox']}")
    print(f"cell:    {config['cell']}")
    print(f"seeds:   {len(config['allowed_seeds'])} development seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
