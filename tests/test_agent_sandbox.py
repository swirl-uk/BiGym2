"""The coding-agent sandbox builder, on a synthetic demonstration dataset.

``bigym-agent sandbox`` writes everything one benchmark session needs: the
prompt, the API document in the rendering that matches the interface and the
action layout, a copy of the harness, the policy template, the development
seeds and the demonstrations. The demonstrations come from the public Hugging
Face dataset, so these tests point :func:`bigym.loco.demos.hub.task_dir` at a
tiny LeRobot v3 export built in ``tmp_path`` (``tests/fixtures/synthetic_dataset.py``,
with the ``full_qpos`` column a sandbox needs) and never touch the network.

The one test that renders video is marked ``slow``: it builds a real
environment and re-renders ten frames of a synthetic episode through the
public code path (run it with ``MUJOCO_GL=egl ... --run-slow``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bigym.loco.agent import (
    cli,
    demo_video,
    evaluate,
    replay,
    sandbox,
    snapshots,
)
from bigym.loco.agent.cli import EnvToolsConfig
from bigym.loco.demos import hub
from tests.fixtures.synthetic_dataset import CAMERAS, make_episode, write_export

NQ = 22

# A 20-float task and a 21-float (torso-pitch) one, so both action layouts and
# both proprioception widths are covered.
FLAT_TASK = "reach_target_single"
PITCH_TASK = "stack_blocks"
SEEDS = (41, 7, 19)
LENGTHS = (6, 4, 9)


@pytest.fixture
def datasets(tmp_path, monkeypatch):
    """Point ``hub.task_dir`` at a synthetic export for each benchmark layout."""
    rng = np.random.default_rng(0)
    built = {}
    for task, action_dim, state_dim in (
        (FLAT_TASK, 20, 50),
        (PITCH_TASK, 21, 56),
    ):
        episodes = []
        for seed, length in zip(SEEDS, LENGTHS, strict=True):
            episode = make_episode(rng, length, action_dim, state_dim, nq=NQ)
            episode["seed"] = np.int64(seed)
            episodes.append(episode)
        built[task] = write_export(tmp_path / "dataset" / task, episodes, task)

    def fake_task_dir(task, repo=None, revision=None):
        if task not in built:
            raise hub.DemosUnavailableError(
                f"Demonstrations for task {task!r} have not been published yet."
            )
        return built[task]

    monkeypatch.setattr(hub, "task_dir", fake_task_dir)
    return built


def build(cell: Path, task: str = FLAT_TASK, *extra: str) -> int:
    """Run the sandbox builder and return its exit status."""
    return sandbox.main(["--task", task, "--cell", str(cell), *extra])


def test_every_benchmark_task_has_a_sentence():
    from bigym.loco.tasks import TASK_MAP

    missing = sorted(set(TASK_MAP) - set(sandbox.TASK_SENTENCES))
    assert not missing, f"no task sentence for {missing}"
    extra = sorted(set(sandbox.TASK_SENTENCES) - set(TASK_MAP))
    assert not extra, f"sentence for unknown task {extra}"
    for task, sentence in sandbox.TASK_SENTENCES.items():
        assert sentence.endswith("."), task
        assert sentence[0].isupper(), task


def test_sandbox_tree_is_complete_for_strict_20dim(tmp_path, datasets):
    cell = tmp_path / "cell"
    assert build(cell, FLAT_TASK, "--demo", "none") == 0
    box = cell / "sandbox"
    for name in (
        "PROMPT.md",
        "README.md",
        "policy.py",
        "run_episodes.py",
        "python",
        "seeds.json",
        "claude_settings.json",
        "docs/api.md",
        ".claude/settings.json",
        "harness/client.py",
        "harness/episode.py",
        "harness/runner.py",
        "harness/wire.py",
    ):
        assert (box / name).is_file(), name
    for name in ("runs", "frames", "sock", "docs", "harness"):
        assert (box / name).is_dir(), name
    assert (cell / "sandbox_config.json").is_file()
    assert (cell / "allowed_seeds.json").is_file()
    assert (cell / "initial_policy.py").is_file()
    # strict withholds the geometry stack, in the document and in the code.
    assert not (box / "harness" / "geometry.py").exists()
    for name in ("client.py", "episode.py"):
        text = (box / "harness" / name).read_text()
        assert "def ik(" not in text
        assert "def camera_info" not in text
        assert "def pixel_to_ray" not in text
    assert "tools.ik" not in (box / "policy.py").read_text()

    api = (box / "docs" / "api.md").read_text()
    assert "## Action: 20 floats, physical units" in api
    assert "| `low_dim_obs` | (50,)" in api
    assert "| 44..45 | left, right gripper state" in api  # the 20-dim layout table
    assert "| 50..51 |" not in api  # ... and not the 21-dim one
    assert "<!-- layout:" not in api
    assert "docs/task.md" not in api
    assert "tools.ik" not in api and "camera_info" not in api

    # Both wrappers are runnable, and the interpreter wrapper is a wrapper (a
    # symlink would lose the virtual environment).
    import os

    assert os.access(box / "run_episodes.py", os.X_OK)
    assert os.access(box / "python", os.X_OK)
    assert not (box / "python").is_symlink()
    assert (box / "python").read_text().startswith("#!/bin/sh")


def test_sandbox_tree_is_complete_for_tools_21dim(tmp_path, datasets):
    cell = tmp_path / "cell"
    assert build(cell, PITCH_TASK, "--demo", "none", "--interface", "tools") == 0
    box = cell / "sandbox"
    assert (box / "harness" / "geometry.py").is_file()
    assert "def ik(" in (box / "harness" / "client.py").read_text()
    assert "def camera_info" in (box / "harness" / "client.py").read_text()

    api = (box / "docs" / "api.md").read_text()
    assert "## Action: 21 floats, physical units" in api
    assert "| 4 | `pitch` torso pitch command" in api
    assert "| `low_dim_obs` | (56,)" in api
    assert "Write the result into `raw[5:19]`." in api
    assert "20 floats" not in api

    policy = (box / "policy.py").read_text()
    assert "must return 21 floats" in policy
    assert "torso_pitch" in policy
    assert "tools.ik(" in policy

    config = json.loads((cell / "sandbox_config.json").read_text())
    assert config["action_dim"] == 21
    assert config["low_dim_obs_dim"] == 56
    assert config["pitch"] is True
    assert config["env_tools"]["interface"] == "tools"


def test_layout_blocks_are_mutually_exclusive(tmp_path, datasets):
    cell = tmp_path / "cell"
    assert build(cell, PITCH_TASK, "--demo", "none") == 0
    api = (cell / "sandbox" / "docs" / "api.md").read_text()
    assert "56 floats" in api
    assert "50 floats" not in api
    assert "<!-- layout:" not in api


def test_prompt_placeholders_are_all_filled(tmp_path, datasets):
    cell = tmp_path / "cell"
    assert build(cell, FLAT_TASK, "--demo", "none", "--budget-steps", "50000") == 0
    prompt = (cell / "sandbox" / "PROMPT.md").read_text()
    assert "{" not in prompt and "}" not in prompt
    assert sandbox.TASK_SENTENCES[FLAT_TASK] in prompt
    assert f"the {len(set(SEEDS))} seeds listed in `seeds.json`" in prompt
    assert "50,000 environment steps" in prompt
    assert prompt == (cell / "sandbox" / "README.md").read_text()


def test_seeds_are_the_dataset_seeds(tmp_path, datasets):
    cell = tmp_path / "cell"
    assert build(cell, FLAT_TASK, "--demo", "none") == 0
    expected = sorted(set(SEEDS))
    assert json.loads((cell / "sandbox" / "seeds.json").read_text()) == expected
    assert json.loads((cell / "allowed_seeds.json").read_text()) == expected
    config = json.loads((cell / "sandbox_config.json").read_text())
    assert config["allowed_seeds"] == expected
    assert config["train_seeds"] == len(expected)
    assert config["dataset_repo"] == hub.dataset_repo()


def test_initial_policy_is_the_untouched_template(tmp_path, datasets):
    cell = tmp_path / "cell"
    assert build(cell, FLAT_TASK, "--demo", "none") == 0
    assert (cell / "initial_policy.py").read_text() == (
        cell / "sandbox" / "policy.py"
    ).read_text()


def test_demo_none_writes_no_demonstration(tmp_path, datasets):
    cell = tmp_path / "cell"
    assert build(cell, FLAT_TASK, "--demo", "none") == 0
    box = cell / "sandbox"
    assert not list(box.glob("demo*.mp4"))
    assert not (box / "demos").exists()
    assert not (box / "demo.json").exists()
    assert not (box / "docs" / "api_demo.md").exists()
    assert not (box / "harness" / "demo.py").exists()
    assert "No demonstrations in this run." in (box / "PROMPT.md").read_text()
    assert "demonstration video" not in (box / "PROMPT.md").read_text()


def test_demo_files_strips_the_simulator_state(tmp_path, datasets):
    cell = tmp_path / "cell"
    assert build(cell, FLAT_TASK, "--demo", "files") == 0
    box = cell / "sandbox"
    files = sorted((box / "demos").glob("*.npz"))
    assert len(files) == len(LENGTHS)
    for path in files:
        with np.load(path) as episode:
            keys = set(episode.files)
        assert "full_qpos" not in keys and "full_qvel" not in keys
        assert keys <= set(sandbox.DEMO_KEEP)
        assert {"rgb_obs", "low_dim_obs", "raw_outer_action", "seed"} <= keys
    doc = (box / "docs" / "api_demo.md").read_text()
    assert f"{len(LENGTHS)} episodes recorded by a human" in doc
    assert "(T, 20)" in doc
    assert "(T, 50)" in doc
    assert doc in (box / "docs" / "api.md").read_text()
    assert "`demos/`" in (box / "PROMPT.md").read_text()
    # The scene description and the controller settings stay out of the sandbox.
    metadata = json.loads((box / "demos" / "metadata.json").read_text())
    assert set(metadata) == {"control_step_seconds", "action_stats", "task"}
    assert set(metadata["task"]) == {"camera_keys", "camera_shape"}


def test_unpublished_task_reports_the_dataset_message(tmp_path, datasets, capsys):
    cell = tmp_path / "cell"
    assert build(cell, "dishwasher_open", "--demo", "none") == 1
    error = capsys.readouterr().err
    assert "have not been published yet" in error
    assert "dishwasher_open" in error
    assert not (cell / "sandbox" / "PROMPT.md").exists()


def test_unknown_task_is_refused_before_any_download(tmp_path, datasets, capsys):
    assert build(tmp_path / "cell", "not_a_task", "--demo", "none") == 1
    assert "no task sentence" in capsys.readouterr().err


def test_force_is_required_to_rebuild(tmp_path, datasets, capsys):
    cell = tmp_path / "cell"
    assert build(cell, FLAT_TASK, "--demo", "none") == 0
    marker = cell / "sandbox" / "policy.py"
    marker.write_text("# the agent's work\n")
    assert build(cell, FLAT_TASK, "--demo", "none") == 1
    assert "use --force" in capsys.readouterr().err
    assert marker.read_text() == "# the agent's work\n"
    assert build(cell, FLAT_TASK, "--demo", "none", "--force") == 0
    assert marker.read_text() != "# the agent's work\n"


def test_isolation_picks_the_interpreter_and_the_settings(tmp_path, datasets):
    cell = tmp_path / "cell"
    assert build(cell, FLAT_TASK, "--demo", "none", "--isolation", "container") == 0
    box = cell / "sandbox"
    assert "/usr/local/bin/python3" in (box / "python").read_text()
    settings = json.loads((box / "claude_settings.json").read_text())
    assert settings["sandbox"]["filesystem"]["allowWrite"] == ["/work"]
    # The container is the isolation; Claude Code's own sandbox cannot start in it.
    assert settings["sandbox"]["enabled"] is False
    assert json.loads((cell / "sandbox_config.json").read_text())["isolation"] == (
        "container"
    )

    soft = tmp_path / "soft"
    assert build(soft, FLAT_TASK, "--demo", "none") == 0
    settings = json.loads((soft / "sandbox" / "claude_settings.json").read_text())
    assert "sandbox" not in settings
    assert "WebFetch" in settings["permissions"]["deny"]
    # On the host the instruction files of the project around the sandbox stay out.
    assert {"**/CLAUDE.md", "**/AGENTS.md"} <= set(settings["claudeMdExcludes"])


def test_image_cap_is_stated_in_the_document(tmp_path, datasets):
    cell = tmp_path / "cell"
    assert build(cell, FLAT_TASK, "--demo", "none", "--image-cap", "320x240") == 0
    api = (cell / "sandbox" / "docs" / "api.md").read_text()
    assert "you may request up to\n  320x240 from the same cameras" in api
    config = json.loads((cell / "sandbox_config.json").read_text())
    assert config["env_tools"]["image_cap"] == "320x240"


def test_episode_listing_and_median_pick(tmp_path, datasets):
    demos = demo_video.TaskDemos(FLAT_TASK)
    assert [episode.length for episode in demos.episodes] == list(LENGTHS)
    assert [episode.seed for episode in demos.episodes] == list(SEEDS)
    assert demos.seeds == sorted(set(SEEDS))
    median = demo_video.pick_episodes(demos.episodes, "median", 1)
    assert [episode.length for episode in median] == [6]
    two = demo_video.pick_episodes(demos.episodes, "median", 2)
    assert len(two) == 2
    explicit = demo_video.pick_episodes(demos.episodes, "2", 1)
    assert explicit[0].index == 2
    with pytest.raises(ValueError):
        demo_video.pick_episodes(demos.episodes, "9", 1)
    qpos = demos.qpos(demos.episodes[0])
    assert qpos.shape == (LENGTHS[0], NQ)


def test_parse_size_rejects_nonsense():
    assert cli.parse_size("84x84") == (84, 84)
    assert cli.parse_size("320X240") == (320, 240)
    for bad in ("84", "0x10", "axb"):
        with pytest.raises(ValueError):
            cli.parse_size(bad)
    with pytest.raises(ValueError):
        EnvToolsConfig(image_cap="84")


# Every environment setting away from its default.
CELL_ENV_TOOLS = EnvToolsConfig(
    interface="tools",
    image_cap="128x96",
    slew=True,
    joint_vmax=4.0,
    accel=None,
    lowpass=0.5,
    ik=False,
    hand_pos=False,
    pitch=False,
    allow_external_cameras=True,
)


def test_a_cell_is_evaluated_and_replayed_in_its_own_environment(
    tmp_path, datasets, monkeypatch
):
    """The settings a cell was built with are the ones its evaluation runs."""
    cell = tmp_path / "cell"
    flags = [
        "--interface",
        "tools",
        "--image-cap",
        "128x96",
        "--slew",
        "--joint-vmax",
        "4.0",
        "--accel",
        "None",
        "--lowpass",
        "0.5",
        "--no-ik",
        "--no-hand-pos",
        "--no-pitch",
        "--allow-external-cameras",
    ]
    assert build(cell, PITCH_TASK, "--demo", "none", *flags) == 0
    assert snapshots.read_env_tools(cell) == CELL_ENV_TOOLS
    # --no-pitch forces the 20-float layout on the 21-float task
    config = json.loads((cell / "sandbox_config.json").read_text())
    assert config["action_dim"] == 20 and config["pitch"] is False
    version = cell / "policies" / "v001"
    version.mkdir(parents=True)
    (version / "policy.py").write_text((cell / "initial_policy.py").read_text())
    snapshots.write_index(
        cell / "policies",
        {"task": PITCH_TASK, "versions": [{"version": 1, "dir": "v001"}]},
    )

    scored = []

    def run_block(block, **kwargs):
        scored.append(block.env_tools)
        recs = [
            {
                "episode": i,
                "seed": 620000 + i,
                "success": 0,
                "length": 3,
                "reward": 0.0,
                "termination": "timeout",
                "fell": False,
            }
            for i in range(block.episodes)
        ]
        return recs, {"task": block.task}, []

    monkeypatch.setattr(evaluate, "run_block", run_block)
    argv = [str(cell), "--version", "1", "--episodes", "2", "--spot-check", "0"]
    assert evaluate.main(argv + ["--no-video"]) == 0
    assert scored == [CELL_ENV_TOOLS]
    summary = json.loads((cell / "eval" / "v001" / "summary.json").read_text())
    assert summary["interface"] == "tools"
    assert summary["image_cap"] == "128x96"
    assert summary["ik"] is False and summary["hand_pos"] is False
    assert summary["onboard_only"] is False
    assert summary["pitch_layout"] == "forced-20dim"
    assert summary["slew"] == {
        "joint_rad_per_s": 4.0,
        "base_per_s": 0.7,
        "joint_accel_rad_per_step2": None,
        "lowpass_alpha": 0.5,
    }
    # the cell decides: an environment flag next to a cell is refused
    with pytest.raises(SystemExit):
        evaluate.main(argv + ["--force", "--slew"])

    built = []

    def env_tools(task, config):
        built.append(config)
        raise RuntimeError("enough")

    monkeypatch.setattr(replay, "EnvTools", env_tools)
    replay.main([str(cell), "--version", "1", "--seeds", "620000", "--no-video"])
    assert built == [CELL_ENV_TOOLS]


def _frame_count(path: Path) -> int:
    """Count the frames of an mp4 by decoding it."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    count = 0
    while capture.read()[0]:
        count += 1
    capture.release()
    return count


@pytest.mark.slow
def test_renders_a_video_from_a_real_environment(tmp_path, monkeypatch):
    """Re-render ten frames of a synthetic episode built on a real env's qpos."""
    from bigym.loco.demos.rerender import BatchRenderer

    task_block = {
        "task_name": FLAT_TASK,
        "robot_model": "g1_dex1",
        "camera_keys": list(CAMERAS),
        "camera_shape": [84, 84],
        "enable_all_floating_dof": True,
        "action_mode": "absolute",
        "demo_down_sample_rate": 10,
        "episode_length": 60000,
        "success_hold_seconds": 1.0,
        "reach_tolerance": 0.05,
    }
    renderer = BatchRenderer(
        {"control_step_seconds": 0.02, "task": task_block}, (84, 84), verbose=False
    )
    try:
        settled = np.asarray(renderer.data.qpos, dtype=np.float64).copy()
    finally:
        renderer.close()

    frames = 10
    rng = np.random.default_rng(1)
    episode = make_episode(rng, frames, 20, 50, nq=NQ)
    episode["full_qpos"] = np.tile(settled, (frames, 1))
    episode["full_qpos"][:, :3] += np.linspace(0, 0.01, frames)[:, None]
    episode["seed"] = np.int64(SEEDS[0])
    root = write_export(
        tmp_path / "dataset" / FLAT_TASK, [episode], FLAT_TASK, task_block
    )
    monkeypatch.setattr(hub, "task_dir", lambda task, repo=None, revision=None: root)

    out = tmp_path / "video"
    record = demo_video.render_demo(
        FLAT_TASK, out, size=(84, 84), views=("head", "left_wrist"), every=1
    )
    assert record["frames"] == frames
    assert record["seeds"] == [SEEDS[0]]
    for name in ("demo_head.mp4", "demo_left_wrist.mp4"):
        path = out / name
        assert path.is_file() and path.stat().st_size > 0
        assert _frame_count(path) == frames
        sidecar = json.loads(path.with_suffix(".json").read_text())
        assert sidecar["size"] == "84x84"
        assert sidecar["fps"] == 25
        assert "dataset_dir" not in sidecar
