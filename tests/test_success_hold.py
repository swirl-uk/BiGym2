"""Cutting a raw replay-format batch on the step the env latches success."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pytest

from bigym.loco import make
from bigym.loco.config import EnvConfig
from bigym.loco.demos import success_hold
from bigym.vr.collect import cli
from bigym.vr.collect.config import CollectConfig

STEP = 0.02  # 50 Hz
DOFS = ["pelvis_x", "pelvis_y", "pelvis_z", "pelvis_rz"]
ACTION_DIM = len(DOFS) + 2
# Asymmetric bounds on pelvis_rz, so its normalized zero is not 0.
STATS = {
    "min": [-0.35, -0.25, 0.4, -0.5, -1.0, -1.0],
    "max": [0.35, 0.25, 1.0, 1.5, 1.0, 1.0],
}


def _episode(frames: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    reward = np.zeros((frames, 1), np.float32)
    reward[-1] = 1.0
    discount = np.ones((frames, 1), np.float32)
    discount[-1] = 0.0
    return {
        "action": rng.uniform(-1, 1, (frames, ACTION_DIM)).astype(np.float32),
        "raw_outer_action": rng.uniform(-1, 1, (frames, ACTION_DIM)).astype(np.float32),
        "lowerbody_command": rng.uniform(-1, 1, (frames, 3)).astype(np.float32),
        "low_dim_obs": rng.normal(size=(frames, 5)).astype(np.float32),
        "rgb_obs": rng.integers(0, 255, (frames, 1, 3, 4, 4), dtype=np.uint8),
        "reward": reward,
        "discount": discount,
        "init_qpos": rng.normal(size=frames),  # episode-level despite its length
        "seed": np.asarray(seed),
    }


def _batch(tmp_path, *, hold: float = 3.0, frames=(300, 420)):
    src = tmp_path / "raw"
    src.mkdir()
    config = EnvConfig(episode_length=30000, success_hold_seconds=hold)
    metadata = dict(config.to_metadata("move_plate"))
    metadata["task"] = dict(metadata["task"], training_success_hold_seconds=1.0)
    metadata.update(
        control_step_seconds=STEP,
        outer_action_floating_dofs=DOFS,
        action_stats=STATS,
    )
    (src / "metadata.json").write_text(json.dumps(metadata))
    (src / "frame_timing.json").write_text("{}")
    for index, n in enumerate(frames):
        np.savez_compressed(
            src / f"20261007T120000_{index}_{n - 1}.npz", **_episode(n, index)
        )
    return src


def _load(path):
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _assert_terminal_marker(episode):
    reward, discount = episode["reward"][:, 0], episode["discount"][:, 0]
    assert reward[-1] == 1.0 and not reward[:-1].any()
    assert discount[-1] == 0.0 and (discount[:-1] == 1.0).all()


def test_cut_ends_each_episode_on_its_own_row(tmp_path):
    src = _batch(tmp_path)
    steps = {"20261007T120000_0_299.npz": 150, "20261007T120000_1_419.npz": 419}
    fingerprint = {"substrate_version": "test-v1", "success_hold_seconds": 1.0}
    out = success_hold.cut_batch(src, tmp_path / "out", steps, fingerprint)

    assert sorted(p.name for p in out.glob("*.npz")) == [
        "20261007T120000_0_150.npz",
        "20261007T120000_1_419.npz",
    ]
    assert (out / "frame_timing.json").is_file()
    raw = _load(src / "20261007T120000_0_299.npz")
    cut = _load(out / "20261007T120000_0_150.npz")
    np.testing.assert_array_equal(cut["action"], raw["action"][:151])
    np.testing.assert_array_equal(cut["rgb_obs"], raw["rgb_obs"][:151])
    # Episode-level arrays are not cut even when their length matches.
    np.testing.assert_array_equal(cut["init_qpos"], raw["init_qpos"])
    _assert_terminal_marker(cut)
    _assert_terminal_marker(_load(out / "20261007T120000_1_419.npz"))

    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["task"]["success_hold_seconds"] == 1.0
    assert metadata["task"]["collect_success_hold_seconds"] == 3.0
    assert metadata["env_config"]["success_hold_seconds"] == 1.0
    # The training view reads back as the env it trains.
    assert EnvConfig.from_metadata(metadata).success_hold_seconds == 1.0
    trim = metadata["success_hold_trim"]
    assert trim["trim_frames"] == {
        "20261007T120000_0_150.npz": 149,
        "20261007T120000_1_419.npz": 0,
    }
    assert trim["substrate_fingerprint"] == fingerprint


def test_cut_refuses_what_it_cannot_do(tmp_path):
    src = _batch(tmp_path)
    fingerprint = {"success_hold_seconds": 1.0}
    last = {"20261007T120000_0_299.npz": 299, "20261007T120000_1_419.npz": 419}
    with pytest.raises(ValueError, match="cannot cut 300 frames to 301"):
        success_hold.cut_batch(
            src, tmp_path / "a", {**last, "20261007T120000_0_299.npz": 300}, fingerprint
        )
    with pytest.raises(ValueError, match="do not match"):
        success_hold.cut_batch(
            src, tmp_path / "b", {**last, "20261007T120000_2_99.npz": 50}, fingerprint
        )
    (tmp_path / "taken").mkdir()
    with pytest.raises(FileExistsError):
        success_hold.cut_batch(src, tmp_path / "taken", last, fingerprint)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["raw", "taken"]


def test_the_collector_trains_on_the_latch_cut(tmp_path, monkeypatch, capsys):
    raw = tmp_path / "20261007_120000"
    cut = tmp_path / "20261007_120000_hold1s"
    monkeypatch.setattr(success_hold, "latch_batch", lambda demo_dir: cut)
    exported = []
    monkeypatch.setattr(cli.subprocess, "run", lambda cmd, check: exported.append(cmd))
    config = CollectConfig(task="move_plate", export_lerobot=True)
    cli._post_process(config, raw, collection_hold=3.0, training_hold=1.0)
    assert "-m bigym.loco.demos.success_hold --demo-dir" in capsys.readouterr().out
    (cmd,) = exported
    assert cmd[cmd.index("--demo-dir") + 1] == str(cut)


def record_episode(env, steps: int) -> dict[str, np.ndarray]:
    """``steps`` near-hold actions from a reset, stored as the collector stores them."""
    data = env.inner_env.data
    env.reset(seed=3)
    episode = {
        "seed": np.asarray([3]),
        **{
            f"init_{name}": getattr(data, name).copy()
            for name in ("qpos", "qvel", "ctrl", "act", "qacc_warmstart")
        },
        **{
            f"lb_state.{key}": np.array(value, copy=True)
            for key, value in env.get_lowerbody_state().items()
        },
    }
    hold = env.normalize_action(env.raw_hold_action())
    rng = np.random.default_rng(0)
    actions = [np.zeros_like(hold, dtype=np.float32)]
    full_qpos = [data.qpos.copy()]
    for _ in range(steps):
        action = np.clip(hold + rng.normal(0.0, 0.02, hold.shape), -1.0, 1.0)
        actions.append(action.astype(np.float32))
        env.step(actions[-1])
        full_qpos.append(data.qpos.copy())
    episode["action"] = np.stack(actions)
    episode["full_qpos"] = np.stack(full_qpos)
    return episode


def test_latch_replays_the_recording_and_refuses_an_episode_without_success(
    tmp_path,
):
    env = make("reach_target_single", camera_keys=()).bigym
    try:
        # Recorded under a narrower envelope than the env's own, as a
        # collector's action_stats are.
        stats = {key: value * 0.5 for key, value in env.action_stats.items()}
        env.set_action_stats(stats["min"], stats["max"])
        episode = record_episode(env, 20)
        with pytest.raises(ValueError, match="within the 20 recorded steps"):
            success_hold.latch_step(env, episode)
        episode["full_qpos"][5, 0] += 1e-9
        with pytest.raises(ValueError, match="full_qpos at row 5"):
            success_hold.latch_step(env, episode)
        episode["full_qpos"][5, 0] -= 1e-9
    finally:
        env.close()
    # From disk, latch_steps replays it under the batch's action_stats.
    path = tmp_path / "20261007T120000_0_20.npz"
    np.savez_compressed(path, **episode)  # ty: ignore[invalid-argument-type]
    metadata = {
        "task": {"task_name": "reach_target_single"},
        "action_stats": {key: value.tolist() for key, value in stats.items()},
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="within the 20 recorded steps"):
        success_hold.latch_steps(tmp_path)


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("BIGYM_DATASET_TESTS"),
    reason="set BIGYM_DATASET_TESTS=1 to download real demonstrations",
)
def test_a_published_demo_latches_on_its_last_row():
    env = make("saucepan_to_hob").bigym
    try:
        episode = env.load_demo_episodes(1)[0]
        assert success_hold.latch_step(env, episode) == len(episode["action"]) - 1
    finally:
        env.close()
