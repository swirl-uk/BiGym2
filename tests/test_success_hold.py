"""Re-timing the terminal success hold of a replay-format batch (no simulation)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from bigym.loco.config import EnvConfig
from bigym.loco.demos import success_hold

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


def test_trim_drops_the_surplus_hold_and_moves_the_marker(tmp_path):
    src = _batch(tmp_path)
    before = sorted(p.name for p in src.iterdir())
    out = success_hold.trim_batch(src)

    assert out == tmp_path / "raw_hold1s"
    assert sorted(p.name for p in src.iterdir()) == before
    assert sorted(p.name for p in out.glob("*.npz")) == [
        "20261007T120000_0_199.npz",
        "20261007T120000_1_319.npz",
    ]
    assert (out / "frame_timing.json").is_file()
    raw = _load(src / "20261007T120000_0_299.npz")
    trimmed = _load(out / "20261007T120000_0_199.npz")
    np.testing.assert_array_equal(trimmed["action"], raw["action"][:200])
    np.testing.assert_array_equal(trimmed["rgb_obs"], raw["rgb_obs"][:200])
    _assert_terminal_marker(trimmed)
    # Episode-level arrays are not cut even when their length matches.
    np.testing.assert_array_equal(trimmed["init_qpos"], raw["init_qpos"])
    assert int(trimmed["seed"]) == 0

    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["task"]["success_hold_seconds"] == 1.0
    assert metadata["task"]["collect_success_hold_seconds"] == 3.0
    assert metadata["env_config"]["success_hold_seconds"] == 1.0
    assert metadata["success_hold_trim"]["trim_frames"] == 100
    # The training view reads back as the env it trains.
    assert EnvConfig.from_metadata(metadata).success_hold_seconds == 1.0


def test_trim_refuses_what_it_cannot_do(tmp_path):
    src = _batch(tmp_path, hold=1.0)
    with pytest.raises(ValueError, match="must be longer"):
        success_hold.plan_trim(src, None)
    with pytest.raises(ValueError, match="not an integer number"):
        success_hold.plan_trim(src, None, source_hold_seconds=1.005)
    with pytest.raises(ValueError, match="at least two frames"):
        success_hold.plan_trim(src, None, source_hold_seconds=7.0)
    (tmp_path / "taken").mkdir()
    with pytest.raises(FileExistsError):
        success_hold.trim_batch(src, tmp_path / "taken", source_hold_seconds=3.0)
    with pytest.raises(ValueError, match="must differ"):
        success_hold.plan_trim(src, src, source_hold_seconds=3.0)


def test_trim_rejects_a_batch_without_one_terminal_reward(tmp_path):
    src = _batch(tmp_path, frames=(300,))
    path = src / "20261007T120000_0_299.npz"
    episode = _load(path)
    episode["reward"][10] = 1.0
    np.savez_compressed(path, **episode)
    with pytest.raises(ValueError, match="one reward=1"):
        success_hold.plan_trim(src, None)


def test_command_line_dry_run_writes_nothing(tmp_path, capsys):
    src = _batch(tmp_path)
    success_hold.main(["trim", "--demo-dir", str(src), "--dry-run"])
    assert "2 episodes validated: 299-419 -> 199-319" in capsys.readouterr().out
    assert sorted(p.name for p in tmp_path.iterdir()) == ["raw"]
    success_hold.main(
        ["trim", "--demo-dir", str(src), "--out-dir", str(tmp_path / "t")]
    )
    assert len(list((tmp_path / "t").glob("*.npz"))) == 2
