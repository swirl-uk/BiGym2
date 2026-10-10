"""LeRobot v3 reader (bigym.loco.demos.dataset) on a synthetic export.

Builds a tiny lossless PNG-mode dataset the way ``bigym-export-lerobot``
lays it out (alignment v2: frame k carries obs[k] plus the transition
executed FROM it, replay row 0 in the ``first_transition`` sidecar) and
checks the reader returns the collector's replay rows bit for bit.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from bigym.loco.demos import dataset
from tests.fixtures.synthetic_dataset import CAMERAS, make_episode, write_export


@pytest.fixture(scope="module")
def synthetic_dataset(tmp_path_factory):
    rng = np.random.default_rng(0)
    episodes = []
    for index, length in enumerate((6, 4, 9)):
        episode = make_episode(rng, length, action_dim=4, state_dim=5)
        episode["seed"] = np.int64(620000 + index)
        episodes.append(episode)
    root = write_export(tmp_path_factory.mktemp("lerobot_v3"), episodes)
    return root, episodes


def test_reader_round_trips_replay_rows(synthetic_dataset):
    root, episodes = synthetic_dataset
    loaded = dataset.load_episodes(root)
    assert [name for name, _ in loaded] == [f"episode_{i}.npz" for i in range(3)]
    for (_, got), want in zip(loaded, episodes, strict=True):
        for key, value in want.items():
            np.testing.assert_array_equal(got[key], value, err_msg=key)


def test_reader_honours_max_episodes(synthetic_dataset):
    root, _ = synthetic_dataset
    assert len(dataset.load_episodes(root, max_episodes=2)) == 2
    assert len(dataset.load_episodes(root, max_episodes=-1)) == 3


def test_undecoded_frames_decode_to_the_same_rows(synthetic_dataset):
    root, episodes = synthetic_dataset
    loaded = dataset.load_episodes(root, decode_images=False)
    for (_, got), want in zip(loaded, episodes, strict=True):
        frames = got["rgb_obs"]
        assert isinstance(frames, dataset.PngFrames)
        assert frames.shape == want["rgb_obs"].shape
        np.testing.assert_array_equal(np.asarray(frames), want["rgb_obs"])
        np.testing.assert_array_equal(frames[-1], want["rgb_obs"][-1])
        np.testing.assert_array_equal(frames[1:3], want["rgb_obs"][1:3])


@pytest.mark.parametrize(
    "index",
    [(2, 0), (-1, 1, 2), (slice(1, 4), 0), (slice(None, None, 2), slice(None), 0),
     [3, 0, 3], np.array([1, 2]), (np.array([0, 2]), 1), slice(3, 3),
     ([0, 2], slice(None), 0), ([0, 2], 1, slice(1, 3)), (slice(1, 4), [0, 2])],
)  # fmt: skip
def test_undecoded_frames_index_like_the_array(synthetic_dataset, index):
    root, episodes = synthetic_dataset
    frames = dataset.load_episodes(root, decode_images=False)[2][1]["rgb_obs"]
    want = episodes[2]["rgb_obs"]
    np.testing.assert_array_equal(frames[index], want[index])


@pytest.mark.parametrize(
    "index",
    [(Ellipsis, 0), (None,), np.array([True, False]), np.zeros((2, 2), int),
     ([0, 2], [0, 1]), (np.array([0, 2]), np.array([1, 0])), ([0, 1], True),
     True, np.True_],
)  # fmt: skip
def test_undecoded_frames_refuse_other_indices(synthetic_dataset, index):
    root, _ = synthetic_dataset
    frames = dataset.load_episodes(root, decode_images=False)[2][1]["rgb_obs"]
    with pytest.raises(IndexError):
        frames[index]


@pytest.mark.parametrize("stack", [1, 3])
def test_undecoded_frames_stack_like_the_env(synthetic_dataset, stack):
    from bigym.loco.env import BiGym

    root, episodes = synthetic_dataset
    frames = dataset.load_episodes(root, decode_images=False)[0][1]["rgb_obs"]
    assert isinstance(frames, dataset.PngFrames)
    want = BiGym._stack_demo_frames(episodes[0]["rgb_obs"], stack, axis=2)
    stacked = frames.stacked(stack)
    assert stacked.shape == want.shape
    for t in range(len(stacked)):
        np.testing.assert_array_equal(stacked[t], want[t])


def test_reader_rejects_video_mode(synthetic_dataset, tmp_path):
    root, _ = synthetic_dataset
    info = json.loads((root / "meta" / "info.json").read_text())
    info["features"]["observation.images.head"]["dtype"] = "video"
    bad = tmp_path / "video"
    bad.mkdir()
    (bad / "meta").mkdir()
    (bad / "meta" / "info.json").write_text(json.dumps(info))
    (bad / "metadata.json").write_text((root / "metadata.json").read_text())
    with pytest.raises(RuntimeError, match="lossy video"):
        dataset.load_episodes(bad)


def test_task_metadata_is_the_root_copy(synthetic_dataset, tmp_path):
    root, _ = synthetic_dataset
    assert dataset.load_task_metadata(root)["task"]["camera_keys"] == list(CAMERAS)
    with pytest.raises(FileNotFoundError):
        dataset.load_task_metadata(tmp_path / "empty")


def test_published_metadata_keeps_only_portable_fields():
    """The exporter drops config placeholders, unused backend settings, local batch dirs and account names."""
    from bigym.loco.demos.lerobot_export import sanitize_published_metadata

    meta = {
        "lowerbody_policy": {
            "backend": "groot_wbc_g1",
            "checkpoint_path": "${hydra:runtime.cwd}/policy.pt",
            "mjlab_g1": {"policy_path": "${hydra:runtime.cwd}/mjlab.onnx"},
            "groot_wbc_g1": {"repo_root": "/home/someone/GR00T"},
        },
        "task": {
            "lowerbody_policy": {"checkpoint_path": "x.pt", "backend": "groot_wbc_g1"}
        },
        "substrate_fingerprint": {"lowerbody_checkpoint": "policy.pt", "x": 1},
        "success_hold_trim": {"source_batch": "runs/batch_a/move_plate"},
        "subset_provenance": {"sources": ["runs/vr/move_plate/session_1"]},
        "collection": {
            "host": "lab-box",
            "operator": "someone",
            "sessions": [{"host": "lab-box", "operator": "someone", "n": 3}],
        },
    }
    out = sanitize_published_metadata(meta)
    assert set(out["lowerbody_policy"]) == {"backend", "groot_wbc_g1"}
    assert out["lowerbody_policy"]["groot_wbc_g1"]["repo_root"] == "<groot-checkout>"
    assert set(out["task"]["lowerbody_policy"]) == {"backend"}
    assert out["substrate_fingerprint"] == {"x": 1}
    assert out["success_hold_trim"]["source_batch"] == "move_plate"
    assert out["subset_provenance"]["sources"] == ["move_plate/session_1"]
    assert "host" not in out["collection"] and "operator" not in out["collection"]
    assert out["collection"]["sessions"][0] == {"n": 3}
    assert "hydra" not in json.dumps(out)
