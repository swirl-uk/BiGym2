"""Collection accounting in the VR collector's metadata.json.

Exercises the `collection` block (attempts / kept / discarded / operator /
session identity) without a headset: the block builder is a pure function and
the writer is driven on a bare CollectorSession instance holding only the
attributes it touches.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from bigym.loco.fingerprint import bigym_git_sha
from bigym.vr.collect.config import CollectConfig
from bigym.vr.collect.session import (
    CollectorSession,
    CollectorStats,
    build_collection_block,
    resolve_operator,
)

IDENTITY: dict[str, Any] = {
    "session_id": "11111111-1111-4111-8111-111111111111",
    "started_at": "2026-09-02T09:00:00Z",
    "operator": "tester",
    "host": "testbox",
    "bigym_git_sha": "deadbee",
}


def _stats(attempted: int = 0, kept: int = 0, **kwargs) -> CollectorStats:
    return CollectorStats(attempted=attempted, saved_successes=kept, **kwargs)


def _session(tmp_path, stats=None, previous=None, unaccounted=0, seed=1):
    """A CollectorSession carrying only what the metadata writer reads."""
    session = object.__new__(CollectorSession)
    session.config = CollectConfig(task="move_plate", seed=seed)
    session.out_dir = tmp_path
    session.stats = stats if stats is not None else CollectorStats()
    session._session_id = IDENTITY["session_id"]
    session._session_started_at = IDENTITY["started_at"]
    session._session_finished_at = None
    session._operator = IDENTITY["operator"]
    session._host = IDENTITY["host"]
    session._bigym_git_sha = IDENTITY["bigym_git_sha"]
    session._unaccounted_episodes = unaccounted
    session._previous_collection = previous
    session._metadata_base = {
        "format": "bigym_replay_npz",
        "pipeline_version": "test",
        "task": {"task_name": "move_plate", "episode_length": 15000},
    }
    return session


# --------------------------------------------------------------------------
# build_collection_block
# --------------------------------------------------------------------------


def test_block_has_the_required_keys_and_arithmetic():
    block = build_collection_block(_stats(attempted=7, kept=5), **IDENTITY)

    for key in (
        "attempts",
        "kept",
        "discarded",
        "success_rate_human",
        "session_id",
        "started_at",
        "finished_at",
        "operator",
        "host",
        "bigym_git_sha",
    ):
        assert key in block, key

    assert block["attempts"] == 7
    assert block["kept"] == 5
    assert block["discarded"] == 7 - 5
    assert block["success_rate_human"] == pytest.approx(5 / 7)
    assert block["session_id"] == IDENTITY["session_id"]
    assert block["operator"] == "tester"
    assert block["host"] == "testbox"
    assert block["bigym_git_sha"] == "deadbee"
    assert block["finished_at"] is None
    assert len(block["sessions"]) == 1  # one entry per session
    assert block["sessions"][0]["attempts"] == 7
    assert block["sessions"][0]["kept"] == 5


def test_zero_attempts_does_not_divide_by_zero():
    block = build_collection_block(_stats(), **IDENTITY)
    assert block["attempts"] == 0
    assert block["kept"] == 0
    assert block["discarded"] == 0
    assert block["success_rate_human"] == 0.0
    assert block["sessions"][0]["seed_first"] is None
    assert block["sessions"][0]["seed_last"] is None


def test_seed_span_matches_the_attempt_counter():
    # Per-episode seeds are config.seed + attempted - 1.
    block = build_collection_block(
        _stats(attempted=4, kept=3), seed_start=17, **IDENTITY
    )
    entry = block["sessions"][0]
    assert entry["seed_first"] == 17
    assert entry["seed_last"] == 17 + 4 - 1


def test_discard_breakdown_is_reported():
    stats = _stats(
        attempted=10,
        kept=4,
        discarded_failed_attempts=3,
        discarded_pending_successes=1,
        restarted_mid_attempt=1,
        rejected_black_rgb=1,
        save_not_stored=0,
    )
    block = build_collection_block(stats, **IDENTITY)
    breakdown = block["sessions"][0]["discard_breakdown"]
    assert breakdown == {
        "failed_attempts": 3,
        "operator_discarded_successes": 1,
        "restarted_mid_attempt": 1,
        "rejected_black_rgb": 1,
        "not_stored": 0,
    }
    # The breakdown is best-effort; `discarded` stays attempts - kept.
    assert block["discarded"] == 6
    assert sum(breakdown.values()) == 6


def test_rewriting_the_same_session_replaces_its_entry():
    first = build_collection_block(_stats(attempted=1, kept=0), **IDENTITY)
    second = build_collection_block(
        _stats(attempted=2, kept=1), previous=first, **IDENTITY
    )
    assert len(second["sessions"]) == 1
    assert second["attempts"] == 2
    assert second["kept"] == 1
    assert second["discarded"] == 1


def test_a_resumed_session_accumulates_across_sessions():
    first = build_collection_block(_stats(attempted=6, kept=4), **IDENTITY)
    second_identity: dict[str, Any] = {
        **IDENTITY,
        "session_id": "22222222-2222-4222-8222-222222222222",
    }
    second_identity["operator"] = "someone_else"
    second_identity["started_at"] = "2026-09-03T09:00:00Z"

    second = build_collection_block(
        _stats(attempted=3, kept=2), previous=first, **second_identity
    )

    assert len(second["sessions"]) == 2
    assert second["attempts"] == 6 + 3
    assert second["kept"] == 4 + 2
    assert second["discarded"] == (6 - 4) + (3 - 2)
    assert second["success_rate_human"] == pytest.approx(6 / 9)
    # Identity fields describe the session that wrote the file last.
    assert second["session_id"] == second_identity["session_id"]
    assert second["operator"] == "someone_else"
    assert [s["operator"] for s in second["sessions"]] == ["tester", "someone_else"]


def test_previous_block_without_a_sessions_list_is_kept_as_one_session():
    legacy = {"attempts": 5, "kept": 3, "operator": "old"}
    block = build_collection_block(
        _stats(attempted=2, kept=2), previous=legacy, **IDENTITY
    )
    assert len(block["sessions"]) == 2
    assert block["attempts"] == 7
    assert block["kept"] == 5
    assert block["discarded"] == 2


def test_unaccounted_episodes_are_reported_separately():
    block = build_collection_block(
        _stats(attempted=2, kept=2), unaccounted_episodes=9, **IDENTITY
    )
    assert block["unaccounted_episodes"] == 9
    assert block["kept"] == 2  # never folded into the accounted counters


# --------------------------------------------------------------------------
# operator / git sha resolution
# --------------------------------------------------------------------------


def test_operator_prefers_the_flag_then_the_env_then_the_login_user(monkeypatch):
    monkeypatch.setenv("BIGYM_OPERATOR", "env_person")
    assert resolve_operator("flag_person") == "flag_person"
    assert resolve_operator(None) == "env_person"
    assert resolve_operator("   ") == "env_person"  # blank flag does not win

    monkeypatch.delenv("BIGYM_OPERATOR", raising=False)
    monkeypatch.setattr(
        "bigym.vr.collect.session.getpass.getuser", lambda: "login_person"
    )
    assert resolve_operator(None) == "login_person"


def test_operator_falls_back_when_the_login_user_is_unknown(monkeypatch):
    monkeypatch.delenv("BIGYM_OPERATOR", raising=False)

    def boom():
        raise KeyError("no pwd entry")

    monkeypatch.setattr("bigym.vr.collect.session.getpass.getuser", boom)
    assert resolve_operator(None) == "unknown"


def test_git_sha_is_a_short_sha_or_none():
    sha = bigym_git_sha()
    if sha is not None:
        sha = sha.removesuffix("-dirty")
    assert sha is None or (sha.isalnum() and 6 <= len(sha) <= 40)


# --------------------------------------------------------------------------
# the writer itself
# --------------------------------------------------------------------------


def _read(tmp_path):
    return json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))


def test_writer_keeps_existing_keys_and_adds_the_collection_block(tmp_path):
    session = _session(tmp_path)
    session._refresh_collection_metadata()

    written = _read(tmp_path)
    assert written["format"] == "bigym_replay_npz"
    assert written["pipeline_version"] == "test"
    assert written["task"]["episode_length"] == 15000
    assert set(written) == {"format", "pipeline_version", "task", "collection"}
    assert written["collection"]["attempts"] == 0
    assert not list(tmp_path.glob("*.tmp"))  # atomic write leaves no debris


def test_counters_are_current_at_every_write_and_final_at_the_end(tmp_path):
    session = _session(tmp_path)
    session._refresh_collection_metadata()
    assert _read(tmp_path)["collection"]["attempts"] == 0

    session.stats.attempted += 1
    session._refresh_collection_metadata()
    started = _read(tmp_path)["collection"]
    assert (started["attempts"], started["kept"], started["discarded"]) == (1, 0, 1)

    session.stats.saved_successes += 1
    session._touch_collection_metadata()
    mid = _read(tmp_path)["collection"]
    assert (mid["attempts"], mid["kept"], mid["discarded"]) == (1, 1, 0)
    assert mid["finished_at"] is None

    session.stats.attempted += 1
    session.finalize_metadata()
    final = _read(tmp_path)["collection"]
    assert (final["attempts"], final["kept"], final["discarded"]) == (2, 1, 1)
    assert final["success_rate_human"] == pytest.approx(0.5)
    assert final["finished_at"] is not None
    assert final["finished_at"].endswith("Z")
    assert final["sessions"][-1]["finished_at"] == final["finished_at"]


def test_resuming_a_batch_dir_accumulates_through_the_writer(tmp_path):
    first = _session(tmp_path, stats=_stats(attempted=6, kept=4))
    first.finalize_metadata()

    second = _session(tmp_path, stats=_stats(attempted=3, kept=2))
    second._session_id = "33333333-3333-4333-8333-333333333333"
    second._previous_collection = second._read_previous_collection()
    assert second._previous_collection is not None
    second.finalize_metadata()

    block = _read(tmp_path)["collection"]
    assert block["attempts"] == 9
    assert block["kept"] == 6
    assert block["discarded"] == 3
    assert [s["attempts"] for s in block["sessions"]] == [6, 3]
    assert block["session_id"] == "33333333-3333-4333-8333-333333333333"


def test_read_previous_collection_tolerates_junk(tmp_path):
    session = _session(tmp_path)
    assert session._read_previous_collection() is None  # no file yet

    (tmp_path / "metadata.json").write_text("{not json", encoding="utf-8")
    assert session._read_previous_collection() is None

    (tmp_path / "metadata.json").write_text('{"format": "x"}', encoding="utf-8")
    assert session._read_previous_collection() is None


def test_writer_does_not_recreate_a_removed_session_dir(tmp_path):
    out_dir = tmp_path / "gone"
    out_dir.mkdir()
    session = _session(out_dir)
    out_dir.rmdir()

    session.finalize_metadata()  # empty-session cleanup already deleted the dir
    assert not out_dir.exists()


def test_a_failed_write_does_not_kill_the_session(tmp_path, monkeypatch):
    session = _session(tmp_path)

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(CollectorSession, "_refresh_collection_metadata", boom)
    session._touch_collection_metadata()  # must not raise
    session.finalize_metadata()  # must not raise


@pytest.mark.parametrize(
    "succeeded, fell, kept",
    [
        (True, False, True),
        (True, True, False),
        # A reward on the last step is not a success by itself.
        (False, False, False),
    ],
)
def test_an_attempt_is_kept_by_the_evaluation_success(tmp_path, succeeded, fell, kept):
    session = _session(
        tmp_path,
        stats=_stats(
            attempted=1,
            recording=True,
            episode_steps=10,
            episode_reward=1.0,
            last_reward=1.0,
        ),
    )
    session.env = SimpleNamespace(
        episode_succeeded=lambda: succeeded, episode_fell=lambda: fell
    )
    session.control_dt = 0.02
    session._current_steps = []
    session._current_fullbody_steps = []
    session._current_seed = 1
    session._pre_engage_steps = 0
    session._engage_state = None
    session._pending = None

    session._finish_attempt(force=False)

    assert (session._pending is not None) == kept
    assert session.stats.discarded_failed_attempts == int(not kept)
    assert not session.stats.recording
