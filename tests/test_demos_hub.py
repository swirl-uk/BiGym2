"""Hub access layer (bigym.loco.demos.hub): repo resolution and lazy fetch.

Network access is mocked; the real dataset is exercised by the
``BIGYM_DATASET_TESTS=1``-gated test in ``test_gym_adapter.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigym.loco.demos import hub


def test_repo_resolution_precedence(monkeypatch):
    monkeypatch.delenv(hub.REPO_ENV_VAR, raising=False)
    assert hub.dataset_repo() == hub.DEFAULT_DATASET_REPO
    monkeypatch.setenv(hub.REPO_ENV_VAR, "someone/else")
    assert hub.dataset_repo() == "someone/else"
    assert hub.dataset_repo("explicit/repo") == "explicit/repo"
    monkeypatch.delenv(hub.REVISION_ENV_VAR, raising=False)
    assert hub.dataset_revision() is None
    monkeypatch.setenv(hub.REVISION_ENV_VAR, "abc123")
    assert hub.dataset_revision() == "abc123"


def _fake_hub(monkeypatch, tmp_path: Path, tasks: list[str]):
    """Stand in for the Hub: a local snapshot with one folder per task."""
    snapshot = tmp_path / "snapshot"
    for task in tasks:
        (snapshot / task / "meta").mkdir(parents=True)
        (snapshot / task / "meta" / "info.json").write_text(
            json.dumps({"codebase_version": "v3.0"})
        )
    calls: list[dict] = []

    def fake_snapshot_download(repo_id, **kwargs):
        calls.append({"repo_id": repo_id, **kwargs})
        return str(snapshot)

    class FakeApi:
        def list_repo_files(self, repo_id, repo_type=None, revision=None):
            return [f"{t}/meta/info.json" for t in tasks] + [".gitattributes"]

    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(hub, "HfApi", FakeApi)
    return snapshot, calls


def test_task_dir_fetches_only_that_task(monkeypatch, tmp_path):
    pytest.importorskip("huggingface_hub")
    snapshot, calls = _fake_hub(monkeypatch, tmp_path, ["move_plate", "pick_box"])
    folder = hub.task_dir("move_plate", repo="org/demos")
    assert folder == snapshot / "move_plate"
    assert calls[-1]["allow_patterns"] == ["move_plate/**"]
    assert calls[-1]["repo_type"] == "dataset"


def test_a_module_attr_task_has_a_path_safe_folder(monkeypatch, tmp_path):
    pytest.importorskip("huggingface_hub")
    snapshot, calls = _fake_hub(monkeypatch, tmp_path, ["my_lab.tasks-SPEC"])
    folder = hub.task_dir("my_lab.tasks:SPEC", repo="org/demos")
    assert folder == snapshot / "my_lab.tasks-SPEC"
    assert calls[-1]["allow_patterns"] == ["my_lab.tasks-SPEC/**"]


def test_unpublished_task_says_so_and_lists_both_groups(monkeypatch, tmp_path):
    pytest.importorskip("huggingface_hub")
    _fake_hub(monkeypatch, tmp_path, ["move_plate", "pick_box"])
    with pytest.raises(hub.DemosUnavailableError) as excinfo:
        hub.task_dir("stack_blocks", repo="org/demos")
    message = str(excinfo.value)
    assert "have not been published yet" in message
    assert "2/40 benchmark tasks" in message
    assert "Published (2): move_plate, pick_box" in message
    assert "Pending (38):" in message and "stack_blocks" in message
    assert "huggingface.co/datasets/org/demos" in message
    assert hub.available_tasks("org/demos") == ("move_plate", "pick_box")
    assert "stack_blocks" in hub.pending_tasks("org/demos")
    assert "move_plate" not in hub.pending_tasks("org/demos")


def test_non_benchmark_task_is_rejected_differently(monkeypatch, tmp_path):
    pytest.importorskip("huggingface_hub")
    _fake_hub(monkeypatch, tmp_path, ["move_plate"])
    with pytest.raises(
        hub.DemosUnavailableError, match="not a BiGym 2.0 benchmark task"
    ):
        hub.task_dir("no_such_task", repo="org/demos")


def test_download_all_fetches_everything(monkeypatch, tmp_path):
    pytest.importorskip("huggingface_hub")
    snapshot, calls = _fake_hub(monkeypatch, tmp_path, ["move_plate"])
    assert hub.download_all("org/demos") == snapshot
    assert calls[-1]["allow_patterns"] is None


def test_cli_list_and_download(monkeypatch, tmp_path, capsys):
    pytest.importorskip("huggingface_hub")
    snapshot, calls = _fake_hub(monkeypatch, tmp_path, ["move_plate", "pick_box"])
    assert hub.main(["--repo", "org/demos", "--list"]) == 0
    out = capsys.readouterr().out
    assert "2/40 benchmark tasks published" in out and "pending (38" in out
    assert hub.main(["--repo", "org/demos", "--task", "pick_box"]) == 0
    assert [call["allow_patterns"] for call in calls[-2:]] == [
        ["pick_box/**"],
        ["agent_demos/pick_box/**"],
    ]
    assert hub.main(["--repo", "org/demos", "--all"]) == 0
    assert calls[-1]["allow_patterns"] is None


def test_cli_agent_fetches_metadata_and_videos_only(monkeypatch, tmp_path):
    pytest.importorskip("huggingface_hub")
    _, calls = _fake_hub(monkeypatch, tmp_path, ["move_plate", "pick_box"])
    assert hub.main(["--repo", "org/demos", "--agent", "--task", "pick_box"]) == 0
    assert [call["allow_patterns"] for call in calls] == [
        ["pick_box/metadata.json", "pick_box/meta/**"],
        ["agent_demos/pick_box/**"],
    ]
    with pytest.raises(SystemExit):
        hub.main(["--agent", "--all", "--local-dir", str(tmp_path / "out")])
