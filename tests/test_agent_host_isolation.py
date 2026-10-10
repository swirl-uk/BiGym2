"""A host session sees none of the instruction files and skills around its sandbox.

These tests start the real Claude Code and Codex CLIs, so they need both
installed and logged in, and they spend a few model calls. They run only with
``BIGYM_AGENT_CLI_TESTS=1``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from bigym.loco.agent.settings import (
    CLAUDE_HOST_FLAGS,
    CODEX_HOST_FLAGS,
    claude_settings_soft,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("BIGYM_AGENT_CLI_TESTS") != "1",
    reason="set BIGYM_AGENT_CLI_TESTS=1 to start the real harness CLIs",
)

SENTINELS = {
    "AGENTS.md": "AMBER-FERRY-31",
    "CLAUDE.md": "COPPER-HERON-58",
    "skill": "ORCHID-LANTERN-74",
}
PROMPT = (
    "Do not use any tools. Quote every codeword (a word like WORD-WORD-12) that "
    "appears anywhere in your instructions or in the descriptions of the skills "
    "available to you. If there is none, answer NONE."
)


@pytest.fixture
def sandbox(tmp_path):
    """A sandbox inside a git project that has instruction files and a skill."""
    project = tmp_path / "project"
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    for name in ("AGENTS.md", "CLAUDE.md"):
        (project / name).write_text(f"The codeword is {SENTINELS[name]}.\n")
    skill = project / ".agents" / "skills" / "codeword"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: codeword\n"
        f"description: Gives the codeword {SENTINELS['skill']}.\n---\n"
        f"The codeword is {SENTINELS['skill']}.\n"
    )
    (project / ".claude").mkdir()
    (project / ".claude" / "skills").symlink_to("../.agents/skills")
    box = project / "bigym-agent-runs" / "move_plate" / "sandbox"
    box.mkdir(parents=True)
    return box


def claude_answer(box, isolated):
    settings = claude_settings_soft(box)
    if not isolated:
        del settings["claudeMdExcludes"]
    (box / "claude_settings.json").write_text(json.dumps(settings))
    cmd = [
        "claude", "-p", PROMPT, "--setting-sources", "project",
        "--settings", str(box / "claude_settings.json"),
        "--permission-mode", "dontAsk", "--allowedTools", "Read,Glob,Grep",
        *(CLAUDE_HOST_FLAGS if isolated else []),
        "--model", "haiku", "--max-turns", "2",
    ]  # fmt: skip
    return ask(cmd, box)


def codex_answer(box, isolated):
    cmd = [
        "codex", "exec", "--skip-git-repo-check", "-C", str(box),
        "--sandbox", "read-only", *(CODEX_HOST_FLAGS if isolated else []), PROMPT,
    ]  # fmt: skip
    return ask(cmd, box)


def ask(cmd, box):
    done = subprocess.run(cmd, cwd=box, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip(), done.stderr
    return done.stdout


# What each harness reads from the project when nothing keeps it out:
# Claude Code loads CLAUDE.md under --setting-sources project, Codex AGENTS.md.
EXPOSED = {"claude": ("CLAUDE.md", "skill"), "codex": ("AGENTS.md", "skill")}


@pytest.mark.parametrize("harness", ["claude", "codex"])
def test_host_session_sees_nothing_from_the_project(sandbox, harness):
    if shutil.which(harness) is None:
        pytest.skip(f"{harness} is not installed")
    answer = {"claude": claude_answer, "codex": codex_answer}[harness]
    exposed = answer(sandbox, isolated=False)
    for source in EXPOSED[harness]:
        assert SENTINELS[source] in exposed, exposed
    isolated = answer(sandbox, isolated=True)
    assert "NONE" in isolated.upper(), isolated
    assert not any(word in isolated for word in SENTINELS.values()), isolated
