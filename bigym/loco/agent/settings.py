"""Agent-harness settings written into a sandbox or an episode directory.

Two isolation levels are offered:

- :func:`claude_settings` is written for a session inside the agent container,
  which is the isolation; the harness's own sandbox is left off there.
- :func:`claude_settings_soft` is the fallback for hosts where that sandbox
  cannot start (unprivileged user namespaces disabled, for instance): tool-level
  rules only, so the transcript has to be audited afterwards.

Both are plain dictionaries serialised to JSON, so a caller can add or drop
entries before writing them.
"""

from __future__ import annotations

from pathlib import Path

# Commands that would fetch code, install packages or reach the network. Under the
# soft level these are tool-permission rules, which Bash can still work around;
# under the full sandbox the network is closed as well.
SOFT_BASH_DENY = [
    "Bash(curl:*)",
    "Bash(wget:*)",
    "Bash(pip:*)",
    "Bash(pip3:*)",
    "Bash(uv:*)",
    "Bash(git:*)",
    "Bash(ssh:*)",
    "Bash(scp:*)",
    "Bash(rsync:*)",
    "Bash(nc:*)",
    "Bash(ncat:*)",
    "Bash(socat:*)",
    "Bash(sudo:*)",
    "Bash(apt:*)",
    "Bash(apt-get:*)",
    "Bash(npm:*)",
    "Bash(npx:*)",
    "Bash(docker:*)",
]

# The shell the agent is left with: its own python wrapper plus read-only text tools.
SOFT_BASH_ALLOW = [
    "Bash(./python:*)", "Bash(/work/python:*)", "Bash(python:*)", "Bash(python3:*)",
    "Bash(./act:*)", "Bash(cd:*)", "Bash(cat:*)", "Bash(ls:*)", "Bash(sed:*)",
    "Bash(grep:*)", "Bash(awk:*)", "Bash(head:*)", "Bash(tail:*)", "Bash(wc:*)",
    "Bash(sort:*)", "Bash(uniq:*)", "Bash(echo:*)", "Bash(printf:*)", "Bash(mkdir:*)",
    "Bash(cp:*)", "Bash(mv:*)", "Bash(rm:*)", "Bash(touch:*)", "Bash(diff:*)",
    "Bash(find:*)", "Bash(pwd:*)", "Bash(date:*)", "Bash(true:*)", "Bash(test:*)",
    "Bash(tee:*)", "Bash(xargs:*)", "Bash(cut:*)", "Bash(tr:*)", "Bash(nl:*)",
    "Bash(stat:*)", "Bash(du:*)", "Bash(sleep:*)",
]  # fmt: skip

# Credential stores that stay unreadable whatever else the session is given.
DENY_READ_ALWAYS = ["~/.ssh", "~/.aws", "~/.kube", "~/.claude", "~/.codex"]


def claude_settings_soft(sandbox: Path, effort: str = "high") -> dict:
    """Tool-level isolation only, for hosts where the full sandbox cannot start.

    Bash network use is NOT enforced at this level; audit the transcript.

    Args:
        sandbox: The sandbox directory the session runs in.
        effort: Reasoning effort level.

    Returns:
        The settings dictionary.
    """
    del sandbox  # the rules are relative to the working directory
    return {
        "permissions": {
            "blockReadsOutsideWorkingDirectories": True,
            "deny": ["WebFetch", "WebSearch", "Agent"] + SOFT_BASH_DENY,
            "allow": list(SOFT_BASH_ALLOW),
            "additionalDirectories": [],
        },
        "disableClaudeAiConnectors": True,
        "effortLevel": effort,
        "autoCompactEnabled": True,
        "autoMemoryEnabled": False,
    }


def claude_settings(
    sandbox: Path,
    extra_read: list[str] | None = None,
    deny_read: list[str] | None = None,
    effort: str = "high",
) -> dict:
    """Settings for a session in the agent container: the sandbox and nothing else.

    Args:
        sandbox: The sandbox directory the session runs in (the only writable
            path, and with ``extra_read`` the only readable one).
        extra_read: Further readable paths (demo videos, for instance).
        deny_read: Paths to deny explicitly on top of
            :data:`DENY_READ_ALWAYS` (the benchmark's own sources, notes and
            results, when they live under a readable parent).
        effort: Reasoning effort level.

    Returns:
        The settings dictionary.
    """
    return {
        # Off: these settings are only written for the container, which is the
        # isolation (mounts, internal network, allowlist proxy), as it is for
        # Codex under --dangerously-bypass-approvals-and-sandbox. The harness
        # sandbox uses bubblewrap, which cannot create a namespace in an
        # unprivileged container, so every command would fail. The paths below
        # record what the container exposes.
        "sandbox": {
            "enabled": False,
            "failIfUnavailable": False,
            "filesystem": {
                "allowRead": [str(sandbox)] + list(extra_read or []),
                "allowWrite": [str(sandbox)],
                "denyRead": list(deny_read or []) + list(DENY_READ_ALWAYS),
                "denyWrite": ["/"],
            },
            "network": {
                "allowedDomains": [],
                "strictAllowlist": True,
                "allowUnixSockets": [str(Path(sandbox) / "sock" / "*")],
            },
        },
        "permissions": {
            "blockReadsOutsideWorkingDirectories": True,
            "deny": ["WebFetch", "WebSearch", "Agent"],
            "additionalDirectories": [],
        },
        "disableClaudeAiConnectors": True,
        "effortLevel": effort,
        "autoCompactEnabled": True,
        "autoMemoryEnabled": False,
    }
