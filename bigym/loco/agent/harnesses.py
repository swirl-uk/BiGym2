"""The agent harnesses ``bigym-agent run`` launches, and their home directories.

Each shipped harness (Codex, Claude Code) runs in its own container on the
internal network, with a per-session home under the cell's ``raw/``, or on
the host (``--isolation soft``); ``--harness custom`` runs any shell command
there.

Credentials are never on a command line (the host process table is readable,
and agents do run ``ps``). An API key is used when one is set
(:data:`API_KEYS`: the environment variable, or a mode-600 file of that name
under :func:`agent_home`); otherwise the harness falls back to its
subscription login. In a container, Codex reads its API key from an
``auth.json`` written into a session-private directory under
:func:`agent_home` and bind-mounted read-only, or, on a subscription, the
ChatGPT-login ``auth.json`` from ``--codex-home``, bind-mounted writable;
Claude Code reads ``ANTHROPIC_API_KEY`` or the OAuth token
(``$CLAUDE_CODE_OAUTH_TOKEN`` or a mode-600
``~/.bigym-agent/claude_oauth_token``) from a mode-600 env file in that
directory. The directory is removed when the session ends, so neither lands
in the cell.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import containers
from .cli import RunConfig

# Per harness: whose models it runs, the harness for the other vendor, and the
# name prefixes (case-insensitive) of that vendor's models. A prefix check, not
# a model list.
FOREIGN_MODELS = {
    "codex": ("OpenAI", "claude", re.compile(r"claude|opus|sonnet|haiku", re.I)),
    "claude": ("Claude", "codex", re.compile(r"gpt|chatgpt|codex|o\d", re.I)),
}


def model_mismatch(harness: str, model: str) -> str | None:
    """Why ``model`` cannot be what ``harness`` runs, or None when it may be."""
    if harness not in FOREIGN_MODELS:
        return None
    runs, other, pattern = FOREIGN_MODELS[harness]
    if not pattern.match(model):
        return None
    return f"--harness {harness} runs {runs} models, not {model}; use --harness {other}"


# What the resumed Claude Code session is told instead of the prompt.
RESUME_MESSAGE = (
    "Your previous session was interrupted (rate limit or timeout); "
    "it has been resumed. The sandbox, your files and the environment "
    "budget are exactly as you left them. Continue."
)


def agent_home() -> Path:
    """Return the directory holding the launcher's credentials and caches.

    Returns:
        ``$BIGYM_AGENT_HOME`` or ``~/.bigym-agent``.
    """
    return Path(
        os.environ.get("BIGYM_AGENT_HOME", Path.home() / ".bigym-agent")
    ).expanduser()


def read_secret(env_var: str, path: Path) -> str | None:
    """Read a credential from the environment or from a mode-600 file."""
    value = os.environ.get(env_var)
    if value:
        return value.strip()
    if path.exists():
        return path.read_text().strip() or None
    return None


# Where each shipped harness finds an API key: the environment variable, or a
# mode-600 file of that name under agent_home().
API_KEYS = {
    "codex": ("OPENAI_API_KEY", "openai_api_key"),
    "claude": ("ANTHROPIC_API_KEY", "anthropic_api_key"),
}


def api_key(harness: str) -> str | None:
    """The harness's API key from its environment variable or key file, or None."""
    env_var, name = API_KEYS[harness]
    return read_secret(env_var, agent_home() / name)


def api_key_source(harness: str) -> str | None:
    """Where :func:`api_key` finds the key (``$VAR`` or the file), or None."""
    env_var, name = API_KEYS[harness]
    if os.environ.get(env_var):
        return f"${env_var}"
    path = agent_home() / name
    if path.exists() and path.read_text().strip():
        return str(path)
    return None


def auth_mode(harness: str) -> str | None:
    """``api_key`` or ``subscription`` for a shipped harness, None for custom."""
    if harness == "custom":
        return None
    return "api_key" if api_key_source(harness) else "subscription"


def auth_report(args: RunConfig) -> tuple[str, str | None]:
    """How the harness will authenticate, and a warning when it cannot.

    Returns:
        ``(line, warning)``: one line naming the credential the session uses
        and what to set for the other mode, and a warning or None.
    """
    harness = args.harness
    if harness == "custom":
        return "custom command: authenticates on its own", None
    env_var, name = API_KEYS[harness]
    key_file = agent_home() / name
    source = api_key_source(harness)
    if source:
        return f"{harness} auth: API key from {source}", None
    hint = f"set {env_var} (or write {key_file}, chmod 600) to use an API key"
    if harness == "codex":
        assert args.codex_home is not None
        line = f"codex auth: ChatGPT login in {args.codex_home / 'auth.json'}; {hint}"
        return line, codex_preflight(args.codex_home)
    if args.isolation != "container":
        return f"claude auth: this host's Claude Code login; {hint}", None
    token_file = agent_home() / "claude_oauth_token"
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        token = "$CLAUDE_CODE_OAUTH_TOKEN"
    elif token_file.exists():
        token = str(token_file)
    else:
        return "claude auth: none", (
            f"no Claude credentials: {hint}, or for a subscription run "
            f"`claude setup-token` and set CLAUDE_CODE_OAUTH_TOKEN (or save it "
            f"to {token_file}, chmod 600)"
        )
    return f"claude auth: OAuth token from {token}; {hint}", None


def session_secrets(name: str, write: bool) -> Path:
    """The session's private credential directory, outside the cell.

    Mode 700 under :func:`agent_home`; the launcher removes it when the
    session ends.

    Args:
        name: The container name, unique per session.
        write: False for a dry run (return the path, create nothing).
    """
    path = agent_home() / "sessions" / name
    if write:
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o700)
    return path


def write_secret_file(path: Path, text: str) -> None:
    """Write a credential file readable by its owner only."""
    path.touch(mode=0o600)
    path.chmod(0o600)
    path.write_text(text)


def codex_token_expiry(auth_json: Path) -> float | None:
    """The ``exp`` claim of the access token in a Codex ``auth.json``, or None.

    The CLI refreshes an expired access token by itself when the refresh
    token is still valid, so this is only a hint for the preflight message;
    an unreadable or non-JWT token gives None.
    """
    try:
        token = json.loads(auth_json.read_text())["tokens"]["access_token"]
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return float(json.loads(base64.urlsafe_b64decode(payload))["exp"])
    except (OSError, KeyError, IndexError, ValueError, TypeError):
        return None


def codex_preflight(shared: Path) -> str | None:
    """A one-line warning about the shared Codex credentials, or None."""
    auth = shared / "auth.json"
    if not auth.exists():
        return (
            f"no {auth}: run `CODEX_HOME={shared} codex login` on this host "
            "(or point --codex-home at a CODEX_HOME that is logged in)"
        )
    exp = codex_token_expiry(auth)
    if exp is not None and exp < time.time():
        when = time.strftime("%Y-%m-%d %H:%M", time.gmtime(exp))
        return (
            f"the Codex access token in {auth} expired at {when} UTC; the CLI "
            "will try to refresh it, and if the refresh token is stale too the "
            f"session ends at once (run `CODEX_HOME={shared} codex login`)"
        )
    return None


def codex_home(
    raw: Path,
    model: str,
    effort: str,
    shared: Path,
    write: bool,
    service_tier: str | None = None,
) -> Path:
    """Build the session's CODEX_HOME under ``raw/``.

    ``auth.json`` is not copied: the shared file is bind-mounted into every
    container, writable, because the CLI rotates its refresh token when it
    refreshes the access token and every session must see the rotated one
    (a stale copy fails with "refresh token was already used"). Sessions and
    rollouts land here, so an audit reads them from the cell.

    Args:
        raw: The cell's ``raw/`` directory the home is created under.
        model: The model name written into ``config.toml``.
        effort: The reasoning effort written into ``config.toml``.
        shared: The shared CODEX_HOME holding ``auth.json`` and the rules.
        write: False for a dry run (return the path, create nothing).
        service_tier: The account speed tier to request, or None for the
            default. ``"fast"`` is a routing priority on the same model, so it
            changes latency and nothing the benchmark records; it is opt-in
            per run so that two models being compared are not silently given
            different tiers (their speed-ups differ, e.g. 2x vs 1.5x).
    """
    home = raw / "codex_home"
    if not write:
        return home
    (home / "rules").mkdir(parents=True, exist_ok=True)
    for rule in sorted((shared / "rules").glob("*.rules")):
        shutil.copy(rule, home / "rules" / rule.name)
    tier = f'service_tier = "{service_tier}"\n' if service_tier else ""
    (home / "config.toml").write_text(
        f'model = "{model}"\nmodel_reasoning_effort = "{effort}"\n'
        f"{tier}"
        'approval_policy = "never"\n\n[projects."/work"]\ntrust_level = "trusted"\n'
    )
    return home


def claude_home(raw: Path, write: bool) -> Path:
    """Build the session's CLAUDE_CONFIG_DIR under ``raw/``.

    Only what Claude Code writes there (its transcript under ``projects/``,
    ``sessions/``) plus a pre-trusted ``/work``: without the trust flag Claude
    Code ignores the sandbox's permission rules ("this workspace has not been
    trusted") and then refuses calls the allow rules would have admitted. No
    credentials file: the token goes in through an env file.
    """
    home = raw / "claude_home"
    if not write:
        return home
    home.mkdir(parents=True, exist_ok=True)
    config = home / ".claude.json"
    data = {}
    if config.exists():
        try:
            data = json.loads(config.read_text())
        except json.JSONDecodeError:
            data = {}
    data["hasCompletedOnboarding"] = True
    data.setdefault("projects", {}).setdefault("/work", {})[
        "hasTrustDialogAccepted"
    ] = True
    config.write_text(json.dumps(data, indent=1))
    os.chmod(config, 0o600)
    return home


def resume_session_id(raw: Path) -> str | None:
    """Read the Claude session id out of an existing stream, for ``--resume``."""
    stream = raw / "claude_stream.jsonl"
    if not stream.exists():
        return None
    for line in stream.read_text(errors="replace").splitlines():
        try:
            session_id = json.loads(line).get("session_id")
        except json.JSONDecodeError:
            continue
        if session_id:
            return str(session_id)
    return None


def harness_version(
    args: RunConfig, image: str | None, dry_run: bool
) -> tuple[str, list[str]]:
    """Probe the agent CLI's version string; returns it with the probe command."""
    if args.harness == "custom":
        # An arbitrary command has no version to probe; the command itself is
        # what run.json records.
        return "", []
    binary = args.harness
    if image:
        probe = containers.docker_argv(
            ["docker", "run", "--rm", image, binary, "--version"]
        )
    else:
        probe = [binary, "--version"]
    if dry_run:
        return "", probe
    try:
        done = subprocess.run(probe, capture_output=True, text=True)
        return done.stdout.strip(), probe
    except OSError as exc:
        return f"{binary} --version failed: {exc}", probe


def container_user() -> str:
    """The uid:gid the container runs as, so its files belong to the caller."""
    return f"{os.getuid()}:{os.getgid()}"


def new_container_name(task: str, cell: Path, args: RunConfig) -> str:
    """Build a unique container name (root, task and the millisecond)."""
    if args.container_name:
        return args.container_name
    stem = f"{cell.parent.name}-{task}-{int(time.time() * 1000)}"
    return containers.CONTAINER_PREFIX + re.sub(r"[^A-Za-z0-9_.-]", "-", stem)


def sock_mount(sandbox: Path) -> list[str]:
    """Bind-mount the real socket directory at ``/work/sock`` when linked.

    ``serve`` moves the worker sockets to a short temp directory (unix socket
    paths are limited to 108 bytes) and leaves ``<sandbox>/sock`` as a symlink
    to it; a symlink into the host filesystem means nothing inside the
    container, so the resolved directory is mounted over ``/work/sock``.
    """
    link = Path(sandbox) / "sock"
    if not link.is_symlink():
        return []
    return ["-v", f"{link.resolve()}:/work/sock"]


@dataclass
class AgentCommand:
    """How to run the agent harness on a cell's sandbox."""

    cmd: list[str] | str
    """The argument vector, or the shell string of ``--harness custom``."""
    env: dict[str, str]
    """The environment the command runs in."""
    stdin: str | None
    """Text fed on stdin (the prompt), or None."""
    out: Path
    """Where its stdout goes."""
    err: Path
    """Where its stderr goes."""
    container: str | None = None
    """The container name, or None on the host."""
    image: str | None = None
    """The harness image, or None on the host."""
    version: str = ""
    """The harness's ``--version`` output (empty in a dry run)."""
    version_cmd: list[str] = field(default_factory=list)
    """The command that probes it."""
    resume: str | None = None
    """The Claude session resumed, or None."""
    auth: str | None = None
    """``api_key`` or ``subscription`` (:func:`auth_mode`), None for custom."""
    secrets: Path | None = None
    """The session-private credential directory to remove afterwards, or None."""

    @property
    def shell(self) -> bool:
        """Whether :attr:`cmd` is a shell string."""
        return isinstance(self.cmd, str)

    @property
    def command(self) -> str | None:
        """The shell command of ``--harness custom``, for ``run.json``."""
        return self.cmd if isinstance(self.cmd, str) else None


@dataclass(frozen=True)
class Session:
    """What a harness command line is built from."""

    args: RunConfig
    """The parsed ``run`` arguments."""
    cell: Path
    """The cell directory."""
    prompt: str
    """The task statement."""
    resume: str | None
    """The Claude session to resume, or None."""
    dry_run: bool
    """Build the command without writing homes or reading credentials."""

    @property
    def sandbox(self) -> Path:
        """``<cell>/sandbox``."""
        return self.cell / "sandbox"

    @property
    def raw(self) -> Path:
        """``<cell>/raw``."""
        return self.cell / "raw"

    def docker_run(
        self,
        name: str,
        image: str,
        options: list[str],
        command: list[str],
        interactive: bool = True,
    ) -> list[str]:
        """Build the ``docker run`` of a harness container.

        The container joins the internal network, reaches the model through
        the proxy, runs as the caller and is limited in memory and processes.

        Args:
            name: The container name.
            image: The harness image.
            options: The harness's own environment and mounts.
            command: The harness command inside the container.
            interactive: Keep stdin open (the prompt is fed on it).

        Returns:
            The command line.
        """
        args = self.args
        return containers.docker_argv(
            ["docker", "run", "--rm"]
            + (["-i"] if interactive else [])
            + ["--name", name, "--user", container_user()]
            + [
                "--network",
                args.docker_network,
                "-e",
                f"HTTPS_PROXY={args.proxy}",
                "-e",
                f"HTTP_PROXY={args.proxy}",
                "-e",
                "NO_PROXY=localhost,127.0.0.1",
            ]
            + options
            + ["--memory", args.memory, "--pids-limit", str(args.pids_limit)]
            + [image]
            + command
        )


def codex_in_container(session: Session, name: str, image: str) -> AgentCommand:
    """Codex in its container, the prompt on stdin."""
    args, sandbox, raw = session.args, session.sandbox, session.raw
    assert args.codex_home is not None and args.model is not None
    shared = args.codex_home
    write = not session.dry_run
    home = codex_home(raw, args.model, args.effort, shared, write, args.service_tier)
    secrets = None
    if auth_mode("codex") == "api_key":
        # The auth.json `codex login --with-api-key` writes, kept out of the
        # cell; read-only, since an API key is never refreshed.
        secrets = session_secrets(name, write)
        auth = secrets / "auth.json"
        key = api_key("codex") if write else None
        if key:
            write_secret_file(
                auth, json.dumps({"auth_mode": "apikey", "OPENAI_API_KEY": key})
            )
        auth_mount = f"{auth}:/codex_home/auth.json:ro"
    else:
        # the one host file every session shares; writable so the CLI's
        # token refresh (which rotates the refresh token) lands in the copy
        # the next session mounts
        auth_mount = f"{shared / 'auth.json'}:/codex_home/auth.json"
    cmd = session.docker_run(
        name,
        image,
        [
            "-e",
            "CODEX_HOME=/codex_home",
            "-e",
            "HOME=/home/agent",
            "-v",
            f"{sandbox}:/work",
            *sock_mount(sandbox),
            "-v",
            f"{home}:/codex_home",
            "-v",
            auth_mount,
            "-w",
            "/work",
        ],
        [
            "codex",
            "exec",
            "--skip-git-repo-check",
            "-C",
            "/work",
            "--dangerously-bypass-approvals-and-sandbox",
            "-m",
            args.model,
            "-c",
            f"model_reasoning_effort={args.effort}",
            "--json",
            "-o",
            "/codex_home/codex_last.txt",
        ],
    )
    # The prompt is fed on stdin: nothing from the host command line
    # reaches the container.
    return AgentCommand(
        cmd,
        dict(os.environ),
        session.prompt,
        raw / "codex_events.jsonl",
        raw / "codex_stderr.txt",
        secrets=secrets,
    )


def claude_in_container(session: Session, name: str, image: str) -> AgentCommand:
    """Claude Code in its container, the credential in a mode-600 env file."""
    args, sandbox, raw = session.args, session.sandbox, session.raw
    assert args.model is not None
    write = not session.dry_run
    home = claude_home(raw, write)
    secrets = session_secrets(name, write)
    env_file = secrets / "claude_env"
    if write:
        key = api_key("claude")
        if key:
            line = f"ANTHROPIC_API_KEY={key}"
        else:
            token = read_secret(
                "CLAUDE_CODE_OAUTH_TOKEN", agent_home() / "claude_oauth_token"
            )
            if not token:
                raise RuntimeError(auth_report(args)[1])
            line = f"CLAUDE_CODE_OAUTH_TOKEN={token}"
        # Never in an argv, and outside the cell and the sandbox.
        write_secret_file(env_file, line + "\n")
    resume = session.resume
    cmd = session.docker_run(
        name,
        image,
        [
            "--env-file",
            str(env_file),
            "-e",
            "CLAUDE_CONFIG_DIR=/claude_home",
            "-e",
            "HOME=/home/agent",
            "-e",
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY=1",
            "-e",
            "DISABLE_TELEMETRY=1",
            "-e",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1",
            "-v",
            f"{sandbox}:/work",
            *sock_mount(sandbox),
            "-v",
            f"{home}:/claude_home",
            "-w",
            "/work",
        ],
        [
            # Approvals are off inside the container, the mirror of Codex's
            # --dangerously-bypass-approvals-and-sandbox: the container is
            # the isolation. Left on, the harness refuses multi-line
            # `./python -c` calls that Codex runs freely, an asymmetry
            # unrelated to the task.
            "claude",
            "-p",
            "--setting-sources",
            "project",
            "--settings",
            "/work/claude_settings.json",
            "--dangerously-skip-permissions",
            "--model",
            args.model,
            "--max-turns",
            str(args.max_turns),
            "--output-format",
            "stream-json",
            "--verbose",
        ]
        + (["--resume", resume] if resume else []),
    )
    return AgentCommand(
        cmd,
        dict(os.environ),
        RESUME_MESSAGE if resume else session.prompt,
        raw / "claude_stream.jsonl",
        raw / "claude_stderr.txt",
        secrets=secrets,
    )


def codex_on_host(session: Session) -> AgentCommand:
    """Codex on this host, under the shared CODEX_HOME."""
    args, raw = session.args, session.raw
    assert args.model is not None
    cmd = [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "-C",
        str(session.sandbox),
        "--dangerously-bypass-approvals-and-sandbox",
        "-m",
        args.model,
        "-c",
        f"model_reasoning_effort={args.effort}",
        "--json",
        "-o",
        str(raw / "codex_last.txt"),
        session.prompt,
    ]
    env = dict(os.environ, CODEX_HOME=str(args.codex_home))
    key = None if session.dry_run else api_key("codex")
    if key:
        # `codex exec` prefers this to the login in CODEX_HOME.
        env["CODEX_API_KEY"] = key
    return AgentCommand(
        cmd, env, None, raw / "codex_events.jsonl", raw / "codex_stderr.txt"
    )


def claude_on_host(session: Session) -> AgentCommand:
    """Claude Code on this host, held to the sandbox's tools by its settings."""
    args, raw = session.args, session.raw
    assert args.model is not None
    resume = session.resume
    cmd = [
        "claude",
        "-p",
        session.prompt,
        "--setting-sources",
        "project",
        "--settings",
        str(session.sandbox / "claude_settings.json"),
        "--permission-mode",
        "dontAsk",
        "--allowedTools",
        "Read,Edit,Write,Bash,Glob,Grep",
        "--model",
        args.model,
        "--max-turns",
        str(args.max_turns),
        "--output-format",
        "stream-json",
        "--verbose",
    ] + (["--resume", resume] if resume else [])
    env = dict(os.environ, CLAUDE_CODE_DISABLE_AUTO_MEMORY="1")
    key = None if session.dry_run else api_key("claude")
    if key:
        env["ANTHROPIC_API_KEY"] = key
    return AgentCommand(
        cmd, env, None, raw / "claude_stream.jsonl", raw / "claude_stderr.txt"
    )


def custom_on_host(session: Session) -> AgentCommand:
    """Any shell command on this host, told where the session is."""
    args, sandbox, raw = session.args, session.sandbox, session.raw
    assert args.command is not None
    # Everything the command needs to find the session is in its
    # environment: the sandbox it works in, the prompt to read and the cell
    # around it. Nothing is passed on the command line, so the command can
    # be anything from a one-line script to another agent's CLI.
    env = dict(
        os.environ,
        AGENT_SANDBOX=str(sandbox),
        BIGYM_AGENT_PROMPT=str(sandbox / "PROMPT.md"),
        BIGYM_AGENT_CELL=str(session.cell),
    )
    return AgentCommand(
        args.command, env, None, raw / "custom_stdout.txt", raw / "custom_stderr.txt"
    )


IN_CONTAINER = {"codex": codex_in_container, "claude": claude_in_container}
ON_HOST = {"codex": codex_on_host, "claude": claude_on_host, "custom": custom_on_host}


def agent_command(
    task: str,
    cell: Path,
    args: RunConfig,
    *,
    dry_run: bool = False,
    container_name: str | None = None,
) -> AgentCommand:
    """Build the command that runs the agent harness on a cell's sandbox.

    Args:
        task: The task name.
        cell: The cell directory.
        args: The parsed ``run`` arguments.
        dry_run: Build the command without writing homes, env files or reading
            credentials.
        container_name: Use this container name instead of generating one (a
            session generates it once, before it claims the registry).

    Returns:
        The command.

    Raises:
        RuntimeError: A credential the harness needs is missing, or the harness
            kind cannot run at the requested isolation level.
    """
    sandbox = cell / "sandbox"
    raw = cell / "raw"
    if not dry_run:
        raw.mkdir(parents=True, exist_ok=True)
    prompt = (
        (sandbox / "PROMPT.md").read_text()
        if (sandbox / "PROMPT.md").exists()
        else "<sandbox/PROMPT.md>"
    )
    container = args.isolation == "container"
    if args.harness == "custom" and container:
        raise RuntimeError(
            "--harness custom runs your command on this host: use --isolation soft "
            "(the benchmark cannot know what your image holds, so it will not "
            "start a container for it)"
        )
    image = (
        (args.container or containers.DEFAULT_IMAGES[args.harness])
        if container
        else None
    )
    name = (
        (container_name or new_container_name(task, cell, args)) if container else None
    )
    version, version_cmd = harness_version(args, image, dry_run)
    resume = resume_session_id(raw) if args.resume else None
    if args.resume and args.harness != "claude":
        raise RuntimeError("--resume is implemented for --harness claude only")
    if args.resume and resume is None and not dry_run:
        raise RuntimeError(f"no session_id in {raw / 'claude_stream.jsonl'}")
    session = Session(args, cell, prompt, resume, dry_run)
    if image is not None and name is not None:
        command = IN_CONTAINER[args.harness](session, name, image)
    else:
        command = ON_HOST[args.harness](session)
    command.container, command.image = name, image
    command.version, command.version_cmd = version, version_cmd
    command.resume = resume
    command.auth = auth_mode(args.harness)
    return command
