"""Docker for ``bigym-agent run``: the agent images, the networks and the proxy.

An agent container joins an internal docker network with no egress and reaches
its model endpoint through an allowlist HTTP proxy on that network. The proxy
alone also joins an egress network. ``run`` creates these on demand when the
defaults below are in use; ``docker/proxy/`` ships the proxy configuration and
the allowlists, and the Dockerfiles of the agent images.
"""

from __future__ import annotations

import grp
import os
import shlex
import subprocess
from pathlib import Path

# Defaults that a host may override; none of them name a particular machine.
DEFAULT_IMAGES = {
    "codex": "bigym-agent-codex",
    "claude": "bigym-agent-claude",
}
DEFAULT_NETWORK = "bigym-agent-internal"
DEFAULT_PROXY = "http://bigym-agent-proxy:8888"
CONTAINER_PREFIX = "bigym-agent-"
# The docker pieces `run` creates on demand when the defaults above are in
# use: the egress network only the proxy joins, the proxy container, and the
# shipped Dockerfiles/allowlist under bigym/loco/agent/docker.
EGRESS_NETWORK = "bigym-agent-egress"
PROXY_CONTAINER = "bigym-agent-proxy"
PROXY_IMAGE = "vimagick/tinyproxy"
DOCKER_DIR = Path(__file__).resolve().parent / "docker"


def docker_argv(argv: list[str]) -> list[str]:
    """Wrap a docker command so it runs with the docker group active.

    Args:
        argv: The docker command line.

    Returns:
        ``argv`` when the calling shell already has the docker group, else the
        same command under ``sg docker``.
    """
    try:
        active = "docker" in {grp.getgrgid(g).gr_name for g in os.getgroups()}
    except KeyError:
        active = False
    return argv if active else ["sg", "docker", "-c", shlex.join(argv)]


def run_docker(argv: list[str], run=None) -> subprocess.CompletedProcess:
    """Run a docker command, capturing its output (never raising on failure)."""
    run = run or subprocess.run
    return run(docker_argv(argv), capture_output=True, text=True)


def docker_has(kind: str, name: str, run=None) -> bool:
    """True when ``docker <kind> inspect <name>`` succeeds."""
    return run_docker(["docker", kind, "inspect", name], run).returncode == 0


def running_containers(run=None) -> set[str]:
    """List the names of the running containers.

    Returns:
        The container names, empty when docker cannot be reached.
    """
    try:
        done = run_docker(["docker", "ps", "--format", "{{.Names}}"], run)
    except OSError:
        return set()
    return {line.strip() for line in done.stdout.splitlines() if line.strip()}


def proxy_networks(run=None) -> set[str]:
    """The networks the proxy container is attached to (empty when absent)."""
    done = run_docker(
        [
            "docker",
            "inspect",
            PROXY_CONTAINER,
            "--format",
            "{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}",
        ],
        run,
    )
    return set(done.stdout.split()) if done.returncode == 0 else set()


def docker_setup_commands(
    harness: str,
    image: str | None,
    network: str,
    proxy: str,
    run=None,
) -> list[list[str]]:
    """The docker commands that bring the session's network, proxy and image up.

    Only the shipped defaults are created: a custom ``--docker-network``,
    ``--proxy`` or ``--container`` means the host runs its own, and nothing
    is touched. Everything that already exists is skipped, so the list is
    empty on a prepared host.

    Args:
        harness: The harness kind (names the Dockerfile).
        image: The harness image, or None for a soft session.
        network: The internal network the agent container joins.
        proxy: The proxy URL the container is given.
        run: ``subprocess.run`` replacement (tests).

    Returns:
        The commands, in order.
    """
    cmds: list[list[str]] = []
    if network == DEFAULT_NETWORK and not docker_has("network", network, run):
        cmds.append(["docker", "network", "create", "--internal", network])
    if proxy == DEFAULT_PROXY:
        if not docker_has("network", EGRESS_NETWORK, run):
            cmds.append(["docker", "network", "create", EGRESS_NETWORK])
        running = PROXY_CONTAINER in running_containers(run)
        exists = running or docker_has("container", PROXY_CONTAINER, run)
        if not running:
            if exists:
                cmds.append(["docker", "start", PROXY_CONTAINER])
            else:
                cmds.append(
                    [
                        "docker",
                        "run",
                        "-d",
                        "--name",
                        PROXY_CONTAINER,
                        "--restart",
                        "unless-stopped",
                        "--network",
                        EGRESS_NETWORK,
                        "-v",
                        f"{DOCKER_DIR / 'proxy' / 'tinyproxy.conf'}"
                        ":/etc/tinyproxy/tinyproxy.conf:ro",
                        "-v",
                        f"{DOCKER_DIR / 'proxy' / 'filter'}:/etc/tinyproxy/filter:ro",
                        PROXY_IMAGE,
                    ]
                )
        if not exists or network not in proxy_networks(run):
            cmds.append(["docker", "network", "connect", network, PROXY_CONTAINER])
    if image is not None and image == DEFAULT_IMAGES.get(harness):
        if not docker_has("image", image, run):
            cmds.append(
                [
                    "docker",
                    "build",
                    "-f",
                    str(DOCKER_DIR / f"Dockerfile.{harness}"),
                    "-t",
                    image,
                    str(DOCKER_DIR),
                ]
            )
    return cmds
