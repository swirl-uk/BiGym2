"""``bigym-agent``: the coding-agent benchmark's command line.

Every subcommand is a dataclass of its settings in this module, whose
``run()`` imports the module that does the work and calls it. The import
waits until the subcommand runs, so a missing optional dependency only
breaks the subcommand that needs it. Each of those modules also takes a
plain argument list, ``main(argv)``, which it parses with
:func:`parse_command`. ``python -m bigym.loco.agent`` is the same command as
``bigym-agent``.

Rendering needs ``MUJOCO_GL=egl`` in the environment the command starts in:
``bigym`` imports MuJoCo, which picks its GL backend on import, before any
code here runs.

The settings of the environment a policy runs against are one dataclass,
:class:`EnvToolsConfig`, nested whole in every subcommand that builds that
environment; its flags (``--interface``, ``--slew``, ...) carry no prefix.

Prompt and document templates live in ``templates/``. The environment API
document ships in two renderings, one per interface, and the sandbox builder
copies the one the session runs:

- ``templates/api_tools.md`` — the ``tools`` interface: named state fields,
  wrist positions, ``camera_info``/``pixel_to_ray`` and the IK solver.
- ``templates/api_strict.md`` — the ``strict`` interface (the default), whose
  policy signs the same I/O contract as the learned baselines: the three
  84x84 cameras plus the raw ``low_dim_obs`` vector in, the physical action
  out, and none of the tools above.

``api_strict.md`` documents the ``low_dim_obs`` layout, which differs between
the 20-dim and the 21-dim (torso-pitch) action layouts, so it carries both
tables fenced by ``<!-- layout: 20 -->`` / ``<!-- layout: 21 -->`` comments;
the sandbox builder keeps the one matching the task's layout
(:func:`bigym.loco.agent.envtools.task_pitch_enabled`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Literal,
    Sequence,
    TypeVar,
    Union,
    cast,
    get_args,
)

import tyro

from bigym.loco.eval.protocol import EVAL_EPISODES, EVAL_SEED_BASE

DESCRIPTION = (
    "The coding-agent benchmark: an agent writes policy.py against a sandboxed "
    "BiGym environment, and the policy is scored on hidden seeds."
)

DEFAULT_ROOT = Path("bigym-agent-runs")
"""Where runs go when ``--root`` is not given: a directory in the current
working directory, so a benchmark's sessions live next to the project that
runs them (and one ``.gitignore`` line keeps them out of its repository),
not in a hidden global directory. Each cell is hundreds of megabytes
(episodes and videos), so keep it out of synced folders."""

Interface = Literal["strict", "tools"]
Tier = Literal["images", "privileged"]
Harness = Literal["codex", "claude", "custom"]
Demo = Literal["video", "files", "none"]


def parse_size(spec: str) -> tuple[int, int]:
    """Parse a ``WxH`` resolution.

    Args:
        spec: Resolution such as ``84x84``.

    Returns:
        ``(width, height)``.

    Raises:
        ValueError: The specification is not ``WxH`` with positive integers.
    """
    width, _, height = str(spec).lower().partition("x")
    try:
        size = (int(width), int(height))
    except ValueError:
        raise ValueError(
            f"resolution {spec!r} is not WxH (for instance 84x84)"
        ) from None
    if size[0] < 1 or size[1] < 1:
        raise ValueError(f"resolution {spec!r} must be positive")
    return size


# BLAS/OpenMP thread pools only spin and fight next to a running simulator.
BLAS_THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def limit_blas_threads() -> None:
    """Give the interpreters this process starts one BLAS/OpenMP thread.

    This interpreter loaded its BLAS when ``bigym`` imported numpy, so the
    setting reaches the ones started afresh, such as the policy process. A
    value the caller set wins.
    """
    for name in BLAS_THREAD_VARIABLES:
        os.environ.setdefault(name, "1")


def default_eval_jobs() -> int:
    """Evaluation processes: ``$BIGYM_AGENT_EVAL_JOBS``, else 3."""
    return int(os.environ.get("BIGYM_AGENT_EVAL_JOBS", "3"))


@dataclass(frozen=True)
class EnvToolsConfig:
    """The environment a policy is developed and scored against.

    A policy tuned against one observation must not be scored against
    another, so the sandbox builder, the server, the evaluation and the
    replays of a cell all take these settings whole. The sandbox builder
    records them in the cell's ``sandbox_config.json`` (under ``env_tools``),
    and a cell's evaluation and replays read them back from there.
    """

    interface: Interface = "strict"
    """strict: the learned baselines' I/O contract - the three onboard cameras
    plus the raw low_dim_obs vector in, the physical action out; no named
    state, no wrist positions, no camera calibration / pixel_to_ray, no IK.
    tools: all of those on top (a separate, labelled row)."""
    tier: Tier = "images"
    """images (the benchmark setting): no object positions in the observation.
    privileged: object state in the observation, for debugging on the tasks
    that implement it."""
    image_cap: str = "84x84"
    """Policy-facing resolution ceiling WxH (84x84 is exactly what the learned
    baselines are rendered at); larger requests are refused."""
    slew: bool = False
    """Enable the command rate limiter (arm joints --joint-vmax, base 0.7/s,
    arm acceleration --accel). Off by default: the learned baselines'
    commands are not rate-limited either."""
    joint_vmax: float = 6.0
    """With --slew: arm joint velocity cap (rad/s)."""
    accel: float | None = 0.03
    """With --slew: arm acceleration cap (rad/step^2); None caps the velocity
    only."""
    lowpass: float | None = None
    """With --slew: first-order low-pass (alpha per step) on the arm targets
    before the caps."""
    ik: bool = True
    """With --interface tools: offer tools.ik (--no-ik withholds it only)."""
    hand_pos: bool = True
    """With --interface tools: offer left_hand_pos/right_hand_pos
    (--no-hand-pos withholds them only)."""
    pitch: bool = True
    """Use the task's action layout (--no-pitch forces the 20-dim layout, only
    to replay a submission written against it)."""
    allow_external_cameras: bool = False
    """Replay-only: the policy may also use the free third_person/front
    cameras."""

    def __post_init__(self) -> None:
        """Reject settings no environment can be built with.

        Raises:
            ValueError: Unknown interface or tier, or a malformed image cap.
        """
        if self.interface not in get_args(Interface):
            raise ValueError(f"interface must be one of {get_args(Interface)}")
        if self.tier not in get_args(Tier):
            raise ValueError(f"tier must be one of {get_args(Tier)}")
        parse_size(self.image_cap)

    @property
    def image_size(self) -> tuple[int, int]:
        """The image cap as ``(width, height)``."""
        return parse_size(self.image_cap)


@dataclass
class SandboxConfig:
    """Build the sandbox directory for one task and one session."""

    task: str
    """Task name."""
    cell: Path
    """The cell directory (one task x one session); the sandbox is written to
    <cell>/sandbox."""
    budget_steps: int = 101_000
    """Environment steps the session may spend in total."""
    workers: int = 3
    """Environment workers the server runs."""
    env_tools: tyro.conf.OmitArgPrefixes[EnvToolsConfig] = field(
        default_factory=EnvToolsConfig
    )
    """The environment the session runs against: the documents are written
    for it and the cell records it for the evaluation. Must match what the
    server runs."""
    harness: Harness = "claude"
    """Which harness will run in this sandbox (recorded in the configuration;
    the harness settings are written either way)."""
    isolation: Literal["soft", "container"] = "soft"
    """soft: the agent runs on this host under tool-level rules, and ./python
    is this interpreter. container: the agent runs inside the benchmark image
    with the sandbox mounted, and ./python is the image's interpreter."""
    client_root: str = "/work"
    """Where the sandbox is mounted inside the container."""
    effort: str = "high"
    """Reasoning effort written into the harness settings."""
    demo: Demo = "video"
    """video: onboard-camera mp4 files of a human demonstration. files: the
    demonstration set as stripped npz. none: no demonstrations (the zero-shot
    condition)."""
    demo_episodes: int = 1
    """With --demo video: how many demonstrations to concatenate."""
    demo_episode: str = "median"
    """With --demo video: 'median' or an episode index."""
    demo_max_files: int = -1
    """With --demo files: cap the number of episodes written (-1: all)."""
    force: bool = False
    """Overwrite a non-empty sandbox."""

    def run(self) -> int | None:
        """Build the sandbox."""
        from bigym.loco.agent import sandbox

        return sandbox.main(self)


@dataclass
class DemoVideoConfig:
    """Render human demonstrations of a task to mp4."""

    task: str
    """Task name."""
    out: Path
    """Output directory."""
    size: str = "84x84"
    """Onboard-camera resolution WxH (84x84 is what the learned baselines see).
    A demonstration rendered above the policy-facing cap would let an agent
    measure the scene offline at a precision its own runtime images cannot
    reach."""
    episode: str = "median"
    """'median' (the demonstration of median length) or an episode index of
    the dataset export."""
    episodes: int = 1
    """How many demonstrations to concatenate."""
    views: str = "head,left_wrist,right_wrist"
    """Comma list of head, left_wrist, right_wrist (the robot's own cameras)
    and third (an outside view for human readers only; the sandbox builder
    never hands it to an agent)."""
    every: int = 2
    """Control steps per frame."""
    fps: int = 25
    """Frame rate of the mp4."""
    crf: str = "24"
    """x264 quality; 0 is lossless."""
    pix_fmt: str = "yuv420p"
    """Pixel format; yuv444p keeps full chroma (use with --crf 0 for a
    bit-exact copy of the observation)."""
    prefix: str = "demo_"
    """Output file name prefix."""

    def run(self) -> int | None:
        """Render the demonstrations."""
        from bigym.loco.agent import demo_video

        return demo_video.main(self)


@dataclass
class ServeConfig:
    """Serve the benchmark environment to an agent sandbox."""

    task: str
    """Task name."""
    sandbox: Path
    """The sandbox directory the agent works in."""
    ledger_dir: Path
    """Where the budget ledger, policy versions and development recordings go
    (the cell)."""
    workers: int = 4
    """Environment worker processes."""
    budget_steps: int = 101_000
    """Environment steps the session may spend in total."""
    record_dir: Path | None = None
    """Record every episode to mp4 under this dir."""
    env_tools: tyro.conf.OmitArgPrefixes[EnvToolsConfig] = field(
        default_factory=EnvToolsConfig
    )
    """The environment every worker holds."""
    reset_cost: int = 200
    """Steps charged per reset (the controller warm-up); 0 = free resets."""
    allowed_seeds: Path | None = None
    """JSON list of the development seeds (the sandbox builder writes
    <ledger>/allowed_seeds.json: the seeds stored in the task's
    demonstrations). Overrides --train-seeds."""
    train_seeds: int = 60
    """Development seeds are [0, N); the published demos cover [0, 60)."""
    client_root: str | None = None
    """Container mode: the path at which the client sees the sandbox (e.g.
    /work); render paths are mapped."""
    policies_dir: Path | None = None
    """Where the policy version snapshots go (default <ledger-dir>/policies)."""
    snapshots: bool = True
    """Record the agent's policy versions."""
    dev_recording: bool = True
    """Record the agent's own episodes as replay batches under
    <ledger-dir>/dev (about 1 MB per episode)."""
    allow_eval_seeds: bool = False
    """In-the-loop evaluation: the model IS the policy, episodes run on the
    protocol seeds."""

    def run(self) -> int | None:
        """Serve until stopped."""
        from bigym.loco.agent import server

        return server.main(self)


@dataclass
class EvaluateConfig:
    """Score a policy.py on the hidden protocol seeds."""

    cell: tyro.conf.Positional[Path | None] = None
    """A run directory (one task x one session): scores one of its recorded
    policy versions into <cell>/eval/vNNN/."""
    version: int | None = None
    """With a cell: the policy version to score (policies/index.json)."""
    force: bool = False
    """With a cell: replace a finished evaluation of that version."""
    task: str | None = None
    """Task name; required without a cell."""
    policy: Path | None = None
    """The policy file; with a cell, an alternative to --version."""
    out: Path | None = None
    """Output directory; required without a cell."""
    episodes: int = EVAL_EPISODES
    """Episodes in the block."""
    seed_offset: int = 0
    """Added to every seed of the block."""
    repeat: int = 1
    """Run the full block this many times (identity check)."""
    spot_check: int = 10
    """After the block, rerun the first N episodes and require identical
    per-episode success (0 disables)."""
    label: str = "scripted"
    """Label recorded in the summary."""
    env_tools: tyro.conf.OmitArgPrefixes[EnvToolsConfig] = field(
        default_factory=EnvToolsConfig
    )
    """The environment the policy is scored in. Without a cell only: a cell
    is scored in the environment its session ran against, as recorded in its
    sandbox_config.json."""
    video: bool = True
    """With a cell: record the success/failure videos."""
    video_dir: Path | None = None
    """After the block: record one success and one failure episode (two of the
    same kind if the other does not exist) to this directory."""
    video_prefix: str = ""
    """File name prefix for --video-dir."""
    video_views: str | None = None
    """Comma pair of recording views for --video-dir; default per task."""
    seed_base: int = EVAL_SEED_BASE
    """First seed; 620000 = hidden protocol block, 0 = the agent's own
    development seeds (in-distribution score)."""
    jobs: int = field(default_factory=default_eval_jobs)
    """Episodes split over this many processes (~2 GB RAM each); records are
    merged in seed order. Default: $BIGYM_AGENT_EVAL_JOBS, else 3."""

    def run(self) -> int | None:
        """Score the policy."""
        from bigym.loco.agent import evaluate

        return evaluate.main(self)


@dataclass
class RecordConfig:
    """Record a policy.py on given seeds to mp4 (our views, not policy-facing)."""

    task: str
    """Task name."""
    policy: Path
    """The policy file."""
    seeds: str
    """Comma-separated seeds, e.g. 620000,620042."""
    out: Path
    """Output directory."""
    prefix: str = ""
    """File name prefix."""
    env_tools: tyro.conf.OmitArgPrefixes[EnvToolsConfig] = field(
        default_factory=EnvToolsConfig
    )
    """The environment the policy runs in."""
    views: str | None = None
    """Comma pair of recording views, e.g. both_tables,rec_third; default is
    per-task (recording only, not policy-facing)."""

    def run(self) -> int | None:
        """Record the seeds."""
        from bigym.loco.agent import evaluate

        return evaluate.main_record(self)


@dataclass
class ReplayConfig:
    """Replay a policy version on given seeds into a viewable batch."""

    cell: tyro.conf.Positional[Path]
    """The run directory of one task x session."""
    version: int | None = None
    """Policy version from policies/index.json."""
    policy: Path | None = None
    """An arbitrary policy.py instead."""
    seeds: str = f"{EVAL_SEED_BASE}-{EVAL_SEED_BASE + 4}"
    """Comma list and/or inclusive a-b ranges."""
    video: bool = True
    """Render the videos (--no-video writes the batch only)."""
    force: bool = False
    """Re-record every seed asked for, replacing the episodes an existing
    replay of this version already holds (without it, seeds already in the
    batch are kept and only the missing ones run)."""
    max_steps: int | None = None
    """Stop each episode after this many steps (short smoke runs)."""
    task: str | None = None
    """Override the cell's task name."""

    def run(self) -> int | None:
        """Replay the policy version."""
        from bigym.loco.agent import replay

        return replay.main(self)


@dataclass
class PoliciesConfig:
    """List the policy versions a session produced."""

    cell: tyro.conf.Positional[Path]
    """The run directory of one task x session."""
    rebuild: bool = False
    """Recompute the index's derived fields (training outcomes from the
    ledger, the helper modules the policy imports, the provenance of each
    version) before printing; the version directories are not touched."""

    def run(self) -> int | None:
        """Print the policy versions."""
        from bigym.loco.agent import snapshots

        return snapshots.main(self)


@dataclass
class InloopConfig:
    """Run the in-the-loop evaluation.

    The model itself acts, one session per episode.
    """

    task: str
    """Task name."""
    root: Path
    """Output root: one directory per episode plus the ledger."""
    episodes: int = 100
    """Episodes in the block."""
    first_episode: int = 0
    """Index of the first episode to run (to resume a block)."""
    seed_offset: int = 0
    """Added to every protocol seed."""
    workers: int = 8
    """Episodes run at once."""
    model: str = "claude-opus-5"
    """Model the harness runs."""
    harness: Literal["claude", "codex"] = "claude"
    """The harness that runs the model."""
    smooth: bool = False
    """Ramp commanded targets at the demo-collection slew limits."""
    effort: str = "medium"
    """codex: model_reasoning_effort."""
    max_commands: int = 30
    """Robot commands per episode."""
    max_turns: int = 80
    """Agent turns per episode."""
    episode_timeout_s: int = 1800
    """Wall-clock limit per episode, in seconds."""
    label: str = "inloop"
    """Label recorded in the results."""
    task_doc: Path | None = None
    """Markdown file describing the task, copied into each episode as task.md;
    without it the model gets the task name only."""
    tier: Tier = "images"
    """privileged: object positions in the observation; images: cameras
    only."""

    def run(self) -> int | None:
        """Run the in-the-loop block."""
        from bigym.loco.agent.inloop import run_inloop

        return run_inloop.main(self)


@dataclass
class RunConfig:
    """Run one coding-agent session per task and score the submission."""

    task: list[str] = field(default_factory=list)
    """Tasks to run, one session each (--task a b c)."""
    root: Path = DEFAULT_ROOT
    """Runs root; cells are <root>/<task>/."""
    harness: Harness = "codex"
    """The harness to run; custom runs --command instead."""
    model: str | None = None
    """Model the harness runs (required except for --harness custom)."""
    command: str | None = None
    """--harness custom: the shell command to run as the agent, from the
    sandbox directory, with AGENT_SANDBOX / BIGYM_AGENT_PROMPT /
    BIGYM_AGENT_CELL in its environment."""
    effort: str = "high"
    """Reasoning effort the harness asks for."""
    service_tier: str | None = None
    """codex account speed tier (e.g. fast); default: the account's own. A
    routing priority on the same model: it changes latency, not any recorded
    metric. Opt-in, because tiers speed models up unequally."""
    env_tools: tyro.conf.OmitArgPrefixes[EnvToolsConfig] = field(
        default_factory=EnvToolsConfig
    )
    """The environment the session runs against; the sandbox, the server and
    the evaluation all get it."""
    budget_steps: int = 101_000
    """Environment steps the session may spend in total."""
    reset_cost: int = 200
    """Steps charged per reset."""
    workers: int = 3
    """Environment workers the server runs."""
    demo: Demo = "video"
    """What demonstration the sandbox gets."""
    demo_episodes: int = 1
    """With --demo video: how many demonstrations to concatenate."""
    spot_check: int = 10
    """Episodes the evaluation reruns to check they are identical."""
    eval_episodes: int = EVAL_EPISODES
    """Hidden seeds the submission is scored on (the benchmark's own figure is
    100; fewer only for a smoke test)."""
    eval_jobs: int = field(default_factory=default_eval_jobs)
    """Evaluation processes. Default: $BIGYM_AGENT_EVAL_JOBS, else 3."""
    isolation: Literal["container", "soft"] | None = None
    """container (the default; soft for --harness custom): the agent runs in
    its image with only the sandbox mounted; soft: on the host, with the
    harness's own tool rules (audit the transcript)."""
    container: str | None = None
    """Harness image; default per harness: codex bigym-agent-codex, claude
    bigym-agent-claude."""
    container_name: str | None = None
    """Explicit container name."""
    docker_network: str = field(
        default_factory=lambda: os.environ.get(
            "BIGYM_AGENT_DOCKER_NETWORK", "bigym-agent-internal"
        )
    )
    """Internal docker network (no egress) the agent container joins. Default:
    $BIGYM_AGENT_DOCKER_NETWORK, else bigym-agent-internal."""
    proxy: str = field(
        default_factory=lambda: os.environ.get(
            "BIGYM_AGENT_PROXY", "http://bigym-agent-proxy:8888"
        )
    )
    """Allowlist HTTP proxy on that network (the model endpoint only).
    Default: $BIGYM_AGENT_PROXY, else http://bigym-agent-proxy:8888."""
    docker_setup: bool = True
    """Create the networks, start the proxy and build the image when they are
    missing (--no-docker-setup leaves them alone)."""
    memory: str = "8g"
    """Container memory limit."""
    pids_limit: int = 1024
    """Container process limit."""
    max_turns: int = 800
    """Claude Code --max-turns."""
    codex_home: Path | None = None
    """Shared CODEX_HOME holding the ChatGPT-login auth.json, used when no
    OpenAI API key is set (default $BIGYM_AGENT_CODEX_HOME or
    ~/.bigym-agent/codex_home)."""
    session_timeout_s: int = 3 * 3600
    """Wall-clock limit of a session, in seconds."""
    server_start_timeout_s: int = 900
    """How long to wait for the environment server, in seconds."""
    gpu: int | None = None
    """GPU to run the session on (default: the one with the most free
    memory)."""
    min_free_mib: int = 6000
    """GPU memory a session needs free before it starts."""
    parallel: int = 1
    """Sessions to run at once."""
    resume: bool = False
    """claude only: continue an interrupted session in the same conversation
    (same sandbox, budget and transcript)."""
    force: bool = False
    """Overwrite a scored cell."""
    eval: bool = True
    """Score the submission (--no-eval skips it)."""
    eval_detached: bool = False
    """Launch the evaluation in the background instead of waiting for it."""
    dry_run: bool = False
    """Print every command that would run and write nothing."""

    def run(self) -> int | None:
        """Run the sessions."""
        from bigym.loco.agent import launch

        return launch.main(self)


@dataclass
class StatusConfig:
    """List the registry's runs with their budget and evaluation progress."""

    root: Path = DEFAULT_ROOT
    """Runs root."""
    all: bool = False
    """Finished runs too."""

    def run(self) -> int | None:
        """Print the runs."""
        from bigym.loco.agent import registry

        return registry.main_status(self)


@dataclass
class ProxyConfig:
    """Inspect, start or remove the egress allowlist proxy.

    Every container session goes through it; `run` brings it up on demand.
    """

    action: tyro.conf.Positional[Literal["status", "up", "down"]]
    """What to do with the proxy."""
    harness: Literal["codex", "claude"] | None = None
    """With `up`: also build this harness's image if missing."""

    def run(self) -> int | None:
        """Act on the proxy."""
        from bigym.loco.agent import launch

        return launch.main_proxy(self)


@dataclass
class KillConfig:
    """Stop runs by registry id or by task name."""

    targets: tyro.conf.Positional[list[str]]
    """Registry ids and/or task names."""
    root: Path = DEFAULT_ROOT
    """Runs root."""

    def run(self) -> int | None:
        """Stop the runs."""
        from bigym.loco.agent import registry

        return registry.main_kill(self)


@dataclass
class GcConfig:
    """Close dead registry rows and list (or remove) orphan containers."""

    root: Path = DEFAULT_ROOT
    """Runs root."""
    remove_orphans: bool = False
    """Remove the orphan containers instead of listing them."""

    def run(self) -> int | None:
        """Collect the dead runs."""
        from bigym.loco.agent import registry

        return registry.main_gc(self)


@dataclass
class ReportConfig:
    """Rebuild a cell's unified transcript from its raw event stream."""

    cell: tyro.conf.Positional[Path]
    """The run cell <root>/<task>/."""
    model: str | None = None
    """Model name for the price table (default: the one in run.json)."""

    def run(self) -> int | None:
        """Rebuild the transcript."""
        from bigym.loco.agent import transcript

        return transcript.main(self)


@dataclass
class AuditConfig:
    """Flag tool calls that name anything outside the sandbox."""

    roots: tyro.conf.Positional[list[Path]]
    """Cells, runs roots or both."""
    client_root: str = "/work"
    """Path the sandbox is mounted at inside the container."""
    show: int = 10
    """Flagged calls to print per session."""

    def run(self) -> int | None:
        """Audit the sessions."""
        from bigym.loco.agent import audit

        return audit.main(self)


@dataclass
class UsageConfig:
    """Token usage and API-equivalent cost of every session under a root."""

    roots: tyro.conf.Positional[list[Path]]
    """Cells, runs roots or both."""
    csv: Path | None = None
    """Also write this CSV file."""

    def run(self) -> int | None:
        """Print the usage table."""
        from bigym.loco.agent import usage

        return usage.main(self)


CONFIGS: dict[str, type] = {
    "sandbox": SandboxConfig,
    "demo-video": DemoVideoConfig,
    "serve": ServeConfig,
    "evaluate": EvaluateConfig,
    "record": RecordConfig,
    "replay": ReplayConfig,
    "policies": PoliciesConfig,
    "inloop": InloopConfig,
    "run": RunConfig,
    "status": StatusConfig,
    "proxy": ProxyConfig,
    "kill": KillConfig,
    "gc": GcConfig,
    "report": ReportConfig,
    "audit": AuditConfig,
    "usage": UsageConfig,
}
"""Subcommand name -> its settings."""

# The subcommands as tyro reads them: a Union of the settings classes, each
# annotated with its name.
Command: Any = Union.__getitem__(
    tuple(
        Annotated[config, tyro.conf.subcommand(name)]  # ty: ignore[invalid-type-form]
        for name, config in CONFIGS.items()
    )
)

C = TypeVar("C")


def parse_command(config: type[C], argv: Sequence[str] | C | None) -> C:
    """One subcommand's settings, from ``bigym-agent``'s dispatch or argv.

    Args:
        config: The subcommand's settings class.
        argv: Settings already parsed by :func:`main`, or command line
            arguments of the subcommand alone (None for ``sys.argv[1:]``).

    Returns:
        The settings.
    """
    if isinstance(argv, config):
        return argv
    name = next(n for n, c in CONFIGS.items() if c is config)
    return tyro.cli(
        config, args=cast("Sequence[str] | None", argv), prog=f"bigym-agent {name}"
    )


def main(argv: list[str] | None = None) -> int:
    """Parse a subcommand and run it.

    Args:
        argv: Command line arguments, or None for ``sys.argv[1:]``.

    Returns:
        The process exit status (0 unless the subcommand returns one).
    """
    args = tyro.cli(Command, args=argv, prog="bigym-agent", description=DESCRIPTION)
    return args.run() or 0
