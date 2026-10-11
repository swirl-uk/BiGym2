"""The slots of a compare view and the stored episode each one plays."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from bigym.loco.agent.snapshots import version_dir
from bigym.vr.viewer.agent.runs import (
    NO_EVAL,
    NO_TRAIN,
    TEMPLATE_MARK,
    VERSION_LAST,
    Cell,
    as_float,
    file_digest,
    find_cell,
    is_template_version,
    percent,
    policy_rows,
    session_prefix,
    sessions_of,
    submission_version,
)
from bigym.vr.viewer.episodes import episode_files, episode_info
from bigym.vr.viewer.layout import COMPARE_GAP, COMPARE_SLOTS, LAYOUT_AUTO, grid_offsets

SLOT_EMPTY = "(empty)"


@dataclass
class CompareSlot:
    """One environment of the compare scene."""

    session: str
    task: str
    cell: Path
    version: int | None = None
    policy: str = ""
    episode: Path | None = None
    source: str = ""
    train_success: float | None = None
    eval_success: float | None = None
    submission: bool = False
    template: bool = False
    prefix: str = ""

    @property
    def name(self) -> str:
        """The session as the labels spell it, with the shared prefix gone."""
        text = str(self.session)
        if self.prefix and text.startswith(self.prefix) and text != self.prefix:
            return text[len(self.prefix) :]
        return text

    @property
    def version_name(self) -> str:
        """``v007`` for a recorded version, ``policy_<sha8>`` for a file."""
        if self.policy:
            return f"policy_{file_digest(Path(self.policy))[:8]}"
        return "—" if self.version is None else f"v{self.version:03d}"

    @property
    def label(self) -> str:
        """The floating label above this slot's pelvis.

        ``s2 · v007 · submission · train 40% · eval 2%`` — who ran, which
        policy, and the two scores the panels already show, so the scene
        alone answers "which of these is the good one". The session drops
        the prefix every slot shares, a v001 that is still the untouched
        template says so, and a score that was never measured is named
        (``no dev episodes``, ``not evaluated``) rather than dashed out.
        """
        version = self.version_name
        if self.template:
            version = f"{version} {TEMPLATE_MARK}"
        parts = [self.name, version]
        if self.submission:
            parts.append("submission")
        train = percent(self.train_success)
        parts.append(f"train {train}" if train else NO_TRAIN)
        rate = percent(self.eval_success)
        parts.append(f"eval {rate}" if rate else NO_EVAL)
        return " · ".join(parts)


@dataclass
class CompareRequest:
    """What the Compare button (or ``--compare``) asked for."""

    slots: list[CompareSlot]
    seed: int
    gap: float = COMPARE_GAP
    layout: str = LAYOUT_AUTO
    prefix: str = ""
    back_to: int = 0
    error: str = ""
    task: str = ""
    missing: list[CompareSlot] = field(default_factory=list)

    def offsets(self) -> list[tuple[float, float, float]]:
        """Where each slot's scene is placed, in slot order."""
        return grid_offsets(len(self.slots), self.gap, self.layout)


def seed_episode(
    cell: Path, version: int | None, seed: int, policy: str = ""
) -> Path | None:
    """The stored episode of one version on one seed, when there is one.

    Evaluation batches hold every hidden seed (620000-620099), so comparing
    two submissions on a hidden seed needs no new rollout at all; replays
    and development batches are searched after them.

    Args:
        cell: The cell directory.
        version: The policy version, or None for ``policy``.
        seed: The seed to find.
        policy: A loose policy file, whose batches live under
            ``replays/policy_<sha8>/``.

    Returns:
        The ``.npz`` path, or None when nothing has rolled this seed out.
    """
    cell = Path(cell)
    candidates: list[Path] = []
    if policy:
        digest = file_digest(Path(policy))[:8]
        candidates.append(cell / "replays" / f"policy_{digest}" / "batch")
    elif version is not None:
        candidates.append(version_dir(cell, version, "eval") / "batch")
        candidates.append(version_dir(cell, version, "replays") / "batch")
        candidates += sorted(cell.glob(f"dev/v{version:03d}_*/batch"))
    for batch in candidates:
        for path in episode_files(batch):
            if episode_info(path)["seed"] == int(seed):
                return path.resolve()
    return None


def episode_source(cell: Path, episode: Path) -> str:
    """``eval/v007`` — where an episode file came from, for the log."""
    try:
        rel = Path(episode).resolve().relative_to(Path(cell).resolve())
    except ValueError:
        return str(episode)
    return str(rel.parent.parent)


def fill_slot(slot: CompareSlot, seed: int) -> CompareSlot:
    """Attach the stored episode, the version's scores and its provenance."""
    rows = policy_rows(slot.cell)
    if slot.version is not None:
        for row in rows:
            if row["version"] == slot.version:
                slot.train_success = as_float(row.get("train_success"))
                slot.eval_success = as_float(row.get("eval_success"))
                slot.template = is_template_version(slot.cell, row)
        slot.submission = slot.version == submission_version(slot.cell, rows)
    slot.episode = seed_episode(slot.cell, slot.version, seed, slot.policy)
    slot.source = episode_source(slot.cell, slot.episode) if slot.episode else ""
    return slot


def resolve_version(cell: Path, text: str) -> int | None:
    """Turn ``v008`` / ``8`` / ``last`` / ``submission`` into a version number."""
    rows = policy_rows(cell)
    name = str(text or "").strip().lower()
    if not name or name in (VERSION_LAST, "submission"):
        return submission_version(cell, rows)
    digits = name[1:] if name.startswith("v") else name
    if digits.isdigit():
        return int(digits)
    return None


def parse_compare(spec: str) -> list[tuple[str, str]]:
    """Parse ``--compare "s1:v008,s2:v007,s3:last"`` into (session, version).

    Args:
        spec: A comma-separated list of ``<session>:<version>`` slots; the
            version may be ``vNNN``, ``NNN``, ``last`` or ``submission``.

    Returns:
        One ``(session, version)`` pair per slot, in the order given.

    Raises:
        ValueError: when a slot is not ``session:version``.
    """
    slots = []
    for chunk in str(spec).split(","):
        text = chunk.strip()
        if not text:
            continue
        if ":" not in text:
            raise ValueError(
                f"--compare slot {text!r} is not '<session>:<version>' "
                "(versions: vNNN, NNN, last, submission)"
            )
        session, version = text.split(":", 1)
        slots.append((session.strip(), version.strip()))
    if not slots:
        raise ValueError("--compare needs at least one '<session>:<version>' slot")
    return slots[:COMPARE_SLOTS]


def match_session(sessions: list[str], text: str) -> str | None:
    """Resolve a session name the user typed: exact, then unique substring."""
    name = str(text).strip()
    if name in sessions:
        return name
    matches = [s for s in sessions if name and name in s]
    return matches[0] if len(matches) == 1 else None


def resolve_compare(
    cells: list[Cell], task: str, specs: list[tuple[str, str]], seed: int
) -> CompareRequest:
    """Turn ``--compare`` / the panel's choices into a filled request.

    Args:
        cells: Every discovered cell.
        task: The task to compare on (compare is one task at a time).
        specs: ``(session, version)`` pairs.
        seed: The seed every slot is posed on.

    Returns:
        The request; ``error`` is set (and ``slots`` may be short) when a
        session or version could not be resolved, and ``missing`` lists the
        slots whose episode has to be produced by ``bigym-agent replay``.
    """
    sessions = sessions_of(cells, task)
    slots: list[CompareSlot] = []
    problems: list[str] = []
    for session_text, version_text in specs:
        session = match_session(sessions, session_text)
        if session is None:
            problems.append(
                f"no session matches {session_text!r} for task {task} "
                f"(have: {', '.join(sessions) or 'none'})"
            )
            continue
        cell = find_cell(cells, task, session)
        if cell is None:
            problems.append(f"session {session} has no cell for task {task}")
            continue
        version = resolve_version(cell.path, version_text)
        if version is None:
            problems.append(f"{session}: no version matches {version_text!r}")
            continue
        slots.append(
            fill_slot(
                CompareSlot(
                    session=session, task=task, cell=cell.path, version=version
                ),
                seed,
            )
        )
    prefix = session_prefix([slot.session for slot in slots])
    for slot in slots:
        slot.prefix = prefix
    return CompareRequest(
        slots=slots,
        seed=int(seed),
        task=task,
        prefix=prefix,
        error="; ".join(problems),
        missing=[s for s in slots if s.episode is None],
    )
