"""Collection session: XR frame loop, attempt lifecycle, HUD, metadata."""

from __future__ import annotations

import datetime
import getpass
import json
import os
import shutil
import socket
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bigym.loco.demos.schema import BATCH_FORMAT
from bigym.loco.demos.writer import ReplayBufferStorage
from bigym.loco.env import make
from bigym.loco.eval.protocol import is_success
from bigym.loco.fingerprint import bigym_git_sha, package_versions
from bigym.loco.timestep import ExtendedTimeStepWrapper
from bigym.vr.collect.config import CollectConfig, collection_env_config
from bigym.vr.collect.constants import (
    DEMO_PIPELINE_VERSION,
    G1_LOWERBODY_BACKENDS,
    GROOT_WBC_MIN_WALK_SPEED,
    GROOT_WBC_POLICY_WALK_THRESHOLD,
)
from bigym.vr.collect.mapper import (
    VRActionMapper,
    planar_space_transform,
    set_posef_position,
    wrap_angle,
)
from bigym.vr.collect.recording import (
    episode_data_specs,
    episode_from_time_steps,
    set_collector_action_stats,
)
from bigym.vr.viewer import Side
from bigym.vr.viewer.spectator import make_spectators


@dataclass
class CollectorStats:
    """Live counters for the HUD and the batch collection metadata."""

    recording: bool = False
    saved_successes: int = 0
    rejected_black_rgb: int = 0
    attempted: int = 0
    # Discard accounting. `attempted - saved_successes` is the
    # authoritative discard count; these four only break it down for the data
    # card and are best-effort (a pending success left unresolved when the
    # operator takes the headset off belongs to none of them).
    discarded_failed_attempts: int = 0
    discarded_pending_successes: int = 0
    restarted_mid_attempt: int = 0
    save_not_stored: int = 0
    episode_steps: int = 0
    episode_reward: float = 0.0
    last_reward: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    yaw: float = 0.0
    height: float = 0.0
    rgb: str = "?"


def _utc_now_iso() -> str:
    """ISO-8601 UTC timestamp, second resolution (``2026-09-02T09:15:31Z``)."""
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def resolve_operator(explicit: str | None = None) -> str:
    """Who ran this session: ``--operator`` > ``$BIGYM_OPERATOR`` > login user."""
    for candidate in (explicit, os.environ.get("BIGYM_OPERATOR")):
        if candidate is not None and str(candidate).strip():
            return str(candidate).strip()
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


def _success_rate(kept: int, attempts: int) -> float:
    return float(kept) / float(attempts) if attempts > 0 else 0.0


def _previous_sessions(previous: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Session entries already recorded in a batch dir's `collection` block."""
    if not isinstance(previous, dict):
        return []
    sessions = previous.get("sessions")
    if isinstance(sessions, list):
        return [dict(entry) for entry in sessions if isinstance(entry, dict)]
    # A block written by hand or by a future/older shape: keep its totals as
    # one anonymous session rather than dropping them.
    if "attempts" in previous or "kept" in previous:
        attempts = int(previous.get("attempts", 0) or 0)
        kept = int(previous.get("kept", 0) or 0)
        return [
            {
                "session_id": previous.get("session_id"),
                "started_at": previous.get("started_at"),
                "finished_at": previous.get("finished_at"),
                "operator": previous.get("operator"),
                "host": previous.get("host"),
                "bigym_git_sha": previous.get("bigym_git_sha"),
                "attempts": attempts,
                "kept": kept,
                "discarded": int(
                    previous.get("discarded", max(attempts - kept, 0)) or 0
                ),
                "success_rate_human": _success_rate(kept, attempts),
            }
        ]
    return []


def build_collection_block(
    stats: CollectorStats,
    *,
    session_id: str,
    started_at: str,
    operator: str,
    host: str,
    finished_at: str | None = None,
    bigym_git_sha: str | None = None,
    seed_start: int | None = None,
    unaccounted_episodes: int = 0,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build metadata.json's `collection` block from live collector stats.

    Pure function (no IO, no clock) so the writer is testable without VR.

    Top-level ``attempts``/``kept``/``discarded``/``success_rate_human`` are
    CUMULATIVE over every accounted session that wrote into the batch dir
    (``sessions`` carries one entry each); the identity fields (``session_id``,
    ``operator``, ``host``, ``started_at``, ``finished_at``, ``bigym_git_sha``)
    describe the session that wrote the file last. For the ordinary
    one-session batch the two readings coincide.

    ``previous`` is the ``collection`` block found in an existing
    metadata.json in the same directory (a resumed batch), or None. Re-writing
    within one session is idempotent: an entry with the same ``session_id`` is
    replaced, never appended twice.
    """
    attempts = int(stats.attempted)
    kept = int(stats.saved_successes)
    entry = {
        "session_id": str(session_id),
        "started_at": started_at,
        "finished_at": finished_at,
        "operator": operator,
        "host": host,
        "bigym_git_sha": bigym_git_sha,
        "attempts": attempts,
        "kept": kept,
        "discarded": max(attempts - kept, 0),
        "success_rate_human": _success_rate(kept, attempts),
        # Episode seeds are config.seed + attempt_index - 1, so the span is the
        # cross-check for the attempt counter.
        "seed_first": None if seed_start is None or attempts == 0 else int(seed_start),
        "seed_last": (
            None
            if seed_start is None or attempts == 0
            else int(seed_start) + attempts - 1
        ),
        "discard_breakdown": {
            "failed_attempts": int(stats.discarded_failed_attempts),
            "operator_discarded_successes": int(stats.discarded_pending_successes),
            "restarted_mid_attempt": int(stats.restarted_mid_attempt),
            "rejected_black_rgb": int(stats.rejected_black_rgb),
            "not_stored": int(stats.save_not_stored),
        },
    }
    sessions = [
        s
        for s in _previous_sessions(previous)
        if s.get("session_id") != entry["session_id"]
    ]
    sessions.append(entry)
    total_attempts = sum(int(s.get("attempts", 0) or 0) for s in sessions)
    total_kept = sum(int(s.get("kept", 0) or 0) for s in sessions)
    total_discarded = sum(
        int(
            s.get(
                "discarded",
                max(int(s.get("attempts", 0) or 0) - int(s.get("kept", 0) or 0), 0),
            )
            or 0
        )
        for s in sessions
    )
    return {
        "attempts": total_attempts,
        "kept": total_kept,
        "discarded": total_discarded,
        "success_rate_human": _success_rate(total_kept, total_attempts),
        "session_id": entry["session_id"],
        "started_at": entry["started_at"],
        "finished_at": entry["finished_at"],
        "operator": entry["operator"],
        "host": entry["host"],
        "bigym_git_sha": entry["bigym_git_sha"],
        # npz files already in the dir when the first accounted session
        # started (a batch collected before this block existed, resumed).
        # Files on disk = kept + unaccounted_episodes.
        "unaccounted_episodes": int(unaccounted_episodes),
        "sessions": sessions,
    }


class CollectorSession:
    """One VR collection session."""

    RESOLUTIONS = {
        "lq": (540, 600),
        "mq": (900, 1000),
        "hq": (1440, 1600),
    }

    def __init__(self, config: CollectConfig, out_dir: Path) -> None:
        """Build the env, demo storage and VR action mapper for one session."""
        self.config = config
        self.env_config, training = collection_env_config(config)
        self.training_hold_seconds = float(training.success_hold_seconds)
        controller = self.env_config.controller
        assert controller is not None
        self.controller = controller
        self.robot_model = str(self.env_config.robot_model)
        self.backend = str(controller.backend)
        if self.backend not in G1_LOWERBODY_BACKENDS:
            raise ValueError(
                f"unsupported lower-body backend {self.backend!r}; expected one "
                f"of {G1_LOWERBODY_BACKENDS}"
            )
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.env: ExtendedTimeStepWrapper = make(config.task, self.env_config)
        set_collector_action_stats(self.env, config)
        self.data_specs = episode_data_specs(
            self.env, store_event_progress=bool(config.store_event_progress)
        )
        self.storage = ReplayBufferStorage(
            self.data_specs,
            self.out_dir,
            use_relabeling=True,
            is_demo_buffer=True,
            # Keep the VR loop live through episode saves; run() flushes on
            # exit so no demo is left unwritten.
            async_write=True,
        )
        self.mapper = VRActionMapper(
            self.env,
            robot_model=self.robot_model,
            lowerbody_backend=self.backend,
            yaw_mode=config.yaw_mode,
            base_vx_scale=config.base_vx_scale,
            base_vy_scale=config.base_vy_scale,
            base_wz_scale=config.base_wz_scale,
            base_z_scale=config.base_z_scale,
            height_cmd=float(controller.default_height_cmd),
            height_cmd_max=self._height_cmd_max(),
            pitch_rate=float(config.pitch_rate),
            stick_deadzone=config.stick_deadzone,
            base_cmd_slew=config.base_cmd_slew,
        )

        # True sim seconds per control step from the compiled model (robot
        # XMLs can override the scene timestep; the env keeps control_frequency
        # true by adjusting substeps).
        self.control_dt = float(self.env.control_step_seconds)
        # Deferred reset warmup: run inline, the ~200-step settle would freeze
        # the XR frame loop ~1 s every attempt. It drains a few control steps
        # per XR frame in _step_for_frame; engage is blocked until it
        # completes, so demos still start from the post-settle state.
        self.env.set_deferred_reset_warmup(True)
        inner_env = self.env.inner_env
        print(
            f"[collector] success_hold_seconds="
            f"{float(self.env_config.success_hold_seconds):g} -> "
            f"{int(inner_env.success_hold_steps)} steps @ "
            f"{1.0 / self.control_dt:.0f} Hz; episode cap "
            f"{int(self.env_config.episode_length or 0)} physics steps"
        )
        self.stats = CollectorStats()
        self._current_steps: list[Any] = []
        self._current_fullbody_steps: list[dict[str, np.ndarray]] = []
        # Wall-clock origin for the per-step `wall_clock` field. The sim time
        # axis is k * control_dt by construction, so this exists only to expose
        # capture-rate jitter the uniform grid hides: XR frame stalls and the
        # catch-up substeps in _step_for_frame.
        self._wall_clock_t0 = time.perf_counter()
        self._pending: dict[str, Any] | None = None
        self._pre_engage_steps = 0
        self._frame_steps = 0
        self._engage_state: dict[str, Any] | None = None
        self._current_seed = int(config.seed)
        self._time_step = self.env.reset(seed=int(config.seed))
        self._sim_accumulator = 0.0
        self._space_offset = None
        self._space_ready = False
        self._space_yaw_offset = 0.0
        self._space_yaw_at_recenter = 0.0
        self._hmd_position_at_recenter: np.ndarray | None = None
        self._robot_yaw_at_recenter: float | None = None
        self._renderer = None
        self._printed_rgb_diag = False

        # Collection accounting: metadata.json records how many
        # attempts the kept demos cost. Identity is fixed here, at session
        # start; the counters are refreshed into the file at every attempt
        # event and finalized when run() returns.
        self._session_id = str(uuid.uuid4())
        self._session_started_at = _utc_now_iso()
        self._session_finished_at: str | None = None
        self._operator = resolve_operator(config.operator)
        self._host = socket.gethostname()
        self._bigym_git_sha = bigym_git_sha()
        self._metadata_base: dict[str, Any] | None = None
        self._previous_collection = self._read_previous_collection()
        if self._previous_collection is not None:
            self._unaccounted_episodes = int(
                self._previous_collection.get("unaccounted_episodes", 0) or 0
            )
            print(
                f"[collection] resuming batch dir; prior sessions: "
                f"{len(_previous_sessions(self._previous_collection))}, "
                f"prior attempts: {self._previous_collection.get('attempts')}"
            )
        else:
            # Episodes already on disk that no session entry accounts for
            # (a batch collected before this block existed).
            self._unaccounted_episodes = len(list(self.out_dir.glob("*.npz")))

        self._write_metadata()

    def run(self) -> None:
        """Run the session and always leave final counters in metadata.json."""
        try:
            self.run_xr_loop()
        finally:
            self.finalize_metadata()

    def run_xr_loop(self) -> None:
        """Run headset frames until enough successes are saved or the session ends."""
        import xr
        from xr import Posef

        from bigym.vr.viewer.vr_mujoco_renderer import VRMujocoRenderer
        from bigym.vr.viewer.xr_context import XRContextObject

        inner = self.env.inner_env
        width_one_eye, height = self.RESOLUTIONS[self.config.resolution]
        width = width_one_eye * 2
        self._space_offset = Posef()
        inner.model.vis.global_.offwidth = width
        inner.model.vis.global_.offheight = height
        self._renderer = VRMujocoRenderer(inner.model, inner.data, height, width)
        # First-person comfort: hide the robot's own head shell from the eye
        # render (the anchor is the head camera, so a few cm of operator
        # drift otherwise puts the eyes inside the mesh). G1 head_link;
        # harmless no-op on robots without that mesh. Recorded rgb_obs is
        # rendered elsewhere and unaffected.
        hidden = self._renderer.hide_first_person_meshes({"head_link"})
        if hidden:
            print(f"[vr] first-person: hid {hidden} head-shell geom(s)")

        def spectator_status() -> dict[str, Any]:
            return {
                "saved / target": f"{self.stats.saved_successes} / "
                f"{int(self.config.episodes)}",
                "attempted": self.stats.attempted,
                "recording": self.stats.recording,
                "episode steps": self.stats.episode_steps,
                "episode reward": f"{self.stats.episode_reward:.3f}",
                "base cmd": f"vx={self.stats.vx:+.2f} vy={self.stats.vy:+.2f} "
                f"yaw={self.stats.yaw:+.2f}",
                "height cmd": f"{self.stats.height:.2f}",
                "rgb": self.stats.rgb,
            }

        spectators = make_spectators(
            inner,
            self.config.spectator,
            hz=float(self.config.spectator_hz),
            status_fn=spectator_status,
            viser_port=int(self.config.spectator_port),
            viser_egl_device_id=(
                None
                if self.config.spectator_gpu is None
                else int(self.config.spectator_gpu)
            ),
        )

        self._print_controls()
        with XRContextObject(
            instance_create_info=xr.InstanceCreateInfo(
                enabled_extension_names=[
                    xr.KHR_OPENGL_ENABLE_EXTENSION_NAME  # ty: ignore[unresolved-attribute]
                ],
            ),
        ) as context:
            self._renderer.set_context(context)
            # Stutter localization: time each pipeline phase per frame; the
            # xr frame_loop wait (vsync/compositor) is deliberately outside
            # "work" but shows up in loop_dt. Spikes go to frame_timing.jsonl.
            spike_s = float(self.config.frame_spike_ms) / 1000.0
            timing_f = (
                (self.out_dir / "frame_timing.jsonl").open("a", encoding="utf-8")
                if spike_s > 0
                else None
            )
            frame_i = 0
            prev_start = None
            work_hist: list[float] = []
            for frame_state in context.frame_loop():
                t0 = time.perf_counter()
                if self._space_ready:
                    self._update_vr_space_follow_robot()
                t1 = time.perf_counter()
                self._handle_input(context)
                t2 = time.perf_counter()
                action = self.mapper.get_action(
                    context, self._space_offset, self._space_yaw_offset
                )
                t3 = time.perf_counter()
                self._step_for_frame(frame_state, action)
                t4 = time.perf_counter()
                self._render(frame_state)
                t4b = time.perf_counter()
                spectators.sync()
                t5 = time.perf_counter()
                if timing_f is not None:
                    work = t5 - t0
                    work_hist.append(work)
                    loop_dt = None if prev_start is None else t0 - prev_start
                    prev_start = t0
                    frame_i += 1
                    if work > spike_s or (loop_dt is not None and loop_dt > 0.040):
                        timing_f.write(
                            json.dumps(
                                {
                                    "frame": frame_i,
                                    "loop_dt_ms": None
                                    if loop_dt is None
                                    else round(loop_dt * 1e3, 2),
                                    "work_ms": round(work * 1e3, 2),
                                    "space_ms": round((t1 - t0) * 1e3, 2),
                                    "input_ms": round((t2 - t1) * 1e3, 2),
                                    "ik_ms": round((t3 - t2) * 1e3, 2),
                                    "sim_ms": round((t4 - t3) * 1e3, 2),
                                    "render_ms": round((t4b - t4) * 1e3, 2),
                                    "spect_ms": round((t5 - t4b) * 1e3, 2),
                                    # Control steps this frame, planar cmd
                                    # speed, and settle-drain flag.
                                    "steps": int(self._frame_steps),
                                    "cmd": round(
                                        (self.stats.vx**2 + self.stats.vy**2) ** 0.5,
                                        3,
                                    ),
                                    "settle": bool(self.env.pending_reset_warmup_steps),
                                }
                            )
                            + "\n"
                        )
                        timing_f.flush()
                    if frame_i % 2000 == 0:
                        w = np.asarray(work_hist[-2000:]) * 1e3
                        print(
                            f"[timing] frames {frame_i - 1999}-{frame_i}: work "
                            f"p50={np.percentile(w, 50):.1f}ms "
                            f"p95={np.percentile(w, 95):.1f}ms "
                            f"max={w.max():.1f}ms "
                            f"spikes>{self.config.frame_spike_ms:g}ms: "
                            f"{int((w > self.config.frame_spike_ms).sum())}",
                            flush=True,
                        )
                if not self._printed_rgb_diag:
                    rgb = self._time_step.rgb_obs
                    if rgb is not None:
                        mx = int(np.asarray(rgb).max())
                        print(
                            f"[diag] MUJOCO_GL={os.environ.get('MUJOCO_GL')} | "
                            f"first-frame rgb_obs.max()={mx} -> camera "
                            f"{'OK' if mx > 0 else 'BLACK (dead this session)'}",
                            flush=True,
                        )
                        self._printed_rgb_diag = True
                if self.stats.saved_successes >= int(self.config.episodes):
                    break
            if timing_f is not None:
                timing_f.close()
        spectators.close()
        print("[save] flushing pending episode writes ...", flush=True)
        self.storage.flush()
        print("[save] all episodes on disk.", flush=True)
        self.env.close()
        self._remove_session_dir_if_empty()

    def _remove_session_dir_if_empty(self) -> None:
        """Drop the session dir when the run saved nothing.

        Debug/look-around sessions leave behind a dir whose complete
        metadata.json makes it indistinguishable from a real batch. Deleting
        is double-guarded: the success counter must be zero AND the dir must
        contain no episode npz files.
        """
        if self.config.keep_empty_session:
            return
        if self.stats.saved_successes > 0:
            return
        if any(self.out_dir.glob("*.npz")):
            print(
                f"[cleanup] saved_successes=0 but npz files exist in "
                f"{self.out_dir}; keeping the directory."
            )
            return
        shutil.rmtree(self.out_dir)
        print(
            f"[cleanup] removed empty session dir (no successful demos): "
            f"{self.out_dir}  (--keep-empty-session overrides)"
        )

    def _height_cmd_max(self) -> float | None:
        """Collector-side height cap; None when disabled (<= 0)."""
        value = float(self.config.height_cmd_max)
        return None if value <= 0 else value

    def _reset_semantics_metadata(self) -> dict[str, Any]:
        controller = self.controller
        init_stance = str(controller.init_stance)
        # The G1 backends spawn in their keyframe only; no per-checkpoint
        # "parked" stance has been measured for them.
        if init_stance != "keyframe":
            raise ValueError(
                f"controller.init_stance={init_stance!r} is not supported by "
                "the G1 lower-body backends (only 'keyframe')"
            )
        warmup_steps = int(controller.reset_warmup_steps or 0)
        outer = self.env.bigym
        return {
            "init_keyframe": "g1_groot_wbc_default_angles",
            # "keyframe" = spawn in the training keyframe. Launch validation
            # aborts on demo/launch mismatch.
            "init_stance": init_stance,
            "init_stance_qpos": None,
            "init_pelvis_z": float(outer.lowerbody.config.init_pelvis_z),
            "default_height_cmd": float(controller.default_height_cmd),
            "reset_warmup_steps": warmup_steps,
            "reset_warmup_seconds": float(warmup_steps * self.control_dt),
            "reset_warmup_source": "official",
            "demo_start": "post_warmup_vr_engage_state",
            "engage_snapshot_fields": (
                "init_qpos, init_qvel, init_ctrl, init_qacc_warmstart, lb_state.*"
            ),
            "pre_engage_steps_saved_per_episode": True,
        }

    def _print_controls(self) -> None:
        backend_label = str(self.backend).upper()
        print(f"\n{backend_label} VR collector controls")
        print(
            "  Quest left X: start/restart attempt; align current HMD "
            "position + heading"
        )
        print(
            "  Quest left Y: finish current attempt (successful attempts go to pending)"
        )
        print(
            "  After a success: right B saves it, left Y discards it, "
            "left X saves & starts next"
        )
        print("  Quest right A: realign current HMD position + heading to robot head")
        print(
            "  Quest right B: engage hand tracking (arms pinned at hold "
            "pose until pressed)"
        )
        print(f"  left stick Y/X: {backend_label} vx/vy")
        if self.backend in G1_LOWERBODY_BACKENDS:
            print(
                "  right stick Y: height (up = stand, down = squat, "
                f"cap {self._height_cmd_max()} m); X: turn - one axis at a time"
            )
            if self.mapper.pitch_slot:
                print(
                    "  HOLD left grip: PITCH mode - left stick Y leans the torso "
                    "(up = forward), walking off while held; release keeps the lean"
                )
        if self.backend == "groot_wbc_g1":
            print(
                "  left stick walk entry: nonzero planar speed >= "
                f"{GROOT_WBC_MIN_WALK_SPEED:.3f} m/s"
            )
        print(
            "  right stick Y: pelvis height (up=stand, down=squat), always live; "
            "the stick acts on one axis at a time (turn OR height)"
        )
        if self.config.yaw_mode == "base":
            print(f"  right stick X: {backend_label} wz")
        else:
            print("  right stick X: ignored")
        if self.robot_model == "g1_dex1":
            print("  Dex1-1 grippers: triggers = left/right gripper open-close")
        print(f"  output: {self.out_dir}\n")

    def _handle_input(self, context: Any) -> None:
        left = context.input.state[Side.LEFT]
        right = context.input.state[Side.RIGHT]
        if right.a_clicked:
            self._recenter_vr_space(context, reason="manual")
            self.mapper.reanchor_hands()
            right.vibration = True
        if self._pending is not None:
            if right.b_clicked:
                self._save_pending()
                right.vibration = True
            elif left.b_clicked:
                self._discard_pending()
                left.vibration = True
            elif left.a_clicked:
                # Default to keeping the demo when jumping straight to the next attempt.
                self._save_pending()
                self._start_attempt(context)
                left.vibration = True
            return
        if left.a_clicked:
            self._start_attempt(context)
            left.vibration = True
        if left.b_clicked:
            self._finish_attempt(force=True)
            left.vibration = True
        if (
            right.b_clicked
            and not self.mapper.engaged
            and self.env.pending_reset_warmup_steps == 0
        ):
            # Engage during the settle drain is ignored (no vibration): the
            # demo must start from the post-settle state.
            self.mapper.engaged = True
            if self.stats.recording:
                # The demo's first frame is this engage-moment state, not the
                # reset state; snapshot it so replay-from-seed can restore it.
                # qpos/qvel alone leave the lower-body controller (obs
                # histories, last actions, rate-limiter anchors) at its
                # post-warmup state instead of the engage state, which is
                # enough divergence to flip some replays; snapshot the full
                # control state as well.
                outer = self.env.bigym
                inner = outer.inner_env
                data = inner.data
                engage_state: dict[str, Any] = {
                    "qpos": np.array(data.qpos, dtype=np.float64),
                    "qvel": np.array(data.qvel, dtype=np.float64),
                    "ctrl": np.array(data.ctrl, dtype=np.float64),
                    "qacc_warmstart": np.array(data.qacc_warmstart, dtype=np.float64),
                    "lowerbody_state": outer.get_lowerbody_state(),
                }
                if np.asarray(data.act).size:
                    engage_state["act"] = np.array(data.act, dtype=np.float64)
                self._engage_state = engage_state
            right.vibration = True
            print("[vr] hand tracking engaged")

    def _start_attempt(self, context: Any | None = None) -> None:
        if self.stats.recording:
            # Left X during a live attempt abandons it: no episode is written
            # and _finish_attempt never runs, so count it here.
            self.stats.restarted_mid_attempt += 1
        self.stats.attempted += 1
        seed = int(self.config.seed) + self.stats.attempted - 1
        self._current_seed = seed
        self._time_step = self.env.reset(seed=seed)
        self.mapper.reset()
        if context is None:
            self._space_ready = False
        else:
            self._recenter_vr_space(context, reason=f"attempt reset seed={seed}")
        self._current_steps = [self._time_step]
        self._current_fullbody_steps = [
            self._capture_fullbody_record(include_action_info=False)
        ]
        self._pre_engage_steps = 0
        self._engage_state = None
        self.stats.recording = True
        self.stats.episode_steps = 0
        self.stats.episode_reward = 0.0
        self.stats.last_reward = 0.0
        self._sim_accumulator = 0.0
        print(
            f"[record] started attempt={self.stats.attempted} seed={seed} "
            "(arms pinned; press right B to engage hand tracking)"
        )
        self._touch_collection_metadata()

    def _robot_head_position(self) -> np.ndarray:
        inner = self.env.inner_env
        robot = inner.robot
        for camera in robot.cameras:
            if camera.name.removeprefix(robot.namespace) == "head":
                return np.asarray(inner.data.bind(camera).xpos, dtype=np.float32)
        pelvis_pos = np.asarray(inner.data.bind(robot.pelvis).xpos, dtype=np.float32)
        return pelvis_pos + np.asarray([0.12, 0.0, 0.69], dtype=np.float32)

    def _recenter_vr_space(self, context: Any, *, reason: str) -> None:
        from bigym.vr.viewer.pyopenxr_to_mujoco_converter import (
            camera_axes_from_pyopenxr,
            vector_from_pyopenxr,
        )

        assert self._space_offset is not None
        hmd_pose = context.input.hmd_pose
        hmd_pos = vector_from_pyopenxr(hmd_pose.position)
        hmd_back, _ = camera_axes_from_pyopenxr(hmd_pose.orientation)
        hmd_forward = -np.asarray(hmd_back, dtype=np.float64)
        target = self._robot_head_position()
        target[2] += float(self.config.recenter_height_offset)
        robot_yaw = self._robot_heading_yaw()
        yaw_offset, offset = planar_space_transform(
            hmd_pos, hmd_forward, target, robot_yaw
        )
        set_posef_position(self._space_offset, offset)
        self._space_yaw_offset = yaw_offset
        self._space_yaw_at_recenter = yaw_offset
        self._hmd_position_at_recenter = np.asarray(hmd_pos, dtype=np.float64)
        self._robot_yaw_at_recenter = robot_yaw
        self._space_ready = True
        print(
            f"[vr] recentered ({reason}): "
            f"hmd={np.round(hmd_pos, 3).tolist()} -> "
            f"head={np.round(target, 3).tolist()} "
            f"yaw_offset={np.rad2deg(yaw_offset):+.1f}deg"
        )

    def _robot_heading_yaw(self) -> float:
        return float(self.env.floating_base_yaw())

    def _update_vr_space_follow_robot(self) -> None:
        if self.config.vr_space_mode != "follow_head":
            return
        if (
            self._space_offset is None
            or self._hmd_position_at_recenter is None
            or self._robot_yaw_at_recenter is None
        ):
            return
        from bigym.vr.viewer.pyopenxr_to_mujoco_converter import rotate_vector_yaw

        robot_yaw_delta = wrap_angle(
            self._robot_heading_yaw() - self._robot_yaw_at_recenter
        )
        self._space_yaw_offset = wrap_angle(
            self._space_yaw_at_recenter + robot_yaw_delta
        )
        current_head = self._robot_head_position()
        current_head[2] += float(self.config.recenter_height_offset)
        offset = current_head - rotate_vector_yaw(
            self._hmd_position_at_recenter, self._space_yaw_offset
        )
        set_posef_position(self._space_offset, offset)

    def _step_for_frame(self, frame_state: Any, action: np.ndarray) -> None:
        # The launch/pending screens are frozen previews. The first real
        # attempt starts with left X, which performs a fresh reset + recenter.
        # Drain a pending settle BEFORE the recording check: the session's
        # initial reset() happens on the preview screen (recording=False),
        # and the preview should show a settled robot, not the pre-warmup
        # spawn pose. During an attempt the drain also gates engage (the
        # HUD shows SETTLE either way).
        self._frame_steps = 0
        pending_warmup = self.env.pending_reset_warmup_steps
        if pending_warmup > 0:
            # Four settle steps per frame: warmup steps run fast=True (no
            # camera renders, ~2.2 ms each on G1), so 4 x 2.2 + eye render
            # still fits the 13.9 ms budget and 200 steps drain in ~0.7 s. The
            # completion TimeStep replaces the placeholder reset() returned.
            time_step = self.env.run_reset_warmup_steps(4)
            self._frame_steps = min(4, pending_warmup)
            # reset() recenters against the pre-settle robot pose, but these
            # warmup steps move the physical head before this frame renders.
            # Refresh follow_head now so the view uses the post-step pose
            # instead of lagging one settle chunk behind. The HMD anchor is
            # still the one captured when the user pressed left X.
            self._update_vr_space_follow_robot()
            if time_step is not None:
                self._time_step = time_step
                if self.stats.recording:
                    self._current_steps = [time_step]
                    self._current_fullbody_steps = [
                        self._capture_fullbody_record(include_action_info=False)
                    ]
                print("[settle] reset warmup drained; engage available")
            self._sim_accumulator = 0.0
            self._update_live_stats()
            return

        if not self.stats.recording:
            self._sim_accumulator = 0.0
            self._update_live_stats()
            return

        period_s = float(frame_state.predicted_display_period) / 1_000_000_000.0
        self._sim_accumulator += max(period_s, 0.0)
        # Catch-up cap: at ~4.4 ms per control step, 2 steps keep the worst
        # frame inside the 13.9 ms 72 Hz budget so a stall does not breed the
        # next one; sim time briefly trails wall time (wall_clock records it).
        max_steps = 2
        steps = 0
        while self._sim_accumulator >= self.control_dt and steps < max_steps:
            self._time_step = self.env.step(action)
            fullbody_record = self._capture_fullbody_record(include_action_info=True)
            self.stats.last_reward = float(
                np.asarray(self._time_step.reward).reshape(-1)[0]
            )
            if self.stats.recording:
                if not self.mapper.engaged:
                    # Arms are pinned at the hold pose; slide the episode
                    # start so the demo begins when tracking engages.
                    self._current_steps = [self._time_step]
                    self._current_fullbody_steps = [fullbody_record]
                    self._pre_engage_steps += 1
                else:
                    self._current_steps.append(self._time_step)
                    self._current_fullbody_steps.append(fullbody_record)
                    self.stats.episode_steps += 1
                    self.stats.episode_reward += self.stats.last_reward
                if self._time_step.last() or self._hit_max_steps():
                    self._finish_attempt(force=False)
            self._sim_accumulator -= self.control_dt
            steps += 1
            self._frame_steps = steps
            if not self.stats.recording:
                self._sim_accumulator = 0.0
                break
        self._update_live_stats()

    def _update_live_stats(self) -> None:
        self.stats.vx = self.mapper.last_vx
        self.stats.vy = self.mapper.last_vy
        self.stats.height = self.mapper.last_height
        self.stats.yaw = self.mapper.last_yaw
        # Live camera health shown in the VR overlay so a dead-camera session
        # (all-black rgb_obs) is caught in-headset, not only in the terminal.
        rgb = self._time_step.rgb_obs
        if rgb is not None:
            self.stats.rgb = "OK" if int(np.asarray(rgb).max()) > 0 else "BLACK!!"

    def _hit_max_steps(self) -> bool:
        return self.config.max_steps is not None and self.stats.episode_steps >= int(
            self.config.max_steps
        )

    def _finish_attempt(self, *, force: bool) -> None:
        if not self.stats.recording:
            return
        if is_success(self.env):
            if force and self._current_steps and not self._current_steps[-1].last():
                self._current_steps[-1] = self._current_steps[-1]._replace(discount=0.0)
            duration_s = self.stats.episode_steps * self.control_dt
            self._pending = {
                "steps": list(self._current_steps),
                "seed": self._current_seed,
                "episode_steps": self.stats.episode_steps,
                "episode_reward": self.stats.episode_reward,
                "pre_engage_steps": self._pre_engage_steps,
                "engage_state": self._engage_state,
                "fullbody_steps": list(self._current_fullbody_steps),
            }
            print(
                f"[pending] success steps={self.stats.episode_steps} "
                f"(~{duration_s:.1f}s) "
                f"reward={self.stats.episode_reward:.3f} -- "
                "right B: save | left Y: discard | left X: save & next attempt"
            )
        else:
            self.stats.discarded_failed_attempts += 1
            print(
                f"[discard] unsuccessful attempt steps={self.stats.episode_steps} "
                f"reward={self.stats.episode_reward:.3f} "
                f"final={self.stats.last_reward:.3f}"
            )
        self.stats.recording = False
        self._current_steps = []
        self._current_fullbody_steps = []
        self._touch_collection_metadata()

    def _save_pending(self) -> None:
        if self._pending is None:
            return
        pending = self._pending
        self._pending = None
        episode = episode_from_time_steps(pending["steps"], self.data_specs)
        if self.config.store_fullbody:
            self._add_fullbody_episode_fields(
                episode, pending.get("fullbody_steps", [])
            )
        # Guard against the all-black RGB failure (camera offscreen render dead
        # for the whole VR session). Such demos are useless for pixel training
        # and can't be salvaged later, so warn loudly on the very first save.
        rgb = episode.get("rgb_obs")
        if rgb is not None and int(np.asarray(rgb).max()) == 0:
            if not self.config.keep_black_rgb:
                self.stats.rejected_black_rgb += 1
                print(
                    "\n[!!! BLACK RGB !!!] episode REJECTED (all-zero rgb_obs) — "
                    "cameras are NOT rendering this session.\n"
                    "                   Stop now, fix rendering, and re-collect. "
                    "(--keep-black-rgb overrides.)\n"
                )
                self._touch_collection_metadata()
                return
            print(
                "\n[!!! BLACK RGB !!!] saved demo has all-zero rgb_obs — cameras are "
                "NOT rendering this session. These demos are useless for "
                "pixel training.\n"
                "                   Stop now, fix rendering, and re-collect.\n"
            )
        episode["seed"] = np.asarray([pending["seed"]], dtype=np.int64)
        episode["pre_engage_steps"] = np.asarray(
            [pending["pre_engage_steps"]], dtype=np.int64
        )
        if pending["engage_state"] is not None:
            engage_state = pending["engage_state"]
            episode["init_qpos"] = engage_state["qpos"]
            episode["init_qvel"] = engage_state["qvel"]
            for key in ("ctrl", "qacc_warmstart", "act"):
                if key in engage_state:
                    episode[f"init_{key}"] = engage_state[key]
            for key, value in (engage_state.get("lowerbody_state") or {}).items():
                episode[f"lb_state.{key}"] = np.asarray(value)
        before = len(self.storage)
        saved = self.storage.add_episode(
            episode,
            require_success=True,
            demo_value=1.0,
            is_expert_value=1.0,
        )
        if saved and len(self.storage) > before:
            self.stats.saved_successes += 1
            print(
                f"[save] success {self.stats.saved_successes}/{self.config.episodes} "
                f"steps={pending['episode_steps']} "
                f"reward={pending['episode_reward']:.3f}"
            )
        else:
            self.stats.save_not_stored += 1
            print(
                "[warn] successful attempt was not stored; check final "
                "reward/episode format"
            )
        self._touch_collection_metadata()

    def _discard_pending(self) -> None:
        if self._pending is None:
            return
        print(
            f"[discard] dropped pending success steps={self._pending['episode_steps']}"
        )
        self._pending = None
        self.stats.discarded_pending_successes += 1
        self._touch_collection_metadata()

    def _capture_fullbody_record(
        self, *, include_action_info: bool
    ) -> dict[str, np.ndarray]:
        wall_clock = time.perf_counter() - self._wall_clock_t0
        outer = self.env.bigym
        inner = outer.inner_env
        data = inner.data
        record: dict[str, np.ndarray] = {
            "wall_clock": np.array([wall_clock], dtype=np.float64),
            "full_qpos": np.asarray(data.qpos, dtype=np.float64).copy(),
            "full_qvel": np.asarray(data.qvel, dtype=np.float64).copy(),
        }
        if not include_action_info:
            return record

        info = outer.get_last_lowerbody_control_info()
        for key in (
            "raw_outer_action",
            "expanded_action",
            "lowerbody_action",
            "leg_joint_targets",
            "torso_target",
            "lowerbody_command",
            "height_command",
        ):
            value = info.get(key)
            if isinstance(value, np.ndarray):
                record[key] = value.copy()
        return record

    def _fullbody_default_record(self) -> dict[str, np.ndarray]:
        outer = self.env.bigym
        inner = outer.inner_env
        qpos_dim = int(inner.model.nq)
        qvel_dim = int(inner.model.nv)
        action_dim = int(self.env.action_spec().shape[0])
        expanded_dim = int(inner.action_space.shape[0])
        leg_names = tuple(outer.leg_joint_names())
        return {
            "wall_clock": np.full((1,), np.nan, dtype=np.float64),
            "full_qpos": np.full((qpos_dim,), np.nan, dtype=np.float64),
            "full_qvel": np.full((qvel_dim,), np.nan, dtype=np.float64),
            "raw_outer_action": np.full((action_dim,), np.nan, dtype=np.float32),
            "expanded_action": np.full((expanded_dim,), np.nan, dtype=np.float32),
            "lowerbody_action": np.full((len(leg_names),), np.nan, dtype=np.float32),
            "leg_joint_targets": np.full((len(leg_names),), np.nan, dtype=np.float32),
            "torso_target": np.full((1,), np.nan, dtype=np.float32),
            "lowerbody_command": np.full((3,), np.nan, dtype=np.float32),
            "height_command": np.full((1,), np.nan, dtype=np.float32),
        }

    def _add_fullbody_episode_fields(
        self, episode: dict[str, np.ndarray], records: list[dict[str, np.ndarray]]
    ) -> None:
        expected_len = int(episode["reward"].shape[0])
        defaults = self._fullbody_default_record()
        if len(records) != expected_len:
            print(
                f"[warn] fullbody record count mismatch: got {len(records)}, "
                f"expected {expected_len}; padding missing rows with NaNs"
            )
        padded = list(records[:expected_len])
        while len(padded) < expected_len:
            padded.append({})

        for key, default in defaults.items():
            rows = []
            for record in padded:
                value = record.get(key)
                if value is None:
                    value = default
                value = np.asarray(value, dtype=default.dtype)
                if value.shape != default.shape:
                    raise ValueError(
                        f"fullbody field {key} shape mismatch: expected "
                        f"{default.shape}, got {value.shape}"
                    )
                rows.append(value)
            episode[key] = np.asarray(rows, dtype=default.dtype)

    def _render(self, frame_state: Any) -> None:
        assert self._renderer is not None
        assert self._space_offset is not None
        if (
            self.env.pending_reset_warmup_steps > 0
            and str(self.config.settle_view) == "curtain"
        ):
            # Headset-only comfort curtain. The deferred warmup has already
            # advanced above in _step_for_frame and continues untouched; only
            # the eye framebuffer is replaced. Recorded rgb_obs comes from the
            # environment renderer after settle, so demo contents and reset
            # semantics are identical to --settle-view live.
            self._renderer.render_solid(frame_state, color=(12, 12, 16))
            return
        info = asdict(self.stats)
        rgb_status = info.pop("rgb", "?")
        hud = str(self.config.hud)
        if hud == "full":
            # 11 lines spanning z 1.8..0.8 at x=2, dead center of the
            # workspace gaze.
            self._renderer.show_stats(info, np.array([2.0, 0.0, 1.8]))
        elif hud == "minimal":
            # Compact, up-and-right of the recentered forward gaze (~20 deg
            # azimuth and elevation): readable at a glance, out of the way.
            # Marker labels render at a FIXED pixel size while the anchors
            # sit ~2.8 m away, so line spacing must be set by ANGLE, not
            # meters: 0.14 m at 2.9 m ~= 2.8 deg/line.
            self._renderer.show_stats(
                {
                    "rec": (
                        "SETTLE"
                        if self.env.pending_reset_warmup_steps > 0
                        else ("REC" if self.stats.recording else "idle")
                    ),
                    "saved": self.stats.saved_successes,
                    "steps": self.stats.episode_steps,
                    "reward": round(self.stats.episode_reward, 2),
                    # visible height so an accidental squat command is
                    # noticed immediately (the channel is rate-integrated)
                    "h": (
                        f"{self.stats.height:.2f} LOCK"
                        if self.mapper.height_locked
                        else round(self.stats.height, 2)
                    ),
                    # Torso pitch (RY slot present): value, and ON while the
                    # left stick drives it (left grip held).
                    **(
                        {
                            "pitch": (
                                f"{self.mapper.last_pitch:+.2f}"
                                + (" ON" if self.mapper.pitch_mode else "")
                            )
                        }
                        if self.mapper.pitch_slot
                        else {}
                    ),
                },
                np.array([2.6, -1.0, 2.5]),
                spacing=np.array([0, 0, -0.14]),
            )
        # Camera health, on its own: green OK / red BLACK. On black, blast a big
        # red banner across the view at several heights so a dead-camera session
        # is impossible to miss in-headset.
        if rgb_status == "OK":
            if hud != "off":
                self._renderer.add_marker(
                    pos=(
                        np.array([2.0, 0.0, 1.9])
                        if hud == "full"
                        else np.array([2.6, -1.0, 2.64])
                    ),
                    label="rgb: OK",
                    rgba=[0.0, 1.0, 0.0, 1.0],
                    size=np.array([0.12, 0.12, 0.12]),
                )
        else:
            for z in (1.4, 1.7, 2.0, 2.3, 2.6):
                self._renderer.add_marker(
                    pos=np.array([2.0, 0.0, z]),
                    label="!!! CAMERA BLACK -- STOP & RESTART !!!",
                    rgba=[1.0, 0.0, 0.0, 1.0],
                    size=np.array([0.5, 0.5, 0.5]),
                )
        self._renderer.render(
            frame_state, self._space_offset, yaw_offset=self._space_yaw_offset
        )

    def _fullbody_metadata_layout(self) -> dict[str, Any]:
        outer = self.env.bigym
        inner = outer.inner_env
        layout = outer.wholebody_action_layout()
        robot = inner.robot
        full_joint_names = [j.name.removeprefix(robot.namespace) for j in robot.joints]
        limb_actuator_names = [
            a.name.removeprefix(robot.namespace) for a in robot.limb_actuators
        ]
        inner_floating_dofs = [dof.value for dof in inner.action_mode.floating_dofs]
        return {
            "outer_action_dim": int(self.env.action_spec().shape[0]),
            "expanded_action_dim": int(inner.action_space.shape[0]),
            "qpos_dim": int(inner.model.nq),
            "qvel_dim": int(inner.model.nv),
            "full_joint_names": full_joint_names,
            "inner_floating_dofs": inner_floating_dofs,
            "inner_limb_actuator_names": limb_actuator_names,
            "base_dim": int(layout["base_dim"]),
            "limb_names": list(layout["limb_names"]),
            "gripper_count": int(layout["gripper_count"]),
            "leg_joint_names": list(layout["leg_joint_names"]),
            "outer_action_low": np.asarray(
                layout["action_low"], dtype=np.float32
            ).tolist(),
            "outer_action_high": np.asarray(
                layout["action_high"], dtype=np.float32
            ).tolist(),
        }

    def _write_metadata(self) -> None:
        outer = self.env.bigym
        layout = outer.wholebody_action_layout()
        backend_label = str(self.backend).upper()
        # env_config is what EnvConfig.from_metadata reads; the flat task and
        # lowerbody_policy blocks repeat it for readers that take single fields.
        blocks = self.env_config.to_metadata(self.config.task)
        task_metadata = dict(blocks["task"])
        task_metadata["collect_success_hold_seconds"] = float(
            self.env_config.success_hold_seconds
        )
        task_metadata["training_success_hold_seconds"] = self.training_hold_seconds
        lowerbody_policy = blocks["lowerbody_policy"]
        metadata = {
            "format": BATCH_FORMAT,
            "pipeline_version": DEMO_PIPELINE_VERSION,
            "package_versions": package_versions(),
            "collector": Path(__file__).name,
            "task": task_metadata,
            "control_step_seconds": float(self.env.control_step_seconds),
            "vr_space": {
                "mode": str(self.config.vr_space_mode),
                "settle_view": str(self.config.settle_view),
                "startup_auto_recenter": False,
                "recenter_trigger": "left_x_or_right_a",
                "heading_alignment": "current_hmd_forward_to_robot_forward",
                "transform_applies_to": [
                    "stereo_views",
                    "left_controller",
                    "right_controller",
                ],
            },
            "substrate_fingerprint": outer.substrate_fingerprint(),
            "env_config": blocks["env_config"],
            "lowerbody_policy": lowerbody_policy,
            "reset_semantics": self._reset_semantics_metadata(),
            "action_semantics": {
                "robot_model": self.robot_model,
                "lowerbody_backend": str(self.backend),
                "base_action_mode": str(lowerbody_policy.get("base_action_mode", "")),
                "base_cmd_slew": (
                    None
                    if self.config.base_cmd_slew is None
                    else float(self.config.base_cmd_slew)
                ),
                "stick_deadzone": float(self.config.stick_deadzone),
                "base_velocity_scale": {
                    "vx": float(self.config.base_vx_scale),
                    "vy": float(self.config.base_vy_scale),
                },
                "policy_walk_switch_speed": (
                    GROOT_WBC_POLICY_WALK_THRESHOLD
                    if self.backend == "groot_wbc_g1"
                    else None
                ),
                "planar_walk_entry_speed": (
                    GROOT_WBC_MIN_WALK_SPEED if self.backend == "groot_wbc_g1" else None
                ),
                "yaw_mode": self.config.yaw_mode,
                "upperbody_ik_backend": "mink",
                "upperbody_ik_conditioning": {
                    "engage_smooth_seconds": float(
                        self.mapper.arm_engage_smooth_seconds
                    ),
                    "max_joint_speed_rad_s": self.mapper.arm_max_joint_speed,
                    "engage_seed": (
                        "current_arm_command"
                        if self.mapper.arm_engage_smooth_seconds > 0.0
                        else None
                    ),
                },
                "left_stick_y": f"{backend_label} vx_body",
                "left_stick_x": f"{backend_label} vy_body",
                "pitch_mode_control": (
                    None if not self.mapper.pitch_slot else "hold left grip"
                ),
                "right_stick_x": (
                    f"{backend_label} wz"
                    if self.config.yaw_mode == "base"
                    else "ignored"
                ),
            },
            "outer_action_floating_dofs": [
                dof.value for dof in outer.outer_action_floating_dofs
            ],
            "outer_limb_actuator_names": list(outer.outer_limb_actuator_names),
            "action_stats": {
                "min": np.asarray(outer.action_stats["min"], dtype=np.float32).tolist(),
                "max": np.asarray(outer.action_stats["max"], dtype=np.float32).tolist(),
            },
            "action_stats_mode": "collector_envelope",
            "raw_outer_action_bounds": {
                "min": layout["action_low"].tolist(),
                "max": layout["action_high"].tolist(),
            },
            "extra_fields": {
                "store_fullbody": bool(self.config.store_fullbody),
                "per_step": [
                    "wall_clock",
                    "full_qpos",
                    "full_qvel",
                    "raw_outer_action",
                    "expanded_action",
                    "lowerbody_action",
                    "leg_joint_targets",
                    "torso_target",
                    "lowerbody_command",
                    "height_command",
                ],
                "training_action_field": "action remains the normalized outer action",
                "wall_clock_semantics": (
                    "seconds since collector start (time.perf_counter). The sim "
                    "time axis is k * control_step_seconds by construction; "
                    "wall_clock is a diagnostic for capture-rate jitter only and "
                    "is never used for training or replay."
                ),
            },
            "fullbody_layout": self._fullbody_metadata_layout(),
        }
        # The static half is built once; only the `collection` counters change
        # during a session, so refreshes stay cheap enough for the VR loop.
        self._metadata_base = metadata
        self._refresh_collection_metadata()

    def _read_previous_collection(self) -> dict[str, Any] | None:
        """The `collection` block of a metadata.json already in this dir.

        Sessions can be resumed into an existing batch directory (an explicit
        --out-dir; ReplayBufferStorage continues the episode numbering), and
        this file is rewritten rather than merged, so prior counters have to
        be carried forward explicitly.
        """
        path = self.out_dir / "metadata.json"
        if not path.is_file():
            return None
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(
                f"[collection] could not read {path} ({exc}); starting fresh counters"
            )
            return None
        collection = previous.get("collection") if isinstance(previous, dict) else None
        return collection if isinstance(collection, dict) else None

    def _collection_block(self) -> dict[str, Any]:
        return build_collection_block(
            self.stats,
            session_id=self._session_id,
            started_at=self._session_started_at,
            finished_at=self._session_finished_at,
            operator=self._operator,
            host=self._host,
            bigym_git_sha=self._bigym_git_sha,
            seed_start=int(self.config.seed),
            unaccounted_episodes=self._unaccounted_episodes,
            previous=self._previous_collection,
        )

    def _refresh_collection_metadata(self, *, finished: bool = False) -> None:
        """Rewrite metadata.json with the current attempt/keep counters."""
        if self._metadata_base is None:
            return
        if not self.out_dir.is_dir():
            # The empty-session cleanup removed the dir; do not recreate it.
            return
        if finished:
            self._session_finished_at = _utc_now_iso()
        metadata = dict(self._metadata_base)
        metadata["collection"] = self._collection_block()
        path = self.out_dir / "metadata.json"
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        tmp.replace(path)

    def _touch_collection_metadata(self) -> None:
        """Best-effort counter refresh from the attempt lifecycle.

        A failed write must never take down a live collection session.
        """
        try:
            self._refresh_collection_metadata()
        except Exception as exc:
            print(f"[collection] metadata refresh failed: {exc}")

    def finalize_metadata(self) -> None:
        """Stamp finished_at and the final counters. Idempotent."""
        try:
            self._refresh_collection_metadata(finished=True)
        except Exception as exc:
            print(f"[collection] final metadata write failed: {exc}")
