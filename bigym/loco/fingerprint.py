"""Substrate identity and the description of what an env simulates.

:func:`substrate_fingerprint` is the record stamped into demo, run and eval
metadata so results can be grouped by comparable substrate;
:func:`describe_model` reads the compiled model's physics values for a
paper's appendix. ``BiGym.substrate_fingerprint()`` and
``BiGym.describe_model()`` return them.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import mujoco
import numpy as np

import bigym

if TYPE_CHECKING:
    from bigym.loco.env import BiGym

# Substrate freeze marker. Bump ONLY for result-affecting changes: obs/action
# layout, reward/success semantics, physics options (solver pin, PD/armature
# tables), or MuJoCo era. Runs are numerically comparable only within one
# substrate version.
SUBSTRATE_VERSION = "bigym2-mj381-v1"


def substrate_fingerprint(env: BiGym) -> dict[str, Any]:
    """Identity of the frozen substrate ``env`` realizes (see SUBSTRATE_VERSION)."""
    model = env.inner_env.model
    lowerbody = env.lowerbody
    return {
        "substrate_version": SUBSTRATE_VERSION,
        "mujoco_version": mujoco.__version__,
        "task": str(env.task_name),
        "initialization_profile": str(env.config.initialization_profile),
        "robot_model": str(env.config.robot_model),
        "lowerbody_backend": (
            str(lowerbody.backend) if lowerbody.controller is not None else None
        ),
        # Two evals that differ in any of the three fields below are not
        # comparable.
        "success_hold_seconds": float(env.config.success_hold_seconds),
        "lowerbody_base_action_mode": (
            str(lowerbody.base_action_mode)
            if lowerbody.controller is not None
            else None
        ),
        "lowerbody_init_stance": (
            str(lowerbody.config.init_stance)
            if lowerbody.controller is not None
            else None
        ),
        # Outer RY torso-pitch slot (21- vs 20-dim action layout).
        "lowerbody_pitch_cmd": (
            bool(lowerbody.pitch_cmd_enabled)
            if lowerbody.controller is not None
            else None
        ),
        # Set for backends that can leave the pelvis roll/pitch passive
        # (BackendBinding.supports_passive_base_tilt); the key name is part
        # of the record schema.
        "g1_passive_base_tilt": (
            bool(lowerbody.config.passive_base_tilt)
            if lowerbody.controller is not None
            and lowerbody.binding.supports_passive_base_tilt
            else None
        ),
        "control_step_seconds": float(env.control_step_seconds),
        # Always None: the backend weights are listed with hashes under
        # lowerbody_weights; the key is part of the record schema.
        "lowerbody_checkpoint": None,
        "lowerbody_weights": weight_hashes(lowerbody.controller),
        "reset_warmup_steps": lowerbody.effective_reset_warmup_steps(),
        "reach_tolerance": (
            float(env.config.reach_tolerance)
            if env.config.reach_tolerance is not None
            else None
        ),
        "compiled_model_sha256": env.compiled_model_sha256,
        "bigym_version": bigym_version(),
        "bigym_git_sha": bigym_git_sha(),
        "episode_length": env.config.episode_length,
        "demo_down_sample_rate": int(env.config.demo_down_sample_rate),
        "action_representation": str(env.config.action_representation),
        "upper_delta_scale_rad": env.upper_delta_scale_rad,
        "action_dim": int(env.action_space.shape[0]),
        "low_dim_component_slices": env.low_dim_component_slices(),
        "solver": int(model.opt.solver),
        # Hash of every compiled contact-relevant field: any geom
        # friction/solref/solimp/condim/priority/margin/gap edit or
        # option change (cone/impratio/noslip/timestep) lands here even
        # when SUBSTRATE_VERSION is forgotten. Forensic identity - the
        # launch validator compares substrate_version, not this hash.
        "contact_signature": contact_signature(model),
    }


def describe_model(env: BiGym) -> dict[str, Any]:
    """What ``env`` actually simulates: compiled options, gains, armature, backend.

    The numbers are read from the COMPILED MjModel (not from config
    tables), so they are what the physics ran with.
    """
    model = env.inner_env.model
    lowerbody = env.lowerbody
    opt = model.opt
    timestep = float(opt.timestep)
    control_dt = float(env.control_step_seconds)
    actuators = []
    for i in range(int(model.nu)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        gain = model.actuator_gainprm[i]
        bias = model.actuator_biasprm[i]
        actuators.append(
            {
                "name": name,
                "kp": float(gain[0]),
                "kv": float(-bias[2]) if len(bias) > 2 else None,
                "ctrlrange": [
                    float(model.actuator_ctrlrange[i][0]),
                    float(model.actuator_ctrlrange[i][1]),
                ],
            }
        )
    armature = {}
    for j in range(int(model.njnt)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        adr = int(model.jnt_dofadr[j])
        armature[name] = float(model.dof_armature[adr])
    backend = None
    if lowerbody.controller is not None:
        backend = {
            "backend": str(lowerbody.backend),
            "controlled_joints": list(lowerbody.controller.controlled_joints),
            "weights": weight_hashes(lowerbody.controller),
            "reset_warmup_steps": lowerbody.effective_reset_warmup_steps(),
            "deterministic_reset": bool(lowerbody.deterministic_reset),
        }
    return {
        "mujoco_version": mujoco.__version__,
        "robot_model": str(env.config.robot_model),
        "task": str(env.task_name),
        "timestep_s": timestep,
        "control_step_s": control_dt,
        "substeps_per_control_step": int(round(control_dt / timestep))
        if timestep > 0
        else None,
        "solver": int(opt.solver),
        "iterations": int(opt.iterations),
        "cone": int(opt.cone),
        "impratio": float(opt.impratio),
        "noslip_iterations": int(opt.noslip_iterations),
        "gravity": [float(g) for g in opt.gravity],
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "actuators": actuators,
        "dof_armature": armature,
        "lowerbody": backend,
        "action_dim": int(env.action_space.shape[0]),
        "action_layout": env.wholebody_action_layout(),
        "low_dim_component_slices": env.low_dim_component_slices(),
    }


def weight_hashes(controller: Any) -> Optional[dict[str, str]]:
    """``{file name: sha256}`` of the backend's weight files (None without one)."""
    if controller is None:
        return None
    return {Path(f).name: file_sha256(f) for f in controller.weight_files()}


def contact_signature(model) -> str:
    """Short hash of the compiled contact parameters and solver options."""
    h = hashlib.sha256()
    for arr in (
        model.geom_friction,
        model.geom_solref,
        model.geom_solimp,
        model.geom_condim,
        model.geom_priority,
        model.geom_margin,
        model.geom_gap,
    ):
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update(
        np.array(
            [
                model.opt.cone,
                model.opt.impratio,
                model.opt.noslip_iterations,
                model.opt.timestep,
                model.opt.solver,
            ],
            dtype=np.float64,
        ).tobytes()
    )
    return h.hexdigest()[:16]


_GIT_SHA_CACHE: dict[str, Optional[str]] = {}


def bigym_version() -> str:
    """The installed bigym package version."""
    return bigym.__version__


def bigym_git_sha() -> Optional[str]:
    """Short SHA of the bigym checkout (None for a wheel install)."""
    if "sha" in _GIT_SHA_CACHE:
        return _GIT_SHA_CACHE["sha"]
    sha: Optional[str] = None
    repo = Path(__file__).resolve().parents[2]
    # git missing from PATH or not answering: record no SHA.
    try:
        if (repo / ".git").exists():
            out = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "--short=12", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if out.returncode == 0:
                sha = out.stdout.strip() or None
                dirty = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(repo),
                        "status",
                        "--porcelain",
                        "--untracked-files=no",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if dirty.returncode == 0 and dirty.stdout.strip():
                    sha = f"{sha}-dirty"
    except (OSError, subprocess.TimeoutExpired):
        sha = None
    _GIT_SHA_CACHE["sha"] = sha
    return sha


_FILE_SHA_CACHE: dict[str, str] = {}


def file_sha256(path) -> str:
    """SHA-256 of a file's bytes, cached per path."""
    key = str(path)
    if key not in _FILE_SHA_CACHE:
        h = hashlib.sha256()
        with open(key, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _FILE_SHA_CACHE[key] = h.hexdigest()
    return _FILE_SHA_CACHE[key]


# The packages whose versions a recorded batch carries.
TRACKED_PACKAGES = ("mujoco", "bigym", "mink", "numpy")


def package_versions() -> dict[str, str | None]:
    """The Python version and each tracked package's (None when absent)."""
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def compiled_model_sha256(model) -> str:
    """SHA-256 of the compiled MjModel (binary MJB): scene + robot + options."""
    buf = np.zeros(int(mujoco.mj_sizeModel(model)), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buf)
    return hashlib.sha256(buf.tobytes()).hexdigest()
