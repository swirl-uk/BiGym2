"""Minimal bigym.loco example: random agent on a G1 controller-in-the-loop env.

Two ways to build the environment (both from docs/getting_started.md):

* ``make(task)`` builds the task's official configuration: G1 with Dex1
  grippers, the GR00T-WBC lower-body policy in the loop, and the task's
  episode budget, success hold, warmup and reset settings. Only these
  numbers are comparable to the leaderboard.
* ``make(task, **overrides)`` changes named ``EnvConfig`` fields on top of
  it: cameras, budget, a floating base (``controller=None``), ...
  ``--research`` runs this path; ``env.config_overrides`` and
  ``protocol_violations(env)`` then list what departs from official.

Either way the frozen lower-body policy keeps the robot balanced from
velocity commands inside every ``env.step``; the (here: random) agent owns
the outer action:

    [base command slots | arm joint targets | grippers]

Run headless:

    MUJOCO_GL=egl uv run python examples/loco_random_agent.py
    MUJOCO_GL=egl uv run python examples/loco_random_agent.py --research
"""

import argparse

import numpy as np

from bigym.loco import make
from bigym.loco.eval import eval_seeds, protocol_violations


def build_env(research: bool):
    """Return the official env, or a variant with a few fields overridden."""
    if not research:
        return make("move_plate")  # official: g1_dex1 + groot_wbc_g1
    # A research variant: head camera only and a shorter budget. Any
    # EnvConfig field can be overridden; see docs/official_configuration.md.
    return make("move_plate", camera_keys=("head",), episode_length=5000)


def main() -> None:
    """Print the outer action layout, then run two short seeded episodes."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--research",
        action="store_true",
        help="override a few EnvConfig fields instead of the official configuration",
    )
    args = parser.parse_args()

    env = build_env(args.research)

    # What am I controlling? (outer action structure)
    layout = env.wholebody_action_layout()
    print(f"outer action dim : {layout['action_dim']}")
    print(
        f"  base cmd slots : {layout['base_dim']} "
        "([vx, vy, height, wz]; +pitch when the task enables it)"
    )
    print(
        f"  limb targets   : {len(layout['limb_names'])} "
        f"({', '.join(layout['limb_names'][:4])}, ...)"
    )
    print(f"  grippers       : {layout['gripper_count']}")
    print(f"obs components   : {env.low_dim_component_slices()}")

    # Is this the official configuration? Empty list = leaderboard-comparable.
    fingerprint = env.substrate_fingerprint()
    print(f"substrate        : {fingerprint['substrate_version']}")
    print(f"overrides        : {env.config_overrides or 'none'}")
    print(f"protocol issues  : {protocol_violations(env) or 'none'}")

    spec = env.action_spec()
    rng = np.random.default_rng(0)

    # The evaluation protocol fixes the reset seed of every episode
    # (620000, 620001, ...); a full evaluation runs eval_seeds(100).
    for seed in eval_seeds(2):
        timestep = env.reset(seed=seed)
        total_reward = 0.0
        for _ in range(100):
            # Small random arm motion, zero base command (stand in place).
            action = rng.uniform(-0.05, 0.05, size=spec.shape).astype(spec.dtype)
            action[: layout["base_dim"]] = 0.0
            timestep = env.step(action)
            total_reward += float(timestep.reward or 0.0)
            if timestep.last():
                break
        print(
            f"seed {seed}: total reward {total_reward:.3f}, "
            f"last step {timestep.step_type.name}"
        )
    env.close()


if __name__ == "__main__":
    main()
