# Determinism and versioning

## Bit-exact replay

Restoring a saved simulator state mid-episode reproduces the rest of the
trajectory to the last bit.

- The G1 scenes use the Newton solver. PGS warmstarts from contact-dependent
  state that cannot be saved.
- A snapshot holds the full controller state: commands, last action, gait
  phase and the policy's observation history.
- The env runs `mj_forward` after every control step and after a restore, so
  kinematics, contacts and sensors match.

Each demonstration stores its engage snapshot: `init_qpos`, `init_qvel`,
`init_ctrl`, `init_qacc_warmstart` and the controller state (`lb_state.*`).
Replay resets the env on the episode's seed, restores the snapshot and applies
the recorded actions.

## Deterministic reset

The official configuration sets `deterministic_reset`, so every reset
restores the controller state from the env's first reset and `reset(seed)`
depends only on the seed.

## Substrate version and fingerprint

`SUBSTRATE_VERSION` is `bigym2-mj381-v1`. It changes when the MuJoCo version,
the physics options, the observation and action layout, or the reward and
success semantics change.

Official results are comparable when their `substrate_version` matches.
Official means `protocol_violations(env)` is empty (see
[Official overrides](official_configuration.md#official-overrides)).

`env.substrate_fingerprint()` adds the identity of the exact run: hashes of
the compiled model and the lower-body weights, the task and controller
settings, `bigym_version` and the git SHA. Leaderboard records, demo batches
and coding-agent evaluations embed it.

The fingerprint does not cover Python packages. Official results use Python
3.12 with the versions pinned in `uv.lock`.
