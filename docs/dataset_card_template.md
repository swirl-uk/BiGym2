# Dataset Card Template — BiGym 2.0 Demonstrations

Copy this file to the root of a demonstration release as `README.md`, fill in
every `<...>` placeholder, and delete the instruction blocks marked
`<!-- how to fill -->`. Most values can be read straight out of the batch's
`metadata.json` (replay-format npz) or `meta/info.json` +
`meta/source_metadata.json` (LeRobot v3 export); the field-by-field pointers
below say which.

Keep the front matter valid YAML — Hugging Face parses it.

---

```yaml
---
pretty_name: "BiGym 2.0 — <task> (<robot>, <n> demos)"
license: cc-by-4.0
language:
  - en
task_categories:
  - robotics
tags:
  - LeRobot
  - bigym
  - humanoid
  - loco-manipulation
  - bi-manual-manipulation
  - imitation-learning
  - teleoperation
  - mujoco
size_categories:
  - n<1K
configs:
  - config_name: default
    data_files: data/*/*.parquet
---
```

# BiGym 2.0 — `<task_name>`

`<one or two sentences: what the operator does in this task, and what counts
as success.>`

- **Homepage / code**: <https://github.com/swirl-uk/BiGym2>
- **Collected with**: `bigym-collect` (BiGym 2.0 VR collector)
- **Format**: LeRobot v3, lossless PNG image mode
  <!-- how to fill: video-mode exports are visualization-only and are rejected
       by the training loader. Only release the lossless export. -->

## At a glance

| Field | Value | Source |
|-------|-------|--------|
| Task | `<task_name>` | `metadata.json` → `task.task_name` |
| Robot model | `g1_dex1` | `task.robot_model` (also `info.json` → `robot_type`) |
| Lower-body backend | `groot_wbc_g1` | `env_config.controller.backend` / `substrate_fingerprint.lowerbody_backend` |
| Lower-body weights | `<file names and sha256>` | `substrate_fingerprint.lowerbody_weights` |
| Substrate version | `<bigym2-mj381-v1>` | `substrate_fingerprint.substrate_version` |
| Substrate fingerprint | see [Substrate fingerprint](#substrate-fingerprint) | `substrate_fingerprint` |
| Episodes released | `<60>` | `info.json` → `total_episodes` |
| Frames | `<37705>` | `info.json` → `total_frames` |
| Control rate | `<50>` Hz (`control_step_seconds = <0.02>`) | `metadata.json` → `control_step_seconds` |
| Attempts / kept | `<attempts>` / `<kept>` (`<xx>%`) | `collection.attempts/kept`; inspect `unaccounted_episodes` |
| Operator(s) | `<name or anonymous id>` | `collection.sessions[].operator`; top-level operator is the latest session |
| Collection dates | `<YYYY-MM-DD>` – `<YYYY-MM-DD>` | `collection.sessions[].started_at/finished_at` |
| Data license | CC BY 4.0 | this card |

## Task and success criterion

- **Task name**: `<task_name>`
- **Success criterion**: the task predicate must hold for
  `<success_hold_seconds>` consecutive seconds
  (`metadata.json` → `task.success_hold_seconds`).
- **Collection hold**: `<collect_success_hold_seconds>` s
  (`task.collect_success_hold_seconds`) — demos are recorded with a *stricter*
  hold than training/eval scores at, so every demo tail carries deliberate
  stay-still supervision.
- **Reach tolerance** (reach-family tasks only): `<0.05 | null>`
  (`task.reach_tolerance`; `null` means the 0.1 class constant).
- **Episode budget**: collected at `<episode_length>` env steps
  (`episode_length / demo_down_sample_rate` outer steps);
  recommended budget for training/eval is `<recommended_episode_length>`
  (`metadata.json` → `recommended_episode_length`, derived by
  `bigym.loco.tasks.BUDGET_RULE`; the rule used is recorded in
  `recommended_episode_length_rule`).
- **Demo down-sample rate**: `<demo_down_sample_rate>`.

<!-- how to fill: raw collector task.success_hold_seconds describes the
     collection hold; task.training_success_hold_seconds records the intended
     training hold. Identify whether this release is the raw or hold-trimmed
     view and read its own metadata. env.get_demos() refuses a dataset whose
     robot_model / backend / demo_down_sample_rate / cameras / observation and
     action dimensions differ from the env. Quote the values here so users do
     not have to open the JSON. -->

## Robot, controller and action space

- **Robot**: `<robot_model>` — `<Unitree G1, 29 DoF, Dex1-1 parallel grippers>`
- **Lower body**: `<backend>` policy running inside the environment step, at
  the same rate as the outer control loop. The agent never actuates the legs
  directly; it sends a base velocity command.
- **Upper body**: `<upperbody_ik_backend>` inverse kinematics from VR
  controller poses (`metadata.json` → `action_semantics`).
- **Outer action** (`action`, dimension `<outer_action_dim>`):
  `<4>` base slots (`<pelvis_x, pelvis_y, pelvis_z, pelvis_rz>`) +
  `<14>` arm joints + `<2>` gripper scalars.
  Bounds are in `fullbody_layout.outer_action_low/high`.
- **Base action mode**: `<lowerbody_cmd>`
  (`action_semantics.base_action_mode`) — demos record **velocity commands**,
  not position deltas.
- **Base velocity scale used during collection**:
  `vx = <0.35>`, `vy = <0.25>`, `wz = <0.5>`
  (`action_semantics.base_velocity_scale`, `right_stick_x`).
- **Height command range**: `<[0.4, 1.0]>` metres in the raw outer action;
  backend default `<0.74>` m. The stored `action` is normalized using the
  collector's `action_stats.min/max`; raw and normalized zero need not match.
- **Action statistics**: `action_stats` with top-level
  `action_stats_mode=<collector_envelope>` defines the collector's
  raw-to-normalized transform. Disclose any downstream training
  normalization separately.
- **Pitch/layout**: state whether `enable_pitch_cmd` is enabled and copy
  the complete `outer_action_floating_dofs` and resolved bounds. The
  example dimensions here are placeholders, not a universal G1 layout.

## Reset semantics

Reproducing this data requires reproducing the reset, not just the task:

- `init_stance`: `<keyframe>`
- `init_pelvis_z`: `<0.75>` m
- `reset_warmup_steps`: `<200>` (`<4.0>` s) — episodes settle before the
  operator engages
- `demo_start`: `<post_warmup_vr_engage_state>`
- Per-episode engage snapshot fields: `init_qpos`, `init_qvel`, `init_ctrl`,
  `init_qacc_warmstart`, `lb_state.*`

All of the above are in `metadata.json` → `reset_semantics`.

## Substrate fingerprint

The fingerprint pins the physics/embodiment world an episode was collected in.
Episodes with different fingerprints are **not** numerically comparable.

```text
{
  "substrate_version": "<bigym2-mj381-v1>",
  "mujoco_version": "<3.8.1>",
  "robot_model": "<g1_dex1>",
  "lowerbody_backend": "<groot_wbc_g1>",
  "lowerbody_weights": <{"<file>": "<sha256>", ...}>,
  "lowerbody_init_stance": "<keyframe>",
  "lowerbody_base_action_mode": "<lowerbody_cmd>",
  "g1_passive_base_tilt": <true>,
  "solver": <2>,
  "contact_signature": "<a3972624f79d3e98>",
  "control_step_seconds": <0.02>,
  "demo_down_sample_rate": <10>,
  "episode_length": <60000>,
  "success_hold_seconds": <3.0>,
  "action_dim": <20>,
  "task": "<move_plate>"
}
```

Package versions at collection time (`metadata.json` → `package_versions`):
`bigym <1.0.0>`, `mujoco <3.8.1>`, `mink <1.2.0>`, `numpy <2.2.6>`,
`python <3.12.3>`.

## Collection

- **Method**: VR teleoperation (`bigym-collect`), operator wearing
  `<headset>`, `<wired USB | Air Link>` link.
- **Operator(s)**: `<name or anonymous id; say if more than one and how
  episodes are split between them>`
- **Attempts / kept**: `<attempts>` attempts, `<kept>` kept
  (`<xx>%` keep rate).
  <!-- how to fill: the collector records attempts and kept counts per session.
       An attempt ends when the operator presses finish; successful attempts go
       to pending and are kept, failures are discarded. Report the totals over
       the whole batch, and say whether discarded attempts are released. -->
- **Discarded attempts**: `<not released | released under data/discarded/>`
- **Success rate of released episodes**: `<100%>` — `<all released episodes
  are successful demonstrations>`
- **VR space**: mode `<follow_head>`, heading alignment
  `<current_hmd_forward_to_robot_forward>` (`metadata.json` → `vr_space`).

## Dataset structure

LeRobot v3 layout:

```
<dataset>/
  meta/
    info.json                  # feature schema, fps, episode/frame counts
    tasks.parquet              # task index -> natural-language task string
    stats.json                 # per-feature statistics
    episodes/                  # per-episode index
    episode_init_states.json   # engage snapshots (BiGym sidecar, not LeRobot)
    alignment.json             # action alignment version (BiGym sidecar)
    source_metadata.json       # copy of the collector metadata.json
  data/chunk-000/file-000.parquet
  images/<camera>/...
  metadata.json                # root copy of source_metadata.json
```

### Per-frame fields

<!-- how to fill: this table is the feature list from meta/info.json. Delete
     rows your export does not contain — the exporter intersects the optional
     extras over every episode, so a batch collected before a field existed
     simply lacks it. -->

| Field | dtype | shape | Notes |
|-------|-------|-------|-------|
| `observation.state` | float32 | (`<50>`,) | low-dim observation; component slices in `substrate_fingerprint.low_dim_component_slices` (`proprioception`, `proprioception_grippers`, `proprioception_floating_base`) |
| `observation.images.head` | image | (3, `<84>`, `<84>`) | head camera, RGB, lossless PNG |
| `observation.images.left_wrist` | image | (3, `<84>`, `<84>`) | left wrist camera |
| `observation.images.right_wrist` | image | (3, `<84>`, `<84>`) | right wrist camera |
| `action` | float32 | (`<20>`,) | **normalized outer action** — the training target |
| `raw_outer_action` | float32 | (`<20>`,) | outer action before normalization |
| `expanded_action` | float32 | (`<35>`,) | outer action expanded to the full joint set |
| `lowerbody_action` | float32 | (`<15>`,) | lower-body policy output |
| `leg_joint_targets` | float32 | (`<15>`,) | leg + waist position targets sent to the actuators |
| `torso_target` | float32 | (1,) | torso yaw target |
| `lowerbody_command` | float32 | (3,) | base velocity command `(vx, vy, wz)` |
| `height_command` | float32 | (1,) | pelvis height command, metres |
| `full_qpos` | float64 | (`<46>`,) | full MuJoCo `qpos` |
| `full_qvel` | float64 | (`<45>`,) | full MuJoCo `qvel` |
| `reward` | float32 | (1,) | task reward |
| `event_progress` | float32 | (1,) | per-task sub-goal progress; all-`NaN` on eventless tasks |
| `discount` | float32 | (1,) | |
| `demo` | float32 | (1,) | demo flag (1 for every frame here) |
| `is_expert` | float32 | (1,) | expert flag |
| `wall_clock` | float64 | (1,) | **diagnostic only** — seconds since collector start. The sim time axis is `k * control_step_seconds` by construction; `wall_clock` exists to expose capture-rate jitter (XR frame stalls) and is never used for training or replay. |
| `timestamp` | float32 | (1,) | LeRobot-synthesized: `frame_index / fps` |
| `frame_index` | int64 | (1,) | |
| `episode_index` | int64 | (1,) | |
| `index` | int64 | (1,) | global frame index |
| `task_index` | int64 | (1,) | |

### Action alignment

`meta/alignment.json` records `action_alignment_version` 2: frame `k`'s
observation is paired with the transition executed *from* it, for every
transition feature; the final frame repeats the last real transition, and
source index 0 lives in `meta/episode_init_states.json` → `first_transition`.

### Per-episode engage snapshots

`meta/episode_init_states.json` stores, per episode, the state needed to replay
it deterministically. Each entry has `source_file` and an `arrays` map where
every entry carries explicit `dtype`, `shape` and `data`:

- `seed`, `pre_engage_steps`
- `init_qpos` (`<46>`), `init_qvel` (`<45>`), `init_ctrl` (`<37>`),
  `init_qacc_warmstart` (`<45>`)
- `lb_state.*` — the full lower-body controller state (observation histories,
  last actions, rate-limiter anchors, torso yaw target, height command)

This sidecar is **not** part of the LeRobot schema; consumers that ignore it
still get valid trajectories, but cannot reproduce the simulation bit-exactly.

## Intended use and limitations

- **Intended use**: imitation learning and offline RL for whole-body humanoid
  loco-manipulation; benchmarking against the BiGym 2.0 evaluation protocol.
- **Reproducibility boundary**: results are only comparable across datasets
  with the same [substrate fingerprint](#substrate-fingerprint). A change to
  the physics version, lower-body weights, control rate, base DoF set or
  success criterion changes the fingerprint.
- **Task coverage**: `<list the tasks actually present in this release>`;
  do not infer coverage from the simulator task list.
- **Scale**: `<n>` episodes from `<one operator>`; this is a demo-driven
  benchmark, not a large-scale pretraining corpus. Expect operator-specific
  strategies and limited coverage of the task's state space.
- **Images**: `<84>`x`<84>` RGB, the raw environment observation. They are the
  observation the policy sees, not a presentation render.

## License

- **Data (this dataset)**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  Attribute BiGym 2.0 and cite the entries below.
- **Code**: the BiGym 2.0 repository is Apache-2.0. Third-party components
  redistributed with the code (robot models, lower-body policy weights, 3D
  props) have their own terms — see `THIRD_PARTY_NOTICES.md` in the
  repository.
- **Scene assets visible in the images**: the rendered observations contain
  3D props under CC0 and CC BY 4.0. Their authors are credited in
  `bigym/envs/xmls/3D_MODELS_ATTRIBUTION.md`; that attribution carries over to
  anyone redistributing these images.

## Citation

Cite both the BiGym 2.0 paper and the original BiGym paper:

```bibtex
@article{zhang2026bigym2,
  title   = {BiGym 2.0: Benchmarking Learned and Agent-Developed Policies for Humanoid Household Manipulation},
  author  = {Zhang, Zexi and Zhu, Zecheng and Chen, Zidong and Tuya, Zulkhuu and James, Stephen},
  journal = {arXiv preprint arXiv:2610.07594},
  year    = {2026}
}

@article{chernyadev2024bigym,
  title   = {{BiGym}: A Demo-Driven Mobile Bi-Manual Manipulation Benchmark},
  author  = {Chernyadev, Nikita and Backshall, Nicholas and Ma, Xiao and Lu, Yunfan and Seo, Younggyo and James, Stephen},
  journal = {arXiv preprint arXiv:2407.07788},
  year    = {2024}
}
```

If the release uses the `groot_wbc_g1` lower-body backend, also credit
NVIDIA's GR00T Whole-Body Control policy — see `THIRD_PARTY_NOTICES.md` §1
for the terms its weights are distributed under.

## Contact

`<maintainer name / issue tracker URL>`
