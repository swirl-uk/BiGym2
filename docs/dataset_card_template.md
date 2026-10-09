# Dataset card template

Copy this file to the root of a demonstration release as `README.md`, fill in
every `<...>` placeholder, and delete the `<!-- how to fill -->` blocks. Most
values come from the batch's `metadata.json`, or from `meta/info.json` and
`meta/source_metadata.json` in a LeRobot v3 export. The pointers below name
the field.

---

```yaml
---
pretty_name: "BiGym 2.0: <task> (<robot>, <n> demos)"
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

# BiGym 2.0: `<task_name>`

`<one or two sentences: what the operator does in this task, and what counts
as success.>`

- Homepage and code: <https://github.com/swirl-uk/BiGym2>
- Format: LeRobot v3, lossless PNG image mode
  <!-- how to fill: the training loader rejects video-mode exports. Release
       the lossless export. -->

## At a glance

| Field | Value | Source |
|-------|-------|--------|
| Episodes | `<60>` | `info.json` → `total_episodes` |
| Frames | `<37705>` | `info.json` → `total_frames` |
| Operator(s) | `<name or anonymous id, and how episodes split between several>` | `collection.sessions[].operator` |
| Collection dates | `<YYYY-MM-DD>` to `<YYYY-MM-DD>` | `collection.sessions[].started_at/finished_at` |
| Data license | CC BY 4.0 | this card |

## Task and success criterion

- Success: the task predicate holds for `<success_hold_seconds>` consecutive
  seconds and the robot never falls (`task.success_hold_seconds`).
- Collection hold: `<collect_success_hold_seconds>` s
  (`task.collect_success_hold_seconds`).
- Reach tolerance (reach tasks only): `<0.05>` m
  (`task.reach_tolerance`).
- Episode budget for training and evaluation: `<episode_length>` env steps,
  the length `bigym.loco.make(task)` sets.
- Demo down-sample rate: `<demo_down_sample_rate>`.

<!-- how to fill: say whether this release is the raw batch or the training
     view cut by bigym.loco.demos.success_hold (its metadata has
     success_hold_trim), and read that release's own metadata. -->

## Robot, controller and action space

- Robot: `<robot_model>`: `<Unitree G1, 29 DoF, Dex1-1 two-finger grippers>`
- Lower body: `<backend>` policy, stepped inside the environment at the
  outer control rate. The agent sends base velocity and height commands and
  never actuates the legs directly.
- Upper body: `<upperbody_ik_backend>` inverse kinematics from VR controller
  poses (`metadata.json` → `action_semantics`).
- Outer action (`action`, dimension `<outer_action_dim>`): `<4>` base slots
  (`<pelvis_x, pelvis_y, pelvis_z, pelvis_rz>`), `<14>` arm joints and `<2>`
  gripper scalars. Copy `outer_action_floating_dofs` and the bounds in
  `fullbody_layout.outer_action_low/high`, and say whether the pitch slot is
  enabled.
- Base velocity scale during collection: `vx = <0.35>`, `vy = <0.25>` m/s
  (`action_semantics.base_velocity_scale`). The yaw scale is the collector's
  `--base-wz-scale` (default 0.5 rad/s) and is not recorded.
- Normalization: the stored `action` is the raw outer action normalized with
  the collector's `action_stats.min/max` (`action_stats_mode =
  <collector_envelope>`). Disclose any further training normalization
  separately.

## Reset semantics

Reproducing this data requires reproducing the reset
(`metadata.json` → `reset_semantics`):

- `init_stance`: `<keyframe>`
- `init_pelvis_z`: `<0.74>` m
- `reset_warmup_steps`: `<200>` (`<4.0>` s). Episodes settle before the
  operator engages.
- `demo_start`: `<post_warmup_vr_engage_state>`

## Substrate fingerprint

Results are comparable when `substrate_version` matches. The main fields:

```text
{
  "substrate_version": "<bigym2-mj381-v1>",
  "mujoco_version": "<3.8.1>",
  "robot_model": "<g1_dex1>",
  "lowerbody_backend": "<groot_wbc_g1>",
  "lowerbody_weights": <{"<file>": "<sha256>", ...}>,
  "g1_passive_base_tilt": <true>,
  "solver": <2>,
  "contact_signature": "<a3972624f79d3e98>"
}
```

The full fingerprint is in `metadata.json` → `substrate_fingerprint`.

Package versions at collection time (`metadata.json` → `package_versions`):
`bigym <1.0.0>`, `mujoco <3.8.1>`, `mink <1.2.0>`, `numpy <2.2.6>`,
`python <3.12.3>`.

## Collection

- Method: VR teleoperation with `bigym-collect`, operator wearing
  `<headset>`.
- Attempts and kept: `<attempts>` attempts, `<kept>` kept (`<xx>%`), from
  `collection.attempts` and `collection.kept`.
  <!-- how to fill: report the totals over the whole batch. -->
- Discarded attempts: `<not released | released under data/discarded/>`

## Dataset structure

Standard LeRobot v3 layout, plus these files:

```
<dataset>/
  meta/
    episode_init_states.json   # engage snapshots (BiGym sidecar)
    alignment.json             # action alignment version (BiGym sidecar)
    source_metadata.json       # the collector's metadata.json
  metadata.json                # root copy of source_metadata.json
```

### Per-frame fields

<!-- how to fill: this table comes from the features in meta/info.json.
     Delete rows your export does not contain. -->

| Field | dtype | shape | Notes |
|-------|-------|-------|-------|
| `observation.state` | float32 | (`<50>`,) | low-dim observation, slices in `substrate_fingerprint.low_dim_component_slices` |
| `observation.images.head` | image | (3, `<84>`, `<84>`) | head camera, RGB, lossless PNG |
| `observation.images.left_wrist` | image | (3, `<84>`, `<84>`) | left wrist camera |
| `observation.images.right_wrist` | image | (3, `<84>`, `<84>`) | right wrist camera |
| `action` | float32 | (`<20>`,) | normalized outer action, the training target |
| `raw_outer_action` | float32 | (`<20>`,) | outer action before normalization |
| `expanded_action` | float32 | (`<35>`,) | outer action expanded to the full joint set |
| `lowerbody_action` | float32 | (`<15>`,) | lower-body policy output |
| `leg_joint_targets` | float32 | (`<15>`,) | leg and waist position targets sent to the actuators |
| `torso_target` | float32 | (1,) | torso yaw target |
| `lowerbody_command` | float32 | (3,) | base velocity command `(vx, vy, wz)` |
| `height_command` | float32 | (1,) | pelvis height command, m |
| `full_qpos` | float64 | (`<46>`,) | full MuJoCo `qpos` |
| `full_qvel` | float64 | (`<45>`,) | full MuJoCo `qvel` |
| `reward` | float32 | (1,) | task reward |
| `event_progress` | float32 | (1,) | per-task sub-goal progress, all `NaN` on tasks without events |
| `discount` | float32 | (1,) | |
| `wall_clock` | float64 | (1,) | seconds since collector start, a diagnostic never used for training or replay |

### Action alignment

Frame `k` pairs its observation with the transition executed from it
(`meta/alignment.json`, version 2). The final frame repeats the last
transition.

### Engage snapshots

`meta/episode_init_states.json` stores each episode's engage snapshot: the
seed and the simulator and controller state needed to replay it bit-exactly.
Consumers that ignore it still get valid trajectories.

## Intended use and limitations

- Task coverage: `<list the tasks actually present in this release>`.
- Scale: `<n>` episodes from `<one operator>`. Expect operator-specific
  strategies and limited coverage of the task's state space.

## License

- Data (this dataset): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  Attribute BiGym 2.0 and cite the entries below.
- Code: the BiGym 2.0 repository is Apache-2.0. Third-party components
  redistributed with the code (robot models, lower-body policy weights, 3D
  props) have their own terms. See `THIRD_PARTY_NOTICES.md` in the
  repository.
- Scene assets visible in the images: the rendered observations contain 3D
  props under CC0 and CC BY 4.0. Their authors are credited in
  `bigym/envs/xmls/3D_MODELS_ATTRIBUTION.md`, and that attribution carries
  over to anyone redistributing these images.

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
NVIDIA's GR00T Whole-Body Control policy. `THIRD_PARTY_NOTICES.md` §1 has
the terms its weights are distributed under.

## Contact

`<maintainer name / issue tracker URL>`
