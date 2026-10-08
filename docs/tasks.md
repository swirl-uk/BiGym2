# Tasks

The benchmark has **20 tasks**, each with 60 human VR demonstrations and an
episode budget derived from them. They are the G1 versions of upstream BiGym
tasks: scenes, spawns, reset distributions and tolerances are adapted to the
Unitree G1, so results are not comparable with BiGym 1.0 tasks of the same
name.

```python
from bigym.loco import make
env = make("move_plate")   # official configuration for this task
```

The task registry (`bigym.loco.tasks.TASKS`)
contains all 40 upstream BiGym tasks. The other 20 are registered so they can
be built, but they are not part of the benchmark yet: they will be released
later together with their demonstrations. They are listed briefly under
[Tasks released later](#tasks-released-later).
`budget_provenance(name) == "data_derived"` holds for exactly the 20
benchmark tasks.

| Scene group | Tasks | Action |
|---|---|---|
| Reaching | `reach_target_single`, `reach_target_multi_modal`, `reach_target_dual` | 20 |
| Table-top | `move_plate` | 20 |
| | `move_two_plates`, `flip_cup`, `flip_cutlery`, `stack_blocks` | 21 |
| Dishwasher | `dishwasher_close`, `dishwasher_load_cups`, `dishwasher_load_cutlery`, `dishwasher_load_plates` | 21 |
| Kitchen counter | `drawer_top_open`, `drawer_top_close` | 20 |
| | `pick_box`, `put_cups`, `saucepan_to_hob`, `sandwich_remove`, `wall_cupboard_open`, `wall_cupboard_close` | 21 |

**Action** is the action dimension. The 21-dim tasks add a torso-pitch
command (`ControllerConfig.pitch_command`, action slot 4,
radians, + = lean forward, clipped to [-0.2, 0.8]) and keep the three waist
joints in the proprioception, so their low-dimensional observation has 56
entries instead of 50. The registry sets this per task and the demonstrations
are recorded in the same layout; read it from `env.wholebody_action_layout()`
rather than hard-coding it.

## How to read a success predicate

Reward is **sparse**: `1.0` on the step where the task predicate has held
for `success_hold_seconds` (1.0 s, **50 consecutive control steps**), `0.0`
otherwise (`BiGymEnv._reward`). The hold counter is reset at the end of the
lower-body warmup, so settling cannot pre-fill it.

An episode ends in one of four ways:

1. **Task reward**: the predicate held for the required time.
2. **Task failure** (`_fail`): every task fails if the pelvis gets more than
   10 m from the origin (`MAX_DISTANCE_FROM_ORIGIN`). The plates, blocks and
   dishwasher cups/cutlery/plates tasks also fail when *any object touches
   the floor*; the sandwich tasks fail when *a gripper holds the sandwich*.
   The other tasks have no extra failure condition, so a dropped object can
   be picked up again.
3. **Physics instability**: MuJoCo reports an unhealthy state. The env
   ends the episode with reward 0, so an exploding simulation never scores.
4. **Time limit**: the episode budget.

```{warning}
**A fall makes the episode a failure.** An episode is a success when the
task reported success (`env.episode_succeeded()`: ending 1 below) and the
robot never fell during the episode (`bigym.loco.eval.is_success`). The robot has fallen when its pelvis tilts
more than about 53° from upright or drops below the controller's height
floor (the bottom of its height command range minus 0.10 m: 0.30 m for
GR00T-WBC).
`LowerBodyBase.is_failed()` checks this after every step and the env latches
it for the episode. A fall does not end the episode; one of the four endings
above still applies, and `env.episode_termination()` labels a fallen
episode `fell` whatever the task reported.
```

Two primitives recur in the predicates: `Prop.is_colliding(other)` means
*any* collider pair is in contact, and "held" means a gripper's **pad geoms**
touch the object. Cabinet and dishwasher states are **normalized** joint
positions in `[0, 1]` (0 = closed), so a tolerance of `0.1` is about ±9° on
a 90° door and ±0.038 m on a 0.38 m drawer.

Each task's environment is the G1 subclass of the upstream class
(`MovePlate` → `MovePlateG1`); the **G1 adaptation** line lists what the
subclass and its preset change.

## Episode budgets

Read `TASKS[name].episode_length` and `budget_provenance(name)` from
`bigym.loco.tasks`; see
[Episode budgets](official_configuration.md#episode-budgets). Budgets are in
env steps; divide by `demo_down_sample_rate` (10) for outer steps.

## Reaching

Source: [`bigym/envs/reach_target.py`](https://github.com/swirl-uk/BiGym2/blob/main/bigym/envs/reach_target.py)

Kinematic reaching: a coloured sphere (radius 0.05 m) must be touched by an
end-effector site. No object physics is involved.

The success radius is `reach_tolerance`, which the registry sets to
**0.05 m** for all three tasks, so the Dex1 pinch centre has to be *inside*
the ball. (The class constant `TOLERANCE = 0.1` would leave a 5 cm shell
around the ball, and a target sampled near the resting hand could be
satisfied without moving.) `robot.get_hand_pos(side)` resolves to the
`*_end_effector` site, which the G1 loader places at the **pinch centre
between the Dex1 pads** (`(0.148, 0, 0)` from `wrist_yaw_link`).

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} `reach_target_single`
```{image} images/tasks_g1/reach_target_single.png
:alt: Reach one target with the left wrist.
```
:::

:::{grid-item-card} `reach_target_multi_modal`
```{image} images/tasks_g1/reach_target_multi_modal.png
:alt: Reach one target with either wrist.
```
:::

:::{grid-item-card} `reach_target_dual`
```{image} images/tasks_g1/reach_target_dual.png
:alt: Reach two targets, one with each arm.
```
:::

::::

### `reach_target_single`

Reach the target with the left wrist. Class `ReachTargetSingleG1`.

- **Success**: `||target_pos − hand_pos|| <= 0.05` for the **left** hand. Reaching with the right hand does not count.
- **Reset randomisation**: target position = base + `U(±0.05, ±0.10, ±0.10)` m in x/y/z. Robot spawn fixed at the origin.
- **G1 adaptation**: target base `[0.45, 0, 0.8]` (upstream `[0.5, 0, 1.0]`).

### `reach_target_multi_modal`

Reach the target with either wrist. Class `ReachTargetG1`.

- **Success**: `Target.is_reached(reach_tolerance)`: `||target_pos − hand_pos|| <= 0.05` for the **left or the right** hand.
- **Reset randomisation**: same `U(±0.05, ±0.10, ±0.10)` target jitter; spawn fixed.
- **G1 adaptation**: target base `[0.45, 0, 0.8]` (upstream `[0.5, 0, 1.0]`).

### `reach_target_dual`

Reach the two targets, one with each wrist. Class `ReachTargetDualG1`.

- **Success**: **both** targets reached at once (left hand on the left target, right hand on the right).
- **Reset randomisation**: each target jitters independently by `U(±0.05, ±0.10, ±0.10)`; spawn fixed.
- **G1 adaptation**: targets `[0.45, ±0.2, 0.8]` (upstream `[0.5, ±0.2, 1.0]`).

## Table-top

Sources: [`bigym/envs/move_plates.py`](https://github.com/swirl-uk/BiGym2/blob/main/bigym/envs/move_plates.py),
[`bigym/envs/manipulation.py`](https://github.com/swirl-uk/BiGym2/blob/main/bigym/envs/manipulation.py)

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} `move_plate`
```{image} images/tasks_g1/move_plate.png
:alt: Move the plate between two draining racks.
```
:::

:::{grid-item-card} `move_two_plates`
```{image} images/tasks_g1/move_two_plates.png
:alt: Move two plates from one draining rack to the other.
```
:::

:::{grid-item-card} `flip_cup`
```{image} images/tasks_g1/flip_cup.png
:alt: Flip an upside-down cup on the table to an upright position.
```
:::

:::{grid-item-card} `flip_cutlery`
```{image} images/tasks_g1/flip_cutlery.png
:alt: Take the cutlery from the holder, flip it, and place it back.
```
:::

:::{grid-item-card} `stack_blocks`
```{image} images/tasks_g1/stack_blocks.png
:alt: Move blocks across the table and stack them in the target area.
```
:::

::::

### `move_plate`

Move the plate between two draining racks. Class `MovePlateG1`.

- **Success**: for the plate, **all** of: (1) within `_SUCCESSFUL_DIST = 0.05` m of at least one of the target rack's 6 `plate_slot_*` sites; (2) the plate's local up-axis within `_SUCCESS_ROT = 20°` of `[0, −1, 0]`; (3) `plate.is_colliding(rack_target)`; (4) **not** `plate.is_colliding(table)`; (5) not held by either gripper. Only one orientation is accepted (`dishwasher_load_plates` accepts either flip); a plate seated 180° rotated scores 0. Fail-fast: a plate touching the floor ends the episode.
- **Reset randomisation**: start and target racks each jitter `U(±0.05, ±0.05, 0)` m independently; the plate spawns in a **randomly chosen** one of the start rack's 6 slots (plus a fixed −5° tilt). Robot spawn fixed at the origin.
- **G1 adaptation**: preset `move_plates_g1.yaml` sinks the table 0.25 m (top ≈0.68 m) and puts the racks at `[0.55, ±0.24, 0.70]` (upstream `[0.7, ±0.3, 0.95]`), so the slots are at ≈0.85 m instead of 1.10 m.

### `move_two_plates`

Move two plates simultaneously from one draining rack to the other. Class `MoveTwoPlatesG1`.

- **Success**: the same five-clause predicate for **both** plates at once. Same floor fail-fast.
- **Reset randomisation**: same rack jitter; two distinct start slots drawn without replacement.
- **G1 adaptation**: same sunk table and rack positions as `move_plate`.

### `flip_cup`

Flip the upside-down cup to an upright position. Class `FlipCupG1`.

- **Success**: `_TOLERANCE = 5°`: the cup's up-axis within 5° of `+z` **and** `cup.is_colliding(cabinet.counter)` **and** not held. The tightest orientation criterion in the suite.
- **Reset randomisation**: one of 10 grid cells around `_CUP_POS`, plus `U(±0.03, ±0.03, 0)`, upside down with yaw `180° ± U(30°)`.
- **G1 adaptation**: spawn `[0.24, 0, 0]`; `_CUP_POS` pulled in to `[0.72, 0, 1]` (upstream `[0.8, 0, 1]`).

### `flip_cutlery`

Take the cutlery from the holder, flip it, and place it back. Class `FlipCutleryG1`.

- **Success**: `_TOLERANCE = 50°`: the spoon's local `+y` within 50° of `−z` **and** `spoon.is_colliding(self.cup)` **and neither the spoon nor the cup is held**. (Upstream BiGym only checks the cup, so a spoon still pinched inside the mug counts there.)
- **Reset randomisation**: mug from the same grid sampler with fully random yaw; the spoon starts at `mug + [0,0,0.15]` rotated 90° about x, with no jitter of its own.
- **G1 adaptation**: spawn `[0.24, 0, 0]`; `_CUP_POS` pulled in and lowered to `[0.72, 0, 0.86]`.

### `stack_blocks`

Move blocks across the table and stack them in the target area. Class `StackBlocksG1`.

- **Success**: sort the three blocks by height; require lowest↔target-pad, middle↔lowest and top↔middle contact, and no block held. Each block must sit at least `_STACK_MIN_RISE = 0.03 m` above the one below it, within `_STACK_MAX_XY_OFFSET = 0.05 m` horizontally. (Upstream BiGym has no such check and accepts a horizontal chain of touching cubes with one on the pad.) Fail-fast: a block touching the floor.
- **G1 adaptation**: preset `stack_blocks_g1.yaml` sinks **both** table tops to `0.70 m`; spawn `[0.18, 0, 0]`. The blocks start in one shallow row around `[0.575, 0, 0.75]` (x always `0.525–0.625`, `|y| ≤ 0.275`), roughly 0.5 m from the settled pelvis. The target is `[1.56, 0, 0.70] + U(±0.04, ±0.08, 0)`, always `0.10–0.18 m` inside the far table's exposed edge, so the task still needs a walk across.

## Dishwasher

Sources: [`bigym/envs/dishwasher.py`](https://github.com/swirl-uk/BiGym2/blob/main/bigym/envs/dishwasher.py),
[`dishwasher_cups.py`](https://github.com/swirl-uk/BiGym2/blob/main/bigym/envs/dishwasher_cups.py),
[`dishwasher_cutlery.py`](https://github.com/swirl-uk/BiGym2/blob/main/bigym/envs/dishwasher_cutlery.py),
[`dishwasher_plates.py`](https://github.com/swirl-uk/BiGym2/blob/main/bigym/envs/dishwasher_plates.py)

`dishwasher.get_state()` returns `[door, bottom_tray, middle_tray]`,
normalized. The three load tasks keep the upstream scene and spawn
(`[0, −0.6, 0]`); `dishwasher_close` raises the dishwasher.

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} `dishwasher_close`
```{image} images/tasks_g1/dishwasher_close.png
:alt: Push back all trays and close the dishwasher door.
```
:::

:::{grid-item-card} `dishwasher_load_cups`
```{image} images/tasks_g1/dishwasher_load_cups.png
:alt: Move cups from the table into the dishwasher's upper tray.
```
:::

:::{grid-item-card} `dishwasher_load_cutlery`
```{image} images/tasks_g1/dishwasher_load_cutlery.png
:alt: Move cutlery from the table holder to the dishwasher's cutlery basket.
```
:::

:::{grid-item-card} `dishwasher_load_plates`
```{image} images/tasks_g1/dishwasher_load_plates.png
:alt: Move plates from the rack to the dishwasher's lower tray.
```
:::

::::

### `dishwasher_close`

Push back all trays and close the door of the dishwasher. Class `DishwasherCloseG1`.

- **Success**: `np.allclose(get_state(), 0, atol=0.05)`. `_TOLERANCE = 0.05` is about ±4.5° on the door, ±0.025 m on the bottom tray and ±0.0225 m on the middle tray.
- **Reset randomisation**: none; reset sets `(door, bottom, middle) = (1, 1, 1)`.
- **G1 adaptation**: preset `dishwasher_g1_raised.yaml` stands the free-standing dishwasher on a 0.25 m plinth, so the open door lies at 0.41–0.50 m instead of 0.16–0.25 m, which the G1 cannot reach even at its 0.40 m minimum height with the torso pitched. The upstream spawn `[0, −0.8, 0]` is kept.

### `dishwasher_load_cups`

Move the cups from the table into the dishwasher's upper tray. Class `DishwasherLoadCupsG1`.

- **Success**: each of the two cups `is_colliding(dishwasher.tray_middle.colliders)` **and** not held. No orientation requirement. Fail-fast: a cup touching the floor.
- **Reset randomisation**: dishwasher at `(door=1, bottom=0, middle=1)`. Cup *i* at `[0.6, −0.6, 1] + i·[0, 0.15, 0] + U(±0.05, ±0.02, 0)`, flipped 180° about x, then `180° ± U(30°)` about z.
- **G1 adaptation**: none. The cups are about 0.71 m from the settled pelvis, one short step.

### `dishwasher_load_cutlery`

Move the cutlery from the holder on the table into the dishwasher's cutlery basket. Class `DishwasherLoadCutleryG1`.

- **Success**: knife and fork both `is_colliding(dishwasher.basket.colliders)` **and** not held. Fail-fast: a cutlery item touching the floor.
- **Reset randomisation**: dishwasher at `(door=1, bottom=1, middle=0)`. Mug at `[0.65, −0.6, 0.86] + U(±0.05, ±0.05, 0)` with yaw `U(±90°)`; knife and fork stand in the mug at `mug + [0,0,0.15] + 0.02·[cos θ, sin θ, 0]`, `θ = i·90° ± U(5°)`.
- **G1 adaptation**: none.

### `dishwasher_load_plates`

Move the plates from the rack into the dishwasher's lower tray. Class `DishwasherLoadPlatesG1`.

- **Success**: per plate (two): local up-axis within `_TOLERANCE = 20°` of `[0, −1, 0]` **or** `[0, +1, 0]` (either flip) **and** `is_colliding(dishwasher.tray_bottom.colliders)` **and** not held. Fail-fast: a plate touching the floor.
- **Reset randomisation**: dishwasher at `(door=1, bottom=1, middle=0)`. The drainer is at `[0.6, −0.6, 0.86] + U(±0.05, ±0.05, 0)`; the plates start on 2 of its sites drawn from `sites[::2][:4]`, offset `[0, −0.01, 0.05]`, with a fixed rotation.
- **G1 adaptation**: none.

## Kitchen counter

Sources: [`bigym/envs/cupboards.py`](https://github.com/swirl-uk/BiGym2/blob/main/bigym/envs/cupboards.py),
[`bigym/envs/pick_and_place.py`](https://github.com/swirl-uk/BiGym2/blob/main/bigym/envs/pick_and_place.py)

The drawer and wall-cupboard tasks measure success on normalized joint
positions from `ModularCabinet.get_state()` with `TOLERANCE = 0.1`, and have
no fail-fast condition. Several G1 presets lower the wall cabinet 0.32 m
and pull it 0.2 m forward so its shelf and handles are within the G1's reach
(the stock cabinet is above its fingertip ceiling).

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} `drawer_top_open`
```{image} images/tasks_g1/drawer_top_open.png
:alt: Open the top drawer of the kitchen cabinet.
```
:::

:::{grid-item-card} `drawer_top_close`
```{image} images/tasks_g1/drawer_top_close.png
:alt: Close the top drawer of the kitchen cabinet.
```
:::

:::{grid-item-card} `pick_box`
```{image} images/tasks_g1/pick_box.png
:alt: Pick up the box from the side table and place it on the counter.
```
:::

:::{grid-item-card} `put_cups`
```{image} images/tasks_g1/put_cups.png
:alt: Pick up cups from the table and put them into the closed wall cabinet.
```
:::

:::{grid-item-card} `saucepan_to_hob`
```{image} images/tasks_g1/saucepan_to_hob.png
:alt: Take the saucepan from the closed cabinet and place it on the hob.
```
:::

:::{grid-item-card} `sandwich_remove`
```{image} images/tasks_g1/sandwich_remove.png
:alt: Take the sandwich out of the frying pan.
```
:::

:::{grid-item-card} `wall_cupboard_open`
```{image} images/tasks_g1/wall_cupboard_open.png
:alt: Open the doors of the wall cabinet.
```
:::

:::{grid-item-card} `wall_cupboard_close`
```{image} images/tasks_g1/wall_cupboard_close.png
:alt: Close the doors of the wall cabinet.
```
:::

::::

### `drawer_top_open`

Open the top drawer of the kitchen cabinet. Class `DrawerTopOpenG1`.

- **Success**: `np.allclose(cabinet_drawers.get_state()[-1], 1, atol=0.1)`: the top drawer at least ~90% out.
- **Reset randomisation**: `initialization_profile="g1_id_v1"` (set by the registry): robot `x ~ U(0.12, 0.14)`, `y ~ U(−0.05, 0.05)`, yaw `~ U(−5°, +5°)`; the top drawer starts at `U(0.0, 0.15)` and is held there through the warmup. The `upstream` profile has no randomisation.
- **G1 adaptation**: spawn `[0.12, 0, 0]` (upstream `[−0.2, 0, 0]`). The wall cabinet stays at stock height, unlike the wall-cupboard tasks; the task never touches it, but it is visible in the camera images.

### `drawer_top_close`

Close the top drawer of the kitchen cabinet. Class `DrawerTopCloseG1`.

- **Success**: `np.allclose(state[-1], 0, atol=0.1)`.
- **Reset randomisation**: `g1_id_v1`, the same robot pose sampling; the top drawer starts at `U(0.85, 1.0)`. The `upstream` profile opens it fully with no jitter.
- **G1 adaptation**: spawn `[0.12, 0, 0]`; stock-height wall cabinet, as in `drawer_top_open`.

### `pick_box`

Pick up the box from the side table and place it on the counter. Class `PickBoxG1`.

- **Success**: the box `is_colliding(cabinet_base.counter)` **and** not held. No fail-fast.
- **Reset randomisation**: box at `[0.70, −1.0] + U(±0.03, ±0.03, 0)`, resting on the side table (centre z `0.6255`, settles at `0.625`), rotated 90° about y, then `U(±180°)` about x.
- **G1 adaptation**: preset `cabinet_door_pick_box_g1.yaml`. The G1 cannot reach the floor (the minimum height command keeps the hands above about 0.41 m), so the box starts on a side table sunk by 0.45 m (top ≈0.50 m), 0.10 m inside its front edge; the pickup is a moderate squat plus torso pitch. The base cabinet is sunk 0.15 m (counter top 0.71 m instead of 0.86 m), so placing the upright box keeps the hands near the G1's relaxed hand height. The box has the stock geometry, texture and friction but weighs **3 kg** instead of 5 kg: Dex1 has no palm, so the box is carried by squeezing it between the open grippers, and a 5 kg box slips out while walking.

### `put_cups`

Pick up the cups from the table and put them into the closed wall cabinet. Class `PutCupsG1`.

- **Success**: both cups `is_colliding(cabinet_wall.shelf_bottom)` **and** not held.
- **Reset randomisation**: two of three grid cells around `[0.8, 0, 1]` (y ∈ {−0.15, 0, +0.15}), each plus `U(±0.03, ±0.03, 0)` and yaw `180° ± U(30°)`. The glass doors start closed.
- **G1 adaptation**: preset `counter_base_wall_1x1_g1.yaml` (lowered wall cabinet); spawn `[0.24, 0, 0]`.

### `saucepan_to_hob`

Take the saucepan from the closed cabinet and place it on the hob. Class `SaucepanToHobG1`.

- **Success**: the saucepan `is_colliding(cabinet_base.hob)` **and** not held.
- **Reset randomisation**: saucepan inside the cabinet at `_SAUCEPAN_POS + U(±0.05, ±0.05, 0)`, yaw 90° `± U(20°)`.
- **G1 adaptation**: spawn `[0.24, 0, 0]`; `_SAUCEPAN_POS = [0.77, 0.1, 0.5]` (upstream `[0.85, 0.1, 0.5]`), the furthest-forward position that keeps the pan fully on the shelf (front edge x = `0.652`). At the upstream position the handle is at the end of a straight arm for a squatting G1.

### `sandwich_remove`

Remove the sandwich from the frying pan. Class `RemoveSandwichG1`.

- **Success**: the sandwich's up-axis within `_TOLERANCE = 10°` of `+z` **or** `−z` (lying flat, either face) **and** `sandwich.is_colliding(self.board)`. `_success` has **no** not-held clause; release is enforced by the fail-fast instead.
- **Fail-fast**: any gripper holding the sandwich ends the episode, so the spatula is mandatory.
- **Reset randomisation**: pan on hob site 1 plus `U(±0.02, ±0.02, 0)`, spatula at a fixed offset from the pan; the toasted sandwich starts on the pan plus `U(±0.01, ±0.01, 0)` with yaw `U(±180°)`; board at `[0.7, −0.6, 0.88]` with yaw `U(±5°)`.
- **G1 adaptation**: spawn `[0.10, 0.25, 0]`, 0.25 m to the left of upstream, so the left shoulder faces the pan handle (which points along the counter to the robot's left) and the right shoulder faces the spatula. Pan yaw jitter is `U(±15°)` instead of `U(±30°)`: with ±30° the handle is out of reach for the yaws that turn it away. The robot walks the last 0.25 m to the counter.

### `wall_cupboard_open`

Open the doors of the wall cabinet. Class `WallCupboardOpenG1`.

- **Success**: `np.allclose(cabinet_wall.get_state(), 1, atol=0.1)`, **both** doors.
- **Reset randomisation**: none; every seed starts from the same state.
- **G1 adaptation**: preset `counter_base_wall_3x1_g1.yaml` (lowered wall cabinet); spawn `[0.24, 0, 0]`.

### `wall_cupboard_close`

Close the doors of the wall cabinet. Class `WallCupboardCloseG1`.

- **Success**: `np.allclose(cabinet_wall.get_state(), 0, atol=0.1)`.
- **Reset randomisation**: none; reset opens both doors fully, the same for every seed.
- **G1 adaptation**: lowered wall-cabinet preset; spawn `[-0.2, 0, 0]`, the upstream value. The doors open toward the robot, and from here their edges are about 0.67 m from the shoulders, one step away; from a closer spawn the doors could be closed without stepping.

## Tasks released later

These tasks are in the registry and can be built with `make`, but
they are not part of the benchmark: they have no published demonstrations
yet, and their budgets are upstream floating-base placeholders
(`budget_provenance(name) == "upstream_placeholder"`). They will be released
later. All of them use the 21-dim action layout.

| Task | Goal | Success predicate |
|---|---|---|
| `drawers_open_all` | Open all three drawers. | `allclose(cabinet_drawers.get_state(), 1, atol=0.1)` |
| `drawers_close_all` | Close all three drawers (they start open). | `allclose(..., 0, atol=0.1)` |
| `cupboards_open_all` | Open all drawers and doors of the kitchen set. | all four cabinets `allclose(get_state(), 1, atol=0.1)`, 7 joints |
| `cupboards_close_all` | Close all drawers and doors (they start open). | all four cabinets at 0, `atol=0.1` |
| `dishwasher_open` | Open the door and pull out both trays. | `allclose(dishwasher.get_state(), 1, atol=0.05)` |
| `dishwasher_open_trays` | Pull out both trays; the door starts open. | trays only: `allclose(get_state()[1:], 1, atol=0.05)` |
| `dishwasher_close_trays` | Push both trays back; the door starts open. | trays only, at 0 |
| `dishwasher_unload_cups` | Move two cups from the upper tray to the counter. | each cup touches any collider of either base cabinet, not held |
| `dishwasher_unload_cups_long` | Put a cup in the closed wall cabinet, then close the dishwasher and the cabinet. | dishwasher and wall cabinet at 0 (`atol=0.1`), cup on `wall_cabinet.shelf_bottom`, not held |
| `dishwasher_unload_cutlery` | Move knife and fork from the basket to a tray on the counter. | each item touches the tray, not held |
| `dishwasher_unload_cutlery_long` | Put the fork in the tray inside a closed drawer, then close the dishwasher and the drawer. | dishwasher and the 4 small drawers at 0 (`atol=0.1`), fork in the tray, not held |
| `dishwasher_unload_plates` | Move two plates from the lower tray to the drainer. | each plate touches the drainer, not held; no orientation check |
| `dishwasher_unload_plates_long` | Put a plate in the rack inside the closed wall cabinet, then close the dishwasher and the cabinet. | dishwasher and wall cabinet at 0 (`_JOINT_TOLERANCE = 0.1`), plate on the drainer, not held |
| `take_cups` | Take two cups out of the closed wall cabinet and put them on the counter. | both cups touch `cabinet_base.counter`, not held |
| `store_box` | Move the box from the counter to the shelf in the cabinet below. | box touches `cabinet_base.shelf`, not held; the door need not be closed |
| `store_kitchenware` | Move the saucepan and the pan from the hob to the cabinet below. | both touch `cabinet_base.shelf`, neither held |
| `sandwich_toast` | Use the spatula to put the sandwich in the frying pan. | sandwich flat (10°, either face), pan on the hob, sandwich on the pan; fails if a gripper holds the sandwich |
| `sandwich_flip` | Flip the sandwich in the frying pan with the spatula. | sandwich upside down (within 10° of `−z`), pan on the hob, sandwich on the pan; same fail-fast |
| `store_groceries_lower` | Put a random set of groceries in the cabinets below the counter. | each of the 4 props (drawn from 7) on a cabinet shelf, not held |
| `store_groceries_upper` | Put a random set of groceries in the wall cabinet and on the open shelf. | each of the 4 props on a wall-cabinet shelf or the open shelf, not held |

The source of each task is in the module listed in
[`bigym/loco/tasks.py`](https://github.com/swirl-uk/BiGym2/blob/main/bigym/loco/tasks.py).
