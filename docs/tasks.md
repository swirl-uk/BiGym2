# Tasks

The task registry `bigym.loco.tasks.TASKS` holds 40 tasks, the G1 versions of
the [BiGym](https://arxiv.org/abs/2407.07788) tasks. 20 of them form the benchmark. Each has 60 human VR
demonstrations and an episode budget derived from them. The other 20 are
listed under [Tasks released later](#tasks-released-later). Scenes, resets and
tolerances are adapted to the Unitree G1 with Dex1-1 grippers, so scores are
not comparable with BiGym tasks of the same name.

```python
from bigym.loco import make
env = make("move_plate")   # official configuration for this task
```

`TASKS[name].episode_length` is the budget in env steps. See
[Episode budgets](official_configuration.md#episode-budgets).

| Scene group | Tasks | Action |
|---|---|---|
| Reaching | `reach_target_single`, `reach_target_multi_modal`, `reach_target_dual` | 20 |
| Table-top | `move_plate` | 20 |
| | `move_two_plates`, `flip_cup`, `flip_cutlery`, `stack_blocks` | 21 |
| Dishwasher | `dishwasher_close`, `dishwasher_load_cups`, `dishwasher_load_cutlery`, `dishwasher_load_plates` | 21 |
| Kitchen counter | `drawer_top_open`, `drawer_top_close` | 20 |
| | `pick_box`, `put_cups`, `saucepan_to_hob`, `sandwich_remove`, `wall_cupboard_open`, `wall_cupboard_close` | 21 |

Action is the action dimension. The 21-dim tasks add a torso-pitch command in
slot 4, and their low-dim observation has 56 entries instead of 50 because it
includes the waist joints. The demonstrations use the same layout.
[Action layout](official_configuration.md#action-layout) lists the slots, and
`env.wholebody_action_layout()` reports them.

## Reward and termination

Reward is sparse: 1.0 once the success condition has held for 1.0 s
(50 control steps), 0 otherwise. An episode ends on the first of:

- success,
- task failure: the pelvis is more than 10 m from the origin, or the task's
  fail-fast condition is met,
- physics instability,
- the episode budget.

A fall does not end the episode. It labels the episode `fell`, which never
counts as a success. [Success and falls](official_configuration.md#success-and-falls)
defines a fall.

Terms used below:

- Held: a gripper pad touches the object. Most success conditions require
  every object to be released.
- Fail-fast: a condition that ends the episode early with no reward. Tasks
  without one let the robot pick up a dropped object again.
- Settled pelvis: the pelvis position after the 200 warmup steps that `reset`
  runs before the policy takes over.
- Doors, drawers and dishwasher trays report a normalised position from 0
  (closed) to 1 (open). A tolerance of 0.1 is 9° on a 90° door and 0.038 m on
  a 0.38 m drawer. The dishwasher trays travel 0.50 m (bottom) and 0.45 m
  (middle).

## Reaching

Code: [`bigym/envs/reach_target.py`](https://github.com/swirl-uk/BiGym2/blob/main/bigym/envs/reach_target.py)

Touch a coloured sphere of radius 0.05 m with the hand. No objects are involved.

- **Success**: the pinch centre between the gripper pads is within 0.05 m of
  the target centre, so it must be inside the ball.
- **Reset**: each target moves ±0.05 m in x, ±0.10 m in y and ±0.10 m in z
  around its base position, 0.45 m ahead and 0.8 m up. The robot spawns at
  the origin.
- **vs. BiGym**: targets sit 0.2 m lower and 0.05 m closer. BiGym's 1.0 m is
  at the G1's shoulder line. BiGym counts 0.1 m as reached, which passes on a
  fingertip graze.

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} `reach_target_single`
```{image} images/tasks/reach_target_single.jpg
:alt: Reach one target with the left wrist.
```
:::

:::{grid-item-card} `reach_target_multi_modal`
```{image} images/tasks/reach_target_multi_modal.jpg
:alt: Reach one target with either wrist.
```
:::

:::{grid-item-card} `reach_target_dual`
```{image} images/tasks/reach_target_dual.jpg
:alt: Reach two targets, one with each arm.
```
:::

::::

### `reach_target_single`

One target in the centre. Only the left hand counts.

### `reach_target_multi_modal`

One target in the centre. Either hand counts.

### `reach_target_dual`

Two targets 0.2 m left and right of centre, each jittered independently. The
left hand must reach the left target and the right hand the right one, at the
same time.

## Table-top

Code: [`bigym/envs/move_plates.py`](https://github.com/swirl-uk/BiGym2/blob/main/bigym/envs/move_plates.py),
[`bigym/envs/manipulation.py`](https://github.com/swirl-uk/BiGym2/blob/main/bigym/envs/manipulation.py)

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} `move_plate`
```{image} images/tasks/move_plate.jpg
:alt: Move the plate between two draining racks.
```
:::

:::{grid-item-card} `move_two_plates`
```{image} images/tasks/move_two_plates.jpg
:alt: Move two plates from one draining rack to the other.
```
:::

:::{grid-item-card} `flip_cup`
```{image} images/tasks/flip_cup.jpg
:alt: Flip an upside-down cup on the table to an upright position.
```
:::

:::{grid-item-card} `flip_cutlery`
```{image} images/tasks/flip_cutlery.jpg
:alt: Take the cutlery from the holder, flip it, and place it back.
```
:::

:::{grid-item-card} `stack_blocks`
```{image} images/tasks/stack_blocks.jpg
:alt: Move blocks across the table and stack them in the target area.
```
:::

::::

### `move_plate`

- **Goal**: move the plate from the left draining rack to the right one.
- **Success**: the plate is within 0.05 m of one of the 6 slots of the target
  rack, stands on edge with its top face toward the robot's right within 20°,
  touches the target rack, does not touch the table, and is not held. A plate
  facing left fails. Fail-fast: the plate touches the floor.
- **Reset**: each rack moves ±0.05 m in x and y. The plate starts in a random
  slot of the left rack.
- **vs. BiGym**: the table is 0.25 m lower (top at 0.70 m) and the racks are
  closer, so the slots sit at about 0.85 m instead of 1.10 m.

### `move_two_plates`

- **Goal**: move both plates from the left draining rack to the right one.
- **Success**: both plates meet the `move_plate` condition at once. Same
  fail-fast.
- **Reset**: as `move_plate`, with the plates in two different slots.
- **vs. BiGym**: same table and racks as `move_plate`.

### `flip_cup`

- **Goal**: turn the upside-down cup upright.
- **Success**: the cup is upright within 5°, touches the counter, and is not
  held. This is the tightest orientation tolerance in the suite.
- **Reset**: the cup starts upside down in one of 8 spots (a 2 × 4 grid,
  0.1 m apart), moved ±0.03 m in x and y, with yaw 180° ± 30°.

### `flip_cutlery`

- **Goal**: take the spoon out of the mug, turn it over, and put it back.
- **Success**: the spoon points down within 50°, touches the mug, and neither
  the spoon nor the mug is held.
- **Reset**: the mug starts in one of the same 8 spots, moved ±0.03 m, with
  random yaw. The spoon stands in it pointing up.
- **vs. BiGym**: BiGym only checks that the mug is released, so a spoon still
  pinched in the mug counts there.

### `stack_blocks`

- **Goal**: carry three blocks to the far table and stack them on the green
  pad.
- **Success**: the lowest block touches the pad, each block touches the one
  below it and sits at least 0.03 m higher and within 0.05 m horizontally of
  it, and no block is held. Fail-fast: a block touches the floor.
- **Reset**: the blocks start in 3 of 4 spots in a row 0.15 m apart on the
  near table, each moved ±0.05 m with random yaw. The pad sits on the far
  table, moved ±0.04 m in x and ±0.08 m in y, always 0.10 to 0.18 m inside its
  edge. The robot has to walk across.
- **vs. BiGym**: both tables are 0.25 m lower (tops at 0.70 m). BiGym also
  accepts a horizontal chain of touching blocks.

## Dishwasher

Code: [`bigym/envs/dishwasher.py`](https://github.com/swirl-uk/BiGym2/blob/main/bigym/envs/dishwasher.py),
[`dishwasher_cups.py`](https://github.com/swirl-uk/BiGym2/blob/main/bigym/envs/dishwasher_cups.py),
[`dishwasher_cutlery.py`](https://github.com/swirl-uk/BiGym2/blob/main/bigym/envs/dishwasher_cutlery.py),
[`dishwasher_plates.py`](https://github.com/swirl-uk/BiGym2/blob/main/bigym/envs/dishwasher_plates.py)

The three load tasks keep the BiGym scene and spawn. The dishwasher door is
open at the start of every task.

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} `dishwasher_close`
```{image} images/tasks/dishwasher_close.jpg
:alt: Push back all trays and close the dishwasher door.
```
:::

:::{grid-item-card} `dishwasher_load_cups`
```{image} images/tasks/dishwasher_load_cups.jpg
:alt: Move cups from the table into the dishwasher's upper tray.
```
:::

:::{grid-item-card} `dishwasher_load_cutlery`
```{image} images/tasks/dishwasher_load_cutlery.jpg
:alt: Move cutlery from the table holder to the dishwasher's cutlery basket.
```
:::

:::{grid-item-card} `dishwasher_load_plates`
```{image} images/tasks/dishwasher_load_plates.jpg
:alt: Move plates from the rack to the dishwasher's lower tray.
```
:::

::::

### `dishwasher_close`

- **Goal**: push both trays in and close the door.
- **Success**: the door is within 4.5° of closed, the bottom tray within
  0.025 m and the middle tray within 0.0225 m.
- **Reset**: none. The door is open and both trays are out on every seed.
- **vs. BiGym**: the dishwasher stands on a 0.25 m plinth, so the open door
  lies at 0.41 to 0.50 m instead of 0.16 to 0.25 m. The G1 cannot reach the
  lower door, even at its 0.40 m minimum height with the torso pitched.

### `dishwasher_load_cups`

- **Goal**: put the two cups from the counter into the upper tray.
- **Success**: both cups touch the middle tray and are not held. Any
  orientation counts. Fail-fast: a cup touches the floor.
- **Reset**: the middle tray is out. The cups start upside down 0.15 m apart,
  each moved ±0.05 m in x and ±0.02 m in y, with yaw 180° ± 30°. They are
  0.65 to 0.75 m from the settled pelvis, one short step.

### `dishwasher_load_cutlery`

- **Goal**: move the knife and fork from the mug on the counter into the
  cutlery basket.
- **Success**: both touch the basket and are not held. Fail-fast: an item
  touches the floor.
- **Reset**: the bottom tray is out. The mug moves ±0.05 m in x and y, with
  yaw ±90°. The knife and fork stand in it at 90° to each other, ± 5°.

### `dishwasher_load_plates`

- **Goal**: move the two plates from the drainer into the lower tray.
- **Success**: each plate stands on edge within 20°, facing either way,
  touches the bottom tray, and is not held. Fail-fast: a plate touches the
  floor.
- **Reset**: the bottom tray is out. The drainer moves ±0.05 m in x and y.
  The plates start in 2 of its 3 odd-numbered slots.

## Kitchen counter

Code: [`bigym/envs/cupboards.py`](https://github.com/swirl-uk/BiGym2/blob/main/bigym/envs/cupboards.py),
[`bigym/envs/pick_and_place.py`](https://github.com/swirl-uk/BiGym2/blob/main/bigym/envs/pick_and_place.py)

The BiGym wall cabinet is above the G1's fingertip ceiling of about 1.55 m.
Where a task uses it, the G1 scene lowers it 0.32 m and pulls it 0.2 m
forward, which puts the bottom shelf at about 1.15 m and the handles at about
1.2 m.

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} `drawer_top_open`
```{image} images/tasks/drawer_top_open.jpg
:alt: Open the top drawer of the kitchen cabinet.
```
:::

:::{grid-item-card} `drawer_top_close`
```{image} images/tasks/drawer_top_close.jpg
:alt: Close the top drawer of the kitchen cabinet.
```
:::

:::{grid-item-card} `pick_box`
```{image} images/tasks/pick_box.jpg
:alt: Pick up the box from the side table and place it on the counter.
```
:::

:::{grid-item-card} `put_cups`
```{image} images/tasks/put_cups.jpg
:alt: Pick up cups from the table and put them into the closed wall cabinet.
```
:::

:::{grid-item-card} `saucepan_to_hob`
```{image} images/tasks/saucepan_to_hob.jpg
:alt: Take the saucepan from the closed cabinet and place it on the hob.
```
:::

:::{grid-item-card} `sandwich_remove`
```{image} images/tasks/sandwich_remove.jpg
:alt: Take the sandwich out of the frying pan.
```
:::

:::{grid-item-card} `wall_cupboard_open`
```{image} images/tasks/wall_cupboard_open.jpg
:alt: Open the doors of the wall cabinet.
```
:::

:::{grid-item-card} `wall_cupboard_close`
```{image} images/tasks/wall_cupboard_close.jpg
:alt: Close the doors of the wall cabinet.
```
:::

::::

### `drawer_top_open`

- **Goal**: open the top drawer.
- **Success**: the top drawer is at least 90% open, within 0.038 m of fully
  out.
- **Reset**: the robot spawns at x 0.12 to 0.14 m, y ±0.05 m, yaw ±5°. The
  top drawer starts 0 to 15% open and is held there through the warmup.

### `drawer_top_close`

- **Goal**: close the top drawer.
- **Success**: the top drawer is within 0.038 m of closed.
- **Reset**: the robot pose is drawn as in `drawer_top_open`. The top drawer
  starts 85 to 100% open.

### `pick_box`

- **Goal**: pick up the box from the side table and put it on the counter.
- **Success**: the box touches the counter, is not held, has its centre over
  the counter, and has its bottom within 0.03 m of the counter top.
- **Reset**: the box stands upright on a side table 0.7 m ahead and 1 m to
  the robot's right, moved ±0.03 m in x and y, with random yaw. The robot has to turn and walk
  to it.
- **vs. BiGym**: the G1 cannot reach the floor. At its minimum height its
  hands stay above about 0.4 m with the torso upright and 0.33 m with full
  pitch. The box therefore starts on a side table with its top at 0.50 m, and
  the counter is 0.15 m lower (top at 0.71 m). The box weighs 3 kg instead of
  5 kg. Dex1-1 has no palm, so the box is carried by squeezing it between the
  open grippers, and a 5 kg box slips out while walking. BiGym also accepts a
  box pressed against the side of the counter.

### `put_cups`

- **Goal**: put the two cups from the counter into the wall cabinet.
- **Success**: both cups touch the bottom shelf of the wall cabinet and are
  not held. The glass doors start closed and need not be closed again.
- **Reset**: the cups take 2 of 3 spots 0.15 m apart across the counter, each
  moved ±0.03 m in x and y, with yaw 180° ± 30°.
- **vs. BiGym**: lowered wall cabinet.

### `saucepan_to_hob`

- **Goal**: take the saucepan out of the closed cabinet and put it on the
  hob.
- **Success**: the saucepan touches the hob and is not held.
- **Reset**: the saucepan moves ±0.05 m in x and y on the shelf, with yaw
  90° ± 20°.
- **vs. BiGym**: the saucepan starts 0.08 m further forward, at the front of
  the shelf. At the BiGym position the handle is at the end of a straight arm
  for a squatting G1.

### `sandwich_remove`

- **Goal**: use the spatula to move the sandwich from the frying pan onto the
  chopping board.
- **Success**: the sandwich lies flat within 10°, either face up, and touches
  the board. Fail-fast: a gripper touches the sandwich, so the spatula is
  required.
- **Reset**: the pan sits on the hob, moved ±0.02 m, with yaw ±15°. The
  spatula keeps a fixed offset from the pan. The sandwich lies on the pan,
  moved ±0.01 m, with random yaw. The board has yaw ±5°.
- **vs. BiGym**: the robot spawns 0.25 m to the left, so the left shoulder
  faces the pan handle and the right shoulder faces the spatula. It walks the
  last 0.25 m to the counter. The pan yaw varies by ±15° instead of ±30°,
  which keeps the handle within reach on every seed.

### `wall_cupboard_open`

- **Goal**: open both doors of the wall cabinet.
- **Success**: both doors are within 9° of fully open.
- **Reset**: none. The doors start closed on every seed.
- **vs. BiGym**: lowered wall cabinet.

### `wall_cupboard_close`

- **Goal**: close both doors of the wall cabinet.
- **Success**: both doors are within 9° of closed.
- **Reset**: none. The doors start fully open on every seed.
- **vs. BiGym**: lowered wall cabinet. The doors open toward the robot, and
  their edges are 0.64 m ahead of the settled pelvis, one step away.

## Tasks released later

These tasks can be built with `make` but are not part of the benchmark. They
have no published demonstrations yet, and their budgets are placeholders taken
from BiGym. All of them use the 21-dim action.

| Task | Goal |
|---|---|
| `drawers_open_all` | Open all three drawers. |
| `drawers_close_all` | Close all three drawers. They start open. |
| `cupboards_open_all` | Open every drawer and door of the kitchen set. |
| `cupboards_close_all` | Close every drawer and door. They start open. |
| `dishwasher_open` | Open the door and pull out both trays. |
| `dishwasher_open_trays` | Pull out both trays. The door starts open. |
| `dishwasher_close_trays` | Push both trays back. The door starts open. |
| `dishwasher_unload_cups` | Move two cups from the upper tray to the counter. |
| `dishwasher_unload_cups_long` | Put a cup in the closed wall cabinet, then close the dishwasher and the cabinet. |
| `dishwasher_unload_cutlery` | Move the knife and fork from the basket to a tray on the counter. |
| `dishwasher_unload_cutlery_long` | Put the fork in the tray inside a closed drawer, then close the dishwasher and the drawer. |
| `dishwasher_unload_plates` | Move two plates from the lower tray to the drainer. |
| `dishwasher_unload_plates_long` | Put a plate in the rack inside the closed wall cabinet, then close the dishwasher and the cabinet. |
| `take_cups` | Take two cups out of the closed wall cabinet and put them on the counter. |
| `store_box` | Move the box from the counter to the shelf in the cabinet below. |
| `store_kitchenware` | Move the saucepan and the pan from the hob to the cabinet below. |
| `sandwich_toast` | Use the spatula to put the sandwich in the frying pan. |
| `sandwich_flip` | Flip the sandwich in the frying pan with the spatula. |
| `store_groceries_lower` | Put a random set of groceries in the cabinets below the counter. |
| `store_groceries_upper` | Put a random set of groceries in the wall cabinet and on the open shelf. |
