# Environment API

You control a Unitree G1 humanoid in MuJoCo. Walking and balance are handled
by a frozen whole-body controller; you command the base and the upper body.
The simulator runs in a separate server process. You talk to it through
`harness/client.py`; `run_episodes.py` does this for you.

## Control loop

- Control rate 50 Hz (one `act` call = 0.02 s of simulated time).
- An episode ends on success, on a task-specific failure (an object on the floor, the robot far from the scene), or at the time limit (`obs["time_limit"]` steps). A fall is recorded in `obs["fell"]` but does not end the episode; a fallen robot runs out the clock.
- Success requires the task condition to hold continuously for 1 s (50 consecutive steps).

## Action: 20 floats, physical units

| index | meaning | range |
|---|---|---|
| 0 | `vx` forward velocity command, body frame, m/s | -1..1 |
| 1 | `vy` lateral velocity command, m/s (+ = left) | -1..1 |
| 2 | `height` pelvis height command, m, absolute | 0.4..1.0 (0.74 = normal stance) |
| 3 | `wz` yaw rate command, rad/s (+ = counter-clockwise) | -1..1 |
| 4..10 | left arm joint targets, rad: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw | joint limits |
| 11..17 | right arm joint targets, same order | joint limits |
| 18 | left gripper command | 0..1 (0 = fully open, 1 = fully closed) |
| 19 | right gripper command | 0..1 |

Notes on the base controller (important):
- Velocity commands are followed with lag and some wobble. Read the pelvis pose from `low_dim_obs` every step and close the loop; do not dead-reckon.
- A planar speed below 0.055 m/s is treated as "stand still". Use 0 or at least 0.06.
- Standing still, the base drifts a few centimetres. Heights above ~0.8 m lock the knees; squatting (lower height) is how you reach low objects. While squatting the controller does not step; walk first, then squat in place.
- Arm targets are absolute joint angles tracked by position control; large jumps are followed over a few tenths of a second. Your commands are applied as given: nothing between you and the robot smooths or rate-limits them.

## Observation: dict returned by reset/step

| key | shape | meaning |
|---|---|---|
| `t` | int | steps since reset |
| `time_limit` | int | max steps in the episode |
| `fell` | bool | the controller reported a fall at some point in this episode (latched; the episode continues) |
| `low_dim_obs` | (50,) | the proprioceptive vector the learned policies get; layout below |

<!-- layout: 20 -->
`low_dim_obs` layout (50 floats; the learned policies get exactly this vector and nothing else about the body):

| index | content |
|---|---|
| 0..6 | left arm joint positions, rad: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw (same order as the action) |
| 7..8 | left finger joint positions |
| 9..15 | right arm joint positions, same order |
| 16..17 | right finger joint positions |
| 18..21 | pelvis x, y, z (m, world frame) and yaw (rad, world frame, 0 = +x axis) |
| 22..43 | velocities of entries 0..21, same order (rad/s, m/s) |
| 44..45 | left, right gripper state (0 open .. 1 closed, the scale of the gripper command) |
| 46..49 | pelvis x, y, z, yaw again, as tracked by the base controller |
<!-- /layout: 20 -->
<!-- layout: 21 -->
`low_dim_obs` layout (56 floats; the learned policies get exactly this vector and nothing else about the body):

| index | content |
|---|---|
| 0..2 | waist joint positions, rad: yaw, roll, pitch |
| 3..9 | left arm joint positions, rad: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw (same order as the action) |
| 10..11 | left finger joint positions |
| 12..18 | right arm joint positions, same order |
| 19..20 | right finger joint positions |
| 21..24 | pelvis x, y, z (m, world frame) and yaw (rad, world frame, 0 = +x axis) |
| 25..49 | velocities of entries 0..24, same order (rad/s, m/s) |
| 50..51 | left, right gripper state (0 open .. 1 closed, the scale of the gripper command) |
| 52..55 | pelvis x, y, z, yaw again, as tracked by the base controller |
<!-- /layout: 21 -->

Task-specific keys are listed in `docs/task.md`.

## Tools available inside the policy

- `tools.hold_action()` -> 20 floats: zero base velocity, current height, current arm and gripper targets. Start from this and overwrite what you need.
- `tools.image(camera, width=84, height=84)` -> uint8 array (height, width, 3).
  Cameras: `head`, `left_wrist`, `right_wrist`, the three cameras mounted on the
  robot (the 84x84 views a learned policy gets; this is also the largest size you
  may request, larger requests are refused). There is no other camera: there is no
  external view of the robot. Rendering costs no budget but takes a few milliseconds.
- `tools.render(camera, path)` saves a PNG from one of the robot's own cameras (`head`, `left_wrist`, `right_wrist`; 84x84, the observation itself) under the sandbox and returns its path. Look at the file with your image viewer. Rendering costs no budget.

## Running episodes

```
python run_episodes.py                      # every seed in seeds.json, sequentially
python run_episodes.py --seeds 7,12,14 --jobs 4
```
Per episode it prints success, steps, whether the robot fell and the
termination reason, and saves `frames/seed<N>_last.png` for failed episodes.
Records go to `runs/<timestamp>.json`; the policy file is snapshotted next to it.

Seeds: any integer in `[0, 200)`. Each seed fixes the initial placement of
the objects. A reset costs 200 steps of budget. The evaluation seed block is reserved and refused by the server.
