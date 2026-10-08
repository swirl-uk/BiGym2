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
- Velocity commands are followed with lag and some wobble. Read `obs["base_pos"]` / `obs["base_yaw"]` every step and close the loop; do not dead-reckon.
- A planar speed below 0.055 m/s is treated as "stand still". Use 0 or at least 0.06.
- Standing still, the base drifts a few centimetres. Heights above ~0.8 m lock the knees; squatting (lower height) is how you reach low objects. While squatting the controller does not step; walk first, then squat in place.
- Arm targets are absolute joint angles tracked by position control; large jumps are followed over a few tenths of a second. Your commands are applied as given: nothing between you and the robot smooths or rate-limits them.

## Observation: dict returned by reset/step

| key | shape | meaning |
|---|---|---|
| `t` | int | steps since reset |
| `time_limit` | int | max steps in the episode |
| `base_pos` | (3,) | pelvis position, world frame, m |
| `base_yaw` | float | pelvis heading, rad (0 = +x axis) |
| `base_quat` | (4,) | pelvis orientation (w, x, y, z) |
| `left_arm_qpos`, `right_arm_qpos` | (7,) | measured arm joint angles, rad, same order as the action |
| `left_hand_pos`, `right_hand_pos` | (3,) | wrist site positions, world frame, m (forward kinematics of the base pose and arm joints above; the point `ik` targets and the reach tasks measure) |
| `left_gripper`, `right_gripper` | float | measured gripper opening (0 open .. 1 closed) |
| `fell` | bool | the controller reported a fall at some point in this episode (latched; the episode continues) |
| `low_dim_obs` | (50,) | the raw proprioceptive vector used by learned policies (joint pos/vel, grippers, base) |

Everything in this table is about the robot's own body. The wrist positions,
the camera poses in `tools.camera_info()` and the `tools.ik` solver are all
computed from the base pose and joint angles with the robot's own kinematic
model; none of them carries information about the objects in the scene.

Task-specific keys are listed in `docs/task.md`.

## Tools available inside the policy

- `tools.hold_action()` -> 20 floats: zero base velocity, current height, current arm and gripper targets. Start from this and overwrite what you need.
- `tools.ik(left_pos=None, left_quat=None, right_pos=None, right_quat=None)` -> 14 floats (left arm 7, right arm 7). Targets are world-frame wrist-site positions (m); orientation is free unless you pass a (w, x, y, z) quaternion. A side whose position you leave as `None` keeps its current wrist position. One call converges on a reachable target (a few ms); the arm then needs a few tenths of a second to physically track the result. Write the result into `raw[4:18]`. Targets outside the arm's reach come back as the closest reachable configuration.
- `tools.image(camera, width=84, height=84)` -> uint8 array (height, width, 3).
  Cameras: `head`, `left_wrist`, `right_wrist`, the three cameras mounted on the
  robot (the 84x84 views a learned policy gets; this is also the largest size you
  may request, larger requests are refused). There is no other camera: there is no
  external view of the robot. Rendering costs no budget but takes a few milliseconds.
- `tools.camera_info()` -> for each of the three cameras: `fovy_deg` (vertical field
  of view), `pos` (world position) and `rot` (3x3 world rotation; the camera looks
  along its -z axis, x right, y up), computed from the robot's own kinematics, i.e.
  what a calibrated camera on a real robot would give you.
- `tools.pixel_to_ray(camera, u, v, width, height)` -> `(origin, direction)`: the world-frame ray
  through pixel (u, v) of an image you requested at `width` x `height` from that camera (pixel
  indices 0-based from the top-left, a pixel's centre is used). This is `camera_info` applied for
  you, the equivalent of ROS `projectPixelTo3dRay` in the world frame; intersecting rays or a ray
  with a plane is up to you.
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
