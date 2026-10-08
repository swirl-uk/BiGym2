
**Information tier: cameras only.** In this run the observation does NOT
include object positions (`target_pos`, `plate_pos`, rack slots are withheld).
You get your own body state (base pose, hand positions, gripper state) as
numbers, and you can look at the scene with `./act look`. Estimate where the
objects are from the images and from how your hand appears relative to them.
`./act look` saves the robot's own three 84x84 cameras (`head`, looking forward
and down; `left_wrist`; `right_wrist`); there is no outside camera. World frame: x forward
from the robot's starting pose, y to its left, z up; your hand positions are
reported in this frame, so use them as visual references.
