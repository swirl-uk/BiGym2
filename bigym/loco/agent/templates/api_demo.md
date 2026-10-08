
## The demonstration

{n_episodes_text} of the task, recorded from the robot's own cameras
({files}; {width}x{height}, {fps} fps; the control loop ran at 50 Hz, so frame k was
recorded at control step {every}k). The files are synchronised: frame k of every camera
is the same instant. `harness/demo.py` reads them:

```python
from harness.demo import Demo
d = Demo()
d.cameras, d.n_frames            # the camera names, frames per camera
im = d.frame("head", k)          # uint8 RGB (height, width, 3), same layout as tools.image()
ims = d.frames("left_wrist")     # uint8 RGB (n_frames, height, width, 3)
d.step_of(k)                     # control step at which frame k was recorded
d.frame_at_step("head", 300)     # frame closest to control step 300
```

This is development material: the demonstration and `harness/demo.py` are not available to
`policy.py` at evaluation, and the policy must not read them.
