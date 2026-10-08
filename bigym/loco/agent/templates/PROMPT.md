You are writing the control policy for a simulated Unitree G1 humanoid robot.
Your submission is `policy.py` in this directory. `docs/api.md` describes the
observation, action and tool interface and how to run episodes.

What you have
- {video_line}
- Training seeds: the {n_seeds} seeds listed in `seeds.json`, the seeds the demonstration
  set was collected on. Object placements are drawn per seed; the hidden evaluation
  seeds draw from the same distribution.
- A budget of {budget} environment steps for everything you run. Each reset
  costs 200 steps plus the steps of the episode.

How you are scored
- After you stop, `policy.py` is run once on 100 hidden seeds. Your score is the
  success rate over those 100 episodes; nothing you report is used. An episode in
  which the robot falls counts as failed.
- Episodes are re-run and must end the same way: no wall-clock time,
  no unseeded randomness.

Rules
- The policy is hand-written control logic. It may not contain learned
  components (no training a network, regression or lookup table from data).
  Take whatever you can from the demonstration video; tuning constants by
  running episodes is fine.
- Libraries: numpy, scipy, OpenCV (cv2), pillow and the Python standard library;
  nothing else is installed and nothing can be installed. No network, no changes
  to the harness. The simulator is reachable only through the client.
- The robot's own cameras (`head`, `left_wrist`, `right_wrist`) and its body
  state are the only view of the scene, for you while developing as much as
  for the policy. There is no outside camera.
- Write code to files and run them with `./python file.py`; inline `python -c '...'`
  is refused by the sandbox and only wastes a turn.

Task: {task_sentence}
