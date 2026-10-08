You are controlling a simulated Unitree G1 humanoid robot, one command at a time.
Task: **{task}** — read `task.md` (goal, success criterion, observation keys).

Commands (run them with Bash from this directory; each one advances the
simulation and prints the new observation):
- `./act status` — current observation and how many commands you have used
- `./act look` — saves the robot's own camera images (head, left_wrist, right_wrist; 84x84) and prints their paths; view them with your file reader
- `./act move_hand left X Y Z [--steps N]` — move the LEFT (or right) wrist point to world coordinates X Y Z (metres) using inverse kinematics, then hold N steps (default 60 = 1.2 s). Targets beyond the arm's reach stop at the closest reachable point: walk closer first.
- `./act walk VX VY WZ --steps N` — base velocity command (forward m/s, left m/s, yaw rad/s) for N steps at 50 Hz, then stop. Speeds below 0.06 m/s are ignored by the controller; 0.15–0.3 m/s is a normal walk. The base wobbles; re-read positions after every command.
- `./act hold --steps N` — stand still for N steps
- `./act gripper left VALUE [--steps N]` — gripper 0 (open) .. 1 (closed)

Rules: at most {max_commands} commands in this episode; the episode also ends on
success, on a fall, or at the time limit. Start with `./act status`, then act.
When the command output says `EPISODE OVER`, stop and state whether it succeeded.
Do not modify any file except by running these commands. Inline `python -c` is refused by the
sandbox; you only need the commands above.
