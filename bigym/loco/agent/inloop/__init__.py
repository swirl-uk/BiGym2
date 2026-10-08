"""In-the-loop evaluation: the model itself is the policy.

Instead of writing a ``policy.py``, the model drives the robot one command at
a time through :mod:`bigym.loco.agent.inloop.act`, one agent session per
episode. :mod:`bigym.loco.agent.inloop.run_inloop` prepares an episode
directory per episode, resets the server's worker to the protocol seed, runs
the session and reads the outcome back from the server.
"""
