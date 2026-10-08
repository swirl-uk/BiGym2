"""Robot configs, and ``ROBOT_MODELS``: ``EnvConfig.robot_model`` name -> robot class."""

from __future__ import annotations

from bigym.robots.configs.g1 import G1Dex1
from bigym.robots.robot import Robot

ROBOT_MODELS: dict[str, type[Robot]] = {"g1_dex1": G1Dex1}
